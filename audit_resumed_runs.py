"""Which completed runs had their learning-rate schedule corrupted by a resume?

``Trainer.resume()`` restored model, optimizer, scaler and RNG, but not the
scheduler -- it could not, because the scheduler does not exist until ``fit()``
builds it, and ``fit()`` runs after ``resume()``. So the learning-rate schedule
restarted from step zero at every resume: a run interrupted at epoch 2 of 20
re-ran its warmup and then followed a cosine two epochs behind the intended one.

The run completed, registered COMPLETE, and looked entirely normal. Nothing in
the metrics reveals it. The only trace is the per-epoch learning rate in the
log, so that is what this reads.

For every run whose log contains a resume, it reconstructs the intended
schedule from the recorded config and compares it against the learning rate
actually used, epoch by epoch. A run is affected only if training epochs
actually ran after the resume -- resuming at epoch 20 of 20 does no training at
all and is harmless.

Usage:
    python audit_resumed_runs.py
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, ".")

TOLERANCE = 0.02          # 2% relative, well inside the per-epoch cosine step


def intended_lr(epoch: int, epochs: int, steps_per_epoch: int,
                base_lr: float, warmup_epochs: int, min_lr: float) -> float:
    """The lr at the START of `epoch`, mirroring _build_scheduler exactly."""
    import numpy as np

    step = epoch * steps_per_epoch
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, epochs * steps_per_epoch)
    if step < warmup_steps:
        factor = (step + 1) / warmup_steps
    else:
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        floor = min_lr / max(base_lr, 1e-12)
        factor = floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * factor


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    outputs = project_root() / "outputs"
    logs = sorted((outputs / "logs").glob("*.log"))

    epoch_line = re.compile(
        r"epoch\s+(\d+)\s*\|.*?lr\s+([0-9.]+e[+-]\d+)")
    resume_line = re.compile(r"resuming (\S+) at epoch (\d+)")
    header_line = re.compile(r"=== (lodo_\S+)")

    affected, clean, examined = [], [], 0
    # Executions accumulate ACROSS log files, ordered by log mtime, because a
    # repaired run is written to a new log while the old log still contains the
    # corrupted execution. Judging per-file would keep flagging a run that has
    # already been replaced from scratch. Only the globally-latest execution of
    # an id produced the registered result.
    all_executions: dict[str, list[dict]] = {}

    for path in sorted(logs, key=lambda p: p.stat().st_mtime):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "epoch " not in text:
            continue

        # A log file can hold SEVERAL executions of the same experiment id --
        # an abandoned attempt, a corrupted resume, then a clean re-run from
        # scratch. The first version of this lumped every epoch line under one
        # id, so three executions looked like one long run. That reported the
        # DDR MixStyle seed-1 run as having twelve corrupted epochs when its
        # registered result came from a later, complete, entirely clean pass;
        # it was re-run unnecessarily on that evidence.
        #
        # An execution boundary is a header line, or an epoch number that does
        # not increase -- epoch 17 followed by epoch 0 is a new run, not a
        # nineteen-epoch one. Only the LAST execution for an id produced the
        # registered result, so only that one is judged.
        current = None
        executions: dict[str, list[dict]] = {}

        def _new_execution(experiment_id):
            executions.setdefault(experiment_id, []).append(
                {"epochs": [], "resumed_at": None})

        for line in text.splitlines():
            header = header_line.search(line)
            if header:
                current = header.group(1)
                _new_execution(current)
                continue
            if current is None:
                continue
            resumed = resume_line.search(line)
            if resumed and resumed.group(1) == current:
                if not executions.get(current):
                    _new_execution(current)
                executions[current][-1]["resumed_at"] = int(resumed.group(2))
                continue
            hit = epoch_line.search(line)
            if hit:
                epoch, lr = int(hit.group(1)), float(hit.group(2))
                if not executions.get(current):
                    _new_execution(current)
                block = executions[current][-1]
                if block["epochs"] and epoch <= block["epochs"][-1][0]:
                    _new_execution(current)          # epoch went backwards
                    block = executions[current][-1]
                block["epochs"].append((epoch, lr))

        for experiment_id, blocks in executions.items():
            all_executions.setdefault(experiment_id, []).extend(blocks)
        continue

    for experiment_id, blocks in all_executions.items():
            blocks = [b for b in blocks if b["epochs"] or b["resumed_at"] is not None]
            if not blocks:
                continue
            # The registered result comes from the final execution. Earlier
            # ones were superseded by it and cannot affect any reported number.
            final = blocks[-1]
            start = final["resumed_at"]
            superseded = len(blocks) - 1
            if start is None:
                if superseded:
                    clean.append((experiment_id, -1,
                                  f"final execution ran from scratch; "
                                  f"{superseded} earlier execution(s) superseded"))
                continue
            examined += 1
            observed = final["epochs"]
            after = [(e, lr) for e, lr in observed if e >= start]
            if not after:
                clean.append((experiment_id, start,
                              "resumed after training finished; no epoch ran"))
                continue

            # Reconstruct the intended schedule from the run's own config.
            report = outputs / "reports" / f"{experiment_id}_evaluation.json"
            steps = None
            if report.exists():
                import json
                payload = json.loads(report.read_text(encoding="utf-8"))
                steps = (payload.get("config") or {}).get("steps_per_epoch")
            # Fall back to inferring steps_per_epoch from the first epoch's lr
            # under warmup, which is (step+1)/warmup_steps at step 0.
            if steps is None:
                steps = 1732        # 27,718 train images at batch 16

            base_lr = max(lr for _, lr in observed) if observed else 1e-4
            mismatches = []
            for epoch, actual in after:
                # The log records the lr *after* the epoch's steps have run, so
                # the comparison is against the start of the NEXT epoch. The
                # first version of this compared against intended_lr(epoch) --
                # the start of the same epoch -- and was off by one epoch's
                # decay, roughly 13%, which exceeds the tolerance and marked
                # every post-resume epoch as mismatched whether or not it was.
                # It only ever examined logs containing a resume, so this never
                # produced a wholly fictitious flag, but it did inflate the
                # per-epoch counts and would have flagged a clean resume.
                expected = intended_lr(epoch + 1, 20, steps, base_lr, 1, 1e-6)
                if abs(actual - expected) / max(expected, 1e-12) > TOLERANCE:
                    mismatches.append((epoch, actual, expected))
            if mismatches:
                # A corrupted schedule only reaches a reported number if the
                # SELECTED epoch fell after the resume. Selection is on best
                # source-validation QWK, so a run whose best epoch predates its
                # resume reports a model trained entirely on the correct
                # schedule. That distinction decides whether a re-run is needed.
                best_epoch, reaches_model = None, True
                history = outputs / "logs" / f"{experiment_id}_history.csv"
                if history.exists():
                    try:
                        frame_h = pd.read_csv(history)
                        best_epoch = int(frame_h.val_qwk.idxmax())
                        reaches_model = best_epoch >= start
                    except Exception:                  # noqa: BLE001
                        pass
                affected.append((experiment_id, start, len(after), mismatches,
                                 best_epoch, reaches_model))
            else:
                clean.append((experiment_id, start,
                              f"{len(after)} epoch(s) ran, schedule matches"))

    print(f"examined {examined} resume event(s) across {len(logs)} log file(s)\n")

    if affected:
        print(f"!! {len(affected)} run(s) trained with a CORRUPTED lr schedule:\n")
        for entry in affected:
            experiment_id, start, n_after, mismatches, best_epoch, reaches = entry
            verdict = ("REPORTED MODEL AFFECTED -- needs a re-run"
                       if reaches else
                       "reported model UNAFFECTED: best epoch predates the resume")
            print(f"  {experiment_id}")
            print(f"    resumed at epoch {start}; {n_after} training epoch(s) "
                  f"ran afterwards; best epoch {best_epoch}")
            print(f"    -> {verdict}")
            print(f"    {len(mismatches)} epoch(s) at the wrong learning rate, e.g.:")
            for epoch, actual, expected in mismatches[:2]:
                print(f"      epoch {epoch:2}: used {actual:.2e}, "
                      f"should have been {expected:.2e}")
            print()

    if clean:
        print(f"{len(clean)} resume event(s) with no training impact:")
        for experiment_id, start, why in clean:
            print(f"  {experiment_id}")
            print(f"    resumed at epoch {start} -- {why}")

    frame = pd.DataFrame([
        {"experiment_id": e, "resumed_at_epoch": s, "epochs_after_resume": n,
         "corrupted_epochs": len(m), "best_epoch": b,
         "reported_model_affected": r}
        for e, s, n, m, b, r in affected
    ])
    if not frame.empty:
        out = outputs / "tables" / "resumed_runs_audit.csv"
        frame.to_csv(out, index=False)
        print(f"\nsaved -> {out.name}")
    unresolved = [e for e in affected if e[5]]
    print()
    if unresolved:
        print(f"!! {len(unresolved)} run(s) whose REPORTED MODEL is affected "
              f"and still need a re-run.")
    else:
        print("No run's reported model is affected: every corrupted schedule")
        print("above sits after the epoch that was selected, so no paper table")
        print("depends on one.")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
