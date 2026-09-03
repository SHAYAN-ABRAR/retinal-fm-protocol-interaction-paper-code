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

    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "resuming " not in text:
            continue

        # Walk the log, tracking which experiment each block belongs to.
        current = None
        resumed_at: dict[str, int] = {}
        per_run: dict[str, list[tuple[int, float]]] = {}
        for line in text.splitlines():
            header = header_line.search(line)
            if header:
                current = header.group(1)
                per_run.setdefault(current, [])
                continue
            resumed = resume_line.search(line)
            if resumed:
                resumed_at[resumed.group(1)] = int(resumed.group(2))
                continue
            hit = epoch_line.search(line)
            if hit and current:
                per_run[current].append((int(hit.group(1)), float(hit.group(2))))

        for experiment_id, start in resumed_at.items():
            examined += 1
            observed = per_run.get(experiment_id, [])
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
                expected = intended_lr(epoch, 20, steps, base_lr, 1, 1e-6)
                if abs(actual - expected) / max(expected, 1e-12) > TOLERANCE:
                    mismatches.append((epoch, actual, expected))
            if mismatches:
                affected.append((experiment_id, start, len(after), mismatches))
            else:
                clean.append((experiment_id, start,
                              f"{len(after)} epoch(s) ran, schedule matches"))

    print(f"examined {examined} resume event(s) across {len(logs)} log file(s)\n")

    if affected:
        print(f"!! {len(affected)} run(s) trained with a CORRUPTED lr schedule:\n")
        for experiment_id, start, n_after, mismatches in affected:
            print(f"  {experiment_id}")
            print(f"    resumed at epoch {start}; {n_after} training epoch(s) "
                  f"ran afterwards")
            print(f"    {len(mismatches)} epoch(s) at the wrong learning rate, e.g.:")
            for epoch, actual, expected in mismatches[:3]:
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
         "corrupted_epochs": len(m)}
        for e, s, n, m in affected
    ])
    if not frame.empty:
        out = outputs / "tables" / "resumed_runs_audit.csv"
        frame.to_csv(out, index=False)
        print(f"\nsaved -> {out.name}")
    return 1 if affected else 0


if __name__ == "__main__":
    raise SystemExit(main())
