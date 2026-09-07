"""The five-seed source-validation table for a full fine-tuning experiment.

`outputs/tables/full_finetune_source_validation.csv` feeds the DDR report and,
through `paper/check_report_numbers.py`, the manuscript -- but it had no
generator in the repository. It was produced ad hoc during the DDR analysis, so
nothing could re-derive it, and nothing would have noticed if it drifted from
the histories it summarises. That is the same gap that let a stale
`protocol_interaction.csv` sit in the tree.

This is that missing generator. On DDR it is a **verifier**: it recomputes the
table from the run histories and compares it against the committed file,
refusing to overwrite it. On APTOS it produces the equivalent table for the
replication.

**Source validation only, and structurally so.** Every column below comes from
a run's own training history or its planned budget. The registry's `test_*`
columns are dropped the moment it is loaded, so this script cannot read a target
metric even by accident -- which matters while the APTOS target is still blind.

Columns match the committed DDR file exactly:

    model, seed, epochs_run, early_stopped, best_epoch, best_val_qwk,
    final_train_loss, val_loss_min, val_loss_final, overfit_ratio,
    peak_vram_gb, runtime_h

Usage:
    python analyse_ft_source_validation.py                 # DDR: verify
    python analyse_ft_source_validation.py --target aptos  # APTOS: generate
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

SOURCES = {"ddr": "aptos-eyepacs-idrid", "aptos": "ddr-eyepacs-idrid"}
OUTPUT = {"ddr": "full_finetune_source_validation.csv",
          "aptos": "aptos_full_finetune_source_validation.csv"}
SEEDS = [42, 1, 2, 3, 4]
# Label order and spelling are the committed file's, not a fresh choice: this
# has to reproduce it, not improve on it.
MODELS = [("ImageNet-MAE", "vit-large-mae-in1k"), ("RETFound", "retfound-cfp")]
TAG = "erm-b16-ftfull-lr0.0001"
BUDGET = 20


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    arguments = sys.argv[1:]
    target = "ddr"
    if "--target" in arguments:
        target = arguments[arguments.index("--target") + 1].lower()
    if target not in SOURCES:
        print(f"unknown target {target!r}; expected one of {sorted(SOURCES)}")
        return 1

    outputs = project_root() / "outputs"
    tables = outputs / "tables"

    registry = pd.read_csv(outputs / "experiment_registry.csv")
    # Target blindness, enforced rather than promised.
    registry = registry[[c for c in registry.columns if not c.startswith("test_")]]
    from src.utils.registry import latest_per_experiment
    registry = latest_per_experiment(registry)
    planned = {str(r.experiment_id): r for _, r in registry.iterrows()}

    rows, missing = [], []
    for label, backbone in MODELS:
        for seed in SEEDS:
            experiment_id = (
                f"lodo_{SOURCES[target]}__{target}_{backbone}_{TAG}_s{seed}")
            path = outputs / "logs" / f"{experiment_id}_history.csv"
            record = planned.get(experiment_id)
            # A history file appears while a run is still training, so its
            # presence is not evidence that the run finished. Only a COMPLETE
            # registry row is, and the two must agree on how many epochs ran --
            # otherwise this would quietly summarise a partial execution as a
            # finished one.
            if not path.exists() or record is None:
                missing.append(f"{experiment_id}  (no "
                               f"{'history' if record is not None else 'registry row'})")
                continue
            if str(record["status"]) != "COMPLETE":
                missing.append(f"{experiment_id}  (status {record['status']})")
                continue
            history = pd.read_csv(path)
            if len(history) != int(record["epochs_run"]):
                missing.append(f"{experiment_id}  (history has {len(history)} "
                               f"epochs, registry recorded "
                               f"{int(record['epochs_run'])})")
                continue
            budget = int(record["epochs_planned"])

            best_index = int(history.val_qwk.idxmax())
            val_loss_min = float(history.val_loss.min())
            val_loss_final = float(history.val_loss.iloc[-1])
            rows.append({
                "model": label,
                "seed": seed,
                "epochs_run": len(history),
                "early_stopped": len(history) < budget,
                "best_epoch": best_index,
                "best_val_qwk": float(history.val_qwk.iloc[best_index]),
                "final_train_loss": float(history.train_loss.iloc[-1]),
                "val_loss_min": val_loss_min,
                "val_loss_final": val_loss_final,
                "overfit_ratio": val_loss_final / val_loss_min,
                "peak_vram_gb": float(history.peak_vram_gb.max()),
                # The committed DDR file sums the per-epoch wall clock rather
                # than taking the registry's train_seconds, which also counts
                # setup. Matched here so the two agree.
                "runtime_h": float(history.seconds.sum()) / 3600.0,
            })

    if missing:
        print(f"NOT RUN -- {len(missing)} {target} full-FT run(s) have no history:")
        for experiment_id in missing:
            print(f"  {experiment_id}")
        return 1

    frame = pd.DataFrame(rows)
    destination = tables / OUTPUT[target]

    # -------------------------------------------------------------- report
    print(f"Source-validation summary, {target.upper()} full fine-tuning, "
          f"{len(frame)} runs.")
    print("The held-out target is not consulted anywhere in this table.\n")
    print(frame.round(4).to_string(index=False))

    print()
    for label, _ in MODELS:
        part = frame[frame.model == label]
        print(f"  {label:<13} best epoch {part.best_epoch.mean():.1f} "
              f"+/- {part.best_epoch.std(ddof=1):.1f}   "
              f"early stopped {int(part.early_stopped.sum())}/{len(part)}   "
              f"overfit ratio {part.overfit_ratio.mean():.2f} "
              f"+/- {part.overfit_ratio.std(ddof=1):.2f}")
    total = frame.runtime_h.sum()
    per_model = "  ".join(
        f"{label} {frame[frame.model == label].runtime_h.sum():.1f} h"
        for label, _ in MODELS)
    print(f"\n  total GPU runtime {total:.1f} h   ({per_model})")

    # ------------------------------------------------------------- persist
    if destination.exists():
        committed = pd.read_csv(destination)
        if list(committed.columns) != list(frame.columns):
            print(f"\n!! STOP -- {destination.name} has different columns")
            return 1
        merged = committed.merge(frame, on=["model", "seed"],
                                 suffixes=("_committed", "_recomputed"))
        if len(merged) != len(frame):
            print(f"\n!! STOP -- {destination.name} does not cover the same runs")
            return 1
        worst, culprit = 0.0, ""
        for column in frame.columns:
            if column in ("model", "seed"):
                continue
            left = merged[f"{column}_committed"]
            right = merged[f"{column}_recomputed"]
            if left.dtype == bool or right.dtype == bool:
                if not (left.astype(bool) == right.astype(bool)).all():
                    print(f"\n!! STOP -- {column} disagrees with the committed file")
                    return 1
                continue
            difference = float((left.astype(float) - right.astype(float)).abs().max())
            if difference > worst:
                worst, culprit = difference, column
        if worst > 1e-9:
            print(f"\n!! STOP -- {destination.name} no longer matches the "
                  f"histories it summarises (worst: {culprit}, {worst:.3e}).")
            print("   The committed file is NOT overwritten. Investigate before "
                  "trusting\n   any number derived from it.")
            return 1
        print(f"\n  {destination.name} verified against the run histories "
              f"(worst drift {worst:.1e}); left unchanged.")
        return 0

    frame.to_csv(destination, index=False)
    print(f"\nsaved -> {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
