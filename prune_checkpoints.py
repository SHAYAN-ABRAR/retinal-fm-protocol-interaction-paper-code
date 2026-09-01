"""Delete resume checkpoints for runs that are provably finished.

Why this exists
---------------
A full fine-tuning run writes ``last.pt`` at 3.4 GB, and ten of them do not fit
on the disk available. ``last.pt`` exists so a crashed run can continue the same
trajectory; once a run has completed and been audited, it can never be resumed
and the file is dead weight. ``best_qwk.pt`` is the reproducible model artifact
and is never deleted here.

Why it is paranoid
------------------
Deleting a resume point from a run that is still going, or that failed, destroys
hours of GPU time with no way back. So a run is pruned only when **all seven**
conditions hold, checked in order and reported individually:

1. the registry's latest record for the id says COMPLETE;
2. an evaluation report exists;
3. target predictions exist;
4. the training history exists;
5. ``best_qwk.pt`` exists;
6. ``best_qwk.pt`` actually loads, and holds a model state dict;
7. the run appears in its summary table -- i.e. the consistency audit's own
   notion of a run that is accounted for.

Any single failure leaves the run untouched and says which condition failed.
A run whose ``last.pt`` is newer than its report is also skipped: that ordering
means training wrote after the evaluation did, which should be impossible and
is worth a human look rather than a deletion.

Default is a dry run. ``--apply`` is required to delete anything.

Usage:
    python prune_checkpoints.py                      # dry run, every candidate
    python prune_checkpoints.py --pattern ftfull     # only full-FT runs
    python prune_checkpoints.py --apply
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def main() -> int:
    import torch

    import pandas as pd

    from src.utils.io import project_root
    from src.utils.registry import latest_per_experiment

    arguments = sys.argv[1:]
    apply = "--apply" in arguments
    pattern = ""
    if "--pattern" in arguments:
        pattern = arguments[arguments.index("--pattern") + 1]

    outputs = project_root() / "outputs"
    registry_path = outputs / "experiment_registry.csv"
    if not registry_path.exists():
        print("NOT RUN -- registry missing")
        return 1

    registry = latest_per_experiment(pd.read_csv(registry_path))
    status_by_id = dict(zip(registry["experiment_id"].astype(str),
                            registry["status"].astype(str)))

    # Every run that appears in any summary table, by reconstructed id. A run
    # missing from its table is exactly the silent-overwrite symptom, and is
    # not something to delete a resume point over.
    from audit_consistency import specifications

    accounted: set[str] = set()
    for spec in specifications():
        for path in sorted((outputs / "tables").glob(spec["file"])):
            frame = pd.read_csv(path)
            for _, row in frame.iterrows():
                try:
                    accounted.add(spec["experiment_id"](row))
                except Exception:                       # noqa: BLE001
                    continue

    checkpoint_root = outputs / "checkpoints"
    if not checkpoint_root.is_dir():
        print("no checkpoint directory")
        return 0

    candidates, skipped = [], []
    for directory in sorted(checkpoint_root.iterdir()):
        if not directory.is_dir():
            continue
        experiment_id = directory.name
        if pattern and pattern not in experiment_id:
            continue
        last = directory / "last.pt"
        if not last.exists():
            continue

        best = directory / "best_qwk.pt"
        report = outputs / "reports" / f"{experiment_id}_evaluation.json"
        predictions = list((outputs / "predictions").glob(
            f"{experiment_id}__target_test*_predictions.csv"))
        # Written to outputs/logs/, not outputs/history/. The first version of
        # this looked in a directory that does not exist, so every run failed
        # the check -- safe, but a condition that can only fail verifies nothing.
        history = list((outputs / "logs").glob(f"{experiment_id}_history.csv"))

        def fail(reason: str) -> None:
            skipped.append((experiment_id, reason))

        if status_by_id.get(experiment_id) != "COMPLETE":
            fail(f"registry status is {status_by_id.get(experiment_id, 'ABSENT')}, "
                 f"not COMPLETE")
            continue
        if not report.exists():
            fail("no evaluation report")
            continue
        if not predictions:
            fail("no target predictions")
            continue
        if not history:
            fail("no training history")
            continue
        if not best.exists():
            fail("no best_qwk.pt -- deleting last.pt would leave no model at all")
            continue
        if last.stat().st_mtime > report.stat().st_mtime + 1:
            fail("last.pt is newer than the evaluation report; ordering is wrong")
            continue
        if experiment_id not in accounted:
            fail("not present in any summary table (possible silent overwrite)")
            continue
        try:
            payload = torch.load(best, map_location="meta", weights_only=False)
            model = payload.get("model")
            if not model:
                raise ValueError("no model state dict")
        except Exception as error:                      # noqa: BLE001
            fail(f"best_qwk.pt does not load ({type(error).__name__}: {error})")
            continue

        candidates.append((experiment_id, last, last.stat().st_size))

    total = sum(size for _, _, size in candidates)
    print(f"{len(candidates)} run(s) safe to prune, "
          f"{len(skipped)} skipped\n")
    for experiment_id, _, size in candidates:
        print(f"  PRUNE  {_gb(size):>9}  {experiment_id}")
    if skipped:
        print()
        for experiment_id, reason in skipped:
            print(f"  KEEP              {experiment_id}\n"
                  f"                    -> {reason}")

    if not candidates:
        print("\nnothing to do")
        return 0

    print(f"\ntotal recoverable: {_gb(total)}")
    if not apply:
        print("dry run -- nothing deleted. Pass --apply to delete.")
        return 0

    freed = 0
    for experiment_id, path, size in candidates:
        path.unlink()
        freed += size
        print(f"  deleted {path}  ({_gb(size)})")
    print(f"\nfreed {_gb(freed)} across {len(candidates)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
