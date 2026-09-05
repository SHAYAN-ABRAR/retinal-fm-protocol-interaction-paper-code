"""What is safely removable, in the order it is safe to remove it.

APTOS full fine-tuning needs roughly 12 GB of retained checkpoints plus room
for a live ``last.pt`` and the temporary file every atomic write creates. There
is not enough free space, so something has to go -- and the failure mode to
avoid is deleting a scientific artifact to force a run through.

Categories, safest first. Only the first four are ever proposed:

1. ``last.pt`` from runs that are COMPLETE and pass every pruning guard. The
   resume point of a finished run can never be used again.
2. checkpoint directories from engineering-only smoke tests, identified by the
   ``-e2`` epoch stamp in the experiment id.
3. checkpoints belonging to SUPERSEDED runs -- a later run of the same
   configuration is the authoritative one.
4. checkpoints belonging to DIVERGED runs, which produced no result.

Never proposed, and listed separately so their size is visible:

* ``best_qwk.pt`` for a current authoritative run -- the reproducible model;
* predictions, histories, reports, registry, result CSVs, provenance files.

Dry run only. It deletes nothing; it prints what could be freed and by which
existing tool.

Usage:
    python audit_storage.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


def gb(n: int) -> float:
    return n / 1e9


def tree_size(path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    import shutil

    import pandas as pd

    from src.utils.io import project_root
    from src.utils.registry import latest_per_experiment

    outputs = project_root() / "outputs"
    root = project_root()

    total, used, free = shutil.disk_usage(root.anchor)
    print(f"disk: {gb(free):.1f} GB free of {gb(total):.1f} GB\n")

    registry = latest_per_experiment(pd.read_csv(outputs / "experiment_registry.csv"))
    status_by_id = dict(zip(registry["experiment_id"].astype(str),
                            registry["status"].astype(str)))

    raw = pd.read_csv(outputs / "experiment_registry.csv")
    superseded_ids = set(raw[raw.status == "SUPERSEDED"]["experiment_id"].astype(str))
    # A superseded row whose id was later re-run is NOT removable: the current
    # row under that id is authoritative.
    superseded_ids -= {k for k, v in status_by_id.items() if v == "COMPLETE"}

    checkpoints = outputs / "checkpoints"
    categories = {
        "1. last.pt from COMPLETE runs": [],
        "2. smoke-test checkpoint dirs": [],
        "3. SUPERSEDED run checkpoints": [],
        "4. DIVERGED run checkpoints": [],
    }
    protected = []

    for directory in sorted(checkpoints.iterdir()) if checkpoints.is_dir() else []:
        if not directory.is_dir():
            continue
        experiment_id = directory.name
        status = status_by_id.get(experiment_id, "ABSENT")
        size = tree_size(directory)

        if "-e2_" in experiment_id:
            categories["2. smoke-test checkpoint dirs"].append(
                (experiment_id, size, "engineering validation only"))
            continue
        if status == "DIVERGED":
            categories["4. DIVERGED run checkpoints"].append(
                (experiment_id, size, "run produced no result"))
            continue
        if experiment_id in superseded_ids:
            categories["3. SUPERSEDED run checkpoints"].append(
                (experiment_id, size, "a later run of this id is authoritative"))
            continue

        last = directory / "last.pt"
        best = directory / "best_qwk.pt"
        if status == "COMPLETE" and last.exists():
            categories["1. last.pt from COMPLETE runs"].append(
                (experiment_id, last.stat().st_size, "resume point of a finished run"))
        if best.exists():
            protected.append((experiment_id, best.stat().st_size, status))

    recoverable = 0
    for name, entries in categories.items():
        subtotal = sum(size for _, size, _ in entries)
        recoverable += subtotal
        print(f"{name}: {len(entries)} item(s), {gb(subtotal):.2f} GB")
        for experiment_id, size, why in sorted(entries, key=lambda e: -e[1])[:8]:
            print(f"    {gb(size):6.2f} GB  {experiment_id[:66]}")
            print(f"               {why}")
        if len(entries) > 8:
            print(f"    ... and {len(entries) - 8} more")
        print()

    protected_total = sum(size for _, size, _ in protected)
    print(f"PROTECTED -- authoritative best_qwk.pt: {len(protected)} file(s), "
          f"{gb(protected_total):.2f} GB. Never proposed.")

    for name in ("predictions", "reports", "logs", "tables", "features",
                 "embeddings", "figures"):
        directory = outputs / name
        if directory.is_dir():
            print(f"PROTECTED -- outputs/{name}: {gb(tree_size(directory)):.2f} GB")

    print(f"\nrecoverable by the four safe categories: {gb(recoverable):.2f} GB")
    print(f"free afterwards would be approximately: "
          f"{gb(free + recoverable):.1f} GB")
    print("\nDRY RUN -- nothing deleted. Category 1 is handled by "
          "prune_checkpoints.py --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
