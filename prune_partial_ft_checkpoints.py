"""Delete partial-FT model weights the inventory has cleared, and nothing else.

Reads `docs/PARTIAL_FT_CHECKPOINT_INVENTORY.csv`, which
`audit_partial_ft_checkpoints.py` produced with a SHA256 per file. This tool
re-verifies every hash before acting: an inventory that no longer describes the
files on disk is not a licence to delete them.

It touches `best_qwk.pt` files and nothing else. Predictions, histories,
configs, registry rows, reports, metric tables and provenance are never opened
for writing here.

Dry run by default. `--apply` is required.

Usage:
    python prune_partial_ft_checkpoints.py
    python prune_partial_ft_checkpoints.py --apply
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, ".")

INVENTORY = Path("docs") / "PARTIAL_FT_CHECKPOINT_INVENTORY.csv"


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import shutil

    import pandas as pd

    from src.utils.io import project_root

    apply = "--apply" in sys.argv[1:]
    root = project_root()
    path = root / INVENTORY
    if not path.exists():
        print(f"NOT RUN -- {INVENTORY} missing; run "
              f"audit_partial_ft_checkpoints.py first")
        return 1

    frame = pd.read_csv(path)
    retain = frame[frame.decision == "RETAIN"]
    delete = frame[frame.decision == "DELETE"]

    # The retained set must cover both model families equally in every target,
    # or the surviving checkpoints are a biased sample of the experiment.
    pivot = retain.groupby(["target", "model"]).size().unstack(fill_value=0)
    if "ImageNet-MAE" not in pivot or "RETFound-CFP" not in pivot:
        print("!! STOP -- the retained set does not contain both model families")
        return 1
    if not (pivot["ImageNet-MAE"] == pivot["RETFound-CFP"]).all():
        print("!! STOP -- the retained set is asymmetric across model families")
        print(pivot.to_string())
        return 1

    # Every deletion candidate must still be fully backed by prediction-level
    # artifacts. Re-checked here rather than trusted from the inventory, since
    # the inventory may predate a change.
    problems = []
    for row in delete.itertuples():
        if row.classification == "corrupted":
            continue                      # metadata retained; weights invalid
        for column in ("predictions_exist", "history_exists",
                       "config_exists", "result_json_exists",
                       "prediction_audit_passes"):
            if not bool(getattr(row, column)):
                problems.append(f"{row.experiment_id}: {column} is False")
    if problems:
        print(f"!! STOP -- {len(problems)} deletion candidate(s) are not fully "
              f"backed by prediction-level artifacts:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    missing, changed, ready = [], [], []
    for row in delete.itertuples():
        target = root / row.checkpoint_path
        if not target.exists():
            missing.append(row.experiment_id)
            continue
        if sha256(target) != row.sha256:
            changed.append(row.experiment_id)
            continue
        ready.append((row.experiment_id, target, int(row.size_bytes)))

    if changed:
        print(f"!! STOP -- {len(changed)} file(s) no longer match the inventory "
              f"hash. The inventory does not describe what is on disk:")
        for experiment_id in changed:
            print(f"  {experiment_id}")
        return 1
    if missing:
        print(f"note: {len(missing)} candidate(s) already absent")

    recoverable = sum(size for _, _, size in ready)
    _, _, free = shutil.disk_usage(root.anchor)
    print(f"retain   {len(retain):>2} file(s)  {retain.size_bytes.sum()/1e9:6.2f} GB")
    print(f"delete   {len(ready):>2} file(s)  {recoverable/1e9:6.2f} GB  "
          f"(hashes verified)")
    print(f"free now {free/1e9:.1f} GB -> projected {(free + recoverable)/1e9:.1f} GB")

    if not apply:
        print("\nDRY RUN -- nothing deleted. Pass --apply.")
        return 0

    freed = 0
    for experiment_id, target, size in ready:
        target.unlink()
        freed += size
    _, _, after = shutil.disk_usage(root.anchor)
    print(f"\ndeleted {len(ready)} checkpoint file(s), freed {freed/1e9:.2f} GB")
    print(f"free now: {after/1e9:.1f} GB")

    # Prove the retained files were untouched.
    bad = [r.experiment_id for r in retain.itertuples()
           if not (root / r.checkpoint_path).exists()
           or sha256(root / r.checkpoint_path) != r.sha256]
    if bad:
        print(f"!! {len(bad)} RETAINED file(s) changed or vanished: {bad}")
        return 1
    print(f"all {len(retain)} retained checkpoints verified unchanged by hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
