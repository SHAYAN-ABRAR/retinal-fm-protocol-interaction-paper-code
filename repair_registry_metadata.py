"""Normalise blank trainable_blocks in the registry to the documented -1.

``KEY_COLUMN_DEFAULTS`` records the convention: "-1 encodes all blocks; NaN is
not a key". ``trainable_blocks`` is part of the dedup key for both the registry
and the summary tables, and NaN never compares equal to NaN -- two runs
differing only in a blank key column merge into one row and the second silently
replaces the first. That is the bug this project has now hit four times.

The defaults dictionary backfills blanks whenever a table is merged, so the
convention held in practice while the stored value did not. ``run_lodo.py``
wrote None for every full-network run, and the Q1 batch-size controls added six
such rows. ``recorded_trainable_blocks()`` fixes the source; this fixes the rows
already written.

Why this is a normalisation and not an edit to a result
------------------------------------------------------
-1 and blank already mean the same thing here -- the whole network trained --
and the reader that matters, ``merge_results_table``, already treats them as
identical. No metric, no seed, no configuration changes. The script proves it:
it compares every other column before and after and refuses to write if
anything but ``trainable_blocks`` moved.

Rows whose recorded adaptation_mode is not a full-network run are never touched,
so a genuinely missing block count is reported rather than filled in.

Usage:
    python repair_registry_metadata.py --dry-run
    python repair_registry_metadata.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

FULL_NETWORK_MODES = {"full_network", "full_finetune"}


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    dry_run = "--dry-run" in sys.argv[1:]
    path = project_root() / "outputs" / "experiment_registry.csv"
    if not path.exists():
        print("NOT RUN -- registry missing")
        return 1

    before = pd.read_csv(path)
    if "trainable_blocks" not in before.columns:
        print("NOT RUN -- registry has no trainable_blocks column")
        return 1

    blank = before["trainable_blocks"].isna()
    if not blank.any():
        print("nothing to repair -- no blank trainable_blocks")
        return 0

    mode = before.get("adaptation_mode")
    if mode is None:
        print("NOT RUN -- registry has no adaptation_mode column, so a blank "
              "cannot be shown to mean 'whole network'")
        return 1

    fixable = blank & mode.isin(FULL_NETWORK_MODES)
    unexplained = blank & ~mode.isin(FULL_NETWORK_MODES)
    if unexplained.any():
        print(f"!! {int(unexplained.sum())} blank row(s) whose adaptation_mode "
              f"is not a full-network run -- left alone, they need a human:")
        for experiment_id in before.loc[unexplained, "experiment_id"]:
            print(f"  {experiment_id}")

    print(f"{int(fixable.sum())} row(s) to normalise to -1:")
    for experiment_id in before.loc[fixable, "experiment_id"]:
        print(f"  {experiment_id}")
    if not fixable.any():
        return 0

    after = before.copy()
    after.loc[fixable, "trainable_blocks"] = -1

    # Nothing but trainable_blocks may have moved.
    for column in before.columns:
        if column == "trainable_blocks":
            continue
        left, right = before[column], after[column]
        same = (left.eq(right) | (left.isna() & right.isna())).all()
        if not same:
            print(f"\n!! STOP -- column {column!r} changed; refusing to write")
            return 1

    if dry_run:
        print(f"\ndry run -- {int(fixable.sum())} row(s) would change; "
              f"nothing written")
        return 0

    after.to_csv(path, index=False)
    print(f"\nregistry updated: {int(fixable.sum())} row(s) normalised, "
          f"{len(after)} rows total, no other column touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
