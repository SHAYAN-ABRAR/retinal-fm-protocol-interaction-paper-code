"""Restore summary rows that a lossy merge key silently overwrote.

What happened
-------------
``lodo_results.csv`` was merged on a key that did not include ``batch_size``,
and the row's ``method`` column holds ``"erm"`` rather than the experiment tag
``erm-b16``. The Q1 batch-size controls therefore matched the existing batch-32
rows on every key field and replaced them. The ``-dbal`` rows survived only
because ``domain_balanced`` happens to be in the key.

This is the fourth occurrence of one bug: a setting that distinguishes two runs
is present in the experiment id but absent from the key that decides whether a
row is new. It cost ``in_domain_results.csv`` a 224 px row (resolution), the
LODO matrix its DenseNet seeds (backbone), and three results under ``-dbal``,
``-a1170`` and ``-tb4-lr0.0001``.

Nothing was lost. The registry is append-only, and predictions and reports are
filed under the full experiment id, which carries the batch tag. This script
rebuilds the missing summary rows from those reports.

What it will not do
-------------------
Existing rows are never touched -- not rewritten, not reordered in value, not
recomputed. A run already present is left exactly as it is, even if the report
disagrees, because silently rewriting a recorded result is the failure this
project guards hardest against. Disagreements are printed and left for a human.

A missing report is a hard stop rather than a skipped row: a summary table that
looks complete but is not is worse than one that is visibly broken.

Usage:
    python repair_lodo_results.py --dry-run
    python repair_lodo_results.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

TABLE = "lodo_results.csv"


def main() -> int:
    import json

    import pandas as pd

    from audit_consistency import specifications
    from src.utils.io import project_root
    from src.utils.registry import latest_per_experiment

    dry_run = "--dry-run" in sys.argv[1:]

    outputs = project_root() / "outputs"
    registry_path = outputs / "experiment_registry.csv"
    # The audit's "file" field is a glob, and it matters: run_lodo.py scopes
    # this table by resolution, so the 512 px runs live in lodo_results_r512.csv.
    # Reading only lodo_results.csv reports all twelve of them as missing and
    # would append them to the 224 px table -- duplicating rows that already
    # exist elsewhere and reintroducing the resolution pooling that the file
    # split exists to prevent.
    table_paths = sorted((outputs / "tables").glob("lodo_results*.csv"))
    if not table_paths or not registry_path.exists():
        print("NOT RUN -- registry or summary table missing")
        return 1

    # Reuse the audit's own id reconstruction. Duplicating the tag rules here is
    # how the audit and the repair would drift apart and disagree about which
    # runs exist.
    spec = next(s for s in specifications() if s["file"].startswith("lodo_results"))
    build_id = spec["experiment_id"]
    in_scope = spec["scope"]

    tables_by_path = {p: pd.read_csv(p) for p in table_paths}
    registry = latest_per_experiment(pd.read_csv(registry_path))

    covered = set()
    for path, frame in tables_by_path.items():
        for _, row in frame.iterrows():
            try:
                covered.add(build_id(row))
            except Exception as error:                  # noqa: BLE001
                print(f"!! cannot rebuild an id for a row of {path.name}: "
                      f"{error}")
                return 1
    total_rows = sum(len(f) for f in tables_by_path.values())

    completed = registry[(registry["protocol"] == "lodo")
                         & (registry["status"] == "COMPLETE")]
    missing = []
    for _, run in completed.iterrows():
        if not in_scope(run):
            continue
        if str(run["experiment_id"]) not in covered:
            missing.append(run)

    print(f"{total_rows} summary rows across "
          f"{', '.join(p.name for p in table_paths)}; "
          f"{len(completed)} completed LODO runs")
    if not missing:
        print("nothing to repair -- every completed run has a summary row")
        return 0

    print(f"{len(missing)} run(s) with no summary row:")
    for run in missing:
        print(f"  {run['experiment_id']}")

    rebuilt = []
    for run in missing:
        experiment_id = str(run["experiment_id"])
        report_path = outputs / "reports" / f"{experiment_id}_evaluation.json"
        if not report_path.exists():
            print(f"\n!! STOP -- no report for {experiment_id}. The row cannot "
                  f"be rebuilt from an authoritative source, and inventing one "
                  f"is not an option.")
            return 1
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        source = payload["results"]["source_val"]
        target = payload["results"]["target_test"]
        rebuilt.append({
            "target": run["target_domain"],
            "method": run["method"],
            "seed": int(run["seed"]),
            "backbone": run["backbone"],
            "image_size": int(run["image_size"]),
            "batch_size": int(run["batch_size"]),
            "n_train": int(run["n_train"]),
            "n_test": int(run["n_test"]),
            "source_qwk": source["metrics"]["qwk"],
            "target_qwk": target["metrics"]["qwk"],
            "source_f1": source["metrics"]["f1_macro"],
            "target_f1": target["metrics"]["f1_macro"],
            "source_ece": source["calibration"]["ece"],
            "target_ece": target["calibration"]["ece"],
            "target_ece_scaled": target["calibration_scaled"]["ece"],
            "target_severe": target["metrics"]["severe_error_rate"],
            "temperature": payload["temperature"]["temperature"],
            "seconds": run.get("train_seconds", float("nan")),
            "domain_balanced": bool(run.get("domain_balanced", False)),
            "irm_anneal_iters": run.get("irm_anneal_iters", 500),
            # -1, not NaN: this column is part of the merge key, and NaN never
            # equals NaN.
            "trainable_blocks": (-1 if pd.isna(run.get("trainable_blocks"))
                                 else int(run["trainable_blocks"])),
            "learning_rate": run.get("learning_rate", 3e-4),
        })

    additions = pd.DataFrame(rebuilt)
    print("\nrebuilt from the evaluation reports:")
    print(additions[["target", "seed", "batch_size", "target_qwk",
                     "target_f1", "target_severe"]].to_string(index=False))

    # Every rebuilt id must differ from every id already present. If two ids
    # collide the key is still lossy and appending would repeat the bug.
    rebuilt_ids = {build_id(row) for _, row in additions.iterrows()}
    clash = rebuilt_ids & covered
    if clash:
        print(f"\n!! STOP -- rebuilt ids already present: {sorted(clash)}")
        return 1

    # Route each restored row to the file its configuration belongs in, by the
    # same rule run_lodo.py uses to choose one.
    def destination(image_size: int):
        suffix = "" if int(image_size) == 224 else f"_r{int(image_size)}"
        return outputs / "tables" / f"lodo_results{suffix}.csv"

    additions["_destination"] = additions["image_size"].map(destination)
    for path in additions["_destination"].unique():
        if path not in tables_by_path:
            print(f"\n!! STOP -- {path.name} does not exist; a restored row "
                  f"belongs to a table this repair did not read.")
            return 1

    if dry_run:
        for path, group in additions.groupby("_destination"):
            print(f"\ndry run -- {len(group)} row(s) would be appended to "
                  f"{path.name}; nothing written")
        return 0

    for path, group in additions.groupby("_destination"):
        before = tables_by_path[path]
        merged = pd.concat([before, group.drop(columns=["_destination"])],
                           ignore_index=True)
        merged = merged.sort_values(
            ["backbone", "target", "seed", "image_size", "batch_size"]
        ).reset_index(drop=True)
        merged.to_csv(path, index=False)
        print(f"\n{path.name}: {len(before)} -> {len(merged)} rows "
              f"({len(group)} restored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
