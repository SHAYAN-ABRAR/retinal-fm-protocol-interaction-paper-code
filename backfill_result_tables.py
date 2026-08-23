"""Fill in what the summary tables record about runs that already finished.

The summary tables are derived; the registry, the evaluation reports and the
saved predictions are the record. Twice now the derived copy has drifted from
the record in ways that were invisible until something downstream broke, so
this reconciles two specific drifts and refuses to touch anything else.

1. One architecture under two labels
   ---------------------------------
   ``lodo_results.csv`` carries ConvNeXt seed 42 as ``convnext-tiny`` and seeds
   1 and 2 as ``convnext_tiny``. The registry calls all twelve runs
   ``convnext_tiny``; the hyphen came from the table being rebuilt by hand after
   the contamination incident, using the experiment-id spelling.

   Nothing is currently mis-scored -- the consistency audit rebuilds ids through
   make_experiment_id, which normalises the separator -- but ``backbone`` is part
   of the dedup key that merge_results_table uses. Under two spellings, re-running
   seed 42 would append a second row rather than replace the first, which is the
   same failure that once let ConvNeXt rows overwrite DenseNet ones and report an
   across-architecture SD as an across-seed SD. The registry's spelling wins.

2. A column no runner writes
   -------------------------
   Both LODO tables have a ``temperature`` column, and ``run_lodo.py`` has never
   populated it: it was added by the same hand-rebuild. So it held values only
   for the rows that rebuild touched, and arrived NaN for every run since --
   all four 512px rows and all eight new ConvNeXt rows. run_lodo.py now records
   it; this backfills the runs that finished before it did.

   The fitted temperature is read from each run's evaluation report, which is
   where it has been correct all along. An existing value is never overwritten:
   it is compared, and a disagreement is an error rather than a silent update.

Idempotent. Changes nothing that is already right. No GPU.

Usage:
    python backfill_result_tables.py                    # report only
    python backfill_result_tables.py --apply
    python backfill_result_tables.py --apply --tables lodo_results.csv
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

# Tables that describe one run per row and carry a temperature column.
TABLES = ["lodo_results.csv", "lodo_results_r512.csv"]
ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]

# The evaluation reports are written through write_json, which rounds to four
# decimal places; the hand-rebuilt rows kept full precision. So 1.896483737651588
# and 1.8965 are the same fitted temperature recorded at two precisions, and the
# tolerance has to admit that. It still refuses anything larger: two genuinely
# different fits differ in the first or second decimal, not the fifth.
TOLERANCE = 5e-4


def main() -> None:
    import pandas as pd

    from src.utils.io import project_root
    from src.utils.registry import make_experiment_id

    arguments = sys.argv[1:]
    apply_changes = "--apply" in arguments
    tables = TABLES
    if "--tables" in arguments:
        tables = arguments[arguments.index("--tables") + 1].split(",")

    outputs = project_root() / "outputs"
    registry = pd.read_csv(outputs / "experiment_registry.csv")
    backbone_by_id = dict(zip(registry["experiment_id"], registry["backbone"]))

    def method_tag(row) -> str:
        tag = f"{row['method']}-b{int(row['batch_size'])}"
        if int(row["image_size"]) != 224:
            tag += f"-r{int(row['image_size'])}"
        return tag

    def temperature_of(experiment_id: str):
        report = outputs / "reports" / f"{experiment_id}_evaluation.json"
        if not report.exists():
            return None
        payload = json.loads(report.read_text(encoding="utf-8"))
        block = payload.get("temperature")
        return None if not block else float(block["temperature"])

    total_changes = 0
    for name in tables:
        path = outputs / "tables" / name
        if not path.exists():
            print(f"{name}: not present -- skipped")
            continue

        frame = pd.read_csv(path)
        relabelled, filled, missing = 0, 0, []

        for index, row in frame.iterrows():
            experiment_id = make_experiment_id(
                protocol="lodo",
                sources=[d for d in ALL_DOMAINS if d != row["target"]],
                target=row["target"], backbone=row["backbone"],
                method=method_tag(row), seed=int(row["seed"]),
            )

            # 1. the registry's spelling of the backbone
            registry_backbone = backbone_by_id.get(experiment_id)
            if registry_backbone is None:
                missing.append(experiment_id)
            elif registry_backbone != row["backbone"]:
                frame.at[index, "backbone"] = registry_backbone
                relabelled += 1

            # 2. the fitted temperature
            if "temperature" not in frame.columns:
                continue
            fitted = temperature_of(experiment_id)
            if fitted is None:
                continue
            current = row["temperature"]
            if pd.isna(current):
                frame.at[index, "temperature"] = fitted
                filled += 1
            elif abs(float(current) - fitted) > TOLERANCE:
                raise SystemExit(
                    f"{name}: {experiment_id} records temperature "
                    f"{float(current)!r} but its evaluation report says "
                    f"{fitted!r}. Not overwriting -- investigate first.")

        changes = relabelled + filled
        total_changes += changes
        print(f"{name}: {relabelled} backbone label(s) normalised, "
              f"{filled} temperature value(s) filled")
        if missing:
            print(f"  !! {len(missing)} row(s) have no registry entry: "
                  f"{missing[0]}" + (" ..." if len(missing) > 1 else ""))

        if changes and apply_changes:
            frame.to_csv(path, index=False)
            print(f"  saved -> {path}")

    if not total_changes:
        print("\nnothing to do -- every row already agrees with the registry "
              "and the evaluation reports")
    elif not apply_changes:
        print(f"\n{total_changes} change(s) pending. Re-run with --apply to write them.")


if __name__ == "__main__":
    main()
