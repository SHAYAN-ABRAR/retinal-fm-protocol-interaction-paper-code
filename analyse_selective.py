"""Selective prediction across the four unseen domains.

The clinical argument the project has been making implicitly: a model deployed on
a new clinic will be wrong more often, so let it *abstain*. This script asks
whether the model's own confidence is a good enough signal to abstain on.

Two numbers per target:

**Error-detection AUROC** -- can confidence separate the model's errors from its
correct predictions? 0.5 is a coin flip; anything near it means abstention is
useless because the model is confidently wrong.

**Risk at coverage** -- if a clinician reviews the least-confident X% by hand,
what is the error rate on the automated remainder? This is the number that
decides whether a deployment is viable.

Why it matters here specifically
--------------------------------
Phase 5 found that on two of four targets a source-fitted temperature does not
fix calibration. Temperature scaling is monotonic, so it changes neither the
predictions nor their ranking -- **every number in this file is identical before
and after scaling**. Selective prediction is therefore the only remaining lever
on those targets, which makes it the practical fallback when recalibration
fails.

Runs entirely from saved predictions. No GPU, no retraining.

Usage:
    python analyse_selective.py                # ERM, all seeds present
    python analyse_selective.py --seed 42
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
BATCH_SIZE = 32
ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def _experiment_id(target: str, method: str, seed: int) -> str:
    from src.utils.registry import make_experiment_id

    sources = [d for d in ALL_TARGETS if d != target]
    return make_experiment_id(
        protocol="lodo", sources=sources, target=target,
        backbone=BACKBONE, method=f"{method}-b{BATCH_SIZE}", seed=seed,
    )


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.selective_prediction import evaluate_selective_prediction
    from src.utils.io import project_root

    arguments = sys.argv[1:]

    def _take(flag: str, default: str | None) -> str | None:
        if flag in arguments:
            index = arguments.index(flag)
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        return default

    method = _take("--method", "erm")
    seed_argument = _take("--seeds", None)

    outputs = project_root() / "outputs"
    results = outputs / "tables" / "lodo_results.csv"
    if not results.exists():
        print("NOT RUN -- no lodo_results.csv")
        return
    frame = pd.read_csv(results)
    frame = frame[frame["method"] == method]
    # lodo_results.csv now also holds domain-balanced-sampler runs, whose
    # "method" column reads exactly the same as their ordinary-sampler
    # counterparts -- only the experiment id distinguishes them. Without this
    # filter an ERM three-seed summary would silently average six rows from two
    # different samplers and report the spread between them as seed noise.
    if "domain_balanced" in frame.columns:
        frame = frame[~frame["domain_balanced"].fillna(False).astype(bool)]
    # lodo_results.csv holds every backbone that has been run. Filtering on
    # method and seed alone would pool DenseNet121 with ConvNeXt-Tiny; rows
    # written before the column existed are DenseNet121.
    if "backbone" in frame.columns:
        frame = frame[frame["backbone"] == BACKBONE]
    seeds = (
        [int(s) for s in seed_argument.split(",")] if seed_argument
        else sorted(frame["seed"].unique().tolist())
    )

    print(f"selective prediction on unseen domains: method={method}, seeds={seeds}")

    rows, curves = [], []
    for target in ALL_TARGETS:
        per_seed = []
        for seed in seeds:
            path = (outputs / "predictions"
                    / f"{_experiment_id(target, method, seed)}"
                      f"__target_test[{target}]_predictions.csv")
            if not path.exists():
                continue
            predictions = pd.read_csv(path)
            result = evaluate_selective_prediction(
                predictions["true_grade"].to_numpy(),
                predictions["predicted_grade"].to_numpy(),
                predictions["confidence"].to_numpy(),
                coverages=COVERAGES,
            )
            record = {"target": target, "seed": seed, "n": result.n_samples,
                      "aurc": result.aurc,
                      "error_auroc": result.error_detection_auroc}
            for entry in result.at_coverage:
                record[f"risk@{entry['coverage']:.1f}"] = entry["risk"]
            per_seed.append(record)
            curves.append({"target": target, "seed": seed,
                           "coverages": result.coverages, "risks": result.risks})
        if not per_seed:
            print(f"\n{target}: NOT RUN")
            continue
        rows.extend(per_seed)

    if not rows:
        return
    table = pd.DataFrame(rows)

    print(f"\n{'=' * 96}\nmean +/- SD across {len(seeds)} seed(s)\n{'=' * 96}")
    risk_columns = [c for c in table.columns if c.startswith("risk@")]
    header = (f"  {'target':10s}{'n':>7s}{'AURC':>16s}{'err-AUROC':>16s}"
              + "".join(f"{c:>14s}" for c in risk_columns))
    print(header)
    print("  " + "-" * (len(header) - 2))
    summary = []
    for target in ALL_TARGETS:
        subset = table[table["target"] == target]
        if subset.empty:
            print(f"  {target:10s}{'NOT RUN':>7s}")
            continue
        cells = (f"  {target:10s}{int(subset['n'].iloc[0]):>7d}"
                 f"{subset['aurc'].mean():>9.4f}+-{subset['aurc'].std(ddof=1):<6.4f}"
                 f"{subset['error_auroc'].mean():>9.4f}+-{subset['error_auroc'].std(ddof=1):<6.4f}")
        record = {"target": target, "n": int(subset["n"].iloc[0]),
                  "aurc_mean": subset["aurc"].mean(),
                  "aurc_sd": subset["aurc"].std(ddof=1),
                  "error_auroc_mean": subset["error_auroc"].mean(),
                  "error_auroc_sd": subset["error_auroc"].std(ddof=1)}
        for column in risk_columns:
            cells += f"{subset[column].mean():>14.4f}"
            record[f"{column}_mean"] = subset[column].mean()
            record[f"{column}_sd"] = subset[column].std(ddof=1)
        print(cells)
        summary.append(record)

    path = outputs / "tables" / f"selective_prediction_{method}.csv"
    pd.DataFrame(summary).to_csv(path, index=False)
    print(f"\nsaved -> {path}")

    # -- what abstention actually buys -------------------------------------
    print(f"\n{'=' * 96}\nwhat abstention buys: full coverage vs 70%\n{'=' * 96}")
    for record in summary:
        full = record.get("risk@1.0_mean")
        seventy = record.get("risk@0.7_mean")
        if full is None or seventy is None or not np.isfinite(full):
            continue
        reduction = (1 - seventy / full) * 100 if full > 0 else float("nan")
        detectable = record["error_auroc_mean"]
        verdict = ("confidence is informative" if detectable >= 0.70 else
                   "confidence is weak" if detectable >= 0.60 else
                   "confidence is near useless")
        # Printed as a reduction, not a signed delta: "+29%" beside a falling
        # error rate reads as an increase.
        print(f"  {record['target']:10s} error rate {full:.3f} -> {seventy:.3f} "
              f"at 70% coverage ({reduction:.0f}% lower)   "
              f"err-AUROC {detectable:.3f} -- {verdict}")

    print("\nTemperature scaling is monotonic: it changes no prediction and no "
          "confidence ranking, so every number above is identical before and "
          "after scaling. On targets where recalibration fails, abstention is "
          "the remaining lever.")

    if curves:
        frames = []
        for c in curves:
            frames.append(pd.DataFrame({"target": c["target"], "seed": c["seed"],
                                        "coverage": c["coverages"], "risk": c["risks"]}))
        path = outputs / "tables" / f"risk_coverage_curves_{method}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
