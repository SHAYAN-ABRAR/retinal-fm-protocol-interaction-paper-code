"""Aggregate the Stage-C methods across seeds: mean +/- SD, and honest verdicts.

The question this answers is the one a reviewer asks first: **does the
single-seed ordering survive training variance?**

Two things are reported side by side, because they answer different questions:

*Across-seed SD* tells you how much a number moves when only the training seed
changes. If the gap between two methods is smaller than that, the gap is not a
finding.

*Paired bootstrap on the pooled predictions* tells you how much a number moves
under resampling of the 507 test images. That is a different source of
uncertainty and does not substitute for the first.

A difference is called real here only when it clears BOTH: the seed means are
separated by more than the pooled across-seed SD, and the paired interval on the
matched seeds excludes zero.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
BATCH_SIZE = 32
METHODS = ["erm", "ordinal", "deep_coral", "mixstyle", "mixstyle_ordinal", "deep_coral_ordinal"]
BASELINE = "erm"
METRICS = ["test_qwk", "test_f1", "test_ece", "test_mae", "test_severe"]
HIGHER_IS_BETTER = {"test_qwk": True, "test_f1": True, "test_ece": False,
                    "test_mae": False, "test_severe": False}


def _experiment_id(method: str, seed: int) -> str:
    from src.utils.registry import make_experiment_id

    return make_experiment_id(
        protocol="lodo", sources=["ddr", "aptos"], target="idrid",
        backbone=BACKBONE, method=f"{method}-b{BATCH_SIZE}", seed=seed,
    )


def load_predictions(method: str, seed: int):
    import pandas as pd

    from src.utils.io import project_root

    path = (
        project_root() / "outputs" / "predictions"
        / f"{_experiment_id(method, seed)}__target_test[idrid]_predictions.csv"
    )
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import paired_bootstrap_difference
    from src.utils.io import project_root, write_json

    outputs = project_root() / "outputs"
    path = outputs / "tables" / "stage_c_method_seeds.csv"
    if not path.exists():
        print("no seed table yet -- run run_method_comparison.py --seeds first")
        return

    seeds_frame = pd.read_csv(path)
    seeds = sorted(seeds_frame["seed"].unique().tolist())
    print(f"seeds present: {seeds}")
    counts = seeds_frame.groupby("method")["seed"].nunique()
    print("runs per method:")
    for method in METHODS:
        n = int(counts.get(method, 0))
        flag = "" if n == len(seeds) else "   <-- INCOMPLETE"
        print(f"  {method:20s} {n}/{len(seeds)}{flag}")

    # -- mean +/- SD per method -------------------------------------------
    print(f"\n=== mean +/- SD across {len(seeds)} seed(s), unseen IDRiD (n=507) ===")
    aggregate = seeds_frame.groupby("method")[METRICS].agg(["mean", "std", "count"])
    rows = []
    header = f"  {'method':20s}" + "".join(f"{m.replace('test_', ''):>22s}" for m in METRICS)
    print(header)
    for method in METHODS:
        if method not in aggregate.index:
            print(f"  {method:20s}" + "  NOT RUN".rjust(22))
            continue
        cells, record = "", {"method": method}
        for metric in METRICS:
            mean = aggregate.loc[method, (metric, "mean")]
            sd = aggregate.loc[method, (metric, "std")]
            n = int(aggregate.loc[method, (metric, "count")])
            cells += f"{mean:>13.4f}+-{'n/a   ' if np.isnan(sd) else f'{sd:<6.4f}'} "
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            record[f"{metric}_n"] = n
        print(f"  {method:20s}{cells}")
        rows.append(record)

    summary = pd.DataFrame(rows)
    summary.to_csv(outputs / "tables" / "stage_c_seed_summary.csv", index=False)

    if BASELINE not in aggregate.index:
        print(f"\nbaseline {BASELINE} missing; cannot compare")
        return

    # -- is the gap bigger than seed noise? --------------------------------
    print(f"\n=== does each gap exceed across-seed variation? (vs {BASELINE}) ===")
    verdict_rows = []
    for metric in METRICS:
        base_mean = aggregate.loc[BASELINE, (metric, "mean")]
        base_sd = aggregate.loc[BASELINE, (metric, "std")]
        base_sd_text = "n/a (single seed)" if np.isnan(base_sd) else f"{base_sd:.4f}"
        print(f"\n  -- {metric} (baseline {base_mean:.4f} +- {base_sd_text}) --")
        for method in METHODS:
            if method == BASELINE or method not in aggregate.index:
                continue
            mean = aggregate.loc[method, (metric, "mean")]
            sd = aggregate.loc[method, (metric, "std")]
            n_method = int(aggregate.loc[method, (metric, "count")])
            n_base = int(aggregate.loc[BASELINE, (metric, "count")])
            delta = mean - base_mean
            better = (delta > 0) == HIGHER_IS_BETTER[metric]
            direction = "better" if better else "worse"

            # With fewer than two seeds the SD is undefined. Reporting "within
            # seed noise" there would claim we measured variance and found the
            # gap smaller -- we measured nothing. Say so instead.
            if n_method < 2 or n_base < 2:
                pooled = float("nan")
                exceeds = None
                verdict = f"CANNOT ASSESS ({min(n_method, n_base)} seed) -- {direction}"
            else:
                # Pooled SD of the two methods: the scale on which a mean
                # difference has to be judged when only the seed changes.
                pooled = float(np.sqrt(np.nanmean([base_sd**2, sd**2])))
                exceeds = bool(abs(delta) > pooled) if pooled > 0 else False
                verdict = (
                    f"EXCEEDS seed noise, {direction}" if exceeds else "within seed noise"
                )

            print(
                f"    {method:20s} {mean:7.4f} +- "
                f"{'   n/a' if np.isnan(sd) else f'{sd:6.4f}'}  "
                f"delta {delta:+7.4f}  pooled SD "
                f"{'   n/a' if np.isnan(pooled) else f'{pooled:6.4f}'}  {verdict}"
            )
            verdict_rows.append({
                "metric": metric, "method": method, "baseline_mean": base_mean,
                "method_mean": mean, "delta": delta, "pooled_seed_sd": pooled,
                "n_seeds": min(n_method, n_base),
                "exceeds_seed_noise": exceeds, "direction": direction,
            })
    pd.DataFrame(verdict_rows).to_csv(
        outputs / "tables" / "stage_c_seed_verdicts.csv", index=False
    )

    # -- paired bootstrap on pooled predictions ----------------------------
    print(f"\n=== paired bootstrap on predictions pooled over seeds (vs {BASELINE}) ===")
    pooled_rows = []
    base_tables = {s: load_predictions(BASELINE, s) for s in seeds}
    base_tables = {s: t for s, t in base_tables.items() if t is not None}
    if not base_tables:
        print("  no baseline prediction files found; skipping")
    else:
        probability_columns = [f"probability_grade_{g}" for g in range(5)]
        for method in METHODS:
            if method == BASELINE:
                continue
            matched = []
            for seed, base_table in base_tables.items():
                table = load_predictions(method, seed)
                if table is None:
                    continue
                table = table.set_index("image_id").loc[base_table["image_id"]].reset_index()
                matched.append((base_table, table))
            if not matched:
                print(f"  {method:20s} NOT RUN")
                continue

            # Stack the matched pairs: each seed contributes the same 507 images,
            # so the comparison stays paired image-for-image within every seed.
            y_true = np.concatenate([b["true_grade"].to_numpy() for b, _ in matched])
            base_pred = np.concatenate([b["predicted_grade"].to_numpy() for b, _ in matched])
            other_pred = np.concatenate([t["predicted_grade"].to_numpy() for _, t in matched])
            base_prob = np.concatenate([b[probability_columns].to_numpy() for b, _ in matched])
            other_prob = np.concatenate([t[probability_columns].to_numpy() for _, t in matched])

            for metric in ("qwk", "f1_macro", "ece"):
                result = paired_bootstrap_difference(
                    y_true, base_pred, other_pred, base_prob, other_prob,
                    metric=metric, n_bootstrap=2000, seed=42,
                )
                print(
                    f"  {method:20s} {metric:9s} {result['difference']:+8.4f}  "
                    f"[{result['ci_lower']:+7.4f}, {result['ci_upper']:+7.4f}]  "
                    f"{'significant' if result['significant'] else 'not significant'}  "
                    f"(n_seeds={len(matched)})"
                )
                pooled_rows.append({"method": method, "n_seeds": len(matched), **result})

    if pooled_rows:
        pd.DataFrame(pooled_rows).to_csv(
            outputs / "tables" / "stage_c_seed_paired_comparison.csv", index=False
        )

    write_json(outputs / "reports" / "stage_c_seed_analysis.json", {
        "seeds": seeds,
        "methods": METHODS,
        "baseline": BASELINE,
        "note": "A difference is treated as real only if it exceeds the pooled "
                "across-seed SD AND its paired bootstrap interval excludes zero. "
                "Seed count is small, so the SD estimate is itself noisy.",
    })
    print("\nsaved -> outputs/tables/stage_c_seed_{summary,verdicts,paired_comparison}.csv")


if __name__ == "__main__":
    main()
