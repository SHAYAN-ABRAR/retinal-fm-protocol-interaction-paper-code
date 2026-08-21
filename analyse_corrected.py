"""Apply Holm-Bonferroni across every paired comparison the project has made.

Each comparison in this project was tested in isolation, and
``paired_bootstrap_difference`` marks its result ``uncorrected for multiple
comparisons``. Taken together they form one family: six methods against ERM,
four deployment costs, and the cross-domain matrix. At alpha = 0.05 with ~35
independent tests the chance of at least one false positive is about 83%, so
significance claimed per-comparison does not survive into a paper.

This script collects the family, recomputes each comparison's bootstrap
p-value from the saved per-image predictions, and applies the step-down
correction. It reports what survives and -- more usefully -- what does not.

Holm rather than plain Bonferroni: same family-wise error control, uniformly
more powerful, and no independence assumption, which matters because these
comparisons share a baseline and a test set.

Runs from saved predictions. No GPU.

Usage:
    python analyse_corrected.py
    python analyse_corrected.py --alpha 0.01
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
BATCH_SIZE = 32
ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]
STAGE_C_METHODS = ["ordinal", "deep_coral", "mixstyle", "mixstyle_ordinal", "deep_coral_ordinal"]
N_BOOTSTRAP = 2000


def _load(path):
    import pandas as pd

    return pd.read_csv(path) if path.exists() else None


def _probabilities(frame):
    import numpy as np

    columns = sorted(
        (c for c in frame.columns if c.startswith("probability_grade_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    return np.asarray(frame[columns].to_numpy(), dtype=np.float64)


def collect(outputs, seed: int = 42):
    """Every paired comparison in the project, as (label, y_true, pred_a, pred_b)."""
    from src.utils.registry import make_experiment_id

    comparisons = []

    # -- Stage C: five methods against ERM on unseen IDRiD ------------------
    def stage_c(method):
        return outputs / "predictions" / (
            make_experiment_id(
                protocol="lodo", sources=["ddr", "aptos"], target="idrid",
                backbone=BACKBONE, method=f"{method}-b{BATCH_SIZE}", seed=seed,
            ) + "__target_test[idrid]_predictions.csv"
        )

    baseline = _load(stage_c("erm"))
    if baseline is not None:
        for method in STAGE_C_METHODS:
            frame = _load(stage_c(method))
            if frame is None:
                continue
            comparisons.append((
                f"stage_c: {method} vs erm (QWK)",
                baseline["true_grade"].to_numpy(),
                baseline["predicted_grade"].to_numpy(),
                frame["predicted_grade"].to_numpy(),
            ))

    # -- Phase 6: LODO against in-domain, on matched images -----------------
    for target in ALL_TARGETS:
        in_id = make_experiment_id(
            protocol="in_domain", sources=[target], target=target,
            backbone=BACKBONE, method=f"erm-b{BATCH_SIZE}", seed=seed,
        )
        lodo_id = make_experiment_id(
            protocol="lodo", sources=[d for d in ALL_TARGETS if d != target], target=target,
            backbone=BACKBONE, method=f"erm-b{BATCH_SIZE}", seed=seed,
        )
        indomain = _load(outputs / "predictions" / f"{in_id}__target_test[{target}]_predictions.csv")
        lodo = _load(outputs / "predictions" / f"{lodo_id}__target_test[{target}]_predictions.csv")
        if indomain is None or lodo is None:
            continue
        shared = set(indomain["image_id"]) & set(lodo["image_id"])
        a = indomain[indomain["image_id"].isin(shared)].sort_values("image_id")
        b = lodo[lodo["image_id"].isin(shared)].sort_values("image_id")
        comparisons.append((
            f"deployment cost: {target} lodo vs in-domain (QWK)",
            a["true_grade"].to_numpy(),
            a["predicted_grade"].to_numpy(),
            b["predicted_grade"].to_numpy(),
        ))

    # -- Phase 7: best single source against the three-source model ---------
    single = _load(outputs / "tables" / "single_source_results.csv")
    if single is not None:
        single = single[single["seed"] == seed]
        for target in ALL_TARGETS:
            subset = single[single["target"] == target]
            if subset.empty:
                continue
            best = subset.loc[subset["target_qwk"].idxmax(), "source"]
            single_id = make_experiment_id(
                protocol="single_source", sources=[best], target=target,
                backbone=BACKBONE, method=f"erm-b{BATCH_SIZE}", seed=seed,
            )
            lodo_id = make_experiment_id(
                protocol="lodo", sources=[d for d in ALL_TARGETS if d != target], target=target,
                backbone=BACKBONE, method=f"erm-b{BATCH_SIZE}", seed=seed,
            )
            one = _load(outputs / "predictions" / f"{single_id}__target_test[{target}]_predictions.csv")
            three = _load(outputs / "predictions" / f"{lodo_id}__target_test[{target}]_predictions.csv")
            if one is None or three is None:
                continue
            shared = set(one["image_id"]) & set(three["image_id"])
            a = one[one["image_id"].isin(shared)].sort_values("image_id")
            b = three[three["image_id"].isin(shared)].sort_values("image_id")
            comparisons.append((
                f"multi-source: {target} 3-source vs {best}-only (QWK)",
                a["true_grade"].to_numpy(),
                a["predicted_grade"].to_numpy(),
                b["predicted_grade"].to_numpy(),
            ))

    return comparisons


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import holm_bonferroni, paired_bootstrap_difference
    from src.utils.io import project_root

    arguments = sys.argv[1:]
    alpha = 0.05
    if "--alpha" in arguments:
        alpha = float(arguments[arguments.index("--alpha") + 1])
    seed = 42
    if "--seed" in arguments:
        seed = int(arguments[arguments.index("--seed") + 1])

    outputs = project_root() / "outputs"
    comparisons = collect(outputs, seed=seed)
    if not comparisons:
        print("NOT RUN -- no prediction files found")
        return

    print(f"family of {len(comparisons)} paired comparisons, seed {seed}, alpha {alpha}")
    print("computing bootstrap p-values ...")

    records = []
    for label, y_true, pred_a, pred_b in comparisons:
        result = paired_bootstrap_difference(
            y_true, pred_a, pred_b, metric="qwk", n_bootstrap=N_BOOTSTRAP, seed=seed,
        )
        records.append({
            "label": label,
            "difference": result["difference"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "p_value": result["p_value"],
            "uncorrected_significant": result["significant"],
            "n": result["n_samples"],
        })

    corrected = holm_bonferroni(
        [r["p_value"] for r in records], [r["label"] for r in records], alpha=alpha
    )
    for record, adjustment in zip(records, corrected):
        record.update({
            "rank": adjustment["rank"],
            "threshold": adjustment["threshold"],
            "p_adjusted": adjustment["p_adjusted"],
            "survives_holm": adjustment["survives"],
        })

    table = pd.DataFrame(records).sort_values("rank")
    print(f"\n{'=' * 112}")
    header = (f"  {'comparison':<48}{'n':>7}{'delta':>9}{'p':>10}{'p_adj':>10}"
              f"{'uncorr':>9}{'Holm':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in table.iterrows():
        print(f"  {row['label']:<48}{int(row['n']):>7}{row['difference']:>+9.4f}"
              f"{row['p_value']:>10.4f}{row['p_adjusted']:>10.4f}"
              f"{'YES' if row['uncorrected_significant'] else 'no':>9}"
              f"{'YES' if row['survives_holm'] else 'no':>8}")

    survived = int(table["survives_holm"].sum())
    uncorrected = int(table["uncorrected_significant"].sum())
    print(f"\n  {uncorrected}/{len(table)} significant uncorrected; "
          f"{survived}/{len(table)} survive Holm-Bonferroni at alpha={alpha}")

    lost = table[table["uncorrected_significant"] & ~table["survives_holm"]]
    if len(lost):
        print(f"\n  !! {len(lost)} comparison(s) lose significance under correction:")
        for _, row in lost.iterrows():
            print(f"       {row['label']}  (p={row['p_value']:.4f}, "
                  f"threshold {row['threshold']:.4f})")
        print("     These must not be reported as significant.")
    else:
        print("\n  No comparison loses significance under correction.")

    path = outputs / "tables" / f"corrected_comparisons_s{seed}.csv"
    table.to_csv(path, index=False)
    print(f"\nsaved -> {path}")
    print("\nNote: this family covers seed-42 comparisons only. Effects that also "
          "need the across-seed bar are listed in the phase reports.")


if __name__ == "__main__":
    main()
