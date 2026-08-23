"""How much of the EyePACS gap is training-set size, and how much is shift?

Reads the sweep written by ``run_subsample_sweep.py`` and turns it into the
decomposition Phase 7 section 6 currently estimates from two points.

Matched images
--------------
The sweep models are tested on all 35,108 EyePACS images; the in-domain model
is tested on EyePACS's own 5,268-image held-out split. Comparing those two
numbers directly would difference two different test sets -- the error Phase 6
exists to avoid. Every QWK here is therefore recomputed on the **in-domain test
split only**, so the sweep curve and the in-domain ceiling live on the same
images and the extrapolation lands somewhere meaningful.

The fit
-------
QWK against log(n), fitted per seed so the extrapolation carries an across-seed
SD rather than a single line through three means. Learning curves are roughly
log-linear over a limited range; that is an assumption, and this script prints
how far beyond the data it is being asked to hold.

Runs from saved predictions. No GPU.

Usage:
    python analyse_subsample.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
BATCH_SIZE = 32
IMAGE_SIZE = 224
TARGET = "eyepacs"
SOURCES = ["ddr", "aptos", "idrid"]


def _matched_qwk(predictions_path, reference_ids, outputs):
    """QWK on the in-domain test split only, or None if the file is absent."""
    import pandas as pd

    from src.evaluation.metrics import quadratic_weighted_kappa

    if not predictions_path.exists():
        return None
    frame = pd.read_csv(predictions_path)
    matched = frame[frame["image_id"].isin(reference_ids)].sort_values("image_id")
    if matched.empty:
        return None
    return quadratic_weighted_kappa(
        matched["true_grade"].to_numpy(), matched["predicted_grade"].to_numpy()
    ), len(matched)


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.utils.io import project_root
    from src.utils.registry import make_experiment_id

    outputs = project_root() / "outputs"
    predictions = outputs / "predictions"
    sweep_path = outputs / "tables" / "subsample_sweep.csv"

    # ---- the in-domain reference and the images everything is scored on ----
    in_frames, in_qwks = [], []
    for seed in (42, 1, 2):
        in_id = make_experiment_id(
            protocol="in_domain", sources=[TARGET], target=TARGET,
            backbone=BACKBONE, method=f"erm-b{BATCH_SIZE}", seed=seed,
        )
        path = predictions / f"{in_id}__target_test[{TARGET}]_predictions.csv"
        if path.exists():
            in_frames.append(pd.read_csv(path))
    if not in_frames:
        print("NOT RUN -- no in-domain EyePACS predictions")
        return

    reference_ids = set(in_frames[0]["image_id"])
    from src.evaluation.metrics import quadratic_weighted_kappa
    for frame in in_frames:
        ordered = frame.sort_values("image_id")
        in_qwks.append(quadratic_weighted_kappa(
            ordered["true_grade"].to_numpy(), ordered["predicted_grade"].to_numpy()))
    in_mean = float(np.mean(in_qwks))
    in_sd = float(np.std(in_qwks, ddof=1)) if len(in_qwks) > 1 else float("nan")
    in_budget = 24574

    # ---- the sweep points, plus 100% from the LODO runs --------------------
    points = []           # (n_train, seed, matched qwk)
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        for _, row in sweep.iterrows():
            experiment_id = make_experiment_id(
                protocol="lodo", sources=SOURCES, target=TARGET, backbone=BACKBONE,
                method=f"erm-b{BATCH_SIZE}-f{int(round(row['fraction'] * 100)):03d}",
                seed=int(row["seed"]),
            )
            got = _matched_qwk(
                predictions / f"{experiment_id}__target_test[{TARGET}]_predictions.csv",
                reference_ids, outputs)
            if got:
                points.append((int(row["n_train"]), int(row["seed"]), got[0]))
    else:
        print(f"note: {sweep_path.name} NOT RUN -- only the 100% point is available")

    for seed in (42, 1, 2):
        lodo_id = make_experiment_id(
            protocol="lodo", sources=SOURCES, target=TARGET,
            backbone=BACKBONE, method=f"erm-b{BATCH_SIZE}", seed=seed,
        )
        got = _matched_qwk(
            predictions / f"{lodo_id}__target_test[{TARGET}]_predictions.csv",
            reference_ids, outputs)
        if got:
            points.append((11841, seed, got[0]))

    if not points:
        print("NOT RUN -- no sweep or LODO predictions found")
        return

    table = pd.DataFrame(points, columns=["n_train", "seed", "qwk"])
    n_matched = len(reference_ids)

    print("=" * 92)
    print("EyePACS: training-set size versus domain shift")
    print("=" * 92)
    print(f"  All QWK on the in-domain test split ({n_matched:,} images), so the sweep")
    print(f"  and the in-domain ceiling are scored on identical data.\n")

    grouped = table.groupby("n_train")["qwk"].agg(["mean", "std", "count"])
    print(f"  {'n_train':>9}{'QWK':>9}{'SD':>9}{'seeds':>7}")
    print("  " + "-" * 34)
    for n, row in grouped.iterrows():
        sd = "n/a" if pd.isna(row["std"]) else f"{row['std']:.4f}"
        print(f"  {n:>9,}{row['mean']:>9.4f}{sd:>9}{int(row['count']):>7}")
    print(f"  {'in-domain':>9}{in_mean:>9.4f}"
          f"{('n/a' if np.isnan(in_sd) else f'{in_sd:.4f}'):>9}{len(in_qwks):>7}"
          f"   (n_train {in_budget:,})")

    if len(grouped) < 3:
        print("\n  Fewer than three distinct training sizes: no curve can be fitted.")
        print("  Run `python run_subsample_sweep.py` first.")
        return

    # ---- fit QWK ~ a + b log(n), one line per seed -------------------------
    slopes, predictions_at_budget = [], []
    for seed, group in table.groupby("seed"):
        if group["n_train"].nunique() < 3:
            continue
        b, a = np.polyfit(np.log(group["n_train"]), group["qwk"], 1)
        slopes.append(b)
        predictions_at_budget.append(a + b * np.log(in_budget))
    if not slopes:
        print("\n  No seed has three distinct sizes; cannot fit.")
        return

    slope = float(np.mean(slopes))
    slope_sd = float(np.std(slopes, ddof=1)) if len(slopes) > 1 else float("nan")
    predicted = float(np.mean(predictions_at_budget))
    predicted_sd = (float(np.std(predictions_at_budget, ddof=1))
                    if len(predictions_at_budget) > 1 else float("nan"))

    largest = int(grouped.index.max())
    observed = float(grouped.loc[largest, "mean"])
    gap = in_mean - observed
    size_share = (predicted - observed) / gap if gap else float("nan")

    print(f"\n  fit: QWK = a + b*log(n),  b = {slope:+.4f}"
          f"{'' if np.isnan(slope_sd) else f' +/- {slope_sd:.4f}'} per e-fold, "
          f"{len(slopes)} seed(s)")
    print(f"  predicted at n = {in_budget:,}: {predicted:.4f}"
          f"{'' if np.isnan(predicted_sd) else f' +/- {predicted_sd:.4f}'}")
    print(f"  extrapolation reaches {in_budget / largest:.2f}x beyond the largest "
          f"training set actually run ({largest:,})")

    print(f"\n  observed at {largest:,}      : {observed:.4f}")
    print(f"  in-domain ceiling        : {in_mean:.4f}")
    print(f"  gap to close             : {gap:+.4f}")
    print(f"  attributable to size     : {predicted - observed:+.4f}  "
          f"({size_share:.0%} of the gap)")
    print(f"  attributable to shift    : {in_mean - predicted:+.4f}  "
          f"({1 - size_share:.0%} of the gap)")

    path = outputs / "tables" / "subsample_decomposition.csv"
    pd.DataFrame([{
        "target": TARGET, "n_matched": n_matched,
        "largest_n_train": largest, "observed_qwk": observed,
        "in_domain_qwk": in_mean, "in_domain_qwk_sd": in_sd,
        "in_domain_budget": in_budget,
        "log_slope": slope, "log_slope_sd": slope_sd,
        "predicted_at_budget": predicted, "predicted_at_budget_sd": predicted_sd,
        "extrapolation_factor": in_budget / largest,
        "gap": gap, "size_share": size_share, "shift_share": 1 - size_share,
        "n_seeds_fitted": len(slopes),
    }]).to_csv(path, index=False)
    print(f"\nsaved -> {path}")

    print("\n  What this does NOT establish:")
    print(f"    * Anything at n > {largest:,}. The budget point is an "
          f"extrapolation {in_budget / largest:.2f}x beyond the data, under a")
    print("      log-linear assumption that no point here tests.")
    print("    * That the remainder is domain shift alone. It is the residual "
          "after size,")
    print("      and still contains label noise, resolution and anything else "
          "unmodelled.")
    if len(slopes) < 2:
        print("    * Any uncertainty at all: one seed was fitted.")


if __name__ == "__main__":
    main()
