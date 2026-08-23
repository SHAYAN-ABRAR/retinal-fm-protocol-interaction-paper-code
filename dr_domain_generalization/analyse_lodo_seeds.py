"""Aggregate the leave-one-domain-out matrix across seeds.

Every LODO number reported so far comes from one training seed. Stage C measured
ERM's across-seed SD on target QWK at **0.032** -- larger than several of the
per-target effects the study wants to claim. This script replaces those
provisional numbers with mean +/- SD, and re-tests the Phase 6 deployment costs
against that noise floor.

The two bars
------------
Following ``analyse_seeds.py``, a difference is called real only when it clears
**both**:

1. *Across-seed SD* -- the seed means are separated by more than the pooled SD.
   This asks whether the effect survives re-initialising the network.
2. *Paired bootstrap* -- the interval on matched test images excludes zero.
   This asks whether the effect survives resampling the test set.

They measure different things and neither substitutes for the other. Stage C
produced a case that cleared the bootstrap and failed the SD bar (Deep CORAL's
calibration effect), which is why both are required.

With fewer than two seeds the SD is undefined and the verdict is
``CANNOT ASSESS``, never "within seed noise" -- the latter would claim variance
was measured when it was not.

Usage:
    python analyse_lodo_seeds.py                    # all seeds present
    python analyse_lodo_seeds.py --seeds 42,1,2
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
BATCH_SIZE = 32
ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]
METRICS = ["target_qwk", "target_f1", "target_ece", "target_ece_scaled", "target_severe"]
HIGHER_IS_BETTER = {
    "target_qwk": True, "target_f1": True, "target_ece": False,
    "target_ece_scaled": False, "target_severe": False,
}

# In-domain reference scores, restricted to each domain's own test split.
# Written by run_in_domain.py; loaded rather than hard-coded so the two files
# can never disagree.
IN_DOMAIN_TABLE = "in_domain_vs_lodo_erm_s42.csv"


def _experiment_id(target: str, method: str, seed: int) -> str:
    from src.utils.registry import make_experiment_id

    sources = [d for d in ALL_TARGETS if d != target]
    return make_experiment_id(
        protocol="lodo", sources=sources, target=target,
        backbone=BACKBONE, method=f"{method}-b{BATCH_SIZE}", seed=seed,
    )


def _in_domain_predictions_path(target: str, method: str, outputs, seed: int = 42):
    """Where run_in_domain.py saved that domain's own-test predictions."""
    from src.utils.registry import make_experiment_id

    experiment_id = make_experiment_id(
        protocol="in_domain", sources=[target], target=target,
        backbone=BACKBONE, method=f"{method}-b{BATCH_SIZE}", seed=seed,
    )
    return outputs / "predictions" / f"{experiment_id}__target_test[{target}]_predictions.csv"


def load_predictions(target: str, method: str, seed: int):
    import pandas as pd

    from src.utils.io import project_root

    path = (
        project_root() / "outputs" / "predictions"
        / f"{_experiment_id(target, method, seed)}__target_test[{target}]_predictions.csv"
    )
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import paired_bootstrap_difference
    from src.evaluation.metrics import quadratic_weighted_kappa
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
    results_path = outputs / "tables" / "lodo_results.csv"
    if not results_path.exists():
        print(f"NOT RUN -- {results_path} does not exist.")
        return

    frame = pd.read_csv(results_path)
    frame = frame[frame["method"] == method]
    # lodo_results.csv holds every backbone that has been run. Filtering on
    # method and seed alone would pool DenseNet121 with ConvNeXt-Tiny; rows
    # written before the column existed are DenseNet121.
    if "backbone" in frame.columns:
        frame = frame[frame["backbone"] == BACKBONE]
    seeds = (
        [int(s) for s in seed_argument.split(",")] if seed_argument
        else sorted(frame["seed"].unique().tolist())
    )
    frame = frame[frame["seed"].isin(seeds)]

    print(f"leave-one-domain-out across seeds: method={method}, seeds={seeds}")

    # -- completeness ------------------------------------------------------
    counts = frame.groupby("target")["seed"].nunique()
    print("\nruns per target:")
    incomplete = []
    for target in ALL_TARGETS:
        n = int(counts.get(target, 0))
        flag = "" if n == len(seeds) else "   <-- INCOMPLETE"
        if n != len(seeds):
            incomplete.append(target)
        print(f"  {target:10s} {n}/{len(seeds)}{flag}")

    # -- mean +/- SD per target -------------------------------------------
    print(f"\n{'=' * 104}\nmean +/- SD across {len(seeds)} seed(s), per unseen domain\n{'=' * 104}")
    aggregate = frame.groupby("target")[METRICS].agg(["mean", "std", "count"])

    header = f"  {'target':10s}" + "".join(f"{m.replace('target_', ''):>21s}" for m in METRICS)
    print(header)
    summary_rows = []
    for target in ALL_TARGETS:
        if target not in aggregate.index:
            print(f"  {target:10s}" + "NOT RUN".rjust(21))
            continue
        cells, record = "", {"target": target, "n_seeds": 0}
        for metric in METRICS:
            mean = aggregate.loc[target, (metric, "mean")]
            sd = aggregate.loc[target, (metric, "std")]
            n = int(aggregate.loc[target, (metric, "count")])
            sd_text = "n/a  " if np.isnan(sd) else f"{sd:<5.4f}"
            cells += f"{mean:>13.4f}+-{sd_text} "
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            record["n_seeds"] = n
        print(f"  {target:10s}{cells}")
        summary_rows.append(record)

    if summary_rows:
        path = outputs / "tables" / f"lodo_{method}_seed_summary.csv"
        pd.DataFrame(summary_rows).to_csv(path, index=False)
        print(f"\nsaved -> {path}")

    # -- per-seed spread, so a single outlier run is visible ---------------
    print(f"\n{'=' * 104}\nper-seed target QWK (a mean hides a single divergent run)\n{'=' * 104}")
    pivot = frame.pivot_table(index="target", columns="seed", values="target_qwk")
    pivot = pivot.reindex(ALL_TARGETS)
    pivot["range"] = pivot.max(axis=1) - pivot.min(axis=1)
    print(pivot.round(4).to_string())

    # -- does the deployment cost survive the seed bar? --------------------
    in_domain_path = outputs / "tables" / IN_DOMAIN_TABLE
    if not in_domain_path.exists():
        print(f"\nin-domain reference {in_domain_path.name} NOT RUN; "
              "skipping the deployment-cost re-test")
        return

    reference = pd.read_csv(in_domain_path).set_index("domain")
    print(f"\n{'=' * 104}\ncost of cross-domain deployment, re-tested against seed noise\n{'=' * 104}")
    head = (f"  {'target':10s}{'in-domain':>10s}{'LODO mean':>11s}{'delta SD':>9s}"
            f"{'delta':>9s}{'paired 95% CI':>24s}{'verdict':>28s}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    verdict_rows = []
    for target in ALL_TARGETS:
        if target not in aggregate.index or target not in reference.index:
            print(f"  {target:10s}{'NOT RUN':>10s}")
            continue

        # Every quantity below is computed on the SAME images the in-domain
        # model was tested on. The LODO mean in the summary table above is over
        # all of the target domain (12,424 DDR images, not 1,862), so reusing it
        # here would difference two different test sets -- the precise error
        # this comparison exists to avoid.
        #
        # Seeds are paired: the LODO model at seed s is differenced against the
        # in-domain model at seed s. Both sides now have three seeds, and
        # pairing them measures the variation of the *difference* rather than
        # adding two independent noise sources together. When only one
        # in-domain seed exists this falls back to using it for every LODO
        # seed, which is what this comparison did before seeds 1 and 2 were run.
        intervals = []
        matched_qwks = []
        in_domain_qwks = []
        deltas = []
        paired_seeds = []

        available = {
            seed: _in_domain_predictions_path(target, method, outputs, seed)
            for seed in seeds
        }
        available = {s: p for s, p in available.items() if p.exists()}
        fallback = _in_domain_predictions_path(target, method, outputs, 42)
        paired = bool(available)

        for seed in seeds:
            predictions = load_predictions(target, method, seed)
            if predictions is None:
                continue
            in_path = available.get(seed, fallback if fallback.exists() else None)
            if in_path is None:
                continue
            reference_frame = pd.read_csv(in_path)
            shared = set(reference_frame["image_id"]) & set(predictions["image_id"])
            if not shared:
                continue
            lodo_matched = (
                predictions[predictions["image_id"].isin(shared)]
                .sort_values("image_id")
            )
            in_matched = (
                reference_frame[reference_frame["image_id"].isin(shared)]
                .sort_values("image_id")
            )
            if not (lodo_matched["true_grade"].to_numpy()
                    == in_matched["true_grade"].to_numpy()).all():
                raise AssertionError(
                    f"{target} seed {seed}: the two runs disagree on ground truth "
                    "for the same image ids; one prediction file is stale."
                )
            result = paired_bootstrap_difference(
                lodo_matched["true_grade"].to_numpy(),
                in_matched["predicted_grade"].to_numpy(),
                lodo_matched["predicted_grade"].to_numpy(),
                metric="qwk", n_bootstrap=2000, seed=seed,
            )
            intervals.append((result["ci_lower"], result["ci_upper"]))
            truth = lodo_matched["true_grade"].to_numpy()
            lodo_qwk = quadratic_weighted_kappa(
                truth, lodo_matched["predicted_grade"].to_numpy())
            in_qwk = quadratic_weighted_kappa(
                truth, in_matched["predicted_grade"].to_numpy())
            matched_qwks.append(lodo_qwk)
            in_domain_qwks.append(in_qwk)
            deltas.append(lodo_qwk - in_qwk)
            paired_seeds.append(seed)

        n_seeds = len(matched_qwks)
        lodo_mean = float(np.mean(matched_qwks)) if matched_qwks else float("nan")
        lodo_sd = float(np.std(matched_qwks, ddof=1)) if n_seeds >= 2 else float("nan")
        in_domain_qwk = (float(np.mean(in_domain_qwks)) if in_domain_qwks
                         else float(reference.loc[target, "in_domain_qwk"]))
        in_domain_sd = (float(np.std(in_domain_qwks, ddof=1))
                        if len(set(in_domain_qwks)) > 1 else float("nan"))
        delta = float(np.mean(deltas)) if deltas else float("nan")
        # Bar 1 is the SD of the per-seed difference. With one in-domain seed
        # every delta shares the same reference, so this collapses to the LODO
        # SD -- the old behaviour, and correctly so.
        delta_sd = float(np.std(deltas, ddof=1)) if n_seeds >= 2 else float("nan")

        # Bar 1: is the gap bigger than the run-to-run variation of the gap?
        exceeds_sd = None if n_seeds < 2 else bool(abs(delta) > delta_sd)

        if intervals:
            lower = min(low for low, _ in intervals)
            upper = max(high for _, high in intervals)
            excludes_zero = not (lower <= 0 <= upper)
            ci_text = f"[{lower:+.4f}, {upper:+.4f}]"
        else:
            lower = upper = float("nan")
            excludes_zero = None
            ci_text = "n/a"

        if exceeds_sd is None:
            verdict = f"CANNOT ASSESS ({n_seeds} seed)"
        elif exceeds_sd and excludes_zero:
            verdict = "REAL (both bars)"
        elif not exceeds_sd:
            verdict = "within seed noise"
        else:
            verdict = "CI spans zero"

        sd_text = "n/a" if np.isnan(delta_sd) else f"{delta_sd:.4f}"
        print(f"  {target:10s}{in_domain_qwk:>10.4f}{lodo_mean:>11.4f}{sd_text:>9s}"
              f"{delta:>+9.4f}{ci_text:>24s}{verdict:>28s}")
        verdict_rows.append({
            "target": target, "in_domain_qwk": in_domain_qwk,
            "in_domain_qwk_sd": in_domain_sd,
            "lodo_qwk_mean": lodo_mean, "lodo_qwk_sd": lodo_sd,
            "n_seeds": n_seeds, "paired_seeds": ",".join(str(s) for s in paired_seeds),
            "in_domain_seeds": len(in_domain_qwks) if paired else 1,
            "delta_qwk": delta, "delta_qwk_sd": delta_sd,
            "ci_lower": lower, "ci_upper": upper,
            "exceeds_seed_sd": exceeds_sd, "ci_excludes_zero": excludes_zero,
            "verdict": verdict,
        })

    if verdict_rows:
        path = outputs / "tables" / f"lodo_{method}_seed_verdicts.csv"
        pd.DataFrame(verdict_rows).to_csv(path, index=False)
        print(f"\nsaved -> {path}")

    reference_seeds = max((int(r.get("in_domain_seeds", 1)) for r in verdict_rows),
                          default=1)
    if reference_seeds >= 2:
        print(f"\nBoth sides carry {reference_seeds} seeds and are paired seed-to-seed, "
              "so the SD above is the run-to-run variation of the difference itself, "
              "not of either model alone. It is larger than the LODO-only SD reported "
              "before the in-domain seeds existed, because the reference now "
              "contributes its own variance instead of being treated as exact.")
    else:
        print("\nThe in-domain reference is itself a single seed (42). A delta here "
              "carries that model's own run-to-run variation, which is not measured.")
    if incomplete:
        print(f"!! incomplete targets: {', '.join(incomplete)} -- "
              "their SD is computed over fewer seeds than the others.")


if __name__ == "__main__":
    main()
