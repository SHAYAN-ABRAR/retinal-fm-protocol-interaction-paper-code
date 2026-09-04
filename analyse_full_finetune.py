"""The five-seed full fine-tuning analysis on DDR.

Scope, fixed before any target metric was read (docs/FULL_FINETUNE_PROTOCOL.md):
matched fixed-budget full fine-tuning, seeds 42/1/2/3/4, both initialisations,
DDR held out. Everything below reads saved predictions. No GPU.

Division of labour between the two uncertainty tools, which is the thing this
project got wrong once and will not get wrong again:

* the **crossed seed x case bootstrap** gives the point estimate and a 95%
  uncertainty interval. It does not give a p-value. Its distribution is built
  around the empirical estimate rather than under H0, so its tail mass is not a
  calibrated test and Holm-correcting it would lend an uncalibrated number the
  appearance of a controlled error rate;
* **seed-level inference** (``src/evaluation/seed_inference.py``) gives the
  formal test: a one-sample t-test on the per-seed paired effects, with the
  exact sign-flip permutation as a sensitivity check.

Four analyses:

**Primary model comparison** (item 7). ImageNet-MAE minus RETFound under full
fine-tuning, QWK, five paired seeds. This is *one* pre-specified comparison, so
no Holm family is invented for it. A non-significant result is reported as
"not demonstrated", never as equivalence.

**Primary protocol interaction** (item 8). I_full(s) = D_full(s) - D_frozen(s),
where D is ImageNet minus RETFound within a protocol. This asks directly
whether full downstream adaptation changes the relative advantage of the two
initialisation states -- which is a stronger question than observing that one
protocol is significant and another is not. The bootstrap here resamples cases
**once per replicate and applies that one sample to all four conditions**, so
the pairing that gives the contrast its power is preserved.

**Secondary interactions** (item 9). D_partial - D_frozen and D_full - D_partial,
Holm-corrected within that two-member family and labelled SECONDARY throughout.
They do not replace the primary full-vs-frozen contrast.

**Secondary outcomes** (item 10). Every metric the evaluation supports, reported
together. QWK stays primary; no metric is promoted because it reads better.

Usage:
    python analyse_full_finetune.py
    python analyse_full_finetune.py --n-bootstrap 5000
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

TARGET = "ddr"
SOURCES = "aptos-eyepacs-idrid"
SEEDS = [42, 1, 2, 3, 4]
REFERENCE = "vit-large-mae-in1k"      # ImageNet-MAE
CANDIDATE = "retfound-cfp"            # RETFound
PROTOCOLS = {
    "frozen": "linprobe-b256",
    "partial": "erm-b16-tb4-lr0.0001",
    "full": "erm-b16-ftfull-lr0.0001",
}
PRIMARY_METRIC = "qwk"
SECONDARY_METRICS = ["f1_macro", "balanced_accuracy", "severe_error_rate",
                     "mae_grade", "within_1_grade", "auroc_macro", "ece", "nll"]
N_BOOTSTRAP = 2000


def main() -> int:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import METRIC_FUNCTIONS
    from src.evaluation.crossed_bootstrap import holm_adjust
    from src.evaluation.metrics import quadratic_weighted_kappa
    from src.evaluation.seed_inference import seed_level_test
    from src.utils.io import project_root
    from src.visualization.calibration_figures import load_target_predictions

    arguments = sys.argv[1:]
    n_bootstrap = N_BOOTSTRAP
    if "--n-bootstrap" in arguments:
        n_bootstrap = int(arguments[arguments.index("--n-bootstrap") + 1])

    outputs = project_root() / "outputs"
    tables = outputs / "tables"

    def qwk(y_true, y_pred):
        return quadratic_weighted_kappa(y_true, y_pred, num_classes=5)

    # ------------------------------------------------------------- load
    runs: dict[tuple[str, str, int], dict] = {}
    missing = []
    for protocol, tag in PROTOCOLS.items():
        for backbone in (REFERENCE, CANDIDATE):
            for seed in SEEDS:
                experiment_id = (f"lodo_{SOURCES}__{TARGET}_{backbone}_{tag}_s{seed}")
                run = load_target_predictions(experiment_id, TARGET,
                                              outputs_dir=outputs)
                if run is None:
                    missing.append(experiment_id)
                else:
                    runs[(protocol, backbone, seed)] = run

    expected = len(PROTOCOLS) * 2 * len(SEEDS)
    print(f"loaded {len(runs)} of {expected} runs")
    if missing:
        print(f"NOT RUN ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
        return 1

    # Every run must have scored the same images with the same labels, or a
    # difference between them is not a difference in the model.
    anchor = runs[("full", REFERENCE, SEEDS[0])]
    truth = anchor["y_true"]
    for key, run in runs.items():
        assert len(run["y_true"]) == len(truth), f"{key}: different test size"
        assert (run["y_true"] == truth).all(), f"{key}: labels disagree"
    n_cases = len(truth)
    print(f"all runs scored the same {n_cases:,} DDR images\n")

    def preds(protocol, backbone, seed):
        return runs[(protocol, backbone, seed)]["y_pred"]

    # ------------------------------------------- crossed bootstrap helpers
    def crossed_ci(per_replicate, seed=7, alpha=0.05):
        """95% interval from a function of (sampled seeds, case indices)."""
        rng = np.random.default_rng(seed)
        draws = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            picked = rng.choice(SEEDS, size=len(SEEDS), replace=True)
            cases = rng.integers(0, n_cases, size=n_cases)
            draws[i] = per_replicate(picked, cases)
        lower, upper = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lower), float(upper)

    def difference_replicate(protocol, metric):
        """One crossed replicate of (reference - candidate) within a protocol."""
        def inner(picked, cases):
            t = truth[cases]
            values = [metric(t, preds(protocol, REFERENCE, s)[cases])
                      - metric(t, preds(protocol, CANDIDATE, s)[cases])
                      for s in picked]
            return float(np.mean(values))
        return inner

    def interaction_replicate(later, earlier, metric):
        """One crossed replicate of D_later - D_earlier.

        The same `cases` sample is applied to all four conditions -- both
        models under both protocols -- which is what preserves the pairing.
        """
        def inner(picked, cases):
            t = truth[cases]
            values = []
            for s in picked:
                d_later = (metric(t, preds(later, REFERENCE, s)[cases])
                           - metric(t, preds(later, CANDIDATE, s)[cases]))
                d_earlier = (metric(t, preds(earlier, REFERENCE, s)[cases])
                             - metric(t, preds(earlier, CANDIDATE, s)[cases]))
                values.append(d_later - d_earlier)
            return float(np.mean(values))
        return inner

    def per_seed_difference(protocol, metric):
        return {s: float(metric(truth, preds(protocol, REFERENCE, s))
                         - metric(truth, preds(protocol, CANDIDATE, s)))
                for s in SEEDS}

    # =================================================== item 7: primary
    print("=" * 92)
    print("PRIMARY -- full fine-tuning, ImageNet-MAE minus RETFound, DDR, QWK")
    print("=" * 92)

    imagenet_qwk = {s: qwk(truth, preds("full", REFERENCE, s)) for s in SEEDS}
    retfound_qwk = {s: qwk(truth, preds("full", CANDIDATE, s)) for s in SEEDS}
    d_full = per_seed_difference("full", qwk)

    print(f"  {'seed':>5} {'ImageNet-MAE':>14} {'RETFound':>12} {'delta':>10}")
    for s in SEEDS:
        print(f"  {s:>5} {imagenet_qwk[s]:>14.4f} {retfound_qwk[s]:>12.4f} "
              f"{d_full[s]:>+10.4f}")

    primary = seed_level_test(list(d_full.values()))
    lo, hi = crossed_ci(difference_replicate("full", qwk))
    print(f"\n  mean paired delta      {primary['mean']:+.4f}")
    print(f"  SD of paired deltas    {primary['seed_sd']:.4f}")
    print(f"  crossed 95% CI         [{lo:+.4f}, {hi:+.4f}]   (uncertainty)")
    print(f"  paired t-test p        {primary['p_ttest']:.4f}   (inference)")
    print(f"  sign-flip p            {primary['p_signflip']:.4f}   "
          f"(sensitivity; floor {primary['min_attainable_p']:.4f} at n=5)")
    print(f"  sign agreement         {primary['sign_agreement']}/5")
    established = primary["p_ttest"] < 0.05 and (lo > 0 or hi < 0)
    if established:
        winner = "ImageNet-MAE" if primary["mean"] > 0 else "RETFound"
        print(f"\n  -> difference established, favouring {winner}")
    else:
        print("\n  -> no difference demonstrated under full fine-tuning.")
        print("     This is NOT evidence of equivalence: a null at five seeds")
        print("     bounds the effect loosely, and the interval above says how")
        print("     loosely.")

    # ============================================ item 8: primary interaction
    print("\n" + "=" * 92)
    print("PRIMARY INTERACTION -- I_full(s) = D_full(s) - D_frozen(s)")
    print("=" * 92)
    d_frozen = per_seed_difference("frozen", qwk)
    d_partial = per_seed_difference("partial", qwk)
    i_full = {s: d_full[s] - d_frozen[s] for s in SEEDS}

    print(f"  {'seed':>5} {'D_frozen':>10} {'D_full':>10} {'I_full':>10}")
    for s in SEEDS:
        print(f"  {s:>5} {d_frozen[s]:>+10.4f} {d_full[s]:>+10.4f} "
              f"{i_full[s]:>+10.4f}")

    interaction = seed_level_test(list(i_full.values()))
    ilo, ihi = crossed_ci(interaction_replicate("full", "frozen", qwk))
    print(f"\n  mean I_full            {interaction['mean']:+.4f}")
    print(f"  SD                     {interaction['seed_sd']:.4f}")
    print(f"  crossed 95% CI         [{ilo:+.4f}, {ihi:+.4f}]   "
          f"(one case sample shared by all four conditions)")
    print(f"  paired t-test p        {interaction['p_ttest']:.4f}")
    print(f"  sign-flip p            {interaction['p_signflip']:.4f}")
    print(f"  sign agreement         {interaction['sign_agreement']}/5")

    # ========================================== item 9: secondary interactions
    print("\n" + "=" * 92)
    print("SECONDARY interactions (Holm-corrected within this two-member family)")
    print("=" * 92)
    secondary = []
    for label, later, earlier in [("D_partial - D_frozen", "partial", "frozen"),
                                  ("D_full - D_partial", "full", "partial")]:
        values = {s: (per_seed_difference(later, qwk)[s]
                      - per_seed_difference(earlier, qwk)[s]) for s in SEEDS}
        test = seed_level_test(list(values.values()))
        clo, chi = crossed_ci(interaction_replicate(later, earlier, qwk))
        secondary.append({"contrast": label, "later": later, "earlier": earlier,
                          "mean": test["mean"], "seed_sd": test["seed_sd"],
                          "ci_lower": clo, "ci_upper": chi,
                          "p_ttest": test["p_ttest"],
                          "p_signflip": test["p_signflip"],
                          "sign_agreement": test["sign_agreement"],
                          **{f"seed{s}": v for s, v in values.items()}})
    holm = holm_adjust([r["p_ttest"] for r in secondary])
    for record, adjusted in zip(secondary, holm):
        record["p_holm"] = adjusted
        print(f"  {record['contrast']:<22} mean {record['mean']:+.4f}  "
              f"CI [{record['ci_lower']:+.4f}, {record['ci_upper']:+.4f}]  "
              f"p {record['p_ttest']:.4f}  Holm {adjusted:.4f}  "
              f"sign {record['sign_agreement']}/5")
    print("\n  These are SECONDARY. They do not replace the primary")
    print("  full-vs-frozen interaction above.")

    # ============================================ item 10: secondary outcomes
    print("\n" + "=" * 92)
    print("SECONDARY OUTCOMES -- full fine-tuning, five seeds. QWK stays primary.")
    print("=" * 92)
    outcome_rows = []
    for name in [PRIMARY_METRIC] + SECONDARY_METRICS:
        function = METRIC_FUNCTIONS[name]

        def scored(backbone, seed, _f=function, _n=name):
            run = runs[("full", backbone, seed)]
            probabilities = run["scaled"] if _n in ("ece", "nll") else run["probabilities"]
            return float(_f(run["y_true"], run["y_pred"], probabilities))

        reference_values = np.array([scored(REFERENCE, s) for s in SEEDS])
        candidate_values = np.array([scored(CANDIDATE, s) for s in SEEDS])
        deltas = reference_values - candidate_values
        test = seed_level_test(list(deltas))
        outcome_rows.append({
            "metric": name,
            "imagenet_mean": float(reference_values.mean()),
            "imagenet_sd": float(reference_values.std(ddof=1)),
            "retfound_mean": float(candidate_values.mean()),
            "retfound_sd": float(candidate_values.std(ddof=1)),
            "delta_mean": test["mean"], "delta_sd": test["seed_sd"],
            "p_ttest": test["p_ttest"], "p_signflip": test["p_signflip"],
            "sign_agreement": test["sign_agreement"],
            "primary": name == PRIMARY_METRIC,
        })
    outcomes = pd.DataFrame(outcome_rows)
    print(outcomes[["metric", "imagenet_mean", "retfound_mean", "delta_mean",
                    "delta_sd", "p_ttest", "sign_agreement"]]
          .round(4).to_string(index=False))
    print("\n  p-values above are uncorrected and exploratory. QWK is the")
    print("  pre-specified primary endpoint; none of these replaces it.")

    # -------------------------------------------------------------- save
    pd.DataFrame([{
        "seed": s, "imagenet_qwk": imagenet_qwk[s], "retfound_qwk": retfound_qwk[s],
        "delta_full": d_full[s], "delta_frozen": d_frozen[s],
        "delta_partial": d_partial[s], "interaction_full_vs_frozen": i_full[s],
    } for s in SEEDS]).to_csv(tables / "full_finetune_per_seed.csv", index=False)

    pd.DataFrame([{
        "analysis": "primary_model_comparison", "protocol": "full",
        "metric": "qwk", "n_seeds": 5, "n_test": n_cases,
        "mean": primary["mean"], "seed_sd": primary["seed_sd"],
        "ci_lower": lo, "ci_upper": hi, "p_ttest": primary["p_ttest"],
        "p_signflip": primary["p_signflip"],
        "sign_agreement": primary["sign_agreement"],
        "min_attainable_signflip_p": primary["min_attainable_p"],
    }, {
        "analysis": "primary_interaction_full_vs_frozen", "protocol": "full-frozen",
        "metric": "qwk", "n_seeds": 5, "n_test": n_cases,
        "mean": interaction["mean"], "seed_sd": interaction["seed_sd"],
        "ci_lower": ilo, "ci_upper": ihi, "p_ttest": interaction["p_ttest"],
        "p_signflip": interaction["p_signflip"],
        "sign_agreement": interaction["sign_agreement"],
        "min_attainable_signflip_p": interaction["min_attainable_p"],
    }]).to_csv(tables / "full_finetune_primary.csv", index=False)

    pd.DataFrame(secondary).to_csv(
        tables / "full_finetune_secondary_interactions.csv", index=False)
    outcomes.to_csv(tables / "full_finetune_secondary_outcomes.csv", index=False)

    print("\nsaved -> full_finetune_per_seed.csv, full_finetune_primary.csv,")
    print("         full_finetune_secondary_interactions.csv,")
    print("         full_finetune_secondary_outcomes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
