"""The APTOS confirmatory replication of the frozen DDR full fine-tuning result.

Pre-registered in `docs/APTOS_FULL_FINETUNE_REPLICATION_PROTOCOL.md`, committed
before the first APTOS full-FT run. **This file was written while three of the
ten APTOS full-FT runs existed and no APTOS target metric had been read**, which
is the only point at which the analysis can be specified without the result
influencing it.

It refuses to produce anything until all thirty APTOS runs (three protocols, two
initialisations, five seeds) are present. There is no partial mode and no
override flag: an interim look at a subset is exactly what target blindness
exists to prevent.

The division of labour between the two uncertainty tools is the one this project
got wrong once:

* the **crossed seed x case bootstrap** gives the estimate and a 95% interval,
  and no p-value. Its distribution is built around the empirical estimate rather
  than under H0, so its tail mass is not calibrated and must never be Holm-
  corrected into something that looks like a controlled error rate;
* **seed-level inference** (`src/evaluation/seed_inference.py`) gives the formal
  test -- a one-sample t-test on the per-seed paired effects, with the exact
  sign-flip permutation as a sensitivity check. At five seeds the smallest
  attainable sign-flip p is 0.0625, so it cannot reach 0.05 however clean the
  data; it is reported as sensitivity, never as the test.

A claim is **established** only when the seed-level test survives Holm *and* the
crossed interval excludes zero.

What it produces:

1. **APTOS primary model comparison** -- ImageNet-MAE minus RETFound under full
   fine-tuning, QWK, five paired seeds.
2. **APTOS primary interaction** -- I_full(s) = D_full(s) - D_frozen(s).
3. **The two-domain primary family** -- the full-vs-frozen interaction on DDR
   and on APTOS, Holm-corrected across exactly those two members. DDR is
   recomputed from its own saved predictions and **asserted equal to the frozen
   `full_finetune_primary.csv`** before it is used, so the family provably rests
   on the result that was frozen rather than a drifted recomputation.
   **No pooled two-domain p-value** -- two domains with different test-set sizes
   and different shift structures pool into a number describing neither.
4. **Adaptation depth** -- frozen / partial / full on APTOS, descriptive.
5. **Secondary outcomes** -- every supported metric under full FT. QWK stays
   primary; no metric is promoted because it reads better.
6. **The central figure table** -- domain x protocol x seed x model QWK.

A null is reported as "not demonstrated". It is never reported as equivalence.

Reads saved predictions only. No GPU.

Usage:
    python analyse_aptos_replication.py
    python analyse_aptos_replication.py --n-bootstrap 5000
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

SOURCES = {"ddr": "aptos-eyepacs-idrid", "aptos": "ddr-eyepacs-idrid"}
SEEDS = [42, 1, 2, 3, 4]
REFERENCE = "vit-large-mae-in1k"          # ImageNet-MAE
CANDIDATE = "retfound-cfp"                # RETFound
PROTOCOLS = {
    "frozen": "linprobe-b256",
    "partial": "erm-b16-tb4-lr0.0001",
    "full": "erm-b16-ftfull-lr0.0001",
}
PRIMARY_METRIC = "qwk"
SECONDARY_METRICS = ["f1_macro", "balanced_accuracy", "severe_error_rate",
                     "mae_grade", "within_1_grade", "auroc_macro", "ece", "nll"]

# Both fixed to the values `analyse_full_finetune.py` used, so recomputing DDR
# here reproduces the frozen interval bit for bit rather than approximately.
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 7


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

    # ------------------------------------------------------------- loading
    def load_domain(target):
        runs, missing = {}, []
        for protocol, tag in PROTOCOLS.items():
            for backbone in (REFERENCE, CANDIDATE):
                for seed in SEEDS:
                    experiment_id = (
                        f"lodo_{SOURCES[target]}__{target}_{backbone}_{tag}_s{seed}")
                    run = load_target_predictions(experiment_id, target,
                                                  outputs_dir=outputs)
                    if run is None:
                        missing.append(experiment_id)
                    else:
                        runs[(protocol, backbone, seed)] = run
        return runs, missing

    expected = len(PROTOCOLS) * 2 * len(SEEDS)
    domains = {}
    for target in ("aptos", "ddr"):
        runs, missing = load_domain(target)
        print(f"{target}: loaded {len(runs)} of {expected} runs")
        if missing:
            print(f"\nNOT RUN -- {len(missing)} {target} run(s) absent:")
            for experiment_id in missing:
                print(f"  {experiment_id}")
            print("\nThe replication analysis does not run on a subset. Until "
                  "every\nrun exists, no APTOS estimate, interval, p-value or "
                  "sign count is\nproduced -- that is what the pre-registered "
                  "target blindness means.")
            return 1

        # A difference between two runs is a difference between two models only
        # if they scored the same images against the same labels.
        anchor = runs[("full", REFERENCE, SEEDS[0])]
        truth = anchor["y_true"]
        for key, run in runs.items():
            assert len(run["y_true"]) == len(truth), f"{target} {key}: size differs"
            assert (run["y_true"] == truth).all(), f"{target} {key}: labels differ"
        domains[target] = {"runs": runs, "truth": truth, "n_cases": len(truth)}
        print(f"{target}: all runs scored the same {len(truth):,} images")

    print()

    # -------------------------------------------------- per-domain machinery
    def make_tools(target):
        runs = domains[target]["runs"]
        truth = domains[target]["truth"]
        n_cases = domains[target]["n_cases"]

        def preds(protocol, backbone, seed):
            return runs[(protocol, backbone, seed)]["y_pred"]

        def crossed_ci(per_replicate, alpha=0.05):
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            draws = np.empty(n_bootstrap, dtype=float)
            for i in range(n_bootstrap):
                picked = rng.choice(SEEDS, size=len(SEEDS), replace=True)
                cases = rng.integers(0, n_cases, size=n_cases)
                draws[i] = per_replicate(picked, cases)
            lower, upper = np.percentile(
                draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            return float(lower), float(upper)

        def difference_replicate(protocol, metric):
            def inner(picked, cases):
                t = truth[cases]
                return float(np.mean([
                    metric(t, preds(protocol, REFERENCE, s)[cases])
                    - metric(t, preds(protocol, CANDIDATE, s)[cases])
                    for s in picked]))
            return inner

        def interaction_replicate(later, earlier, metric):
            """One crossed replicate of D_later - D_earlier.

            The same `cases` sample is applied to all four conditions -- both
            models under both protocols -- which is what preserves the pairing
            the contrast draws its power from.
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

        return {"runs": runs, "truth": truth, "n_cases": n_cases, "preds": preds,
                "crossed_ci": crossed_ci, "difference": difference_replicate,
                "interaction": interaction_replicate,
                "per_seed_difference": per_seed_difference}

    aptos = make_tools("aptos")
    ddr = make_tools("ddr")

    # ============================================== APTOS primary comparison
    print("=" * 92)
    print("APTOS PRIMARY -- full fine-tuning, ImageNet-MAE minus RETFound, QWK")
    print("=" * 92)

    imagenet_qwk = {s: qwk(aptos["truth"], aptos["preds"]("full", REFERENCE, s))
                    for s in SEEDS}
    retfound_qwk = {s: qwk(aptos["truth"], aptos["preds"]("full", CANDIDATE, s))
                    for s in SEEDS}
    d_full = aptos["per_seed_difference"]("full", qwk)
    d_frozen = aptos["per_seed_difference"]("frozen", qwk)
    d_partial = aptos["per_seed_difference"]("partial", qwk)

    print(f"  {'seed':>5} {'ImageNet-MAE':>14} {'RETFound':>12} {'delta':>10}")
    for s in SEEDS:
        print(f"  {s:>5} {imagenet_qwk[s]:>14.4f} {retfound_qwk[s]:>12.4f} "
              f"{d_full[s]:>+10.4f}")

    primary = seed_level_test(list(d_full.values()))
    lo, hi = aptos["crossed_ci"](aptos["difference"]("full", qwk))
    print(f"\n  mean paired delta      {primary['mean']:+.4f}")
    print(f"  SD of paired deltas    {primary['seed_sd']:.4f}")
    print(f"  crossed 95% CI         [{lo:+.4f}, {hi:+.4f}]   (uncertainty)")
    print(f"  paired t-test p        {primary['p_ttest']:.4f}   (inference)")
    print(f"  sign-flip p            {primary['p_signflip']:.4f}   "
          f"(sensitivity; floor {primary['min_attainable_p']:.4f} at n=5)")
    print(f"  sign agreement         {primary['sign_agreement']}/5")
    if primary["p_ttest"] < 0.05 and (lo > 0 or hi < 0):
        winner = "ImageNet-MAE" if primary["mean"] > 0 else "RETFound"
        print(f"\n  -> difference established, favouring {winner}")
    else:
        print("\n  -> no difference demonstrated under full fine-tuning on APTOS.")
        print("     NOT evidence of equivalence: a null at five seeds bounds the")
        print("     effect only as tightly as the interval above.")

    # ============================================== APTOS primary interaction
    print("\n" + "=" * 92)
    print("APTOS PRIMARY INTERACTION -- I_full(s) = D_full(s) - D_frozen(s)")
    print("=" * 92)
    i_full = {s: d_full[s] - d_frozen[s] for s in SEEDS}
    print(f"  {'seed':>5} {'D_frozen':>10} {'D_full':>10} {'I_full':>10}")
    for s in SEEDS:
        print(f"  {s:>5} {d_frozen[s]:>+10.4f} {d_full[s]:>+10.4f} "
              f"{i_full[s]:>+10.4f}")

    aptos_interaction = seed_level_test(list(i_full.values()))
    aptos_lo, aptos_hi = aptos["crossed_ci"](
        aptos["interaction"]("full", "frozen", qwk))
    print(f"\n  mean I_full            {aptos_interaction['mean']:+.4f}")
    print(f"  SD                     {aptos_interaction['seed_sd']:.4f}")
    print(f"  crossed 95% CI         [{aptos_lo:+.4f}, {aptos_hi:+.4f}]")
    print(f"  paired t-test p        {aptos_interaction['p_ttest']:.4f}")
    print(f"  sign-flip p            {aptos_interaction['p_signflip']:.4f}")
    print(f"  sign agreement         {aptos_interaction['sign_agreement']}/5")

    # =================================================== two-domain family
    print("\n" + "=" * 92)
    print("PRIMARY FAMILY -- full-vs-frozen interaction on DDR and APTOS")
    print("=" * 92)

    ddr_d_full = ddr["per_seed_difference"]("full", qwk)
    ddr_d_frozen = ddr["per_seed_difference"]("frozen", qwk)
    ddr_i_full = {s: ddr_d_full[s] - ddr_d_frozen[s] for s in SEEDS}
    ddr_interaction = seed_level_test(list(ddr_i_full.values()))
    ddr_lo, ddr_hi = ddr["crossed_ci"](ddr["interaction"]("full", "frozen", qwk))

    # The frozen DDR result is the thing being replicated. If recomputing it
    # from its own predictions no longer reproduces it, the family is not
    # resting on what was frozen and nothing below should be believed.
    frozen_path = tables / "full_finetune_primary.csv"
    if not frozen_path.exists():
        print(f"!! STOP -- {frozen_path.name} missing; the frozen DDR result "
              f"cannot be verified")
        return 1
    frozen = pd.read_csv(frozen_path)
    record = frozen[frozen.analysis == "primary_interaction_full_vs_frozen"]
    if len(record) != 1:
        print("!! STOP -- frozen DDR primary interaction row not found exactly once")
        return 1
    record = record.iloc[0]
    drift = {
        "mean": abs(ddr_interaction["mean"] - float(record["mean"])),
        "p_ttest": abs(ddr_interaction["p_ttest"] - float(record["p_ttest"])),
    }
    if n_bootstrap == N_BOOTSTRAP:
        drift["ci_lower"] = abs(ddr_lo - float(record["ci_lower"]))
        drift["ci_upper"] = abs(ddr_hi - float(record["ci_upper"]))
    worst = max(drift.values())
    if worst > 1e-9:
        print("!! STOP -- recomputed DDR interaction does not match the frozen "
              "result:")
        for key, value in drift.items():
            print(f"    {key}: differs by {value:.3e}")
        return 1
    checked = ", ".join(sorted(drift))
    print(f"  DDR recomputed from its own predictions and verified against "
          f"frozen\n  full_finetune_primary.csv ({checked}; worst drift "
          f"{worst:.1e}).")
    if n_bootstrap != N_BOOTSTRAP:
        print(f"  Interval check skipped: --n-bootstrap {n_bootstrap} is not the "
              f"frozen {N_BOOTSTRAP}.")

    family = [
        {"domain": "DDR", "n_test": ddr["n_cases"], "mean": ddr_interaction["mean"],
         "seed_sd": ddr_interaction["seed_sd"], "ci_lower": ddr_lo,
         "ci_upper": ddr_hi, "p_ttest": ddr_interaction["p_ttest"],
         "p_signflip": ddr_interaction["p_signflip"],
         "sign_agreement": ddr_interaction["sign_agreement"],
         "min_attainable_signflip_p": ddr_interaction["min_attainable_p"],
         **{f"seed{s}": ddr_i_full[s] for s in SEEDS}},
        {"domain": "APTOS", "n_test": aptos["n_cases"],
         "mean": aptos_interaction["mean"], "seed_sd": aptos_interaction["seed_sd"],
         "ci_lower": aptos_lo, "ci_upper": aptos_hi,
         "p_ttest": aptos_interaction["p_ttest"],
         "p_signflip": aptos_interaction["p_signflip"],
         "sign_agreement": aptos_interaction["sign_agreement"],
         "min_attainable_signflip_p": aptos_interaction["min_attainable_p"],
         **{f"seed{s}": i_full[s] for s in SEEDS}},
    ]
    for entry, adjusted in zip(family, holm_adjust([e["p_ttest"] for e in family])):
        entry["p_holm"] = adjusted
        entry["ci_excludes_zero"] = bool(entry["ci_lower"] > 0
                                         or entry["ci_upper"] < 0)
        entry["established"] = bool(adjusted < 0.05 and entry["ci_excludes_zero"])

    print(f"\n  {'domain':<7} {'n_test':>7} {'mean':>9} {'95% CI':>21} "
          f"{'p':>8} {'Holm':>8} {'sign':>6}")
    for entry in family:
        interval = f"[{entry['ci_lower']:+.4f}, {entry['ci_upper']:+.4f}]"
        print(f"  {entry['domain']:<7} {entry['n_test']:>7,} "
              f"{entry['mean']:>+9.4f} {interval:>21} "
              f"{entry['p_ttest']:>8.4f} {entry['p_holm']:>8.4f} "
              f"{entry['sign_agreement']:>4}/5")
    for entry in family:
        verdict = ("established (Holm-adjusted test and interval agree)"
                   if entry["established"] else
                   "not demonstrated -- and not a demonstration of equivalence")
        print(f"    {entry['domain']:<6} -> {verdict}")

    agree = {entry["domain"]: np.sign(entry["mean"]) for entry in family}
    print(f"\n  Direction: DDR {'negative' if agree['DDR'] < 0 else 'positive'}, "
          f"APTOS {'negative' if agree['APTOS'] < 0 else 'positive'}.")
    print("  No pooled two-domain p-value is computed. The two test sets differ in")
    print("  size and in shift structure; a pooled number would describe neither.")
    print("  The two estimates with their intervals are the honest presentation.")

    # =================================================== adaptation depth
    print("\n" + "=" * 92)
    print("APTOS adaptation depth -- descriptive; the primary family is above")
    print("=" * 92)
    depth = []
    for protocol in ("frozen", "partial", "full"):
        reference_values = np.array(
            [qwk(aptos["truth"], aptos["preds"](protocol, REFERENCE, s))
             for s in SEEDS])
        candidate_values = np.array(
            [qwk(aptos["truth"], aptos["preds"](protocol, CANDIDATE, s))
             for s in SEEDS])
        deltas = aptos["per_seed_difference"](protocol, qwk)
        test = seed_level_test(list(deltas.values()))
        plo, phi = aptos["crossed_ci"](aptos["difference"](protocol, qwk))
        depth.append({
            "protocol": protocol,
            "imagenet_mean": float(reference_values.mean()),
            "imagenet_sd": float(reference_values.std(ddof=1)),
            "retfound_mean": float(candidate_values.mean()),
            "retfound_sd": float(candidate_values.std(ddof=1)),
            "delta_mean": test["mean"], "delta_sd": test["seed_sd"],
            "ci_lower": plo, "ci_upper": phi, "p_ttest": test["p_ttest"],
            "p_signflip": test["p_signflip"],
            "sign_agreement": test["sign_agreement"],
            **{f"seed{s}": deltas[s] for s in SEEDS},
        })
    print(pd.DataFrame(depth)[["protocol", "imagenet_mean", "retfound_mean",
                               "delta_mean", "ci_lower", "ci_upper", "p_ttest",
                               "sign_agreement"]].round(4).to_string(index=False))
    print("\n  Uncorrected. These three are descriptive context for the")
    print("  interaction, not a second family of tests.")

    # =================================================== secondary outcomes
    print("\n" + "=" * 92)
    print("APTOS secondary outcomes -- full fine-tuning. QWK stays primary.")
    print("=" * 92)
    outcome_rows = []
    for name in [PRIMARY_METRIC] + SECONDARY_METRICS:
        function = METRIC_FUNCTIONS[name]

        def scored(backbone, seed, _f=function, _n=name):
            run = aptos["runs"][("full", backbone, seed)]
            probabilities = (run["scaled"] if _n in ("ece", "nll")
                             else run["probabilities"])
            return float(_f(run["y_true"], run["y_pred"], probabilities))

        reference_values = np.array([scored(REFERENCE, s) for s in SEEDS])
        candidate_values = np.array([scored(CANDIDATE, s) for s in SEEDS])
        test = seed_level_test(list(reference_values - candidate_values))
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
    print("\n  Uncorrected and exploratory. QWK is the pre-specified primary")
    print("  endpoint and none of these replaces it.")

    # ================================================= central figure table
    figure_rows = []
    for target, tools in (("DDR", ddr), ("APTOS", aptos)):
        for protocol in ("frozen", "partial", "full"):
            for seed in SEEDS:
                reference_value = qwk(tools["truth"],
                                      tools["preds"](protocol, REFERENCE, seed))
                candidate_value = qwk(tools["truth"],
                                      tools["preds"](protocol, CANDIDATE, seed))
                figure_rows.append({
                    "domain": target, "protocol": protocol, "seed": seed,
                    "imagenet_qwk": float(reference_value),
                    "retfound_qwk": float(candidate_value),
                    "delta": float(reference_value - candidate_value),
                    "n_test": tools["n_cases"],
                })

    # ------------------------------------------------------------- saving
    pd.DataFrame([{
        "seed": s, "imagenet_qwk": imagenet_qwk[s], "retfound_qwk": retfound_qwk[s],
        "delta_full": d_full[s], "delta_frozen": d_frozen[s],
        "delta_partial": d_partial[s], "interaction_full_vs_frozen": i_full[s],
    } for s in SEEDS]).to_csv(tables / "aptos_full_finetune_per_seed.csv",
                              index=False)

    pd.DataFrame([{
        "analysis": "primary_model_comparison", "protocol": "full",
        "metric": "qwk", "n_seeds": 5, "n_test": aptos["n_cases"],
        "mean": primary["mean"], "seed_sd": primary["seed_sd"],
        "ci_lower": lo, "ci_upper": hi, "p_ttest": primary["p_ttest"],
        "p_signflip": primary["p_signflip"],
        "sign_agreement": primary["sign_agreement"],
        "min_attainable_signflip_p": primary["min_attainable_p"],
    }, {
        "analysis": "primary_interaction_full_vs_frozen", "protocol": "full-frozen",
        "metric": "qwk", "n_seeds": 5, "n_test": aptos["n_cases"],
        "mean": aptos_interaction["mean"], "seed_sd": aptos_interaction["seed_sd"],
        "ci_lower": aptos_lo, "ci_upper": aptos_hi,
        "p_ttest": aptos_interaction["p_ttest"],
        "p_signflip": aptos_interaction["p_signflip"],
        "sign_agreement": aptos_interaction["sign_agreement"],
        "min_attainable_signflip_p": aptos_interaction["min_attainable_p"],
    }]).to_csv(tables / "aptos_full_finetune_primary.csv", index=False)

    pd.DataFrame(family).to_csv(
        tables / "two_domain_interaction_holm.csv", index=False)
    pd.DataFrame(depth).to_csv(
        tables / "aptos_adaptation_depth.csv", index=False)
    outcomes.to_csv(tables / "aptos_secondary_outcomes.csv", index=False)
    pd.DataFrame(figure_rows).to_csv(
        tables / "figure_qwk_by_domain_protocol_seed.csv", index=False)

    print("\nsaved -> aptos_full_finetune_per_seed.csv,")
    print("         aptos_full_finetune_primary.csv,")
    print("         two_domain_interaction_holm.csv,")
    print("         aptos_adaptation_depth.csv,")
    print("         aptos_secondary_outcomes.csv,")
    print("         figure_qwk_by_domain_protocol_seed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
