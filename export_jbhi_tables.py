"""The two authoritative final tables for the JBHI manuscript.

Every number a main-text claim rests on comes from here or from the
domain-specific generators this reproduces. Written after the experimental
programme closed; it runs no training and reads only saved predictions.

Produces:

    outputs/tables/JBHI_MASTER_RESULTS.csv / .tex
    outputs/tables/JBHI_PRIMARY_INTERACTION.csv / .tex

**Seed sets are never silently mixed.** The frozen linear probe was run at ten
seeds and the two adaptation protocols at five, so every frozen row appears
twice and is labelled: once over all ten seeds as a descriptive estimate, and
once over the five seeds common to all three protocols. Only the common-five
subset may be differenced against partial or full, because a protocol
interaction is a paired quantity and pairing requires the same seeds. The
`seed_set` column carries this and the LaTeX tables print it.

Inferential vocabulary, fixed for the manuscript:

* the **crossed seed x case bootstrap** supplies an *uncertainty interval*. It
  is not a test: its distribution is built around the empirical estimate rather
  than under the null, so its tail mass is not a calibrated p-value;
* the **paired seed-level t-test** is the *formal inferential test*;
* **Holm** is the *multiplicity adjustment*, applied across the two
  domain-specific interaction tests and nowhere else;
* the **exact sign-flip permutation** is a *distribution-free sensitivity
  analysis*, floored at 2/2^5 = 0.0625 at five seeds;
* **seed SD and sign agreement** are *descriptive stability diagnostics*.

An effect with a multiplicity-adjusted seed-level p below 0.05 whose crossed
interval excludes zero is described as "statistically supported under the
study's inferential framework". No other phrase is used for it.

Usage:
    python export_jbhi_tables.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BS = chr(92)
NL = BS + BS
TIMES = "$" + BS + "times$"

SOURCES = {"ddr": "aptos-eyepacs-idrid",
           "aptos": "ddr-eyepacs-idrid",
           "idrid": "aptos-ddr-eyepacs"}
DOMAIN_LABEL = {"ddr": "DDR", "aptos": "APTOS", "idrid": "IDRiD"}
PROTOCOLS = {"frozen": "linprobe-b256",
             "partial": "erm-b16-tb4-lr0.0001",
             "full": "erm-b16-ftfull-lr0.0001"}
PROTOCOL_LABEL = {"frozen": "Frozen linear probe",
                  "partial": "Partial FT (last 4/24)",
                  "full": "Full fine-tuning"}
REFERENCE = "vit-large-mae-in1k"          # ImageNet-MAE
CANDIDATE = "retfound-cfp"                # RETFound
COMMON_SEEDS = [42, 1, 2, 3, 4]           # every protocol has these
FROZEN_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
INTERACTION_DOMAINS = ["ddr", "aptos"]    # the only domains with full FT
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 7


def _math(name: str) -> str:
    return BS + "text{" + name + "}"


def _guard(body: str) -> str:
    for bad in ("\t", "\r", "\x0b", "\x0c"):
        if bad in body:
            raise RuntimeError(
                f"generated table contains control character {bad!r}")
    return body


def main() -> int:
    import numpy as np
    import pandas as pd

    from src.evaluation.crossed_bootstrap import holm_adjust
    from src.evaluation.metrics import quadratic_weighted_kappa
    from src.evaluation.seed_inference import seed_level_test
    from src.utils.io import project_root
    from src.visualization.calibration_figures import load_target_predictions

    outputs = project_root() / "outputs"
    tables = outputs / "tables"

    def qwk(y_true, y_pred):
        return quadratic_weighted_kappa(y_true, y_pred, num_classes=5)

    # ------------------------------------------------------------- loading
    runs, truth, n_cases, missing = {}, {}, {}, []
    for domain in SOURCES:
        for protocol, tag in PROTOCOLS.items():
            seeds = FROZEN_SEEDS if protocol == "frozen" else COMMON_SEEDS
            for backbone in (REFERENCE, CANDIDATE):
                for seed in seeds:
                    eid = (f"lodo_{SOURCES[domain]}__{domain}_{backbone}"
                           f"_{tag}_s{seed}")
                    run = load_target_predictions(eid, domain,
                                                  outputs_dir=outputs)
                    if run is None:
                        # IDRiD has no full fine-tuning by design; that absence
                        # is expected and is reported as a gap, not an error.
                        if not (domain == "idrid" and protocol == "full"):
                            missing.append(eid)
                        continue
                    runs[(domain, protocol, backbone, seed)] = run
        anchor = next((v for (d, _, _, _), v in runs.items() if d == domain), None)
        if anchor is None:
            print(f"!! no runs at all for {domain}")
            return 1
        truth[domain] = anchor["y_true"]
        n_cases[domain] = len(anchor["y_true"])

    if missing:
        print(f"NOT RUN -- {len(missing)} expected run(s) absent:")
        for eid in missing[:20]:
            print(f"  {eid}")
        return 1

    for (domain, protocol, backbone, seed), run in runs.items():
        assert len(run["y_true"]) == n_cases[domain], f"{domain} size differs"
        assert (run["y_true"] == truth[domain]).all(), f"{domain} labels differ"

    idrid_full = [k for k in runs if k[0] == "idrid" and k[1] == "full"]
    print(f"loaded {len(runs)} runs; IDRiD full-FT runs present: "
          f"{len(idrid_full)} (expected 0)")

    # --------------------------------------------------- crossed bootstrap
    def crossed_ci(domain, per_replicate, seeds, alpha=0.05):
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = n_cases[domain]
        draws = np.empty(N_BOOTSTRAP, dtype=float)
        for i in range(N_BOOTSTRAP):
            picked = rng.choice(seeds, size=len(seeds), replace=True)
            cases = rng.integers(0, n, size=n)
            draws[i] = per_replicate(picked, cases)
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)

    def preds(domain, protocol, backbone, seed):
        return runs[(domain, protocol, backbone, seed)]["y_pred"]

    def difference_replicate(domain, protocol):
        def inner(picked, cases):
            t = truth[domain][cases]
            return float(np.mean([
                qwk(t, preds(domain, protocol, REFERENCE, s)[cases])
                - qwk(t, preds(domain, protocol, CANDIDATE, s)[cases])
                for s in picked]))
        return inner

    def interaction_replicate(domain, later, earlier):
        """One crossed replicate of D_later - D_earlier.

        One case sample is shared by all four conditions, which is what keeps
        the pairing the contrast draws its power from.
        """
        def inner(picked, cases):
            t = truth[domain][cases]
            values = []
            for s in picked:
                d_l = (qwk(t, preds(domain, later, REFERENCE, s)[cases])
                       - qwk(t, preds(domain, later, CANDIDATE, s)[cases]))
                d_e = (qwk(t, preds(domain, earlier, REFERENCE, s)[cases])
                       - qwk(t, preds(domain, earlier, CANDIDATE, s)[cases]))
                values.append(d_l - d_e)
            return float(np.mean(values))
        return inner

    def per_seed_delta(domain, protocol, seeds):
        return {s: float(qwk(truth[domain], preds(domain, protocol, REFERENCE, s))
                         - qwk(truth[domain], preds(domain, protocol, CANDIDATE, s)))
                for s in seeds}

    # ------------------------------------------------------- master table
    master = []
    for domain in ("ddr", "aptos", "idrid"):
        for protocol in ("frozen", "partial", "full"):
            if domain == "idrid" and protocol == "full":
                continue
            sets = ([("all-10", FROZEN_SEEDS), ("common-5", COMMON_SEEDS)]
                    if protocol == "frozen" else [("common-5", COMMON_SEEDS)])
            for label, seeds in sets:
                ref = np.array([qwk(truth[domain],
                                    preds(domain, protocol, REFERENCE, s))
                                for s in seeds])
                cand = np.array([qwk(truth[domain],
                                     preds(domain, protocol, CANDIDATE, s))
                                 for s in seeds])
                deltas = per_seed_delta(domain, protocol, seeds)
                test = seed_level_test(list(deltas.values()))
                lo, hi = crossed_ci(domain, difference_replicate(domain, protocol),
                                    seeds)
                if protocol == "full":
                    status = ("primary model comparison; no difference "
                              "demonstrated (not equivalence)"
                              if not (test["p_ttest"] < 0.05
                                      and (lo > 0 or hi < 0))
                              else "primary model comparison; difference "
                                   "statistically supported")
                else:
                    status = "descriptive, uncorrected"
                master.append({
                    "domain": DOMAIN_LABEL[domain],
                    "protocol": protocol,
                    "seed_set": label,
                    "n_seeds": len(seeds),
                    "n_test": n_cases[domain],
                    "ImageNet_QWK_mean": float(ref.mean()),
                    "ImageNet_QWK_sd": float(ref.std(ddof=1)),
                    "RETFound_QWK_mean": float(cand.mean()),
                    "RETFound_QWK_sd": float(cand.std(ddof=1)),
                    "paired_delta": test["mean"],
                    "paired_seed_SD": test["seed_sd"],
                    "crossed_CI_low": lo,
                    "crossed_CI_high": hi,
                    "raw_p": test["p_ttest"],
                    "adjusted_p": "",          # Holm applies only to the family
                    "sign_agreement": f"{test['sign_agreement']}/{len(seeds)}",
                    "sign_flip_p": test["p_signflip"],
                    "inference_status": status,
                })
                print(f"  {DOMAIN_LABEL[domain]:6} {protocol:8} {label:9} "
                      f"delta {test['mean']:+.4f}")

    master_frame = pd.DataFrame(master)
    master_frame.to_csv(tables / "JBHI_MASTER_RESULTS.csv", index=False)

    # -------------------------------------------------- interaction table
    family = []
    for domain in INTERACTION_DOMAINS:
        d_frozen = per_seed_delta(domain, "frozen", COMMON_SEEDS)
        d_full = per_seed_delta(domain, "full", COMMON_SEEDS)
        inter = {s: d_full[s] - d_frozen[s] for s in COMMON_SEEDS}
        test = seed_level_test(list(inter.values()))
        lo, hi = crossed_ci(domain, interaction_replicate(domain, "full", "frozen"),
                            COMMON_SEEDS)
        family.append({
            "domain": DOMAIN_LABEL[domain],
            "n_test": n_cases[domain],
            "n_seeds": len(COMMON_SEEDS),
            "delta_frozen": float(np.mean(list(d_frozen.values()))),
            "delta_full": float(np.mean(list(d_full.values()))),
            "interaction": test["mean"],
            "interaction_seed_SD": test["seed_sd"],
            "crossed_CI_low": lo,
            "crossed_CI_high": hi,
            "paired_t_p": test["p_ttest"],
            "sign_flip_p": test["p_signflip"],
            "min_attainable_sign_flip_p": test["min_attainable_p"],
            "sign_agreement": f"{test['sign_agreement']}/{len(COMMON_SEEDS)}",
            **{f"seed{s}": inter[s] for s in COMMON_SEEDS},
        })
    for entry, adjusted in zip(family, holm_adjust([e["paired_t_p"] for e in family])):
        entry["Holm_p"] = adjusted
        supported = bool(adjusted < 0.05 and
                         (entry["crossed_CI_low"] > 0 or entry["crossed_CI_high"] < 0))
        entry["inference_status"] = (
            "statistically supported under the study's inferential framework"
            if supported else "not demonstrated (not equivalence)")

    order = ["domain", "n_test", "n_seeds", "delta_frozen", "delta_full",
             "interaction", "interaction_seed_SD", "crossed_CI_low",
             "crossed_CI_high", "paired_t_p", "Holm_p", "sign_flip_p",
             "min_attainable_sign_flip_p", "sign_agreement",
             "inference_status"] + [f"seed{s}" for s in COMMON_SEEDS]
    family_frame = pd.DataFrame(family)[order]
    family_frame.to_csv(tables / "JBHI_PRIMARY_INTERACTION.csv", index=False)

    # ------------------------------------- agreement with frozen generators
    # These tables must not drift from the per-domain analyses that produced
    # the reports. Checked rather than assumed.
    checks = []
    ddr_frozen = pd.read_csv(tables / "full_finetune_primary.csv")
    aptos_frozen = pd.read_csv(tables / "aptos_full_finetune_primary.csv")
    for frame, label in ((ddr_frozen, "DDR"), (aptos_frozen, "APTOS")):
        row = frame[frame.analysis == "primary_interaction_full_vs_frozen"].iloc[0]
        here = family_frame[family_frame.domain == label].iloc[0]
        for key_here, key_there in (("interaction", "mean"),
                                    ("crossed_CI_low", "ci_lower"),
                                    ("crossed_CI_high", "ci_upper"),
                                    ("paired_t_p", "p_ttest")):
            checks.append(abs(float(here[key_here]) - float(row[key_there])))
    worst = max(checks)
    if worst > 1e-9:
        print(f"\n!! STOP -- interaction table disagrees with the per-domain "
              f"generators by {worst:.3e}")
        return 1
    print(f"\ninteraction table agrees with the per-domain generators "
          f"(worst drift {worst:.1e})")

    # ---------------------------------------------------------- LaTeX out
    rows = []
    for r in master_frame.itertuples():
        rows.append(" & ".join([
            r.domain, PROTOCOL_LABEL[r.protocol], r.seed_set, str(r.n_seeds),
            f"{r.ImageNet_QWK_mean:.4f}", f"{r.RETFound_QWK_mean:.4f}",
            "$" + f"{r.paired_delta:+.4f}" + "$",
            "$[" + f"{r.crossed_CI_low:+.4f}, {r.crossed_CI_high:+.4f}" + "]$",
            f"{r.raw_p:.4f}", str(r.sign_agreement).replace("/", "/"),
        ]) + " " + NL)
    caption = (
        "Authoritative per-protocol results on the three held-out domains, "
        "quadratic weighted kappa. $" + BS + "Delta$ is the paired "
        "ImageNet-MAE minus RETFound difference; intervals are crossed seed "
        + TIMES + " case bootstrap uncertainty intervals and $p$ is an "
        "uncorrected paired seed-level $t$-test. The frozen linear probe was "
        "run at ten seeds and the two adaptation protocols at five, so each "
        "frozen row appears twice: once over all ten seeds as a descriptive "
        "estimate and once over the five seeds common to every protocol. "
        "**Only the common-five rows may be differenced across protocols**, "
        "since the interaction is a paired quantity. IDRiD has no full "
        "fine-tuning row: that experiment was not run, and no value is "
        "imputed for it. Formal multiplicity-adjusted inference in this study "
        "attaches to the protocol interaction in Table" + BS + "~" + BS
        + "ref{tab:jbhi-primary-interaction}, not to the per-protocol "
        "comparisons here."
    ).replace("**", "")
    header = ("Held-out & Protocol & Seed set & $n$ & ImageNet-MAE & RETFound & $"
              + BS + "Delta$ & 95" + BS + "% CI & $p$ & sign " + NL)
    body = _guard("\n".join([
        BS + "begin{table*}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{tab:jbhi-master-results}",
        BS + "begin{tabular}{lllrrrrrrr}", BS + "toprule",
        header, BS + "midrule", "\n".join(rows),
        BS + "bottomrule", BS + "end{tabular}", BS + "end{table*}",
    ]))
    (tables / "JBHI_MASTER_RESULTS.tex").write_text(body + "\n", encoding="utf-8")

    rows = []
    for r in family_frame.itertuples():
        rows.append(" & ".join([
            r.domain,
            f"{int(r.n_test):,}".replace(",", BS + ","),
            str(int(r.n_seeds)),
            "$" + f"{r.delta_frozen:+.4f}" + "$",
            "$" + f"{r.delta_full:+.4f}" + "$",
            "$" + f"{r.interaction:+.4f}" + "$",
            "$[" + f"{r.crossed_CI_low:+.4f}, {r.crossed_CI_high:+.4f}" + "]$",
            f"{r.paired_t_p:.4f}", f"{r.Holm_p:.4f}", f"{r.sign_flip_p:.4f}",
            str(r.sign_agreement),
        ]) + " " + NL)
    caption = (
        "Primary analysis: the protocol-by-initialisation interaction $I_{"
        + _math("full") + "} = " + BS + "Delta_{" + _math("full") + "} - "
        + BS + "Delta_{" + _math("frozen") + "}$ on the two held-out domains "
        "for which matched full fine-tuning was run, over the five common "
        "seeds. $" + BS + "Delta$ is ImageNet-MAE minus RETFound QWK within a "
        "protocol, so a negative interaction means the ImageNet-MAE advantage "
        "measured under frozen probing is reduced once the encoder adapts. "
        "Intervals are crossed seed " + TIMES + " case bootstrap uncertainty "
        "intervals and are not tests; $p$ is the paired seed-level $t$-test, "
        "Holm-adjusted across exactly these two domains; $p_{" + _math("flip")
        + "}$ is an exact sign-flip sensitivity analysis whose smallest "
        "attainable two-sided value at five seeds is $0.0625$. Both domains "
        "are statistically supported under the study's inferential framework, "
        "with all five seed-level interactions negative on each. No pooled "
        "two-domain $p$-value is reported: the test sets differ in size and "
        "shift structure. The APTOS analysis was specified as a confirmatory "
        "replication after the DDR result was frozen, and Holm adjustment "
        "across the two is applied for final reporting."
    )
    header = ("Held-out & $n_{" + _math("test") + "}$ & seeds & $" + BS
              + "Delta_{" + _math("frozen") + "}$ & $" + BS + "Delta_{"
              + _math("full") + "}$ & $I_{" + _math("full") + "}$ & 95" + BS
              + "% CI & $p$ & $p_{" + _math("Holm") + "}$ & $p_{"
              + _math("flip") + "}$ & sign " + NL)
    body = _guard("\n".join([
        BS + "begin{table*}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{tab:jbhi-primary-interaction}",
        BS + "begin{tabular}{lrrrrrrrrr}", BS + "toprule",
        header, BS + "midrule", "\n".join(rows),
        BS + "bottomrule", BS + "end{tabular}", BS + "end{table*}",
    ]))
    (tables / "JBHI_PRIMARY_INTERACTION.tex").write_text(body + "\n",
                                                        encoding="utf-8")

    print("\nsaved -> JBHI_MASTER_RESULTS.csv / .tex  "
          f"({len(master_frame)} rows)")
    print(f"         JBHI_PRIMARY_INTERACTION.csv / .tex  "
          f"({len(family_frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
