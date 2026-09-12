"""Post-hoc secondary screening-oriented analysis: referable DR (ICDR grade >= 2).

**This is not a primary outcome and does not replace QWK.** It exists so a
clinical reader can interpret the same frozen predictions through the
operating definition screening programmes actually use.

Constraints honoured here, and worth stating because they are what keep this
from becoming a second headline:

* the threshold is the **fixed ICDR referable definition, grade >= 2**. It was
  not chosen, tuned or moved after seeing target performance, and it is the
  same constant (`REFERABLE_THRESHOLD`) the evaluation code has always used;
* **no significance test is run.** Means and across-seed SDs only. Testing
  eight conditions after the primary analysis has concluded would be searching
  for another positive result, which is exactly what a post-hoc analysis must
  not do;
* whatever it shows is kept.

Reuses `src.evaluation.metrics.referable_dr_metrics` rather than reimplementing
the definition, so the threshold and the AUROC construction cannot drift from
the rest of the project. AUROC uses the summed probability of grades >= 2 on
the **raw** probabilities, matching how macro AUROC is computed everywhere else
(temperature-scaled probabilities are used only for ECE and NLL).

Writes:
    outputs/tables/JBHI_REFERABLE_DR_SECONDARY.csv   one row per condition
    outputs/tables/JBHI_REFERABLE_DR_SECONDARY.tex   the manuscript table
    outputs/tables/JBHI_REFERABLE_DR_PER_SEED.csv    the five seeds per condition

Reads frozen predictions only. No training, no GPU.

Usage:
    python export_referable_dr_secondary.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BS = chr(92)
NL = BS + BS
PM = "$" + BS + "pm$"

SOURCES = {"ddr": "aptos-eyepacs-idrid", "aptos": "ddr-eyepacs-idrid"}
DOMAIN_LABEL = {"ddr": "DDR", "aptos": "APTOS"}
PROTOCOLS = {"frozen": "linprobe-b256", "full": "erm-b16-ftfull-lr0.0001"}
PROTOCOL_LABEL = {"frozen": "Frozen probe", "full": "Full fine-tuning"}
ARMS = {"vit-large-mae-in1k": "ImageNet-MAE", "retfound-cfp": "RETFound"}
SEEDS = [42, 1, 2, 3, 4]

METRICS = [
    ("referable_sensitivity", "sensitivity"),
    ("referable_specificity", "specificity"),
    ("referable_precision", "precision_ppv"),
    ("referable_f1", "f1"),
    ("referable_auroc", "auroc"),
]


def main() -> int:
    import numpy as np
    import pandas as pd

    from src.evaluation.metrics import REFERABLE_THRESHOLD, referable_dr_metrics
    from src.utils.io import project_root
    from src.visualization.calibration_figures import load_target_predictions

    outputs = project_root() / "outputs"
    tables = outputs / "tables"

    per_seed, missing = [], []
    for domain in ("ddr", "aptos"):
        for protocol, tag in PROTOCOLS.items():
            for backbone, arm in ARMS.items():
                for seed in SEEDS:
                    eid = (f"lodo_{SOURCES[domain]}__{domain}_{backbone}"
                           f"_{tag}_s{seed}")
                    run = load_target_predictions(eid, domain,
                                                  outputs_dir=outputs)
                    if run is None:
                        missing.append(eid)
                        continue
                    # Raw probabilities, matching the AUROC convention used by
                    # the primary secondary-outcome tables.
                    result = referable_dr_metrics(
                        run["y_true"], run["y_pred"], run["probabilities"])
                    per_seed.append({
                        "domain": DOMAIN_LABEL[domain],
                        "protocol": protocol,
                        "arm": arm,
                        "seed": seed,
                        "n": int(len(run["y_true"])),
                        "referable_prevalence": result["referable_prevalence"],
                        **{out: result.get(key, float("nan"))
                           for key, out in METRICS},
                    })

    if missing:
        print(f"NOT RUN -- {len(missing)} run(s) absent:")
        for eid in missing[:10]:
            print(f"  {eid}")
        return 1

    seeds_frame = pd.DataFrame(per_seed)
    seeds_frame.to_csv(tables / "JBHI_REFERABLE_DR_PER_SEED.csv", index=False)

    # The test set is identical across arms and protocols within a domain, so
    # n and prevalence are constants of the domain, not of the condition.
    for domain, group in seeds_frame.groupby("domain"):
        assert group.n.nunique() == 1, f"{domain}: test size varies"
        assert group.referable_prevalence.nunique() == 1, \
            f"{domain}: prevalence varies across conditions"

    rows = []
    for (domain, protocol, arm), group in seeds_frame.groupby(
            ["domain", "protocol", "arm"], sort=False):
        row = {
            "domain": domain,
            "protocol": protocol,
            "arm": arm,
            "n_seeds": len(group),
            "n": int(group.n.iloc[0]),
            "referable_prevalence": float(group.referable_prevalence.iloc[0]),
            "referable_threshold": int(REFERABLE_THRESHOLD),
        }
        for _, out in METRICS:
            row[f"{out}_mean"] = float(group[out].mean())
            row[f"{out}_sd"] = float(group[out].std(ddof=1))
        rows.append(row)

    order = {"DDR": 0, "APTOS": 1}
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        by=["domain", "protocol", "arm"],
        key=lambda s: s.map(order) if s.name == "domain"
        else s.map({"frozen": 0, "full": 1}) if s.name == "protocol" else s
    ).reset_index(drop=True)
    frame.to_csv(tables / "JBHI_REFERABLE_DR_SECONDARY.csv", index=False)

    # ------------------------------------------------------------ LaTeX
    lines = []
    for r in frame.itertuples():
        lines.append(" & ".join([
            r.domain, PROTOCOL_LABEL[r.protocol], r.arm,
            f"{r.sensitivity_mean:.3f} " + PM + f" {r.sensitivity_sd:.3f}",
            f"{r.specificity_mean:.3f} " + PM + f" {r.specificity_sd:.3f}",
            f"{r.precision_ppv_mean:.3f} " + PM + f" {r.precision_ppv_sd:.3f}",
            f"{r.f1_mean:.3f} " + PM + f" {r.f1_sd:.3f}",
            f"{r.auroc_mean:.3f} " + PM + f" {r.auroc_sd:.3f}",
        ]) + " " + NL)

    prevalence = {r.domain: (r.n, r.referable_prevalence)
                  for r in frame.itertuples()}
    prevalence_text = "; ".join(
        f"{d}: $n={n:,}$".replace(",", BS + ",") + f", prevalence {p:.3f}"
        for d, (n, p) in prevalence.items())

    caption = (
        "\\textbf{Post-hoc secondary screening-oriented analysis using the "
        "fixed ICDR referable-DR definition (grade $\\geq 2$).} Reported for "
        "clinical interpretability only; it is not a primary outcome and does "
        "not replace quadratic weighted kappa. The threshold is the fixed "
        "ICDR definition and was not tuned on either held-out domain, nor "
        "selected after inspecting target performance. Values are the mean "
        "$\\pm$ standard deviation over the five common seeds "
        "(42, 1, 2, 3, 4). AUROC uses the summed predicted probability of "
        "grades $\\geq 2$. The held-out test set is identical across arms and "
        "protocols within a domain, so $n$ and referable prevalence are "
        "properties of the domain (" + prevalence_text + "). "
        "\\textbf{No significance test is reported here}: the pre-specified "
        "inference is the protocol interaction on quadratic weighted kappa, "
        "and testing these conditions post hoc would amount to searching for "
        "an additional positive result."
    ).replace("\\textbf", BS + "textbf").replace("\\geq", BS + "geq") \
     .replace("\\pm", BS + "pm").replace("$\\", "$" + BS)

    header = ("Held-out & Protocol & Initialisation & Sensitivity & "
              "Specificity & PPV & F1 & AUROC " + NL)
    body = "\n".join([
        BS + "begin{table*}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{tab:jbhi-referable-dr}",
        BS + "begin{tabular}{lllrrrrr}", BS + "toprule",
        header, BS + "midrule", "\n".join(lines),
        BS + "bottomrule", BS + "end{tabular}", BS + "end{table*}",
    ])
    for bad in ("\t", "\r", "\x0b", "\x0c"):
        if bad in body:
            raise RuntimeError(f"control character {bad!r} in generated table")
    (tables / "JBHI_REFERABLE_DR_SECONDARY.tex").write_text(
        body + "\n", encoding="utf-8")

    # ----------------------------------------------------------- report
    print(f"referable-DR threshold: grade >= {REFERABLE_THRESHOLD} (fixed)")
    print(f"conditions: {len(frame)}  (2 domains x 2 protocols x 2 arms)")
    print(f"seeds per condition: {frame.n_seeds.unique().tolist()}\n")
    show = frame[["domain", "protocol", "arm", "n", "referable_prevalence",
                  "sensitivity_mean", "specificity_mean", "precision_ppv_mean",
                  "f1_mean", "auroc_mean"]]
    print(show.round(4).to_string(index=False))
    print("\nsaved -> JBHI_REFERABLE_DR_SECONDARY.csv / .tex")
    print("         JBHI_REFERABLE_DR_PER_SEED.csv "
          f"({len(seeds_frame)} rows)")
    print("\nPost-hoc secondary. QWK remains the pre-specified primary "
          "endpoint; no significance test was run here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
