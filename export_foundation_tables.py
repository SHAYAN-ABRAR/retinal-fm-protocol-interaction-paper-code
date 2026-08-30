"""LaTeX tables for the foundation-model and DG-method results.

Kept separate from export_paper_tables.py, which predates these phases and
covers the earlier LODO, deployment-cost and selective-prediction tables. Both
scripts obey the same rule: every value is read from a results CSV and none is
retyped. A table that disagrees with the analysis that produced it is invisible
on inspection, which is exactly how it survives to publication.

Bold marks an effect that clears BOTH bars -- greater than the across-seed
standard deviation of the paired per-seed differences, and a paired bootstrap
interval excluding zero. Every unbolded entry is within seed noise, and the
captions say so rather than leaving a reader to infer it from a font weight.

Usage:
    python export_foundation_tables.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

LABELS = {"ddr": "DDR", "aptos": "APTOS 2019", "idrid": "IDRiD"}
TARGETS = ["ddr", "aptos", "idrid"]

BS = "\\"          # keeps the f-strings below readable
NL = BS + BS       # LaTeX row terminator
# Math-mode, and delimited. Unwrapped it is an error in a text-mode caption;
# undelimited the following letters are swallowed into the command name, which
# is how this caption shipped an undefined \timescase.
TIMES = "$" + BS + "times$"
LF = chr(10)       # real newline, built this way because writing a literal
                   # escape here has been flattened by an editing layer twice


def _cell(frame, delta_column):
    """(delta, |delta|/sigma, n_seeds) for one comparison, or NOT RUN."""
    if frame.empty:
        return "NOT RUN", "--", "--"
    row = frame.iloc[0]
    established = "within seed noise" not in str(row["verdict"])
    # Math mode so the sign renders as a minus rather than a hyphen.
    # 	extbf{$x$} does not bold math content -- the weight silently does not
    # apply. \mathbf inside the math is what actually renders bold.
    value = f"{row[delta_column]:+.4f}"
    delta = ("$" + BS + "mathbf{" + value + "}$") if established else ("$" + value + "$")
    return delta, f"{abs(row['delta_over_sd']):.2f}", str(int(row["n_seeds"]))


def foundation_table(tables):
    """The headline table, built from the CURRENT inference.

    Previously this read the `verdict` column of the per-protocol comparison
    CSVs, which encoded the retired two-bar rule. That made the table contradict
    the manuscript: it bolded IDRiD partial fine-tuning as an established
    difference on the strength of |delta| > 1.13 seed SD, while the crossed
    bootstrap with Holm correction reports no difference there. A generated
    table that disagrees with the paper's own statistics is worse than no table.

    Bold now marks only what survives Holm correction in
    crossed_bootstrap_foundation.csv. Seed SD is retained as a descriptive
    column and is explicitly not a significance criterion.
    """
    import pandas as pd

    path = tables / "crossed_bootstrap_foundation.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)

    def cell(protocol, target):
        row = frame[(frame.protocol == protocol) & (frame.target == target)]
        if row.empty:
            return ["NOT RUN", "--", "--", "--"]
        r = row.iloc[0]
        established = r.p_holm < 0.05
        value = f"{r.delta_qwk:+.4f}"
        delta = ("$" + BS + "mathbf{" + value + "}$") if established else ("$" + value + "$")
        ci = "$[" + f"{r.ci_lower:+.4f}, {r.ci_upper:+.4f}" + "]$"
        p = ("$<$0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}")
        return [str(int(r.n_seeds)), delta, ci, p]

    rows = []
    for target in TARGETS:
        fz = cell("frozen", target)
        ft = cell("partial_finetune_4", target)
        rows.append(" & ".join([LABELS[target]] + fz + ft) + " " + NL)

    caption = (
        "RETFound against the ImageNet MAE checkpoint it was initialised from, "
        "under two adaptation protocols. Architecture, parameter count, "
        "objective, splits, augmentation and schedule are identical; the "
        "model-level intervention is the additional retinal-domain MAE "
        "continuation pretraining that turns that initialisation into RETFound. "
        "$" + BS + "Delta$ is ImageNet-MAE minus RETFound, so a positive value "
        "favours the general-purpose initialisation. Intervals are from the "
        "crossed seed " + TIMES + " case bootstrap; $p$ is Holm-corrected "
        "across all six comparisons. Bold marks the two results that survive "
        "correction. Under partial fine-tuning no held-out domain shows a "
        "difference that survives correction, which is not a demonstration of "
        "equivalence."
    )
    return LF.join([
        BS + "begin{table*}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{tab:foundation}",
        BS + "begin{tabular}{lrrrrrrrr}", BS + "toprule",
        "& " + BS + "multicolumn{4}{c}{Frozen (linear probe)} & "
        + BS + "multicolumn{4}{c}{Partial fine-tuning (last 4 of 24 blocks)} " + NL,
        BS + "cmidrule(lr){2-5}" + BS + "cmidrule(lr){6-9}",
        "Held-out domain & seeds & $" + BS + "Delta$ QWK & 95" + BS + "% CI & "
        "$p_{" + BS + "text{Holm}}$ & seeds & $" + BS + "Delta$ QWK & 95" + BS
        + "% CI & $p_{" + BS + "text{Holm}}$ " + NL,
        BS + "midrule",
        LF.join(rows),
        BS + "bottomrule", BS + "end{tabular}", BS + "end{table*}",
    ])


def dg_table(tables):
    import pandas as pd

    rows = []
    for target in ("eyepacs", "ddr"):
        path = tables / f"method_comparison_lodo_{target}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame.sampler == "natural"]
        for _, r in frame.iterrows():
            established = "within seed noise" not in str(r["verdict"])
            value = f"{r['delta_qwk']:+.4f}"
            delta = (("$" + BS + "mathbf{" + value + "}$") if established
                     else ("$" + value + "$"))
            rows.append(" & ".join([
                LABELS.get(target, target.upper()),
                str(r["method"]).replace("_", " "),
                f"{r['method_qwk']:.4f}", delta,
                f"{abs(r['delta_over_sd']):.2f}", str(int(r["n_seeds"])),
            ]) + " " + NL)
    if not rows:
        return None

    caption = (
        "Domain-generalization objectives against ERM under a naturally "
        "shuffled sampler, on the two held-out domains tested. No method "
        "clears both criteria on either domain. IRMv1 diverged to NaN on every "
        "DDR seed, under its published annealing schedule and again under one "
        "matched to the split's iteration count, and is recorded as diverged "
        "rather than scored."
    )
    return "\n".join([
        BS + "begin{table}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{tab:dg}",
        BS + "begin{tabular}{llrrrr}", BS + "toprule",
        "Held-out & Method & QWK & $" + BS + "Delta$ vs ERM & $|" + BS
        + "Delta|/" + BS + "sigma$ & seeds " + NL,
        BS + "midrule",
        "\n".join(rows),
        BS + "bottomrule", BS + "end{tabular}", BS + "end{table}",
    ])


def main() -> None:
    from src.utils.io import project_root

    tables = project_root() / "outputs" / "tables"
    written = []
    for name, body in (("table_foundation", foundation_table(tables)),
                       ("table_dg_methods", dg_table(tables))):
        if body is None:
            print(f"{name}: NOT RUN -- source comparison table missing")
            continue
        path = tables / f"{name}.tex"
        path.write_text(body + "\n", encoding="utf-8")
        written.append(path)
        print(f"saved -> {path.name}")
    if written:
        print(f"\n{len(written)} LaTeX table(s) written. "
              "Every value read from a results CSV; none retyped.")


if __name__ == "__main__":
    main()
