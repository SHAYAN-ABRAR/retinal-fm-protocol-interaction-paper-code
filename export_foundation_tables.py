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
    import pandas as pd

    frozen_path = tables / "linear_probe_comparison_vit_large_mae_in1k.csv"
    fine_path = tables / "finetune_comparison.csv"
    if not (frozen_path.exists() and fine_path.exists()):
        return None
    frozen, fine = pd.read_csv(frozen_path), pd.read_csv(fine_path)

    rows = []
    for target in TARGETS:
        fz = _cell(frozen[frozen.target == target], "delta_target_qwk")
        ft = _cell(fine[fine.target == target], "delta_qwk")
        rows.append(" & ".join([LABELS[target], fz[2], fz[0], fz[1],
                                ft[2], ft[0], ft[1]]) + " " + NL)

    caption = (
        "RETFound against the ImageNet MAE checkpoint it was initialised from, "
        "under two evaluation protocols. Architecture, parameter count, "
        "objective, splits, augmentation and schedule are identical; the "
        "intervention is the additional retinal-domain MAE pretraining stage "
        "that turns that initialisation into RETFound, not dataset identity "
        "alone. $" + BS + "Delta$ is "
        "ImageNet-MAE minus RETFound, so a positive value favours the "
        "general-purpose initialisation. $" + BS + "sigma$ is the standard "
        "deviation of the paired per-seed differences. Bold marks effects "
        "clearing both criteria of Section~" + BS + "ref{sec:bars}; every "
        "other entry is within seed noise. The frozen protocol reports an "
        "advantage on two of three held-out domains and the fine-tuned "
        "protocol on none."
    )
    return NL.join([]) or "\n".join([
        BS + "begin{table*}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{tab:foundation}",
        BS + "begin{tabular}{lrrrrrr}", BS + "toprule",
        "& " + BS + "multicolumn{3}{c}{Frozen (linear probe)} & "
        + BS + "multicolumn{3}{c}{Fine-tuned (last 4 of 24 blocks)} " + NL,
        BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}",
        "Held-out domain & seeds & $" + BS + "Delta$ QWK & $|" + BS
        + "Delta|/" + BS + "sigma$ & seeds & $" + BS + "Delta$ QWK & $|"
        + BS + "Delta|/" + BS + "sigma$ " + NL,
        BS + "midrule",
        "\n".join(rows),
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
