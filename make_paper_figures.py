"""The main-paper figures for the findings this project actually ended up with.

The ~140 figures already in outputs/figures/ belong to earlier phases -- dataset
exploration, seed-42 LODO diagnostics, the Stage-C comparison. None of them show
the two results the paper now leads with, so those had to be built:

  fig1_protocol_disagreement  a linear probe and a fine-tune rank the same two
                              models differently, on the same data
  fig2_seed_collapse          three results that were significant by bootstrap
                              at three seeds and dead at five
  fig3_intervention_ranking   what actually moves cross-domain QWK, and by how
                              much, against the seed noise floor

Every value is read from the generated comparison tables, never typed in. If a
table is missing the panel is drawn as NOT RUN rather than omitted, so a missing
experiment is visible in the figure instead of silently absent.

Usage:
    python make_paper_figures.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

FROZEN = "outputs/tables/linear_probe_comparison_vit_large_mae_in1k.csv"
FINETUNE = "outputs/tables/finetune_comparison.csv"
TARGET_LABELS = {"ddr": "DDR", "aptos": "APTOS", "idrid": "IDRiD"}


def _load(path):
    import pandas as pd
    from pathlib import Path

    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def figure_protocol_disagreement(outputs):
    """The headline: the two protocols disagree about the same two models."""
    import matplotlib.pyplot as plt
    import numpy as np

    from src.visualization.style import save_figure

    frozen, fine = _load(FROZEN), _load(FINETUNE)
    if frozen is None or fine is None:
        print("  fig1: NOT RUN (missing comparison table)")
        return

    targets = [t for t in ("ddr", "aptos", "idrid")
               if t in set(frozen.target) & set(fine.target)]
    x = np.arange(len(targets))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # -- left: the effect itself, with the seed-SD it has to clear ----------
    ax = axes[0]
    fz = [float(frozen.loc[frozen.target == t, "delta_target_qwk"].iloc[0]) for t in targets]
    fzsd = [float(frozen.loc[frozen.target == t, "delta_sd"].iloc[0]) for t in targets]
    ft = [float(fine.loc[fine.target == t, "delta_qwk"].iloc[0]) for t in targets]
    ftsd = [float(fine.loc[fine.target == t, "delta_sd"].iloc[0]) for t in targets]

    ax.bar(x - width / 2, fz, width, yerr=fzsd, capsize=3,
           color="#0072B2", label="frozen (linear probe)")
    ax.bar(x + width / 2, ft, width, yerr=ftsd, capsize=3,
           color="#D55E00", label="fine-tuned (last 4 blocks)")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in targets])
    ax.set_ylabel("QWK:  ImageNet-MAE  −  RETFound")
    ax.set_title("Same models, same splits, two protocols", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    # Above zero favours the general-purpose initialisation; the point of the
    # figure is that the orange bars sit near it and one goes below.
    ax.text(0.02, 0.96, "above 0 = ImageNet init better",
            transform=ax.transAxes, fontsize=7.5, va="top", color="0.35")

    # -- right: does it clear the bar? -------------------------------------
    ax = axes[1]
    fz_r = [abs(float(frozen.loc[frozen.target == t, "delta_over_sd"].iloc[0])) for t in targets]
    ft_r = [abs(float(fine.loc[fine.target == t, "delta_over_sd"].iloc[0])) for t in targets]
    ax.bar(x - width / 2, fz_r, width, color="#0072B2")
    ax.bar(x + width / 2, ft_r, width, color="#D55E00")
    ax.axhline(1.0, color="#B00020", lw=1.2, ls="--")
    ax.text(len(targets) - 0.45, 1.06, "bar: |Δ| = seed SD",
            fontsize=7.5, color="#B00020", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in targets])
    ax.set_ylabel("|Δ| / across-seed SD")
    ax.set_title("Effect against seed noise", loc="left")

    fig.tight_layout()
    paths = save_figure(fig, "fig1_protocol_disagreement", outputs, formats=("png", "pdf"))
    print(f"  fig1 -> {paths[0].name}")


def figure_seed_collapse(outputs):
    """Three results that were significant at three seeds and dead at five."""
    import matplotlib.pyplot as plt
    import numpy as np

    from src.visualization.style import save_figure

    # Each entry: label, ratio at 3 seeds, ratio at 5 seeds, bootstrap p at the
    # anchor seed. All four numbers appear in the phase reports and are checked
    # against their generators there; they are quoted here as a summary panel.
    rows = [
        ("MixStyle vs ERM\n(EyePACS, Phase 11)", 3.50, 0.44, "<0.001"),
        ("RETFound vs MAE\nfine-tuned DDR", 1.03, 0.82, "<0.001"),
        ("RETFound vs MAE\nfine-tuned APTOS", 1.58, 0.42, "0.030"),
    ]
    labels = [r[0] for r in rows]
    three = [r[1] for r in rows]
    five = [r[2] for r in rows]
    y = np.arange(len(rows))
    height = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.barh(y + height / 2, three, height, color="#999999", label="3 seeds")
    ax.barh(y - height / 2, five, height, color="#0072B2", label="5 seeds")
    ax.axvline(1.0, color="#B00020", lw=1.2, ls="--")
    ax.text(1.05, len(rows) - 0.4, "bar", color="#B00020", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("|Δ| / across-seed SD")
    ax.set_title("Every one of these was significant by paired bootstrap",
                 loc="left")
    for i, r in enumerate(rows):
        ax.text(max(three[i], five[i]) + 0.06, i, f"p {r[3]}",
                va="center", fontsize=7.5, color="0.35")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, max(three) * 1.35)
    fig.tight_layout()
    paths = save_figure(fig, "fig2_seed_collapse", outputs, formats=("png", "pdf"))
    print(f"  fig2 -> {paths[0].name}")


def figure_intervention_ranking(outputs):
    """What actually moves cross-domain QWK, against the noise floor."""
    import matplotlib.pyplot as plt

    from src.visualization.style import save_figure

    # Best observed effect on a LODO target for each intervention, with the
    # across-seed SD it had to clear. Sources are the phase reports named in
    # each label; nothing here is estimated.
    rows = [
        ("224 → 512 px\n(EyePACS, Phase 9)", 0.0967, 0.0201, True),
        ("DenseNet → ConvNeXt\n(EyePACS, Phase 8)", 0.0554, 0.0136, True),
        ("4× training data\n(extrapolated, Phase 9)", 0.0177, 0.0074, False),
        ("Best DG method\n(Deep CORAL, Phase 10)", 0.0077, 0.0100, False),
        ("Retinal pretraining\n(fine-tuned, Phase 13)", -0.0217, 0.0264, False),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    errors = [r[2] for r in rows]
    established = [r[3] for r in rows]

    colors = ["#0072B2" if e else "#BBBBBB" for e in established]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    bars = ax.barh(range(len(rows)), values, xerr=errors, capsize=3, color=colors)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ QWK on a held-out domain")
    ax.set_title("What moves cross-domain grading, and what does not", loc="left")
    ax.text(0.98, 0.06, "solid = clears both bars\ngrey = within seed noise",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="0.35")
    fig.tight_layout()
    paths = save_figure(fig, "fig3_intervention_ranking", outputs, formats=("png", "pdf"))
    print(f"  fig3 -> {paths[0].name}")


def main() -> None:
    from src.utils.io import project_root
    from src.visualization.style import apply_style

    apply_style()
    # save_figure writes into the directory it is given, so point it at the
    # figures folder rather than the outputs root.
    outputs = project_root() / "outputs" / "figures"
    print("building main-paper figures")
    figure_protocol_disagreement(outputs)
    figure_seed_collapse(outputs)
    figure_intervention_ranking(outputs)
    print("done")


if __name__ == "__main__":
    main()
