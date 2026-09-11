"""The four candidate main figures for the JBHI manuscript.

Data only -- no prose, no captions asserting causation. Every value is read
from an authoritative generator CSV.

    fig1_study_design         the lineage intervention and the three protocols
    fig2_adaptation_depth     paired per-seed deltas at each adaptation depth
    fig3_primary_interaction  forest plot of the two domain interactions
    fig4_source_validation    optimisation behaviour (descriptive context)

Exports PNG at 300 dpi and PDF (vector) into outputs/figures/jbhi_final/.

Usage:
    python make_jbhi_figures.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

OUT = "jbhi_final"
IMAGENET = "#2E5A87"      # ImageNet-MAE
RETFOUND = "#B5651D"      # RETFound
NEUTRAL = "#444444"
GRID = "#DDDDDD"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.bbox": "tight",
    })
    return plt


def _save(fig, name, directory):
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, format=suffix)
        paths.append(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def figure_study_design(plt, directory):
    """The comparison is one lineage step, not two unrelated architectures."""
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, colour, weight="normal", size=8.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="white",
                                   edgecolor=colour, linewidth=1.6, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, color=colour, fontweight=weight, zorder=3)

    # The lineage: one checkpoint continued into another.
    box(0.3, 4.1, 2.5, 1.0, "ImageNet-MAE\nViT-L/16", IMAGENET, "bold")
    ax.annotate("", xy=(4.3, 4.6), xytext=(2.9, 4.6),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, linewidth=1.4))
    ax.text(3.6, 4.95, "retinal-domain MAE\ncontinuation pretraining",
            ha="center", va="bottom", fontsize=7.2, color=NEUTRAL)
    box(4.4, 4.1, 2.5, 1.0, "RETFound-CFP\nViT-L/16", RETFOUND, "bold")
    ax.text(3.6, 3.72, "identical architecture; same starting weights",
            ha="center", va="top", fontsize=7.2, color=NEUTRAL, style="italic")

    # The three downstream protocols, applied identically to both arms.
    labels = ["Frozen\nlinear probe", "Partial FT\nlast 4 of 24 blocks",
              "Full FT\nall 303.3M encoder params"]
    for i, label in enumerate(labels):
        x = 0.3 + i * 2.35
        box(x, 1.75, 2.1, 0.95, label, NEUTRAL)
        for src, colour in ((1.55, IMAGENET), (5.65, RETFOUND)):
            ax.annotate("", xy=(x + 1.05, 2.72), xytext=(src, 4.08),
                        arrowprops=dict(arrowstyle="-", color=colour,
                                        linewidth=0.7, alpha=0.45))
    ax.text(3.6, 3.05, "each initialisation × each protocol, matched recipe",
            ha="center", va="center", fontsize=7.2, color=NEUTRAL)

    box(7.3, 1.75, 2.4, 3.35,
        "Leave-one-domain-out\nexternal evaluation\n\nheld out:\nDDR  •  APTOS  •  IDRiD\n\n"
        "target labels never used\nfor selection, stopping\nor calibration",
        NEUTRAL, size=7.4)
    ax.annotate("", xy=(7.25, 2.25), xytext=(6.9, 2.25),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, linewidth=1.4))

    ax.text(0.3, 0.85, "The contrast is a lineage intervention: the only difference "
                       "between the two arms is RETFound's additional\nretinal-domain "
                       "MAE continuation from the ImageNet-MAE checkpoint on the left.",
            ha="left", va="center", fontsize=7.4, color=NEUTRAL)
    return _save(fig, "fig1_study_design", directory)


def figure_adaptation_depth(plt, directory, figure_data):
    """Paired per-seed deltas, so the reader sees the data, not only a bar."""
    import numpy as np

    domains = ["DDR", "APTOS"]
    protocols = ["frozen", "partial", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)

    for ax, domain in zip(axes, domains):
        part = figure_data[figure_data.domain == domain]
        for i, protocol in enumerate(protocols):
            sub = part[part.protocol == protocol]
            values = sub.delta.to_numpy(dtype=float)
            jitter = np.linspace(-0.13, 0.13, len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, s=22,
                       facecolor="white", edgecolor=NEUTRAL, linewidth=0.9,
                       zorder=3)
            ax.hlines(values.mean(), i - 0.28, i + 0.28, color=NEUTRAL,
                      linewidth=2.0, zorder=4)
        # Join each seed across protocols: the interaction is a paired quantity.
        for seed in sorted(part.seed.unique()):
            line = [float(part[(part.protocol == p) & (part.seed == seed)].delta.iloc[0])
                    for p in protocols]
            ax.plot(range(3), line, color=NEUTRAL, alpha=0.22, linewidth=0.8,
                    zorder=2)
        ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Frozen", "Partial", "Full"])
        ax.set_title(f"{domain} held out", fontsize=9.5)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("QWK difference\nImageNet-MAE − RETFound")
    axes[0].text(0.04, 0.96, "ImageNet-MAE ahead", transform=axes[0].transAxes,
                 fontsize=7, va="top", color=IMAGENET)
    axes[0].text(0.04, 0.04, "RETFound ahead", transform=axes[0].transAxes,
                 fontsize=7, va="bottom", color=RETFOUND)
    fig.suptitle("Paired per-seed difference at each adaptation depth "
                 "(five common seeds)", fontsize=9.5, y=1.02)
    return _save(fig, "fig2_adaptation_depth", directory)


def figure_primary_interaction(plt, directory, family):
    """Forest plot: two domain-specific estimates, never pooled."""
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    rows = list(family.itertuples())
    positions = list(range(len(rows)))[::-1]

    for y, r in zip(positions, rows):
        ax.plot([r.crossed_CI_low, r.crossed_CI_high], [y, y], color=NEUTRAL,
                linewidth=1.8, solid_capstyle="round", zorder=2)
        ax.plot([r.crossed_CI_low, r.crossed_CI_low], [y - 0.09, y + 0.09],
                color=NEUTRAL, linewidth=1.4)
        ax.plot([r.crossed_CI_high, r.crossed_CI_high], [y - 0.09, y + 0.09],
                color=NEUTRAL, linewidth=1.4)
        ax.scatter([r.interaction], [y], s=62, marker="D", color=IMAGENET,
                   zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(0.012, y + 0.22,
                f"{r.interaction:+.4f}  [{r.crossed_CI_low:+.4f}, "
                f"{r.crossed_CI_high:+.4f}]",
                fontsize=7.4, va="center", color=NEUTRAL)
        ax.text(0.012, y - 0.20,
                f"Holm $p$ = {r.Holm_p:.4f}   signs {r.sign_agreement}   "
                f"$n$ = {int(r.n_test):,}",
                fontsize=7.0, va="center", color="#777777")

    ax.axvline(0.0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([r.domain for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.set_xlim(-0.20, 0.075)
    ax.set_xlabel("Interaction  $I_{full}$ = $\\Delta_{full}$ − "
                  "$\\Delta_{frozen}$   (QWK)")
    ax.set_title("Negative = the ImageNet-MAE advantage measured under frozen "
                 "probing\nis reduced once the encoder adapts", fontsize=8.6)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return _save(fig, "fig3_primary_interaction", directory)


def figure_source_validation(plt, directory, ddr, aptos):
    """Descriptive optimisation context. No causal annotation."""
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    combined = []
    for frame, domain in ((ddr, "DDR"), (aptos, "APTOS")):
        part = frame.copy()
        part["domain"] = domain
        combined.append(part)
    import pandas as pd
    allruns = pd.concat(combined, ignore_index=True)
    allruns["arm"] = allruns.model.replace({"RETFound": "RETFound",
                                            "ImageNet-MAE": "ImageNet-MAE"})

    ax = axes[0]
    for arm, colour in (("ImageNet-MAE", IMAGENET), ("RETFound", RETFOUND)):
        sub = allruns[allruns.arm == arm]
        jitter = np.linspace(-0.14, 0.14, len(sub))
        x = (0 if arm == "ImageNet-MAE" else 1) + jitter
        ax.scatter(x, sub.best_epoch, s=22, facecolor="white",
                   edgecolor=colour, linewidth=1.0, zorder=3)
        ax.hlines(sub.best_epoch.mean(), (0 if arm == "ImageNet-MAE" else 1) - 0.3,
                  (0 if arm == "ImageNet-MAE" else 1) + 0.3, color=colour,
                  linewidth=2.0, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ImageNet-MAE", "RETFound"])
    ax.set_ylabel("best source-validation epoch")
    ax.set_title("Epoch of best source-validation QWK", fontsize=8.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    for arm, colour in (("ImageNet-MAE", IMAGENET), ("RETFound", RETFOUND)):
        sub = allruns[allruns.arm == arm]
        jitter = np.linspace(-0.14, 0.14, len(sub))
        x = (0 if arm == "ImageNet-MAE" else 1) + jitter
        ax.scatter(x, sub.overfit_ratio, s=22, facecolor="white",
                   edgecolor=colour, linewidth=1.0, zorder=3)
        ax.hlines(sub.overfit_ratio.mean(), (0 if arm == "ImageNet-MAE" else 1) - 0.3,
                  (0 if arm == "ImageNet-MAE" else 1) + 0.3, color=colour,
                  linewidth=2.0, zorder=4)
    ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ImageNet-MAE", "RETFound"])
    ax.set_ylabel("final ÷ minimum source-validation loss")
    ax.set_title("Source-validation loss inflation", fontsize=8.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    stopped = allruns.groupby("arm").early_stopped.sum().to_dict()
    total = allruns.groupby("arm").size().to_dict()
    fig.suptitle(
        f"Optimisation behaviour under the shared fixed budget, DDR + APTOS "
        f"(early stopped: RETFound {int(stopped.get('RETFound',0))}/"
        f"{total.get('RETFound',0)}, ImageNet-MAE "
        f"{int(stopped.get('ImageNet-MAE',0))}/{total.get('ImageNet-MAE',0)})",
        fontsize=8.6, y=1.04)
    return _save(fig, "fig4_source_validation", directory)


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    plt = _style()
    outputs = project_root() / "outputs"
    tables = outputs / "tables"
    directory = outputs / "figures" / OUT
    directory.mkdir(parents=True, exist_ok=True)

    needed = ["figure_qwk_by_domain_protocol_seed.csv",
              "JBHI_PRIMARY_INTERACTION.csv",
              "full_finetune_source_validation.csv",
              "aptos_full_finetune_source_validation.csv"]
    for name in needed:
        if not (tables / name).exists():
            print(f"NOT RUN -- {name} missing")
            return 1

    figure_data = pd.read_csv(tables / "figure_qwk_by_domain_protocol_seed.csv")
    family = pd.read_csv(tables / "JBHI_PRIMARY_INTERACTION.csv")
    ddr = pd.read_csv(tables / "full_finetune_source_validation.csv")
    aptos = pd.read_csv(tables / "aptos_full_finetune_source_validation.csv")

    written = []
    written += figure_study_design(plt, directory)
    written += figure_adaptation_depth(plt, directory, figure_data)
    written += figure_primary_interaction(plt, directory, family)
    written += figure_source_validation(plt, directory, ddr, aptos)

    for path in written:
        print(f"  saved {path.relative_to(project_root())} "
              f"({path.stat().st_size/1024:.0f} KB)")
    print(f"\n{len(written)} file(s) in outputs/figures/{OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
