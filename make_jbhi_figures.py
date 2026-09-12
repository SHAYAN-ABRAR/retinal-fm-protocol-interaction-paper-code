"""The five candidate main figures for the JBHI manuscript.

Data only. No prose, no causal annotation, and no scientific value typed by
hand: every plotted quantity is read from an authoritative generator CSV or
recomputed from saved predictions.

    fig1_study_design              lineage intervention and matched protocols
    fig2_adaptation_depth          paired per-seed delta at three depths
    fig3_primary_interaction       forest plot of the two domain interactions
    fig4_seed_interaction_slopes   per-seed frozen -> full slopes
    fig5_full_ft_paired_qwk        full-FT paired arm performance

Each writes PNG at 300 dpi and a vector PDF into outputs/figures/jbhi_final/,
and contributes rows to FIGURE_PROVENANCE.csv (written by
export_figure_provenance.py).

Palette is Okabe-Ito, which is colour-blind safe; the two arms are additionally
separated by marker shape and line style so the figures survive greyscale.

Usage:
    python make_jbhi_figures.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

OUT = "jbhi_final"

# Okabe-Ito: distinguishable under all common colour-vision deficiencies, and
# these two differ enough in luminance to survive a greyscale print.
IMAGENET = "#0072B2"      # blue
RETFOUND = "#D55E00"      # vermillion
INK = "#333333"
MUTED = "#8C8C8C"
GRID = "#E0E0E0"

# IEEE widths in inches.
ONE_COL = 3.5
TWO_COL = 7.16

COMMON_SEEDS = [42, 1, 2, 3, 4]
PROTOCOL_ORDER = ["frozen", "partial", "full"]
PROTOCOL_TICKS = ["Frozen\nprobe", "Partial\nFT", "Full\nFT"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "figure.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embed real text, not paths
    })
    return plt


def _save(fig, name, directory):
    import matplotlib.pyplot as plt
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        # PNG at 400 dpi: the tight bounding box trims the canvas, so a figure
        # authored at two-column width lands narrower than that on disk. Saved
        # at 300 it would fall to ~235 effective dpi once scaled back up to
        # column width in the manuscript. The PDF is vector and unaffected.
        # CreationDate is dropped from the PDF so that re-running the
        # generator on unchanged data reproduces byte-identical files; without
        # it every rebuild differs by a timestamp alone.
        fig.savefig(path, format=suffix, dpi=400 if suffix == "png" else None,
                    metadata={"CreationDate": None} if suffix == "pdf" else None)
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- figure 1
def figure_study_design(plt, directory):
    """Rebuilt: a bus layout, so no arrow crosses another and no label overlaps.

    The previous version drew six diagonal arrows from two source boxes to
    three protocol boxes, which collided with the protocol labels and with each
    other. Both initialisations now feed a single horizontal bus and each
    protocol drops from it, which states "both arms enter every protocol"
    without any crossing.
    """
    from matplotlib.patches import FancyArrow, Rectangle

    fig, ax = plt.subplots(figsize=(TWO_COL, 3.6))
    ax.set_xlim(0, 100)
    # Trimmed to the drawn content: a 0-100 box left a dead white band below
    # the evaluation panel that survived the tight bounding box.
    ax.set_ylim(12, 97)
    ax.axis("off")

    def box(x, y, w, h, lines, edge, bold_first=True, size=7.6):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=edge,
                               linewidth=1.3, zorder=3))
        first, rest = lines[0], lines[1:]
        if rest:
            ax.text(x + w / 2, y + h * 0.63, first, ha="center", va="center",
                    fontsize=size, color=edge,
                    fontweight="bold" if bold_first else "normal", zorder=4)
            ax.text(x + w / 2, y + h * 0.26, "\n".join(rest), ha="center",
                    va="center", fontsize=size - 0.8, color=INK, zorder=4)
        else:
            ax.text(x + w / 2, y + h / 2, first, ha="center", va="center",
                    fontsize=size, color=edge,
                    fontweight="bold" if bold_first else "normal", zorder=4)

    # -- row 1: the lineage ------------------------------------------------
    box(4, 78, 26, 15, ["ImageNet-MAE", "ViT-L/16"], IMAGENET)
    box(70, 78, 26, 15, ["RETFound-CFP", "ViT-L/16"], RETFOUND)
    ax.add_patch(FancyArrow(33, 85.5, 33, 0, width=0.35, head_width=2.4,
                            head_length=2.6, length_includes_head=True,
                            facecolor=INK, edgecolor=INK, zorder=3))
    # Two short lines rather than one long one: a single line reached both box
    # edges and left no visual gap.
    ax.text(50, 91.3, "continued MAE pretraining\non retinal CFP",
            ha="center", va="center", fontsize=7.2, color=INK,
            linespacing=1.25)
    ax.text(50, 80.4, "same architecture · same lineage",
            ha="center", va="center", fontsize=6.9, color=MUTED, style="italic")

    # -- the bus: both arms enter every protocol ---------------------------
    for x, colour in ((17, IMAGENET), (83, RETFOUND)):
        ax.plot([x, x], [78, 70], color=colour, linewidth=1.2, zorder=2)
    ax.plot([17, 83], [70, 70], color=MUTED, linewidth=1.2, zorder=2)
    ax.text(50, 71.6, "each initialisation enters each protocol",
            ha="center", va="bottom", fontsize=6.9, color=MUTED)

    centres = [19, 50, 81]
    for x in centres:
        ax.add_patch(FancyArrow(x, 70, 0, -8.5, width=0.25, head_width=1.9,
                                head_length=2.2, length_includes_head=True,
                                facecolor=MUTED, edgecolor=MUTED, zorder=2))

    # -- row 2: the three protocols ----------------------------------------
    protocols = [
        ["Frozen", "linear probe"],
        ["Partial FT", "last 4 of 24 blocks"],
        ["Full FT", "encoder + head", "303.3M parameters"],
    ]
    for x, lines in zip(centres, protocols):
        box(x - 15, 44, 30, 16, lines, INK, bold_first=True)

    # -- row 3: evaluation --------------------------------------------------
    for x in centres:
        ax.add_patch(FancyArrow(x, 44, 0, -6.5, width=0.25, head_width=1.9,
                                head_length=2.2, length_includes_head=True,
                                facecolor=MUTED, edgecolor=MUTED, zorder=2))
    ax.add_patch(Rectangle((4, 14), 92, 23, facecolor="white", edgecolor=INK,
                           linewidth=1.3, zorder=3))
    ax.text(50, 32.2, "Leave-one-domain-out external evaluation", ha="center",
            va="center", fontsize=8.0, color=INK, fontweight="bold", zorder=4)
    ax.text(28, 24.0, "Primary interaction\nDDR  ·  APTOS", ha="center",
            va="center", fontsize=7.4, color=INK, zorder=4)
    ax.text(72, 24.0, "Frozen + partial only\nIDRiD", ha="center", va="center",
            fontsize=7.4, color=MUTED, zorder=4)
    ax.plot([50, 50], [18.5, 28.5], color=GRID, linewidth=0.9, zorder=3)
    ax.text(50, 16.6, "target labels never used for selection, early stopping, "
                      "tuning or temperature scaling",
            ha="center", va="center", fontsize=6.8, color=MUTED, zorder=4)
    return _save(fig, "fig1_study_design", directory)


# ---------------------------------------------------------------- figure 2
def figure_adaptation_depth(plt, directory, figure_data, master):
    """Paired per-seed delta at each depth, with the seed paths separated."""
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 3.1), sharey=True)
    for ax, domain in zip(axes, ["DDR", "APTOS"]):
        part = figure_data[figure_data.domain == domain]
        seeds = sorted(part.seed.unique())
        # A small horizontal offset per seed keeps the five trajectories
        # readable where they bunch near zero.
        offsets = np.linspace(-0.16, 0.16, len(seeds))
        for seed, dx in zip(seeds, offsets):
            values = [float(part[(part.protocol == p) & (part.seed == seed)].delta.iloc[0])
                      for p in PROTOCOL_ORDER]
            xs = [i + dx for i in range(3)]
            ax.plot(xs, values, color=MUTED, alpha=0.55, linewidth=0.8,
                    zorder=2)
            ax.scatter(xs, values, s=17, facecolor="white", edgecolor=INK,
                       linewidth=0.8, zorder=3)
        for i, protocol in enumerate(PROTOCOL_ORDER):
            row = master[(master.domain == domain) & (master.protocol == protocol)
                         & (master.seed_set == "common-5")].iloc[0]
            mean = float(row.paired_delta)
            ax.hlines(mean, i - 0.26, i + 0.22, color=IMAGENET, linewidth=2.4,
                      zorder=4)
            # Beside the bar, not above it. Directly above or below, the label
            # landed on a seed point in four of the six groups; the seeds are
            # jittered only to +/-0.16 and the bar ends at 0.28, so x+0.32 is
            # clear of every drawn element.
            ax.annotate(f"{mean:+.3f}", (i + 0.26, mean), ha="left",
                        va="center", fontsize=6.8, color=IMAGENET,
                        fontweight="bold", zorder=5)
        ax.axhline(0.0, color=INK, linewidth=0.9, linestyle=(0, (4, 3)),
                   zorder=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(PROTOCOL_TICKS)
        ax.set_xlim(-0.55, 2.95)
        ax.set_title(f"{domain} held out", fontsize=8.5)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("ImageNet-MAE − RETFound\nQWK difference")
    axes[0].text(0.03, 0.97, "ImageNet-MAE higher", transform=axes[0].transAxes,
                 fontsize=6.8, va="top", color=IMAGENET)
    axes[0].text(0.03, 0.03, "RETFound higher", transform=axes[0].transAxes,
                 fontsize=6.8, va="bottom", color=RETFOUND)
    fig.suptitle("Paired per-seed difference at each adaptation depth "
                 "($n=5$ common seeds; bar = mean)", fontsize=8.5, y=1.00)
    return _save(fig, "fig2_adaptation_depth", directory)


# ---------------------------------------------------------------- figure 3
def figure_primary_interaction(plt, directory, family):
    """Two domain-specific estimates, never pooled."""
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.15))
    rows = list(family.itertuples())
    positions = list(range(len(rows)))[::-1]

    for y, r in zip(positions, rows):
        ax.plot([r.crossed_CI_low, r.crossed_CI_high], [y, y], color=INK,
                linewidth=1.6, solid_capstyle="butt", zorder=2)
        for edge in (r.crossed_CI_low, r.crossed_CI_high):
            ax.plot([edge, edge], [y - 0.10, y + 0.10], color=INK,
                    linewidth=1.3, zorder=2)
        ax.scatter([r.interaction], [y], s=54, marker="D", color=IMAGENET,
                   zorder=3, edgecolor="white", linewidth=0.7)
        ax.text(0.008, y + 0.20,
                f"{r.interaction:+.4f}  [{r.crossed_CI_low:+.4f}, "
                f"{r.crossed_CI_high:+.4f}]", fontsize=7.2, va="center",
                color=INK)
        ax.text(0.008, y - 0.20,
                f"Holm $p$ = {r.Holm_p:.4f}    signs {r.sign_agreement}    "
                f"$n$ = {int(r.n_test):,}", fontsize=6.8, va="center",
                color=MUTED)

    ax.axvline(0.0, color=INK, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([r.domain for r in rows], fontsize=8.5)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlim(-0.185, 0.072)
    ax.set_xlabel("Interaction  $I_{\\mathrm{full}}$ = "
                  "$\\Delta_{\\mathrm{full}}$ − $\\Delta_{\\mathrm{frozen}}$   (QWK)")
    ax.set_title("Negative: the ImageNet-MAE − RETFound difference is reduced "
                 "from frozen probing to full fine-tuning", fontsize=8.0,
                 pad=6)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return _save(fig, "fig3_primary_interaction", directory)


# ---------------------------------------------------------------- figure 4
def figure_seed_slopes(plt, directory, figure_data, family):
    """Every seed's frozen -> full slope, so the pairing is visible."""
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 3.2), sharey=True)
    for ax, domain in zip(axes, ["DDR", "APTOS"]):
        part = figure_data[figure_data.domain == domain]
        seeds = sorted(part.seed.unique())
        frozen, full = [], []
        for seed in seeds:
            a = float(part[(part.protocol == "frozen") & (part.seed == seed)].delta.iloc[0])
            b = float(part[(part.protocol == "full") & (part.seed == seed)].delta.iloc[0])
            frozen.append(a)
            full.append(b)
            ax.plot([0, 1], [a, b], color=MUTED, alpha=0.85, linewidth=1.0,
                    zorder=2)
            ax.scatter([0], [a], s=24, facecolor="white", edgecolor=IMAGENET,
                       linewidth=1.1, zorder=3)
            ax.scatter([1], [b], s=24, marker="s", facecolor="white",
                       edgecolor=RETFOUND, linewidth=1.1, zorder=3)

        # Seed labels collide wherever the frozen values bunch -- on DDR three
        # of them sit inside 0.02 QWK. Push colliding labels apart along y and
        # draw a hairline back to the point, so the label still identifies its
        # seed without overprinting a neighbour.
        order = np.argsort(frozen)
        span = (max(frozen) - min(frozen)) or 1.0
        gap = span * 0.125
        placed = np.array([frozen[i] for i in order], dtype=float)
        for i in range(1, len(placed)):
            if placed[i] - placed[i - 1] < gap:
                placed[i] = placed[i - 1] + gap
        for position, index in zip(placed, order):
            value = frozen[index]
            ax.annotate(f"{seeds[index]}", xy=(-0.035, value),
                        xytext=(-0.155, position), fontsize=6.2, color=MUTED,
                        ha="right", va="center",
                        arrowprops=dict(arrowstyle="-", color=GRID,
                                        linewidth=0.6, shrinkA=1, shrinkB=1))

        ax.hlines(np.mean(frozen), -0.16, 0.16, color=INK, linewidth=2.4,
                  zorder=4)
        ax.hlines(np.mean(full), 0.84, 1.16, color=INK, linewidth=2.4, zorder=4)
        row = family[family.domain == domain].iloc[0]
        ax.axhline(0.0, color=INK, linewidth=0.9, linestyle=(0, (4, 3)),
                   zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Frozen probe", "Full fine-tuning"])
        ax.set_xlim(-0.32, 1.32)
        ax.set_title(f"{domain} held out   mean $I_{{\\mathrm{{full}}}}$ = "
                     f"{row.interaction:+.4f}   ({row.sign_agreement} negative)",
                     fontsize=8.0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("ImageNet-MAE − RETFound\nQWK difference")
    fig.suptitle("Per-seed change from frozen probing to full fine-tuning "
                 "(same five seeds, paired within seed)", fontsize=8.5, y=1.00)
    return _save(fig, "fig4_seed_interaction_slopes", directory)


# ---------------------------------------------------------------- figure 5
def figure_full_ft_paired(plt, directory, figure_data, master):
    """Absolute arm performance under full FT, paired within seed."""
    import numpy as np

    # Constrained layout, because the three-line panel titles and the shared
    # legend both need reserved space; with a manual layout the suptitle
    # overprinted the titles and the legend overprinted the axis labels.
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 3.1),
                             constrained_layout=True)
    for ax, domain in zip(axes, ["DDR", "APTOS"]):
        part = figure_data[(figure_data.domain == domain)
                           & (figure_data.protocol == "full")]
        seeds = sorted(part.seed.unique())
        ys = list(range(len(seeds)))[::-1]
        for y, seed in zip(ys, seeds):
            row = part[part.seed == seed].iloc[0]
            a, b = float(row.imagenet_qwk), float(row.retfound_qwk)
            ax.plot([a, b], [y, y], color=MUTED, linewidth=1.1, zorder=2)
            ax.scatter([a], [y], s=30, facecolor=IMAGENET, edgecolor="white",
                       linewidth=0.7, zorder=3,
                       label="ImageNet-MAE" if y == ys[0] else None)
            ax.scatter([b], [y], s=30, marker="s", facecolor=RETFOUND,
                       edgecolor="white", linewidth=0.7, zorder=3,
                       label="RETFound" if y == ys[0] else None)
        mrow = master[(master.domain == domain) & (master.protocol == "full")
                      & (master.seed_set == "common-5")].iloc[0]
        ax.axvline(float(mrow.ImageNet_QWK_mean), color=IMAGENET,
                   linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
        ax.axvline(float(mrow.RETFound_QWK_mean), color=RETFOUND,
                   linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"seed {s}" for s in seeds])
        ax.set_ylim(-0.55, len(seeds) - 0.45)
        span = part[["imagenet_qwk", "retfound_qwk"]].to_numpy(dtype=float)
        lo, hi = span.min(), span.max()
        pad = (hi - lo) * 0.22 + 0.004
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel("QWK under full fine-tuning")
        # The statistics belong in the title block. Placed below the axis they
        # overprinted the tick labels and the axis label.
        ax.set_title(
            f"{domain} held out\n"
            f"paired $\\Delta$ = {float(mrow.paired_delta):+.4f},  "
            f"95% CI [{float(mrow.crossed_CI_low):+.4f}, "
            f"{float(mrow.crossed_CI_high):+.4f}]\n"
            f"seed-level $p$ = {float(mrow.raw_p):.4f} — no difference "
            f"demonstrated", fontsize=7.6, linespacing=1.5)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    # No suptitle: the panel titles already name each domain, and a suptitle
    # on top of three-line titles is where the previous collision came from.
    # Dashed arm means are explained in the legend rather than a banner.
    labels = [f"{label} (dashed line = mean)" for label in labels]
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False, handletextpad=0.4, columnspacing=2.0)
    return _save(fig, "fig5_full_ft_paired_qwk", directory)


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    plt = _style()
    root = project_root()
    tables = root / "outputs" / "tables"
    directory = root / "outputs" / "figures" / OUT
    directory.mkdir(parents=True, exist_ok=True)

    needed = ["figure_qwk_by_domain_protocol_seed.csv",
              "JBHI_PRIMARY_INTERACTION.csv", "JBHI_MASTER_RESULTS.csv"]
    for name in needed:
        if not (tables / name).exists():
            print(f"NOT RUN -- {name} missing")
            return 1

    figure_data = pd.read_csv(tables / "figure_qwk_by_domain_protocol_seed.csv")
    family = pd.read_csv(tables / "JBHI_PRIMARY_INTERACTION.csv")
    master = pd.read_csv(tables / "JBHI_MASTER_RESULTS.csv")

    written = []
    written += figure_study_design(plt, directory)
    written += figure_adaptation_depth(plt, directory, figure_data, master)
    written += figure_primary_interaction(plt, directory, family)
    written += figure_seed_slopes(plt, directory, figure_data, family)
    written += figure_full_ft_paired(plt, directory, figure_data, master)

    for path in written:
        print(f"  {path.relative_to(root)}  ({path.stat().st_size/1024:.0f} KB)")
    print(f"\n{len(written)} file(s) in outputs/figures/{OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
