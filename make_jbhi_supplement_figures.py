"""The six supplementary figures for the JBHI manuscript.

Descriptive and secondary evidence only. Nothing here creates a new claim, and
no panel is annotated with a causal reading.

    s1_source_validation   optimisation behaviour under the shared budget
    s2_domain_profile      class composition and acquisition heterogeneity
    s3_configuration       Q1 batch / resolution / combined decomposition
    s4_calibration         reliability curves, frozen and full, DDR and APTOS
    s5_risk_coverage       retrospective selective prediction
    s6_error_pattern       normalised confusion matrices under full FT

Every quantity is read from an authoritative CSV or recomputed from saved
predictions. No retinal image thumbnails are included: redistribution terms for
the four public datasets were not verified, so S2 is quantitative only.

Usage:
    python make_jbhi_supplement_figures.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

OUT = "jbhi_final"
IMAGENET = "#0072B2"
RETFOUND = "#D55E00"
INK = "#333333"
MUTED = "#8C8C8C"
GRID = "#E0E0E0"
DOMAIN_COLOURS = {"DDR": "#0072B2", "APTOS": "#D55E00",
                  "IDRiD": "#009E73", "EyePACS": "#CC79A7"}

TWO_COL = 7.16
SOURCES = {"ddr": "aptos-eyepacs-idrid", "aptos": "ddr-eyepacs-idrid"}
TAGS = {"frozen": "linprobe-b256", "full": "erm-b16-ftfull-lr0.0001"}
REFERENCE = "vit-large-mae-in1k"
CANDIDATE = "retfound-cfp"
SEEDS = [42, 1, 2, 3, 4]
# Reliability bins thinner than this (pooled over seeds) are not drawn.
MIN_BIN_COUNT = 50


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
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


def _load(outputs, domain, protocol, backbone, seed):
    from src.visualization.calibration_figures import load_target_predictions
    eid = f"lodo_{SOURCES[domain]}__{domain}_{backbone}_{TAGS[protocol]}_s{seed}"
    return load_target_predictions(eid, domain, outputs_dir=outputs)


# ------------------------------------------------------------------- S1
def s1_source_validation(plt, directory, ddr, aptos):
    """Descriptive optimisation behaviour. No causal annotation anywhere."""
    import numpy as np
    import pandas as pd

    frames = []
    for frame, domain in ((ddr, "DDR"), (aptos, "APTOS")):
        part = frame.copy()
        part["domain"] = domain
        frames.append(part)
    runs = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.6),
                             constrained_layout=True)
    arms = [("ImageNet-MAE", IMAGENET), ("RETFound", RETFOUND)]

    for ax, column, title, ylabel in (
            (axes[0], "best_epoch", "Best source-validation epoch", "epoch"),
            (axes[1], "overfit_ratio", "Source-validation loss inflation",
             "final ÷ minimum val loss")):
        for i, (arm, colour) in enumerate(arms):
            sub = runs[runs.model == arm]
            jitter = np.linspace(-0.15, 0.15, len(sub))
            ax.scatter(i + jitter, sub[column], s=20, facecolor="white",
                       edgecolor=colour, linewidth=1.0, zorder=3)
            ax.hlines(sub[column].mean(), i - 0.3, i + 0.3, color=colour,
                      linewidth=2.2, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([a for a, _ in arms])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8.0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1].axhline(1.0, color=INK, linewidth=0.8, linestyle=(0, (4, 3)),
                    zorder=1)

    ax = axes[2]
    counts = [int(runs[runs.model == arm].early_stopped.sum()) for arm, _ in arms]
    totals = [int((runs.model == arm).sum()) for arm, _ in arms]
    bars = ax.bar([0, 1], counts, width=0.55,
                  color=[c for _, c in arms], edgecolor="white", zorder=3)
    for bar, count, total in zip(bars, counts, totals):
        ax.annotate(f"{count}/{total}", (bar.get_x() + bar.get_width() / 2,
                                         count), ha="center", va="bottom",
                    fontsize=7.4, color=INK, fontweight="bold",
                    xytext=(0, 2), textcoords="offset points")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([a for a, _ in arms])
    ax.set_ylim(0, max(totals) * 1.22)
    ax.set_ylabel("runs early-stopped")
    ax.set_title("Early stopping (DDR + APTOS)", fontsize=8.0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.suptitle("Optimisation behaviour under the shared fixed budget "
                 "(source validation only; descriptive)", fontsize=8.5)
    return _save(fig, "s1_source_validation", directory)


# ------------------------------------------------------------------- S2
def s2_domain_profile(plt, directory, characteristics, image_stats):
    """Class composition and acquisition heterogeneity. Quantitative only."""
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.7),
                             constrained_layout=True)
    label_map = {"DDR": "DDR", "APTOS 2019": "APTOS", "IDRiD": "IDRiD",
                 "EyePACS": "EyePACS"}
    order = ["DDR", "APTOS", "IDRiD", "EyePACS"]

    # Panel A: normalised grade composition.
    ax = axes[0]
    grade_columns = [f"Grade {g}" for g in range(5)]
    bottom = np.zeros(len(order))
    shades = ["#F0F0F0", "#CFCFCF", "#9E9E9E", "#6B6B6B", "#333333"]
    for grade, shade in zip(range(5), shades):
        values = []
        for name in order:
            row = characteristics[characteristics.Domain.map(
                lambda d: label_map.get(d, d)) == name].iloc[0]
            total = sum(float(row[c]) for c in grade_columns)
            values.append(float(row[f"Grade {grade}"]) / total * 100.0)
        values = np.array(values)
        ax.bar(order, values, bottom=bottom, color=shade, edgecolor="white",
               linewidth=0.6, label=f"grade {grade}", zorder=3)
        bottom += values
    ax.set_ylabel("% of images")
    ax.set_ylim(0, 100)
    ax.set_title("DR grade composition", fontsize=8.0)
    ax.tick_params(axis="x", rotation=30)
    # The grade key goes at figure level. Anchored under this panel it landed
    # on the rotated "EyePACS" tick label.
    grade_handles, grade_labels = ax.get_legend_handles_labels()

    # Panels B and C: acquisition statistics from the sampled image audit.
    stats = image_stats[image_stats.readable == True]  # noqa: E712
    name_of = {"ddr": "DDR", "aptos": "APTOS", "idrid": "IDRiD",
               "eyepacs": "EyePACS"}
    for ax, column, title, xlabel in (
            (axes[1], "megapixels", "Image size", "megapixels (log)"),
            (axes[2], "brightness", "Mean brightness", "mean intensity")):
        data, labels, colours = [], [], []
        for key, name in name_of.items():
            values = stats[stats.domain == key][column].to_numpy(dtype=float)
            if len(values):
                data.append(values)
                labels.append(name)
                colours.append(DOMAIN_COLOURS[name])
        parts = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                           widths=0.6, showfliers=False,
                           medianprops=dict(color=INK, linewidth=1.2),
                           whiskerprops=dict(color=MUTED, linewidth=0.9),
                           capprops=dict(color=MUTED, linewidth=0.9),
                           boxprops=dict(linewidth=0.8))
        for patch, colour in zip(parts["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.35)
            patch.set_edgecolor(colour)
        if column == "megapixels":
            ax.set_yscale("log")
        ax.set_ylabel(xlabel)
        ax.set_title(title, fontsize=8.0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=30)

    n_per = int(stats.groupby("domain").size().min())
    fig.legend(grade_handles, grade_labels, loc="outside lower center", ncol=5,
               frameon=False, fontsize=6.8, columnspacing=1.4,
               handlelength=1.1, handletextpad=0.4)
    fig.suptitle("Dataset profiles: class composition and acquisition "
                 f"heterogeneity ($n={n_per}$ sampled images per domain)",
                 fontsize=8.3)
    return _save(fig, "s2_domain_profile", directory)


# ------------------------------------------------------------------- S3
def s3_configuration(plt, directory, decomposition, per_seed):
    """Q1: batch, resolution and combined contrasts, per held-out domain."""
    import numpy as np

    qwk = decomposition[decomposition.metric == "qwk"]
    contrasts = ["batch", "resolution", "combined"]
    labels = {"batch": "Batch\n224/b32 → 224/b16",
              "resolution": "Resolution\n224/b16 → 512/b16",
              "combined": "Combined\n224/b32 → 512/b16"}

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 3.0), sharey=True,
                             constrained_layout=True)
    for ax, target in zip(axes, ["eyepacs", "idrid"]):
        for i, contrast in enumerate(contrasts):
            row = qwk[(qwk.contrast == contrast) & (qwk.target == target)].iloc[0]
            seeds = [float(row[f"delta_seed{s}"]) for s in (1, 2, 42)]
            jitter = np.linspace(-0.13, 0.13, len(seeds))
            ax.scatter(np.full(len(seeds), i) + jitter, seeds, s=20,
                       facecolor="white", edgecolor=MUTED, linewidth=0.9,
                       zorder=3)
            ax.plot([i, i], [row.ci_lower, row.ci_upper], color=INK,
                    linewidth=1.5, zorder=4, solid_capstyle="butt")
            supported = bool(row.p_holm < 0.05)
            ax.scatter([i], [row.delta], s=52, marker="D",
                       color=INK if supported else "white", edgecolor=INK,
                       linewidth=1.1, zorder=5)
            ax.annotate(f"Holm $p$ = {row.p_holm:.3f}", (i, row.ci_upper),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6.4,
                        color=INK if supported else MUTED,
                        fontweight="bold" if supported else "normal")
        ax.axhline(0.0, color=INK, linewidth=0.9, linestyle=(0, (4, 3)),
                   zorder=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels([labels[c] for c in contrasts], fontsize=6.8)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(f"{target.upper() if target=='idrid' else 'EyePACS'} "
                     f"held out", fontsize=8.2)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1].set_title("IDRiD held out", fontsize=8.2)
    axes[0].set_ylabel("QWK difference (higher = better configuration)")
    fig.suptitle("Configuration decomposition, DenseNet-121, three seeds "
                 "(filled marker: survives Holm; open: does not)",
                 fontsize=8.2)
    return _save(fig, "s3_configuration", directory)


# ------------------------------------------------------------------- S4
def s4_calibration(plt, directory, outputs):
    """Reliability curves, averaged over the five common seeds."""
    import numpy as np

    from src.evaluation.calibration import calibration_metrics, reliability_curve

    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL, 4.6),
                             constrained_layout=True)
    rows = [("ddr", "DDR"), ("aptos", "APTOS")]
    cols = [("frozen", "Frozen linear probe"), ("full", "Full fine-tuning")]

    for r, (domain, domain_label) in enumerate(rows):
        for c, (protocol, protocol_label) in enumerate(cols):
            ax = axes[r][c]
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8,
                    linestyle=(0, (3, 3)), zorder=1)
            for backbone, label, colour, marker in (
                    (REFERENCE, "ImageNet-MAE", IMAGENET, "o"),
                    (CANDIDATE, "RETFound", RETFOUND, "s")):
                curves, eces = [], []
                for seed in SEEDS:
                    run = _load(outputs, domain, protocol, backbone, seed)
                    if run is None:
                        continue
                    # Temperature-scaled probabilities, matching how ECE is
                    # reported everywhere else in this study.
                    curve = reliability_curve(run["scaled"], run["y_true"])
                    curves.append(curve)
                    eces.append(calibration_metrics(run["scaled"],
                                                    run["y_true"])["ece"])
                if not curves:
                    continue
                centres = curves[0]["confidence"]
                accuracy = np.vstack([c["accuracy"] for c in curves])
                counts = np.vstack([c["count"] for c in curves]).sum(axis=0)
                with np.errstate(invalid="ignore"):
                    mean_accuracy = np.where(
                        np.all(np.isnan(accuracy), axis=0), np.nan,
                        np.nanmean(accuracy, axis=0))
                # A bin holding a handful of images produces an accuracy of 0
                # or 1 that is noise, and it drew a vertical spike into two of
                # these panels. Bins with fewer than 50 pooled images across
                # the five seeds are not drawn; the annotated ECE is still
                # computed on all of the data.
                mean_accuracy = np.where(counts >= MIN_BIN_COUNT,
                                         mean_accuracy, np.nan)
                ax.plot(centres, mean_accuracy, color=colour, linewidth=1.3,
                        marker=marker, markersize=3.0, zorder=3,
                        label=f"{label}  ECE {np.mean(eces):.3f}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title(f"{domain_label} — {protocol_label}", fontsize=8.0)
            ax.grid(color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, loc="upper left", fontsize=6.6)
            if r == 1:
                ax.set_xlabel("confidence")
            if c == 0:
                ax.set_ylabel("empirical accuracy")

    fig.suptitle("Reliability after temperature scaling on source validation "
                 "(mean over five seeds; secondary evidence)", fontsize=8.3)
    return _save(fig, "s4_calibration", directory)


# ------------------------------------------------------------------- S5
def s5_risk_coverage(plt, directory, outputs):
    """Retrospective selective prediction. Not a deployment claim."""
    import numpy as np

    from src.evaluation.selective_prediction import risk_coverage_curve

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 3.0),
                             constrained_layout=True)
    grid = np.linspace(0.05, 1.0, 96)

    for ax, (domain, label) in zip(axes, [("ddr", "DDR"), ("aptos", "APTOS")]):
        for protocol, dash in (("frozen", (0, (4, 2))), ("full", (0, ()))):
            for backbone, arm, colour in ((REFERENCE, "ImageNet-MAE", IMAGENET),
                                          (CANDIDATE, "RETFound", RETFOUND)):
                stacked = []
                for seed in SEEDS:
                    run = _load(outputs, domain, protocol, backbone, seed)
                    if run is None:
                        continue
                    confidence = run["scaled"].max(axis=1)
                    coverage, risk, _ = risk_coverage_curve(
                        run["y_true"], run["y_pred"], confidence)
                    stacked.append(np.interp(grid, coverage, risk))
                if not stacked:
                    continue
                ax.plot(grid, np.mean(stacked, axis=0), color=colour,
                        linestyle=dash, linewidth=1.3, zorder=3,
                        label=f"{arm}, {protocol}")
        ax.set_xlabel("coverage (fraction retained)")
        ax.set_title(f"{label} held out", fontsize=8.2)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("risk (error rate among retained)")
    axes[0].legend(frameon=False, fontsize=6.4, loc="upper left")
    fig.suptitle("Retrospective selective prediction, mean over five seeds — "
                 "not a deployment claim", fontsize=8.3)
    return _save(fig, "s5_risk_coverage", directory)


# ------------------------------------------------------------------- S6
def s6_error_pattern(plt, directory, outputs):
    """Row-normalised confusion under full FT, pooled over the five seeds."""
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL, 5.0),
                             constrained_layout=True)
    for r, (domain, domain_label) in enumerate([("ddr", "DDR"),
                                                ("aptos", "APTOS")]):
        for c, (backbone, arm) in enumerate([(REFERENCE, "ImageNet-MAE"),
                                             (CANDIDATE, "RETFound")]):
            ax = axes[r][c]
            matrix = np.zeros((5, 5), dtype=float)
            severe = []
            for seed in SEEDS:
                run = _load(outputs, domain, "full", backbone, seed)
                if run is None:
                    continue
                truth = np.asarray(run["y_true"], dtype=int)
                predicted = np.asarray(run["y_pred"], dtype=int)
                for t, p in zip(truth, predicted):
                    matrix[t, p] += 1
                severe.append(float(np.mean(np.abs(truth - predicted) >= 2)))
            normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
            image = ax.imshow(normalised, cmap="Greys", vmin=0, vmax=1,
                              aspect="equal")
            for i in range(5):
                for j in range(5):
                    value = normalised[i, j]
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                            fontsize=6.2,
                            color="white" if value > 0.55 else INK)
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels(range(5), fontsize=7)
            ax.set_yticklabels(range(5), fontsize=7)
            ax.set_title(f"{domain_label} — {arm}\nsevere-error rate "
                         f"{np.mean(severe):.4f}", fontsize=7.8)
            if r == 1:
                ax.set_xlabel("predicted grade")
            if c == 0:
                ax.set_ylabel("true grade")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_color(GRID)

    fig.colorbar(image, ax=axes, shrink=0.55, label="row-normalised frequency",
                 pad=0.02)
    fig.suptitle("Error pattern under full fine-tuning, pooled over five seeds "
                 "(rows sum to 1; severe error = |true − predicted| ≥ 2)",
                 fontsize=8.3)
    return _save(fig, "s6_error_pattern", directory)


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    plt = _style()
    root = project_root()
    outputs = root / "outputs"
    tables = outputs / "tables"
    directory = outputs / "figures" / OUT
    directory.mkdir(parents=True, exist_ok=True)

    required = {
        "ddr_sv": tables / "full_finetune_source_validation.csv",
        "aptos_sv": tables / "aptos_full_finetune_source_validation.csv",
        "characteristics": tables / "table1_dataset_characteristics.csv",
        "image_stats": outputs / "reports" / "image_statistics.csv",
        "decomposition": tables / "configuration_decomposition.csv",
        "per_seed": tables / "configuration_per_seed.csv",
    }
    for key, path in required.items():
        if not path.exists():
            print(f"NOT RUN -- {key}: {path.name} missing")
            return 1

    written = []
    written += s1_source_validation(
        plt, directory, pd.read_csv(required["ddr_sv"]),
        pd.read_csv(required["aptos_sv"]))
    written += s2_domain_profile(
        plt, directory, pd.read_csv(required["characteristics"]),
        pd.read_csv(required["image_stats"]))
    written += s3_configuration(
        plt, directory, pd.read_csv(required["decomposition"]),
        pd.read_csv(required["per_seed"]))
    written += s4_calibration(plt, directory, outputs)
    written += s5_risk_coverage(plt, directory, outputs)
    written += s6_error_pattern(plt, directory, outputs)

    for path in written:
        print(f"  {path.relative_to(root)}  ({path.stat().st_size/1024:.0f} KB)")
    print(f"\n{len(written)} file(s) in outputs/figures/{OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
