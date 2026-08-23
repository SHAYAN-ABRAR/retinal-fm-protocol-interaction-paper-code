"""Dataset-level quality-control and domain-shift figures (Phase 2).

Every function takes data and returns a matplotlib Figure; saving is the
caller's job via :func:`..style.save_figure`.  :func:`generate_all` runs the lot
and reports what it wrote.

These figures answer three questions:

1. *What is in each dataset?*  Counts, class balance, example images.
2. *Is anything broken?*  Unreadable files, duplicates, degenerate aspect ratios.
3. *How different do the domains look before any model is involved?*  Geometry
   and photometry distributions, and a PCA of those statistics.

Question 3 is descriptive.  Separation in a PCA of hand-picked image statistics
shows the domains differ in appearance; it does **not** show a model will rely on
that difference.  The figures are labelled accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.schema import GRADE_NAMES, N_GRADES
from ..utils.logging import get_logger
from .style import (
    DOMAIN_ORDER,
    GRADE_LABELS,
    apply_style,
    domain_color,
    domain_label,
    grade_color,
    save_figure,
)

log = get_logger("viz.dataset")

__all__ = [
    "figure_images_per_dataset",
    "figure_class_distribution",
    "figure_class_distribution_normalized",
    "figure_domain_class_heatmap",
    "figure_class_distribution_stacked",
    "figure_split_composition",
    "figure_grade_examples_grid",
    "figure_statistic_distribution",
    "figure_rgb_channel_statistics",
    "figure_domain_shift_boxplots",
    "figure_statistics_pca",
    "figure_preprocessing_examples",
    "figure_augmentation_examples",
    "figure_data_quality_summary",
    "generate_all",
]


def _ordered_domains(frame: "Any") -> list[str]:
    present = set(frame["domain"].unique())
    return [d for d in DOMAIN_ORDER if d in present] + sorted(present - set(DOMAIN_ORDER))


# ---------------------------------------------------------------------------
# 1-5: composition
# ---------------------------------------------------------------------------

def figure_images_per_dataset(manifest: "Any") -> Any:
    import matplotlib.pyplot as plt

    domains = _ordered_domains(manifest)
    counts = [int((manifest["domain"] == d).sum()) for d in domains]

    figure, axis = plt.subplots(figsize=(6.0, 3.8))
    bars = axis.bar(
        [domain_label(d) for d in domains], counts,
        color=[domain_color(d) for d in domains], edgecolor="black", linewidth=0.5,
    )
    for bar, count in zip(bars, counts):
        axis.annotate(
            f"{count:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points",
        )
    axis.set_ylabel("Number of images")
    axis.set_title("Dataset size by domain")
    axis.set_ylim(0, max(counts) * 1.15)
    axis.set_yscale("log")
    axis.set_ylabel("Number of images (log scale)")
    figure.tight_layout()
    return figure


def figure_class_distribution(manifest: "Any") -> Any:
    """Grouped bars: absolute counts per grade per domain."""
    import matplotlib.pyplot as plt

    domains = _ordered_domains(manifest)
    width = 0.8 / len(domains)
    positions = np.arange(N_GRADES)

    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    for i, domain in enumerate(domains):
        subset = manifest[manifest["domain"] == domain]
        counts = [int((subset["grade"] == g).sum()) for g in range(N_GRADES)]
        axis.bar(
            positions + i * width - 0.4 + width / 2, counts, width,
            label=domain_label(domain), color=domain_color(domain),
            edgecolor="black", linewidth=0.4,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([GRADE_LABELS[g] for g in range(N_GRADES)])
    axis.set_xlabel("DR severity grade (ICDR)")
    axis.set_ylabel("Number of images (log scale)")
    axis.set_yscale("log")
    axis.set_title("Class distribution by domain (absolute counts)")
    axis.legend(title="Domain", ncol=2)
    figure.tight_layout()
    return figure


def figure_class_distribution_normalized(manifest: "Any") -> Any:
    """Grouped bars: within-domain class proportions.

    This is the figure that shows *label shift*: the domains do not merely differ
    in appearance, they differ in how common each grade is.
    """
    import matplotlib.pyplot as plt

    domains = _ordered_domains(manifest)
    width = 0.8 / len(domains)
    positions = np.arange(N_GRADES)

    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    for i, domain in enumerate(domains):
        subset = manifest[manifest["domain"] == domain]
        share = [100.0 * (subset["grade"] == g).sum() / len(subset) for g in range(N_GRADES)]
        axis.bar(
            positions + i * width - 0.4 + width / 2, share, width,
            label=domain_label(domain), color=domain_color(domain),
            edgecolor="black", linewidth=0.4,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([GRADE_LABELS[g] for g in range(N_GRADES)])
    axis.set_xlabel("DR severity grade (ICDR)")
    axis.set_ylabel("Share of the domain (%)")
    axis.set_title("Normalised class distribution by domain (label shift)")
    axis.legend(title="Domain", ncol=2)
    figure.tight_layout()
    return figure


def figure_domain_class_heatmap(manifest: "Any", *, normalize: bool = True) -> Any:
    import matplotlib.pyplot as plt

    domains = _ordered_domains(manifest)
    matrix = np.zeros((len(domains), N_GRADES))
    for i, domain in enumerate(domains):
        subset = manifest[manifest["domain"] == domain]
        for grade in range(N_GRADES):
            count = int((subset["grade"] == grade).sum())
            matrix[i, grade] = 100.0 * count / len(subset) if normalize else count

    figure, axis = plt.subplots(figsize=(6.5, 3.4))
    image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(range(N_GRADES))
    axis.set_xticklabels([GRADE_LABELS[g] for g in range(N_GRADES)], rotation=20, ha="right")
    axis.set_yticks(range(len(domains)))
    axis.set_yticklabels([domain_label(d) for d in domains])
    axis.grid(False)

    threshold = matrix.max() * 0.6
    for i in range(len(domains)):
        for j in range(N_GRADES):
            axis.text(
                j, i, f"{matrix[i, j]:.1f}" + ("%" if normalize else ""),
                ha="center", va="center", fontsize=8.5,
                color="white" if matrix[i, j] > threshold else "black",
            )
    axis.set_title("Class composition by domain" + (" (% within domain)" if normalize else ""))
    figure.colorbar(image, ax=axis, label="% of domain" if normalize else "images")
    figure.tight_layout()
    return figure


def figure_class_distribution_stacked(manifest: "Any") -> Any:
    import matplotlib.pyplot as plt

    domains = _ordered_domains(manifest)
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    bottom = np.zeros(len(domains))
    for grade in range(N_GRADES):
        share = np.array(
            [
                100.0 * (manifest[manifest["domain"] == d]["grade"] == grade).sum()
                / len(manifest[manifest["domain"] == d])
                for d in domains
            ]
        )
        axis.bar(
            [domain_label(d) for d in domains], share, bottom=bottom,
            label=GRADE_LABELS[grade], color=grade_color(grade),
            edgecolor="white", linewidth=0.6,
        )
        bottom += share
    axis.set_ylabel("Share of the domain (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Stacked class composition by domain")
    axis.legend(title="Grade", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    return figure


def figure_split_composition(manifest: "Any") -> Any:
    """Train/val/test sizes per domain, to show the splits are sane."""
    import matplotlib.pyplot as plt

    if "split" not in manifest.columns:
        raise ValueError("manifest has no 'split' column")

    domains = _ordered_domains(manifest)
    splits = ["train", "val", "test"]
    shades = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    width = 0.8 / len(splits)
    positions = np.arange(len(domains))
    for i, split in enumerate(splits):
        counts = [
            int(((manifest["domain"] == d) & (manifest["split"] == split)).sum())
            for d in domains
        ]
        bars = axis.bar(
            positions + i * width - 0.4 + width / 2, counts, width,
            label=split, color=shades[split], edgecolor="black", linewidth=0.4,
        )
        for bar, count in zip(bars, counts):
            axis.annotate(
                f"{count:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=7.5,
                xytext=(0, 1.5), textcoords="offset points", rotation=90,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels([domain_label(d) for d in domains])
    axis.set_ylabel("Number of images (log scale)")
    axis.set_yscale("log")
    axis.set_title("Split composition by domain")
    axis.legend(title="Split")
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# 6-7: example images
# ---------------------------------------------------------------------------

def figure_grade_examples_grid(
    manifest: "Any",
    *,
    image_size: int = 224,
    seed: int = 42,
    preprocess: bool = True,
) -> Any:
    """Grid of domains (rows) x grades (columns), one representative each.

    This is the figure that makes cross-domain appearance difference visible at a
    glance -- and equally, shows how similar the same grade can look across
    domains.
    """
    import matplotlib.pyplot as plt

    from ..data.preprocessing import PreprocessConfig, load_and_preprocess

    config = PreprocessConfig(image_size=image_size)
    domains = _ordered_domains(manifest)
    rng = np.random.default_rng(seed)

    figure, axes = plt.subplots(
        len(domains), N_GRADES, figsize=(2.0 * N_GRADES, 2.15 * len(domains))
    )
    axes = np.atleast_2d(axes)

    for row, domain in enumerate(domains):
        for column in range(N_GRADES):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)

            candidates = manifest[
                (manifest["domain"] == domain) & (manifest["grade"] == column)
            ]
            if not len(candidates):
                axis.text(0.5, 0.5, "no example", ha="center", va="center", fontsize=8)
                axis.set_facecolor("#EEEEEE")
            else:
                path = candidates.iloc[int(rng.integers(len(candidates)))]["path"]
                try:
                    array = (
                        load_and_preprocess(path, config)
                        if preprocess
                        else np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(path))
                    )
                    axis.imshow(array)
                except Exception as exc:  # noqa: BLE001
                    axis.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=7)
                    log.warning("example image failed: %s (%s)", path, exc)

            if row == 0:
                axis.set_title(GRADE_LABELS[column], fontsize=10)
            if column == 0:
                axis.set_ylabel(domain_label(domain), fontsize=10, fontweight="bold")

    figure.suptitle(
        "Representative images by domain (rows) and DR grade (columns)"
        + (f" -- after preprocessing at {image_size}px" if preprocess else " -- raw"),
        fontsize=12, y=1.0,
    )
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# 8-13: appearance statistics
# ---------------------------------------------------------------------------

def figure_statistic_distribution(
    stats: "Any",
    column: str,
    *,
    title: str,
    xlabel: str,
    log_x: bool = False,
    bins: int = 45,
) -> Any:
    """Overlaid per-domain histogram of one image statistic."""
    import matplotlib.pyplot as plt

    usable = stats[stats["readable"]]
    domains = _ordered_domains(usable)
    values = usable[column].astype(float)
    finite = values[np.isfinite(values)]

    # Widen the range by a hair before building the edges. Several domains are
    # degenerate on the geometry columns -- every IDRiD image is exactly
    # 4288x2848, which is also the global maximum -- and with exact edges those
    # values land on (or just outside, after floating-point rounding) the last
    # boundary. The domain's histogram then sums to zero and `density=True`
    # divides by it, producing an all-NaN bar and a RuntimeWarning.
    low, high = float(finite.min()), float(finite.max())
    if high <= low:                       # a single unique value across the corpus
        low, high = low * 0.99, high * 1.01 if high else 1.0
    else:
        margin = (high - low) * 1e-3
        low, high = low - margin, high + margin

    if log_x:
        edges = np.logspace(np.log10(max(low, 1e-6)), np.log10(high), bins)
    else:
        edges = np.linspace(low, high, bins)

    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    for domain in domains:
        subset = usable[usable["domain"] == domain][column].astype(float)
        axis.hist(
            subset, bins=edges, alpha=0.55, label=domain_label(domain),
            color=domain_color(domain), edgecolor="none", density=True,
        )
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.set_title(title)
    axis.legend(title="Domain")
    figure.tight_layout()
    return figure


def figure_rgb_channel_statistics(stats: "Any") -> Any:
    """Mean and standard deviation of each RGB channel, by domain."""
    import matplotlib.pyplot as plt

    usable = stats[stats["readable"]]
    domains = _ordered_domains(usable)
    channels = ["r", "g", "b"]
    channel_names = ["Red", "Green", "Blue"]

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    for panel, (suffix, label) in enumerate(
        [("mean", "Channel mean (0-255)"), ("std", "Channel std. dev.")]
    ):
        axis = axes[panel]
        width = 0.8 / len(domains)
        positions = np.arange(len(channels))
        for i, domain in enumerate(domains):
            subset = usable[usable["domain"] == domain]
            values = [subset[f"{c}_{suffix}"].astype(float).mean() for c in channels]
            errors = [subset[f"{c}_{suffix}"].astype(float).std() for c in channels]
            axis.bar(
                positions + i * width - 0.4 + width / 2, values, width,
                yerr=errors, capsize=2.5, label=domain_label(domain),
                color=domain_color(domain), edgecolor="black", linewidth=0.4,
                error_kw={"linewidth": 0.7},
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(channel_names)
        axis.set_ylabel(label)
        axis.set_title(label.split(" (")[0])
    axes[0].legend(title="Domain", fontsize=8)
    figure.suptitle("RGB channel statistics by domain (mean ± SD across images)", fontsize=12)
    figure.tight_layout()
    return figure


def figure_domain_shift_boxplots(stats: "Any") -> Any:
    """Box plots of the appearance statistics that differ most across domains."""
    import matplotlib.pyplot as plt

    usable = stats[stats["readable"]]
    domains = _ordered_domains(usable)
    panels = [
        ("brightness", "Mean luminance (0-255)"),
        ("contrast", "RMS contrast"),
        ("aspect_ratio", "Aspect ratio (w/h)"),
        ("megapixels", "Resolution (megapixels)"),
        ("b_mean", "Blue channel mean"),
        ("r_mean", "Red channel mean"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    for axis, (column, label) in zip(axes.ravel(), panels):
        data = [usable[usable["domain"] == d][column].astype(float).dropna() for d in domains]
        parts = axis.boxplot(
            data, patch_artist=True, showfliers=False,
            medianprops={"color": "black", "linewidth": 1.3},
            tick_labels=[domain_label(d) for d in domains],
        )
        for patch, domain in zip(parts["boxes"], domains):
            patch.set_facecolor(domain_color(domain))
            patch.set_alpha(0.75)
        axis.set_ylabel(label)
        axis.set_title(label.split(" (")[0])
        axis.tick_params(axis="x", rotation=20)
        if column == "megapixels":
            axis.set_yscale("log")

    figure.suptitle(
        "Image appearance statistics by domain (descriptive; outliers hidden)", fontsize=12
    )
    figure.tight_layout()
    return figure


def figure_statistics_pca(stats: "Any", *, seed: int = 42) -> Any:
    """PCA of standardised image statistics, coloured by domain and by grade.

    A caveat belongs on this figure and is printed on it: these are *hand-picked
    summary statistics*, not learned features.  Clean separation by domain shows
    the datasets differ in basic appearance. It does not by itself show that a
    trained model depends on those differences -- the learned-embedding figures
    in Phase 4 address that.
    """
    import matplotlib.pyplot as plt

    usable = stats[stats["readable"]].copy()
    columns = [
        "brightness", "contrast", "aspect_ratio", "megapixels",
        "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    ]
    matrix = usable[columns].astype(float).to_numpy()
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    keep = np.isfinite(usable[columns].astype(float).to_numpy()).all(axis=1)
    usable = usable[keep]

    # Standardise, then PCA via SVD (no sklearn dependency needed here).
    centred = matrix - matrix.mean(axis=0)
    scale = centred.std(axis=0)
    scale[scale == 0] = 1.0
    standardised = centred / scale
    _u, singular, components = np.linalg.svd(standardised, full_matrices=False)
    projected = standardised @ components[:2].T
    explained = (singular**2 / (singular**2).sum())[:2] * 100

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    for domain in _ordered_domains(usable):
        mask = (usable["domain"] == domain).to_numpy()
        axes[0].scatter(
            projected[mask, 0], projected[mask, 1], s=9, alpha=0.55,
            label=domain_label(domain), color=domain_color(domain), linewidths=0,
        )
    axes[0].set_title("Coloured by domain")
    axes[0].legend(title="Domain", fontsize=8)

    for grade in range(N_GRADES):
        mask = (usable["grade"] == grade).to_numpy()
        if not mask.any():
            continue
        axes[1].scatter(
            projected[mask, 0], projected[mask, 1], s=9, alpha=0.55,
            label=GRADE_LABELS[grade], color=grade_color(grade), linewidths=0,
        )
    axes[1].set_title("Coloured by DR grade")
    axes[1].legend(title="Grade", fontsize=8)

    for axis in axes:
        axis.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")

    figure.suptitle(
        "PCA of hand-crafted image statistics\n"
        "Descriptive only: separation by domain shows appearance differs, "
        "not that a model relies on it",
        fontsize=11,
    )
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# 14-15: preprocessing and augmentation
# ---------------------------------------------------------------------------

def figure_preprocessing_examples(
    manifest: "Any", *, image_size: int = 224, seed: int = 42
) -> Any:
    """Before/after panels of the retina crop, one row per domain."""
    import matplotlib.pyplot as plt
    from PIL import Image

    from ..data.preprocessing import (
        PreprocessConfig,
        estimate_retina_bbox,
        preprocess_array,
    )

    config = PreprocessConfig(image_size=image_size)
    domains = _ordered_domains(manifest)
    rng = np.random.default_rng(seed)

    figure, axes = plt.subplots(len(domains), 3, figsize=(7.6, 2.6 * len(domains)))
    axes = np.atleast_2d(axes)

    for row, domain in enumerate(domains):
        candidates = manifest[manifest["domain"] == domain]
        path = candidates.iloc[int(rng.integers(len(candidates)))]["path"]
        with Image.open(path) as handle:
            original = np.asarray(handle.convert("RGB"))

        bbox = estimate_retina_bbox(original, threshold=config.background_threshold)
        processed = preprocess_array(original, config)

        axes[row, 0].imshow(original)
        axes[row, 0].set_ylabel(domain_label(domain), fontsize=10, fontweight="bold")
        axes[row, 0].set_title(
            f"original {original.shape[1]}x{original.shape[0]}" if row == 0 else
            f"{original.shape[1]}x{original.shape[0]}", fontsize=8.5,
        )

        cropped = original if bbox is None else original[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        axes[row, 1].imshow(cropped)
        retained = 100.0 * cropped.size / original.size
        axes[row, 1].set_title(
            ("retina crop " if row == 0 else "") + f"({retained:.0f}% kept)", fontsize=8.5
        )
        if bbox is not None:
            axes[row, 0].add_patch(
                plt.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    fill=False, edgecolor="#00FF66", linewidth=1.4,
                )
            )

        axes[row, 2].imshow(processed)
        axes[row, 2].set_title(
            ("square pad + resize " if row == 0 else "") + f"{image_size}x{image_size}",
            fontsize=8.5,
        )

        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)

    figure.suptitle("Harmonised preprocessing: retina localisation, square framing, resize", fontsize=12)
    figure.tight_layout()
    return figure


def figure_augmentation_examples(
    manifest: "Any", *, image_size: int = 224, n_examples: int = 7, seed: int = 42
) -> Any:
    """One image, its deterministic preprocessing, and several augmented draws."""
    import matplotlib.pyplot as plt

    from ..data.augmentations import AugmentationConfig, build_train_transform
    from ..data.preprocessing import PreprocessConfig, load_and_preprocess

    config = PreprocessConfig(image_size=image_size)
    rng = np.random.default_rng(seed)
    row = manifest.iloc[int(rng.integers(len(manifest)))]
    base = load_and_preprocess(row["path"], config)

    try:
        transform = build_train_transform(AugmentationConfig(image_size=image_size))
    except ImportError:
        figure, axis = plt.subplots(figsize=(6, 2))
        axis.text(
            0.5, 0.5, "albumentations is not installed\n(pip install -r requirements.txt)",
            ha="center", va="center",
        )
        axis.set_axis_off()
        return figure

    mean = np.array(AugmentationConfig().mean)
    std = np.array(AugmentationConfig().std)

    columns = n_examples + 1
    figure, axes = plt.subplots(1, columns, figsize=(1.75 * columns, 2.35))
    axes[0].imshow(base)
    axes[0].set_title("preprocessed\n(deterministic)", fontsize=8)
    for i in range(1, columns):
        tensor = transform(image=base)["image"].numpy().transpose(1, 2, 0)
        axes[i].imshow(np.clip(tensor * std + mean, 0, 1))
        axes[i].set_title(f"aug {i}", fontsize=8)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)

    figure.suptitle(
        f"Training augmentation draws -- {domain_label(row['domain'])}, "
        f"grade {int(row['grade'])} ({GRADE_NAMES[int(row['grade'])]})",
        fontsize=11,
    )
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# 16-17: data quality
# ---------------------------------------------------------------------------

def figure_data_quality_summary(
    stats: "Any",
    *,
    deduplication: Any | None = None,
) -> Any:
    """Unreadable-image counts and the duplicate-removal outcome."""
    import matplotlib.pyplot as plt

    domains = _ordered_domains(stats)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    unreadable = [int((~stats[stats["domain"] == d]["readable"]).sum()) for d in domains]
    sampled = [int((stats["domain"] == d).sum()) for d in domains]
    bars = axes[0].bar(
        [domain_label(d) for d in domains], unreadable,
        color=[domain_color(d) for d in domains], edgecolor="black", linewidth=0.5,
    )
    for bar, bad, total in zip(bars, unreadable, sampled):
        axes[0].annotate(
            f"{bad}/{total}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points",
        )
    axes[0].set_ylabel("Unreadable images")
    axes[0].set_title("Corrupt / unreadable images (in the inspected sample)")
    axes[0].set_ylim(0, max(max(unreadable), 1) * 1.4)

    axis = axes[1]
    if deduplication is None:
        axis.text(0.5, 0.5, "deduplication report not provided", ha="center", va="center")
        axis.set_axis_off()
    else:
        labels = ["duplicate\ngroups", "consistent\n(kept 1)", "conflicting\n(dropped)"]
        values = [
            deduplication.n_groups,
            deduplication.n_consistent_groups,
            deduplication.n_conflicting_groups,
        ]
        bars = axis.bar(
            labels, values, color=["#666666", "#0072B2", "#D73027"],
            edgecolor="black", linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points",
            )
        axis.set_ylabel("Groups of byte-identical images")
        axis.set_title(
            f"Exact duplicates removed before splitting\n"
            f"({deduplication.n_removed:,} images dropped from "
            f"{deduplication.n_input:,})",
            fontsize=10,
        )
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def generate_all(
    manifest: "Any",
    stats: "Any",
    figures_dir: Path | str,
    *,
    deduplication: Any | None = None,
    image_size: int = 224,
    seed: int = 42,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Generate every Phase-2 dataset figure. Returns the written paths."""
    apply_style()
    written: list[Path] = []

    def emit(name: str, figure: Any) -> None:
        written.extend(save_figure(figure, name, figures_dir, formats=formats))

    emit("01_images_per_dataset", figure_images_per_dataset(manifest))
    emit("02_class_distribution_counts", figure_class_distribution(manifest))
    emit("03_class_distribution_normalized", figure_class_distribution_normalized(manifest))
    emit("04_domain_class_heatmap", figure_domain_class_heatmap(manifest))
    emit("05_class_distribution_stacked", figure_class_distribution_stacked(manifest))
    if "split" in manifest.columns:
        emit("06_split_composition", figure_split_composition(manifest))
    emit(
        "07_grade_examples_by_domain",
        figure_grade_examples_grid(manifest, image_size=image_size, seed=seed),
    )

    emit(
        "08_image_width_distribution",
        figure_statistic_distribution(
            stats, "width", title="Image width by domain",
            xlabel="Width (pixels, log scale)", log_x=True,
        ),
    )
    emit(
        "09_image_height_distribution",
        figure_statistic_distribution(
            stats, "height", title="Image height by domain",
            xlabel="Height (pixels, log scale)", log_x=True,
        ),
    )
    emit(
        "10_aspect_ratio_distribution",
        figure_statistic_distribution(
            stats, "aspect_ratio", title="Aspect ratio by domain",
            xlabel="Aspect ratio (width / height)",
        ),
    )
    emit(
        "11_brightness_distribution",
        figure_statistic_distribution(
            stats, "brightness", title="Brightness by domain (after retina crop)",
            xlabel="Mean luminance (0-255)",
        ),
    )
    emit(
        "12_contrast_distribution",
        figure_statistic_distribution(
            stats, "contrast", title="Contrast by domain (after retina crop)",
            xlabel="RMS contrast (luminance std. dev.)",
        ),
    )
    emit("13_rgb_channel_statistics", figure_rgb_channel_statistics(stats))
    emit("14_domain_shift_boxplots", figure_domain_shift_boxplots(stats))
    emit("15_image_statistics_pca", figure_statistics_pca(stats, seed=seed))

    emit(
        "16_preprocessing_examples",
        figure_preprocessing_examples(manifest, image_size=image_size, seed=seed),
    )
    emit(
        "17_augmentation_examples",
        figure_augmentation_examples(manifest, image_size=image_size, seed=seed),
    )
    emit("18_data_quality_summary", figure_data_quality_summary(stats, deduplication=deduplication))

    log.info("generated %d dataset figures in %s", len(written), figures_dir)
    return written
