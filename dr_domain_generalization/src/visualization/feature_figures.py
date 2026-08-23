"""Learned-representation figures: projections coloured by domain and by grade.

These are the figures that make the domain-invariance claim visible. Each one
carries the **quantitative** summary (linear-probe accuracy, silhouette scores)
in its title, because a projection alone is not evidence -- UMAP in particular
can manufacture apparent clusters from noise, and its global geometry is not
meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.schema import N_GRADES
from ..utils.logging import get_logger
from .style import GRADE_LABELS, apply_style, domain_color, domain_label, grade_color, save_figure

log = get_logger("viz.features")

__all__ = [
    "figure_embedding",
    "figure_domain_centroids",
    "figure_domain_separability_comparison",
    "generate_embedding_figures",
]


def figure_embedding(
    result: Any,
    method: str = "pca",
    *,
    title_prefix: str = "",
) -> Any:
    """One projection, shown twice: coloured by domain and by DR grade.

    Reading it: if the left panel separates cleanly and the right does not, the
    representation encodes the dataset more strongly than the disease -- which is
    what a domain-generalization method should reduce.
    """
    import matplotlib.pyplot as plt

    if method not in result.projections:
        raise KeyError(f"no {method!r} projection; available: {sorted(result.projections)}")
    points = result.projections[method]

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    for domain_id in sorted(set(result.domain_id.tolist())):
        mask = result.domain_id == domain_id
        name = str(result.domain[mask][0]) if mask.any() else str(domain_id)
        axes[0].scatter(
            points[mask, 0], points[mask, 1], s=7, alpha=0.5, linewidths=0,
            color=domain_color(name), label=domain_label(name),
        )
    axes[0].set_title("Coloured by domain")
    axes[0].legend(title="Domain", fontsize=8, markerscale=2)

    for grade in range(N_GRADES):
        mask = result.y_true == grade
        if not mask.any():
            continue
        axes[1].scatter(
            points[mask, 0], points[mask, 1], s=7, alpha=0.5, linewidths=0,
            color=grade_color(grade), label=GRADE_LABELS[grade],
        )
    axes[1].set_title("Coloured by DR grade")
    axes[1].legend(title="Grade", fontsize=8, markerscale=2)

    for axis in axes:
        axis.set_xlabel(f"{method.upper()} 1")
        axis.set_ylabel(f"{method.upper()} 2")
        if method != "pca":
            # UMAP/t-SNE axes have no interpretable scale.
            axis.set_xticks([])
            axis.set_yticks([])

    separability = result.domain_separability
    clusters = result.cluster_scores
    figure.suptitle(
        f"{title_prefix}Learned features -- {method.upper()} projection\n"
        f"domain linear probe {separability.get('probe_accuracy', float('nan')):.3f} "
        f"(chance {separability.get('chance', float('nan')):.3f}, "
        f"lift {separability.get('lift', float('nan')):.3f})  |  "
        f"silhouette: domain {clusters.get('silhouette_by_domain', float('nan')):.3f}, "
        f"grade {clusters.get('silhouette_by_grade', float('nan')):.3f}",
        fontsize=10.5,
    )
    figure.tight_layout()
    return figure


def figure_domain_centroids(result: Any, *, title_prefix: str = "") -> Any:
    """Pairwise distance between domain centroids, relative to within-domain spread.

    A ratio near 0 means the domains occupy the same region of feature space;
    values above ~1 mean the centroids are further apart than the domains are
    wide, i.e. the representation still separates them.
    """
    import matplotlib.pyplot as plt

    ratios = result.centroid_distances.get("distance_over_within_domain_spread", {})
    if not ratios:
        raise ValueError("no centroid distances in this result")

    labels = list(ratios)
    values = [ratios[k] for k in labels]

    figure, axis = plt.subplots(figsize=(max(5.5, 1.5 * len(labels) + 2.5), 3.8))
    bars = axis.bar(labels, values, color="#0072B2", edgecolor="black", linewidth=0.5)
    for bar, value in zip(bars, values):
        axis.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", fontsize=8.5,
                      xytext=(0, 2), textcoords="offset points")
    axis.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    axis.annotate("centroid gap = within-domain spread", (0.01, 1.0),
                  xycoords=("axes fraction", "data"), fontsize=8, color="grey", va="bottom")
    axis.set_ylabel("Centroid distance / within-domain spread")
    axis.set_title(f"{title_prefix}Domain separation in feature space")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    return figure


def figure_domain_separability_comparison(rows: Sequence[dict[str, Any]]) -> Any:
    """Compare methods by how much domain information their features retain.

    The headline DG figure: a method that reduces the probe's lift toward zero
    has produced a more domain-invariant representation. Whether that *helps*
    target accuracy is a separate question the results table answers.
    """
    import matplotlib.pyplot as plt

    labels = [r["method"] for r in rows]
    lift = [r["lift"] for r in rows]
    silhouette_domain = [r.get("silhouette_by_domain", float("nan")) for r in rows]
    silhouette_grade = [r.get("silhouette_by_grade", float("nan")) for r in rows]

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    positions = np.arange(len(rows))

    bars = axes[0].bar(positions, lift, color="#D55E00", edgecolor="black", linewidth=0.5)
    for bar, value in zip(bars, lift):
        axes[0].annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=8,
                         xytext=(0, 2), textcoords="offset points")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Domain-probe lift over chance")
    axes[0].set_title("Domain information retained (lower = more invariant)")
    axes[0].set_ylim(0, max(max(lift) * 1.25, 0.1))

    width = 0.38
    axes[1].bar(positions - width / 2, silhouette_domain, width, label="by domain",
                color="#D55E00", edgecolor="black", linewidth=0.4)
    axes[1].bar(positions + width / 2, silhouette_grade, width, label="by grade",
                color="#009E73", edgecolor="black", linewidth=0.4)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Clustering by domain vs by grade")
    axes[1].legend(fontsize=8)

    figure.suptitle("Representation analysis across methods", fontsize=12)
    figure.tight_layout()
    return figure


def generate_embedding_figures(
    result: Any,
    figures_dir: Path | str,
    *,
    prefix: str = "embedding",
    title_prefix: str = "",
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Write every available projection plus the centroid-distance figure."""
    apply_style()
    written: list[Path] = []
    for method in result.projections:
        written.extend(save_figure(
            figure_embedding(result, method, title_prefix=title_prefix),
            f"{prefix}_{method}", figures_dir, formats=formats,
        ))
    if result.centroid_distances:
        written.extend(save_figure(
            figure_domain_centroids(result, title_prefix=title_prefix),
            f"{prefix}_centroids", figures_dir, formats=formats,
        ))
    return written
