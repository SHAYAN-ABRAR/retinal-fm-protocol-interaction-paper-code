"""Cross-domain result matrices: the central figures of the paper.

Three views, each answering a different question:

:func:`figure_cross_domain_matrix`
    Train-domain x test-domain heatmap for one metric. The diagonal is
    in-domain, the off-diagonal is external. The *drop* from diagonal to
    off-diagonal within a row is the generalization gap for that training set.

:func:`figure_lodo_summary`
    Leave-one-domain-out: per-target source-validation score against unseen-target
    score, so the four gaps are directly comparable. This is the figure that
    answers "how much does performance fall on an unseen clinic?".

:func:`figure_metric_heatmap_grid`
    The same matrix for several metrics at once (QWK, macro F1, ECE, NLL), so a
    reader can see that discrimination and calibration degrade differently -- the
    study's main claim.

A presentation rule applied throughout: **an unrun cell is drawn as hatched grey
and labelled "NOT RUN"**, never as a blank or a zero. A white cell in a heatmap
reads as "bad result"; an absent experiment must not be mistaken for one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..utils.logging import get_logger
from .style import DOMAIN_LABELS, DOMAIN_ORDER, apply_style, save_figure

log = get_logger("viz.domain")

__all__ = [
    "build_cross_domain_matrix",
    "figure_cross_domain_matrix",
    "figure_lodo_summary",
    "figure_metric_heatmap_grid",
    "generate_domain_figures",
]

LOWER_IS_BETTER = {"ece", "nll", "brier", "mae_grade", "severe_error_rate"}
NOT_RUN_COLOR = "#DDDDDD"


def build_cross_domain_matrix(
    registry: Any,
    *,
    metric: str = "test_qwk",
    domains: Sequence[str] = tuple(DOMAIN_ORDER),
    method: str | None = None,
    seed: int | None = None,
    aggregate: str = "mean",
) -> tuple[np.ndarray, list[str]]:
    """Assemble a (train-domain x test-domain) matrix from the registry.

    ``NaN`` marks an experiment that has not been run, and is rendered
    distinctly rather than being imputed or shown as zero.

    ``method`` filters to a single method. **Pass it for any figure that goes in
    the paper.** Without it, several methods and seeds can target the same cell,
    and the cell then shows their ``aggregate`` (mean by default) rather than one
    experiment -- which is a different quantity and must not be labelled as a
    method's result. A warning is emitted whenever a cell aggregates more than
    one run, so this cannot happen silently.

    ``seed`` filters further to a single training seed. Several seeds of the
    same method land in the same cell, and their mean is not any one run's
    result either -- pass this for paper figures alongside ``method``.
    """
    from collections import defaultdict

    domains = list(domains)
    matrix = np.full((len(domains), len(domains)), np.nan)
    collected: dict[tuple[int, int], list[float]] = defaultdict(list)

    if registry is None or len(registry) == 0 or metric not in registry.columns:
        return matrix, domains

    completed = registry[registry.get("status", "COMPLETE") == "COMPLETE"]
    if method is not None and "method" in completed.columns:
        normalised = completed["method"].astype(str).str.split("-b").str[0]
        completed = completed[normalised == method]
    if seed is not None and "seed" in completed.columns:
        completed = completed[completed["seed"].astype("Int64") == seed]

    for _, row in completed.iterrows():
        target = row.get("target_domain")
        if target not in domains:
            continue
        sources = row.get("source_domains")
        if isinstance(sources, str):
            try:
                import json

                sources = json.loads(sources)
            except (ValueError, TypeError):
                sources = [s.strip() for s in sources.strip("[]").replace("'", "").split(",")]
        if not isinstance(sources, (list, tuple)):
            continue

        value = row.get(metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        # A LODO run trains on several domains at once; it contributes to the
        # row of every source it used, which is the honest reading of "a model
        # trained including domain X, evaluated on Y".
        for source in sources:
            if source in domains:
                collected[(domains.index(source), domains.index(target))].append(float(value))

    multiply_filled = 0
    for (i, j), values in collected.items():
        if len(values) > 1:
            multiply_filled += 1
        matrix[i, j] = float(np.mean(values) if aggregate == "mean" else np.median(values))

    if multiply_filled:
        log.warning(
            "%d matrix cell(s) aggregate more than one run (%s over %s). Pass "
            "method=... to build a single-method matrix; an aggregated cell is not "
            "any one experiment's result.",
            multiply_filled, aggregate,
            {f"{domains[i]}->{domains[j]}": len(v) for (i, j), v in collected.items() if len(v) > 1},
        )
    return matrix, domains


def _draw_matrix(axis, matrix, domains, *, metric, cmap, fmt="{:.3f}"):
    import matplotlib.pyplot as plt

    lower_better = any(key in metric for key in LOWER_IS_BETTER)
    display = np.ma.masked_invalid(matrix)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(NOT_RUN_COLOR)

    image = axis.imshow(display, cmap=colormap, aspect="auto")
    axis.set_xticks(range(len(domains)))
    axis.set_yticks(range(len(domains)))
    axis.set_xticklabels([DOMAIN_LABELS.get(d, d) for d in domains], rotation=20, ha="right")
    axis.set_yticklabels([DOMAIN_LABELS.get(d, d) for d in domains])
    axis.set_xlabel("Tested on")
    axis.set_ylabel("Trained including")
    axis.grid(False)

    finite = matrix[np.isfinite(matrix)]
    threshold = (finite.max() + finite.min()) / 2 if finite.size else 0.0
    for i in range(len(domains)):
        for j in range(len(domains)):
            value = matrix[i, j]
            if not np.isfinite(value):
                axis.text(j, i, "NOT\nRUN", ha="center", va="center",
                          fontsize=7, color="#666666", style="italic")
                # Hatching so the cell is distinguishable in greyscale print too.
                axis.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                    edgecolor="#AAAAAA", linewidth=0.0,
                ))
                continue
            bright = value > threshold if not lower_better else value < threshold
            axis.text(j, i, fmt.format(value), ha="center", va="center",
                      fontsize=8.5, color="white" if bright else "black")
    return image


def figure_cross_domain_matrix(
    registry: Any,
    *,
    metric: str = "test_qwk",
    domains: Sequence[str] = tuple(DOMAIN_ORDER),
    method: str | None = None,
    seed: int | None = None,
    title: str | None = None,
) -> Any:
    """One train-domain x test-domain heatmap."""
    import matplotlib.pyplot as plt

    matrix, domains = build_cross_domain_matrix(
        registry, metric=metric, domains=domains, method=method, seed=seed
    )
    lower_better = any(key in metric for key in LOWER_IS_BETTER)

    figure, axis = plt.subplots(figsize=(6.6, 5.0))
    image = _draw_matrix(
        axis, matrix, domains, metric=metric,
        cmap="YlOrRd" if lower_better else "YlGnBu",
    )
    figure.colorbar(
        image, ax=axis,
        label=f"{metric.replace('test_', '').upper()} "
              f"({'lower' if lower_better else 'higher'} is better)",
    )
    axis.set_title(
        title or f"Cross-domain {metric.replace('test_', '').upper()}"
        + (f" -- {method}" if method else " -- ALL METHODS POOLED")
        + (f" (seed {seed})" if seed is not None else "")
    )
    figure.tight_layout()
    return figure


def figure_metric_heatmap_grid(
    registry: Any,
    *,
    metrics: Sequence[str] = ("test_qwk", "test_f1_macro", "test_ece", "test_nll"),
    domains: Sequence[str] = tuple(DOMAIN_ORDER),
    method: str | None = None,
    seed: int | None = None,
) -> Any:
    """Several cross-domain matrices side by side.

    Placing discrimination and calibration next to each other is the point: they
    do not degrade together, and a single-metric heatmap hides that.
    """
    import matplotlib.pyplot as plt

    columns = min(len(metrics), 2)
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6.4 * columns, 5.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, metric in zip(axes, metrics):
        matrix, resolved = build_cross_domain_matrix(
            registry, metric=metric, domains=domains, method=method, seed=seed
        )
        lower_better = any(key in metric for key in LOWER_IS_BETTER)
        image = _draw_matrix(
            axis, matrix, resolved, metric=metric,
            cmap="YlOrRd" if lower_better else "YlGnBu",
        )
        axis.set_title(
            f"{metric.replace('test_', '').upper()} "
            f"({'lower' if lower_better else 'higher'} better)", fontsize=10.5
        )
        figure.colorbar(image, ax=axis, fraction=0.046)

    for axis in axes[len(metrics):]:
        axis.set_visible(False)

    figure.suptitle(
        f"Cross-domain result matrices{f' -- {method}' if method else ' -- ALL METHODS POOLED'}. "
        "Hatched cells are experiments that have NOT been run -- not zero, not a poor result.",
        fontsize=11,
    )
    figure.tight_layout()
    return figure


def figure_lodo_summary(rows: Sequence[dict[str, Any]], *, metric: str = "qwk") -> Any:
    """Per-target source-validation vs unseen-target score, for the four LODO runs."""
    import matplotlib.pyplot as plt

    if not rows:
        figure, axis = plt.subplots(figsize=(6, 2.2))
        axis.text(0.5, 0.5, "NOT RUN -- no leave-one-domain-out results yet",
                  ha="center", va="center", fontsize=11, style="italic")
        axis.set_axis_off()
        return figure

    labels = [DOMAIN_LABELS.get(r["target"], r["target"]) for r in rows]
    source = [r.get(f"source_{metric}", np.nan) for r in rows]
    target = [r.get(f"target_{metric}", np.nan) for r in rows]

    figure, axis = plt.subplots(figsize=(max(6.5, 1.7 * len(rows) + 3), 4.4))
    positions = np.arange(len(rows))
    width = 0.36

    bars_a = axis.bar(positions - width / 2, source, width, label="source validation",
                      color="#0072B2", edgecolor="black", linewidth=0.5)
    bars_b = axis.bar(positions + width / 2, target, width, label="unseen target",
                      color="#D55E00", edgecolor="black", linewidth=0.5)

    for bar_a, bar_b, a, b in zip(bars_a, bars_b, source, target):
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        for bar, value in ((bar_a, a), (bar_b, b)):
            axis.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          ha="center", va="bottom", fontsize=8,
                          xytext=(0, 2), textcoords="offset points")
        axis.annotate(
            f"Δ {b - a:+.3f}",
            ((bar_a.get_x() + bar_b.get_x() + bar_b.get_width()) / 2, max(a, b)),
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color="#B22222" if b < a else "#227722",
            xytext=(0, 15), textcoords="offset points",
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([f"hold out\n{label}" for label in labels])
    axis.set_ylabel(metric.upper())
    axis.set_title(
        f"Leave-one-domain-out {metric.upper()}: source validation vs unseen target"
    )
    axis.legend()
    finite = [v for v in source + target if np.isfinite(v)]
    axis.set_ylim(0, max(finite) * 1.3 if finite else 1.0)
    figure.tight_layout()
    return figure


def generate_domain_figures(
    registry: Any,
    figures_dir: Path | str,
    *,
    lodo_rows: Sequence[dict[str, Any]] = (),
    method: str | None = "erm",
    seed: int | None = None,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Write the cross-domain matrices and the LODO summary.

    ``method`` defaults to "erm" so the paper figure shows one method rather than
    a pooled average across methods and seeds.
    """
    apply_style()
    written: list[Path] = []
    suffix = f"_{method}" if method else "_pooled"
    if seed is not None:
        suffix += f"_s{seed}"

    written.extend(save_figure(
        figure_metric_heatmap_grid(registry, method=method, seed=seed),
        f"domain_matrix_grid{suffix}", figures_dir, formats=formats,
    ))
    for metric in ("test_qwk", "test_ece"):
        written.extend(save_figure(
            figure_cross_domain_matrix(registry, metric=metric, method=method, seed=seed),
            f"domain_matrix_{metric.replace('test_', '')}{suffix}", figures_dir,
            formats=formats,
        ))
    written.extend(save_figure(
        figure_lodo_summary(list(lodo_rows)), "lodo_summary_qwk", figures_dir,
        formats=formats,
    ))
    log.info("generated %d cross-domain figures", len(written))
    return written
