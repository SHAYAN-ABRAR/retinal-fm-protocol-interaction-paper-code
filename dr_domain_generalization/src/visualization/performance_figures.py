"""Classification-performance and calibration figures for a fitted model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.schema import N_GRADES
from ..utils.logging import get_logger
from .style import GRADE_LABELS, apply_style, grade_color, save_figure

log = get_logger("viz.performance")

__all__ = [
    "figure_confusion_matrices",
    "figure_per_class_metrics",
    "figure_roc_curves",
    "figure_reliability_diagram",
    "figure_confidence_histogram",
    "figure_indomain_vs_external",
    "generate_performance_figures",
]


def figure_confusion_matrices(metrics: dict[str, Any], *, title_suffix: str = "") -> Any:
    """Raw counts and row-normalised confusion matrices, side by side.

    Row normalisation (recall per true grade) is the informative view under
    severe imbalance: the raw matrix is dominated by grade 0 and hides how the
    rare grades are handled.
    """
    import matplotlib.pyplot as plt

    counts = np.asarray(metrics["confusion_matrix"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = counts / counts.sum(axis=1, keepdims=True)
    normalised = np.nan_to_num(normalised)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for axis, matrix, label, fmt in (
        (axes[0], counts, "Counts", "{:.0f}"),
        (axes[1], normalised, "Row-normalised (recall)", "{:.2f}"),
    ):
        image = axis.imshow(matrix, cmap="Blues", aspect="auto")
        axis.set_xticks(range(N_GRADES))
        axis.set_yticks(range(N_GRADES))
        axis.set_xticklabels([GRADE_LABELS[g] for g in range(N_GRADES)], rotation=35, ha="right")
        axis.set_yticklabels([GRADE_LABELS[g] for g in range(N_GRADES)])
        axis.set_xlabel("Predicted grade")
        axis.set_ylabel("True grade")
        axis.set_title(label)
        axis.grid(False)
        threshold = matrix.max() * 0.55 if matrix.max() else 1
        for i in range(N_GRADES):
            for j in range(N_GRADES):
                axis.text(
                    j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=8.5,
                    color="white" if matrix[i, j] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046)

    figure.suptitle(f"Confusion matrix{title_suffix}", fontsize=12)
    figure.tight_layout()
    return figure


def figure_per_class_metrics(metrics: dict[str, Any], *, title_suffix: str = "") -> Any:
    """Precision / recall / specificity / F1 per grade, with support annotated."""
    import matplotlib.pyplot as plt

    rows = metrics["per_class"]
    names = ["precision", "recall", "specificity", "f1"]
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    figure, axis = plt.subplots(figsize=(8.2, 4.0))
    width = 0.8 / len(names)
    positions = np.arange(N_GRADES)
    for i, (name, colour) in enumerate(zip(names, colours)):
        values = [row[name] if row[name] == row[name] else 0.0 for row in rows]
        axis.bar(positions + i * width - 0.4 + width / 2, values, width,
                 label=name.capitalize(), color=colour, edgecolor="black", linewidth=0.4)

    for grade, row in enumerate(rows):
        axis.annotate(f"n={row['support']}", (grade, 1.02), ha="center", fontsize=8, color="grey")

    axis.set_xticks(positions)
    axis.set_xticklabels([GRADE_LABELS[g] for g in range(N_GRADES)])
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Score")
    axis.set_xlabel("DR severity grade")
    axis.set_title(f"Per-class performance{title_suffix}")
    axis.legend(ncol=4, loc="lower center")
    figure.tight_layout()
    return figure


def figure_roc_curves(
    y_true: np.ndarray, probabilities: np.ndarray, *, title_suffix: str = ""
) -> Any:
    """One-vs-rest ROC per grade, plus the macro average.

    Grades absent from ``y_true`` are skipped and named in the legend rather
    than drawn as a meaningless diagonal.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    figure, axis = plt.subplots(figsize=(5.8, 5.2))
    y_true = np.asarray(y_true).astype(int)

    all_fpr = np.linspace(0, 1, 200)
    interpolated: list[np.ndarray] = []
    skipped: list[int] = []

    for grade in range(N_GRADES):
        positive = (y_true == grade).astype(int)
        if positive.sum() == 0 or positive.sum() == len(positive):
            skipped.append(grade)
            continue
        fpr, tpr, _ = roc_curve(positive, probabilities[:, grade])
        score = auc(fpr, tpr)
        axis.plot(fpr, tpr, color=grade_color(grade), linewidth=1.6,
                  label=f"{GRADE_LABELS[grade]} (AUC {score:.3f})")
        interpolated.append(np.interp(all_fpr, fpr, tpr))

    if interpolated:
        macro = np.mean(interpolated, axis=0)
        axis.plot(all_fpr, macro, color="black", linewidth=2.2, linestyle="--",
                  label=f"Macro average (AUC {auc(all_fpr, macro):.3f})")

    axis.plot([0, 1], [0, 1], color="grey", linewidth=0.9, linestyle=":")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    suffix = f"  [grades absent from this split: {skipped}]" if skipped else ""
    axis.set_title(f"One-vs-rest ROC{title_suffix}{suffix}", fontsize=10)
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    return figure


def figure_reliability_diagram(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    scaled_probabilities: np.ndarray | None = None,
    n_bins: int = 15,
    title_suffix: str = "",
) -> Any:
    """Reliability diagram, optionally comparing before/after temperature scaling.

    Bars below the diagonal mean over-confidence -- the model claims more
    certainty than its accuracy justifies -- which is the failure mode that
    matters clinically when deploying on an unseen domain.
    """
    import matplotlib.pyplot as plt

    from ..evaluation.calibration import calibration_metrics, reliability_curve

    figure, axes = plt.subplots(
        1, 2 if scaled_probabilities is not None else 1,
        figsize=(9.8 if scaled_probabilities is not None else 5.2, 4.6),
        squeeze=False,
    )
    panels = [(axes[0][0], probabilities, "Uncalibrated")]
    if scaled_probabilities is not None:
        panels.append((axes[0][1], scaled_probabilities, "After temperature scaling"))

    for axis, probs, label in panels:
        curve = reliability_curve(probs, y_true, n_bins=n_bins)
        metrics = calibration_metrics(probs, y_true, n_bins=n_bins)
        centres = np.linspace(0, 1, n_bins + 1)
        centres = (centres[:-1] + centres[1:]) / 2

        axis.bar(centres, np.nan_to_num(curve["accuracy"]), width=1.0 / n_bins * 0.92,
                 color="#0072B2", edgecolor="black", linewidth=0.5, label="accuracy")
        gap = np.nan_to_num(curve["confidence"]) - np.nan_to_num(curve["accuracy"])
        axis.bar(centres, gap, width=1.0 / n_bins * 0.92,
                 bottom=np.nan_to_num(curve["accuracy"]),
                 color="#D55E00", alpha=0.4, edgecolor="black", linewidth=0.4,
                 label="gap (confidence - accuracy)")
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.1,
                  label="perfect calibration")

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Confidence")
        axis.set_ylabel("Accuracy")
        axis.set_title(
            f"{label}\nECE {metrics['ece']:.4f} | adaptive {metrics['adaptive_ece']:.4f} | "
            f"NLL {metrics['nll']:.4f}",
            fontsize=9.5,
        )
        axis.legend(fontsize=7.5, loc="upper left")

    figure.suptitle(f"Reliability diagram{title_suffix}", fontsize=12)
    figure.tight_layout()
    return figure


def figure_confidence_histogram(
    probabilities: np.ndarray, y_true: np.ndarray, *, title_suffix: str = ""
) -> Any:
    """Confidence distribution, split by whether the prediction was correct.

    Well-separated distributions mean confidence is usable for abstention; heavy
    overlap means it is not, whatever the ECE says.
    """
    import matplotlib.pyplot as plt

    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(y_true).astype(int)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    axes[0].hist(confidence, bins=30, color="#0072B2", edgecolor="black", linewidth=0.4)
    axes[0].axvline(confidence.mean(), color="red", linestyle="--", linewidth=1.2,
                    label=f"mean {confidence.mean():.3f}")
    axes[0].axvline(correct.mean(), color="green", linestyle=":", linewidth=1.4,
                    label=f"accuracy {correct.mean():.3f}")
    axes[0].set_xlabel("Predicted confidence")
    axes[0].set_ylabel("Images")
    axes[0].set_title("Confidence distribution")
    axes[0].legend(fontsize=8)

    bins = np.linspace(min(confidence.min(), 0.2), 1.0, 30)
    if correct.any():
        axes[1].hist(confidence[correct], bins=bins, alpha=0.6, density=True,
                     color="#009E73", label=f"correct (n={int(correct.sum())})")
    if (~correct).any():
        axes[1].hist(confidence[~correct], bins=bins, alpha=0.6, density=True,
                     color="#D55E00", label=f"incorrect (n={int((~correct).sum())})")
    axes[1].set_xlabel("Predicted confidence")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Confidence: correct vs incorrect predictions")
    axes[1].legend(fontsize=8)

    figure.suptitle(f"Confidence analysis{title_suffix}", fontsize=12)
    figure.tight_layout()
    return figure


def figure_indomain_vs_external(rows: Sequence[dict[str, Any]], *, metric: str = "qwk") -> Any:
    """Paired bar chart of a metric on the source validation vs the unseen target.

    The central figure of the paper's first claim: the size of the drop between
    the two bars *is* the domain-generalization gap.
    """
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    source = [r[f"source_{metric}"] for r in rows]
    target = [r[f"target_{metric}"] for r in rows]

    figure, axis = plt.subplots(figsize=(max(6.0, 1.6 * len(rows) + 3), 4.2))
    positions = np.arange(len(rows))
    width = 0.36
    bars_a = axis.bar(positions - width / 2, source, width, label="source validation",
                      color="#0072B2", edgecolor="black", linewidth=0.5)
    bars_b = axis.bar(positions + width / 2, target, width, label="unseen target domain",
                      color="#D55E00", edgecolor="black", linewidth=0.5)

    for bar_a, bar_b, a, b in zip(bars_a, bars_b, source, target):
        for bar, value in ((bar_a, a), (bar_b, b)):
            axis.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          ha="center", va="bottom", fontsize=8,
                          xytext=(0, 2), textcoords="offset points")
        axis.annotate(
            f"Δ {b - a:+.3f}",
            ((bar_a.get_x() + bar_b.get_x() + bar_b.get_width()) / 2, max(a, b)),
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color="#B22222" if b < a else "#227722",
            xytext=(0, 16), textcoords="offset points",
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=12, ha="right")
    axis.set_ylabel(metric.upper().replace("_", " "))
    axis.set_title(f"In-domain vs unseen-domain {metric.upper()} (domain-generalization gap)")
    axis.legend()
    axis.set_ylim(0, max(max(source + target) * 1.28, 0.1))
    figure.tight_layout()
    return figure


def generate_performance_figures(
    evaluation: dict[str, Any],
    outputs_by_split: dict[str, dict[str, np.ndarray]],
    figures_dir: Path | str,
    *,
    prefix: str = "performance",
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Write the performance and calibration figures for one evaluated model."""
    apply_style()
    written: list[Path] = []

    for split_name, result in evaluation["results"].items():
        outputs = outputs_by_split.get(split_name)
        if outputs is None:
            continue
        suffix = f" -- {split_name}"
        tag = split_name.replace("[", "_").replace("]", "").replace("/", "-")

        written.extend(save_figure(
            figure_confusion_matrices(result.metrics, title_suffix=suffix),
            f"{prefix}_{tag}_confusion", figures_dir, formats=formats))
        written.extend(save_figure(
            figure_per_class_metrics(result.metrics, title_suffix=suffix),
            f"{prefix}_{tag}_per_class", figures_dir, formats=formats))
        written.extend(save_figure(
            figure_roc_curves(outputs["y_true"], outputs["probabilities"], title_suffix=suffix),
            f"{prefix}_{tag}_roc", figures_dir, formats=formats))
        written.extend(save_figure(
            figure_reliability_diagram(
                outputs["probabilities"], outputs["y_true"],
                scaled_probabilities=outputs.get("probabilities_scaled"),
                title_suffix=suffix),
            f"{prefix}_{tag}_reliability", figures_dir, formats=formats))
        written.extend(save_figure(
            figure_confidence_histogram(
                outputs["probabilities"], outputs["y_true"], title_suffix=suffix),
            f"{prefix}_{tag}_confidence", figures_dir, formats=formats))

    return written
