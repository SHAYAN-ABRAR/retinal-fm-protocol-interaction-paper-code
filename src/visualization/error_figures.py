"""Error-analysis galleries built from the sample-level prediction tables.

These are the figures that support the discussion section: not "the model scored
0.68" but *which* images it got wrong and how confidently.

Four galleries, chosen because each answers a different question:

``high_confidence_errors``
    The dangerous failures. A wrong prediction made at 0.95 confidence is worse
    than one made at 0.4, because no abstention rule would catch it.
``low_confidence_correct``
    The wasted deferrals. These are cases selective prediction would reject even
    though the model was right -- the cost side of the coverage trade-off.
``largest_ordinal_errors``
    Sorted by |true - predicted|. A 0-vs-4 confusion is a different clinical
    event from a 2-vs-3 one, and QWK weights them 16:1.
``confusion_cell``
    Every image in one cell of the confusion matrix, e.g. true 4 predicted 0.

An explicit caveat
------------------
These galleries show *what* the model got wrong, never *why*. Reading a cause
into a handful of hand-picked images is exactly the failure mode that makes
qualitative sections unreliable, so the figures are labelled with the facts
(true grade, prediction, confidence) and nothing is asserted about mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.schema import GRADE_NAMES
from ..utils.logging import get_logger
from .style import apply_style, save_figure

log = get_logger("viz.errors")

__all__ = [
    "figure_error_gallery",
    "figure_confusion_cell",
    "figure_error_summary",
    "generate_error_figures",
]


def _load_image(path: str, size: int = 224) -> np.ndarray | None:
    from PIL import Image

    try:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            if max(image.size) != size:
                image = image.resize((size, size), Image.LANCZOS)
            return np.asarray(image)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s: %s", path, exc)
        return None


def _gallery(
    rows: Any,
    *,
    title: str,
    n_cols: int = 6,
    n_rows: int = 2,
    image_size: int = 224,
) -> Any:
    """Render up to ``n_cols * n_rows`` prediction rows as a labelled grid."""
    import matplotlib.pyplot as plt

    rows = rows.head(n_cols * n_rows)
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(1.85 * n_cols, 2.15 * n_rows))
    axes = np.atleast_2d(axes)

    for index, axis in enumerate(axes.ravel()):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        if index >= len(rows):
            axis.set_visible(False)
            continue

        row = rows.iloc[index]
        image = _load_image(str(row["path"]), image_size)
        if image is None:
            axis.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=8)
            continue
        axis.imshow(image)

        true_grade, predicted = int(row["true_grade"]), int(row["predicted_grade"])
        correct = true_grade == predicted
        axis.set_title(
            f"true {true_grade} -> pred {predicted}\n"
            f"conf {row['confidence']:.2f} | {row['dataset']}",
            fontsize=7.5,
            color="#1a7f37" if correct else "#B22222",
        )
        for spine in axis.spines.values():
            spine.set_edgecolor("#1a7f37" if correct else "#B22222")
            spine.set_linewidth(2.0)

    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    return figure


def figure_error_gallery(
    predictions: Any,
    kind: str,
    *,
    n_cols: int = 6,
    n_rows: int = 2,
    image_size: int = 224,
) -> Any:
    """One of the four standard galleries.

    ``kind`` is one of ``high_confidence_errors``, ``low_confidence_correct``,
    ``largest_ordinal_errors``, ``high_confidence_correct``.
    """
    frame = predictions.copy()
    frame["correct"] = frame["true_grade"] == frame["predicted_grade"]
    frame["absolute_grade_error"] = (frame["true_grade"] - frame["predicted_grade"]).abs()

    if kind == "high_confidence_errors":
        selected = frame[~frame["correct"]].sort_values("confidence", ascending=False)
        title = ("Highest-confidence MISTAKES -- the failures abstention would not catch")
    elif kind == "low_confidence_correct":
        selected = frame[frame["correct"]].sort_values("confidence")
        title = "Lowest-confidence CORRECT predictions -- the cost of deferring"
    elif kind == "largest_ordinal_errors":
        selected = frame.sort_values(
            ["absolute_grade_error", "confidence"], ascending=[False, False]
        )
        title = "Largest ordinal errors (|true - predicted|), most confident first"
    elif kind == "high_confidence_correct":
        selected = frame[frame["correct"]].sort_values("confidence", ascending=False)
        title = "Highest-confidence correct predictions"
    else:
        raise ValueError(f"unknown gallery kind {kind!r}")

    if selected.empty:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 2))
        axis.text(0.5, 0.5, f"no samples match {kind!r}", ha="center", va="center")
        axis.set_axis_off()
        return figure

    return _gallery(
        selected, title=title, n_cols=n_cols, n_rows=n_rows, image_size=image_size
    )


def figure_confusion_cell(
    predictions: Any,
    true_grade: int,
    predicted_grade: int,
    *,
    n_cols: int = 6,
    n_rows: int = 1,
    image_size: int = 224,
) -> Any:
    """Every image in one confusion-matrix cell, most confident first."""
    frame = predictions[
        (predictions["true_grade"] == true_grade)
        & (predictions["predicted_grade"] == predicted_grade)
    ].sort_values("confidence", ascending=False)

    label = (
        f"True {true_grade} ({GRADE_NAMES[true_grade]}) predicted as "
        f"{predicted_grade} ({GRADE_NAMES[predicted_grade]}) -- n={len(frame)}"
    )
    if frame.empty:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 1.8))
        axis.text(0.5, 0.5, f"{label}\n(no such errors)", ha="center", va="center", fontsize=9)
        axis.set_axis_off()
        return figure

    return _gallery(frame, title=label, n_cols=n_cols, n_rows=n_rows, image_size=image_size)


def figure_error_summary(predictions: Any, *, title_suffix: str = "") -> Any:
    """How errors distribute over grade distance, true grade and confidence."""
    import matplotlib.pyplot as plt

    from .style import grade_color

    frame = predictions.copy()
    frame["correct"] = frame["true_grade"] == frame["predicted_grade"]
    frame["absolute_grade_error"] = (frame["true_grade"] - frame["predicted_grade"]).abs()

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))

    distances = frame["absolute_grade_error"].value_counts().sort_index()
    bars = axes[0].bar(distances.index, distances.values, color="#0072B2",
                       edgecolor="black", linewidth=0.5)
    for bar, value in zip(bars, distances.values):
        axes[0].annotate(f"{100 * value / len(frame):.1f}%",
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=8,
                         xytext=(0, 2), textcoords="offset points")
    axes[0].set_xlabel("|true grade - predicted grade|")
    axes[0].set_ylabel("Images")
    axes[0].set_title("Ordinal error distance")
    axes[0].set_xticks(sorted(distances.index))

    by_grade = frame.groupby("true_grade")["correct"].agg(["mean", "size"])
    bars = axes[1].bar(by_grade.index, by_grade["mean"],
                       color=[grade_color(g) for g in by_grade.index],
                       edgecolor="black", linewidth=0.5)
    for bar, (_, row) in zip(bars, by_grade.iterrows()):
        axes[1].annotate(f"n={int(row['size'])}",
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=8,
                         xytext=(0, 2), textcoords="offset points")
    axes[1].set_xlabel("True grade")
    axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Recall by true grade")
    axes[1].set_xticks(sorted(by_grade.index))

    errors = frame[~frame["correct"]]
    if len(errors):
        severe = errors[errors["absolute_grade_error"] >= 2]
        axes[2].hist(errors["confidence"], bins=20, alpha=0.65, color="#D55E00",
                     label=f"all errors (n={len(errors)})")
        if len(severe):
            axes[2].hist(severe["confidence"], bins=20, alpha=0.8, color="#7B1010",
                         label=f"severe (>=2 grades, n={len(severe)})")
    axes[2].set_xlabel("Confidence")
    axes[2].set_ylabel("Errors")
    axes[2].set_title("Confidence of incorrect predictions")
    axes[2].legend(fontsize=8)

    figure.suptitle(f"Error analysis{title_suffix}", fontsize=12)
    figure.tight_layout()
    return figure


def generate_error_figures(
    predictions: Any,
    figures_dir: Path | str,
    *,
    prefix: str = "errors",
    title_suffix: str = "",
    confusion_cells: Sequence[tuple[int, int]] = ((4, 0), (0, 4), (3, 0), (2, 0)),
    image_size: int = 224,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Write the full error-analysis set for one prediction table."""
    apply_style()
    written: list[Path] = []

    written.extend(save_figure(
        figure_error_summary(predictions, title_suffix=title_suffix),
        f"{prefix}_summary", figures_dir, formats=formats,
    ))

    for kind in (
        "high_confidence_errors",
        "low_confidence_correct",
        "largest_ordinal_errors",
    ):
        written.extend(save_figure(
            figure_error_gallery(predictions, kind, image_size=image_size),
            f"{prefix}_{kind}", figures_dir, formats=formats,
        ))

    for true_grade, predicted_grade in confusion_cells:
        written.extend(save_figure(
            figure_confusion_cell(
                predictions, true_grade, predicted_grade, image_size=image_size
            ),
            f"{prefix}_cell_true{true_grade}_pred{predicted_grade}",
            figures_dir, formats=formats,
        ))

    log.info("generated %d error-analysis figures", len(written))
    return written
