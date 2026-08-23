"""Training-curve figures, built from the epoch history table."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..utils.logging import get_logger
from .style import apply_style, save_figure

log = get_logger("viz.training")

__all__ = [
    "figure_loss_curves",
    "figure_metric_curves",
    "figure_learning_rate",
    "figure_training_diagnostics",
    "generate_training_figures",
]


def figure_loss_curves(history: Any) -> Any:
    """Train and validation loss per epoch, with the best epoch marked."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 3.9))
    axis.plot(history["epoch"], history["train_loss"], marker="o", markersize=3,
              label="train", color="#0072B2")
    axis.plot(history["epoch"], history["val_loss"], marker="s", markersize=3,
              label="validation", color="#D55E00")

    if len(history):
        best = int(np.nanargmin(history["val_loss"].to_numpy()))
        axis.axvline(history["epoch"].iloc[best], color="grey", linestyle="--", linewidth=1)
        axis.annotate(
            f"best val loss\nepoch {history['epoch'].iloc[best]}",
            (history["epoch"].iloc[best], history["val_loss"].iloc[best]),
            textcoords="offset points", xytext=(8, 12), fontsize=8, color="grey",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Training and validation loss")
    axis.legend()
    figure.tight_layout()
    return figure


def figure_metric_curves(history: Any) -> Any:
    """Validation QWK, macro F1 and accuracy per epoch.

    QWK gets its own axis because it lives on a different scale from the two
    proportions and would otherwise be visually compressed.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    axes[0].plot(history["epoch"], history["val_qwk"], marker="o", markersize=3,
                 color="#009E73", label="validation QWK")
    if len(history):
        best = int(np.nanargmax(history["val_qwk"].to_numpy()))
        axes[0].scatter([history["epoch"].iloc[best]], [history["val_qwk"].iloc[best]],
                        s=70, facecolors="none", edgecolors="black", zorder=5)
        axes[0].annotate(
            f"best {history['val_qwk'].iloc[best]:.4f}\nepoch {history['epoch'].iloc[best]}",
            (history["epoch"].iloc[best], history["val_qwk"].iloc[best]),
            textcoords="offset points", xytext=(6, -22), fontsize=8,
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Quadratic weighted kappa")
    axes[0].set_title("Validation QWK (model-selection metric)")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_f1_macro"], marker="s", markersize=3,
                 color="#CC79A7", label="macro F1")
    axes[1].plot(history["epoch"], history["val_accuracy"], marker="^", markersize=3,
                 color="#0072B2", label="accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation macro F1 and accuracy")
    axes[1].legend()

    figure.suptitle("Validation metrics per epoch (source domains only)", fontsize=12)
    figure.tight_layout()
    return figure


def figure_learning_rate(history: Any) -> Any:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.0, 3.4))
    axis.plot(history["epoch"], history["learning_rate"], marker="o", markersize=3,
              color="#666666")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Learning rate")
    axis.set_yscale("log")
    axis.set_title("Learning-rate schedule (cosine with warmup)")
    figure.tight_layout()
    return figure


def figure_training_diagnostics(history: Any) -> Any:
    """Gradient norm, epoch time and peak VRAM -- the health checks.

    Worth plotting: a gradient norm that collapses to zero or explodes shows a
    problem the loss curve can hide, and the VRAM trace confirms the run stayed
    inside the 8 GB budget rather than nearly OOM-ing.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    axes[0].plot(history["epoch"], history["grad_norm"], marker="o", markersize=3,
                 color="#E69F00")
    axes[0].set_ylabel("Gradient L2 norm (pre-clip)")
    axes[0].set_title("Gradient norm")

    axes[1].plot(history["epoch"], history["seconds"], marker="s", markersize=3,
                 color="#0072B2")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Epoch wall-clock time")
    axes[1].set_ylim(bottom=0)

    axes[2].plot(history["epoch"], history["peak_vram_gb"], marker="^", markersize=3,
                 color="#D55E00")
    axes[2].axhline(8.0, color="red", linestyle="--", linewidth=1)
    axes[2].annotate("8 GB card limit", (0.02, 8.0), xycoords=("axes fraction", "data"),
                     fontsize=8, color="red", va="bottom")
    axes[2].set_ylabel("Peak allocated (GB)")
    axes[2].set_title("Peak GPU memory")
    axes[2].set_ylim(0, 8.6)

    for axis in axes:
        axis.set_xlabel("Epoch")
    figure.suptitle("Training diagnostics", fontsize=12)
    figure.tight_layout()
    return figure


def generate_training_figures(
    history: Any,
    figures_dir: Path | str,
    *,
    prefix: str = "training",
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Write every training figure for one run."""
    apply_style()
    written: list[Path] = []
    for name, figure in (
        ("loss_curves", figure_loss_curves(history)),
        ("metric_curves", figure_metric_curves(history)),
        ("learning_rate", figure_learning_rate(history)),
        ("diagnostics", figure_training_diagnostics(history)),
    ):
        written.extend(save_figure(figure, f"{prefix}_{name}", figures_dir, formats=formats))
    return written
