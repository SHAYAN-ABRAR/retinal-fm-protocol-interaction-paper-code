"""Cross-domain calibration figures -- the paper's central claim in two panels.

The project's argument is that **discrimination and calibration do not degrade
together**. A model can keep almost all of its ordinal agreement on an unseen
domain while its confidence becomes badly wrong, and a study that reports only
QWK or accuracy will not notice. These figures are built to make that visible
in one look, across all four held-out domains at once.

Recovering the temperature-scaled probabilities
-----------------------------------------------
The saved prediction files hold post-softmax probabilities, not logits, so
scaled probabilities are reconstructed as ``softmax(log(p) / T)``. This is
exact, not an approximation: ``log(p) = z - logsumexp(z)`` and the constant
shift cancels inside the softmax, so ``softmax(log(p)/T) == softmax(z/T)``.
Verified against the stored ``calibration_scaled.ece`` for all four targets to
within 8e-6 (float32 rounding in the CSVs).

The temperature is always the one fitted on **source validation**. It never
sees the target domain -- that is the whole point of the protocol, and the
figure captions say so.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .style import apply_style, domain_label, save_figure

log = logging.getLogger(__name__)

__all__ = [
    "scaled_probabilities",
    "load_target_predictions",
    "figure_reliability_grid",
    "figure_discrimination_vs_calibration",
    "generate_calibration_figures",
]

DOMAIN_ORDER = ["ddr", "aptos", "idrid", "eyepacs"]


def scaled_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a fitted temperature to saved softmax probabilities.

    Exact reconstruction -- see the module docstring. ``temperature`` must be
    positive; a non-positive value would silently invert or flatten the
    distribution rather than scale it.
    """
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    logits = np.log(np.clip(probabilities, 1e-12, None)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def load_target_predictions(
    experiment_id: str, target: str, *, outputs_dir: Path | str
) -> dict[str, Any] | None:
    """Load one run's target predictions, temperature and both ECEs.

    Returns ``None`` when the run has not happened, so callers can render a
    NOT RUN panel rather than dropping the domain from the figure.
    """
    import pandas as pd

    outputs = Path(outputs_dir)
    predictions = outputs / "predictions" / f"{experiment_id}__target_test[{target}]_predictions.csv"
    report = outputs / "reports" / f"{experiment_id}_evaluation.json"
    if not predictions.exists() or not report.exists():
        return None

    frame = pd.read_csv(predictions)
    payload = json.loads(report.read_text(encoding="utf-8"))
    columns = sorted(
        (c for c in frame.columns if c.startswith("probability_grade_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    probabilities = frame[columns].to_numpy(dtype=np.float64)
    temperature = float(payload["temperature"]["temperature"])
    result = payload["results"]["target_test"]

    return {
        "target": target,
        "experiment_id": experiment_id,
        "y_true": frame["true_grade"].to_numpy(),
        "y_pred": frame["predicted_grade"].to_numpy(),
        "probabilities": probabilities,
        "scaled": scaled_probabilities(probabilities, temperature),
        "temperature": temperature,
        "fitted_on": payload["temperature"].get("fitted_on", "unknown"),
        "qwk": result["metrics"]["qwk"],
        "ece": result["calibration"]["ece"],
        "ece_scaled": result["calibration_scaled"]["ece"],
        "n": len(frame),
    }


def _reliability_panel(axis, probabilities, y_true, *, n_bins, title, show_legend):
    from ..evaluation.calibration import calibration_metrics, reliability_curve

    curve = reliability_curve(probabilities, y_true, n_bins=n_bins)
    metrics = calibration_metrics(probabilities, y_true, n_bins=n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    width = 0.92 / n_bins

    accuracy = np.nan_to_num(curve["accuracy"])
    confidence = np.nan_to_num(curve["confidence"])
    gap = confidence - accuracy

    axis.bar(centres, accuracy, width=width, color="#0072B2",
             edgecolor="black", linewidth=0.4, label="accuracy", zorder=3)

    # The gap is drawn signed, and the two signs get different colours. Over-
    # confidence (the clinically dangerous direction: the model claims more
    # certainty than its accuracy earns) sits above the bar in orange.
    # Under-confidence sits below it in purple. Plotting |gap| would hide which
    # way the model is wrong, and drawing both in one colour makes the negative
    # bars overlap the accuracy bars into an unreadable blend.
    over = np.where(gap > 0, gap, 0.0)
    under = np.where(gap < 0, gap, 0.0)
    axis.bar(centres, over, width=width, bottom=accuracy, color="#D55E00",
             alpha=0.55, edgecolor="black", linewidth=0.3,
             label="over-confident", zorder=4)
    axis.bar(centres, under, width=width, bottom=accuracy, color="#9467BD",
             alpha=0.9, edgecolor="black", linewidth=0.3,
             label="under-confident", zorder=5)
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1.0,
              label="perfect calibration", zorder=6)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal")
    axis.set_title(f"{title}\nECE {metrics['ece']:.3f}", fontsize=9)
    if show_legend:
        axis.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    return metrics


def figure_reliability_grid(
    runs: Sequence[dict[str, Any] | None],
    *,
    domains: Sequence[str] = tuple(DOMAIN_ORDER),
    n_bins: int = 15,
) -> Any:
    """Reliability across every held-out domain, before and after temperature.

    Rows are the two calibration states; columns are the four unseen domains.
    Reading down a column shows what a source-fitted temperature can and cannot
    repair -- which is the finding that the single-domain version of this figure
    cannot show.
    """
    import matplotlib.pyplot as plt

    domains = list(domains)
    figure, axes = plt.subplots(2, len(domains), figsize=(3.3 * len(domains), 7.0),
                                squeeze=False)

    for column, (domain, run) in enumerate(zip(domains, runs)):
        top, bottom = axes[0][column], axes[1][column]
        if run is None:
            for axis in (top, bottom):
                axis.text(0.5, 0.5, "NOT RUN", ha="center", va="center",
                          fontsize=11, style="italic", color="#888888")
                axis.set_xticks([])
                axis.set_yticks([])
            top.set_title(domain_label(domain), fontsize=11, fontweight="bold")
            continue

        _reliability_panel(
            top, run["probabilities"], run["y_true"], n_bins=n_bins,
            title="uncalibrated", show_legend=(column == 0),
        )
        _reliability_panel(
            bottom, run["scaled"], run["y_true"], n_bins=n_bins,
            title=f"after T = {run['temperature']:.3f}", show_legend=False,
        )
        top.annotate(
            f"{domain_label(domain)}  (n = {run['n']:,})",
            xy=(0.5, 1.22), xycoords="axes fraction", ha="center",
            fontsize=11, fontweight="bold",
        )
        # Name the targets where the source-fitted temperature does not finish
        # the job, so the figure cannot be read as "temperature scaling works".
        if run["ece_scaled"] > 0.09:
            bottom.annotate(
                "still miscalibrated", xy=(0.04, 0.90), xycoords="axes fraction",
                ha="left", fontsize=8.5, fontweight="bold", color="#B22222",
            )

    for row, label in enumerate(("Uncalibrated", "Source-fitted temperature")):
        axes[row][0].set_ylabel(f"{label}\n\nAccuracy", fontsize=9)
    for axis in axes[1]:
        axis.set_xlabel("Confidence", fontsize=9)
    # The top row's tick labels sit directly under the bottom row's panel
    # titles; they are redundant with the bottom row's identical axis.
    for axis in axes[0]:
        axis.set_xticklabels([])

    figure.suptitle(
        "Calibration on each unseen domain, before and after temperature scaling\n"
        "temperature fitted on source validation only -- it never sees the target",
        fontsize=12.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.subplots_adjust(hspace=0.32)
    return figure


def figure_discrimination_vs_calibration(runs: Sequence[dict[str, Any] | None]) -> Any:
    """QWK against ECE, with an arrow per domain showing what temperature moves.

    The thesis in one panel: horizontal position (discrimination) barely moves
    across three of the four domains while vertical position (miscalibration)
    varies by a factor of three. Temperature scaling moves points straight down
    -- it is monotonic, so it changes no prediction and cannot move a point
    sideways.
    """
    import matplotlib.pyplot as plt

    from .style import domain_color

    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    present = [r for r in runs if r is not None]

    if not present:
        axis.text(0.5, 0.5, "NOT RUN -- no evaluated runs yet", ha="center",
                  va="center", fontsize=11, style="italic")
        axis.set_axis_off()
        return figure

    qwks = [r["qwk"] for r in present]
    eces = [r["ece"] for r in present]
    midpoint = (min(qwks) + max(qwks)) / 2

    for run in present:
        colour = domain_color(run["target"])
        axis.annotate(
            "", xy=(run["qwk"], run["ece_scaled"]), xytext=(run["qwk"], run["ece"]),
            arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.6,
                            linestyle="--", alpha=0.85),
        )
        axis.scatter(run["qwk"], run["ece"], s=150, color=colour, edgecolor="black",
                     linewidth=0.8, zorder=5)
        axis.scatter(run["qwk"], run["ece_scaled"], s=150, color=colour,
                     edgecolor="black", linewidth=0.8, marker="s", zorder=5)
        # Labels go beside the point, not above it: a point near the top of the
        # range would otherwise print its label over the title, and a label
        # placed above one domain's marker lands on another domain's arrow.
        on_right = run["qwk"] < midpoint
        axis.annotate(
            f"{domain_label(run['target'])}\nn={run['n']:,}",
            xy=(run["qwk"], run["ece"]),
            xytext=(12 if on_right else -12, 0), textcoords="offset points",
            ha="left" if on_right else "right", va="center",
            fontsize=9, fontweight="bold", color=colour,
        )

    # Headroom so the topmost label and the legend do not meet.
    span = max(qwks) - min(qwks)
    axis.set_xlim(min(qwks) - 0.22 * span, max(qwks) + 0.16 * span)
    axis.set_ylim(0, max(eces) * 1.32)

    axis.scatter([], [], s=110, color="#777777", edgecolor="black",
                 label="uncalibrated (circle)")
    axis.scatter([], [], s=110, color="#777777", edgecolor="black", marker="s",
                 label="after source-fitted temperature (square)")
    axis.legend(fontsize=8.5, loc="upper right")

    axis.set_xlabel("QWK on the unseen domain  (higher is better)")
    axis.set_ylabel("ECE on the unseen domain  (lower is better)")
    axis.set_title(
        "Discrimination and calibration do not degrade together\n"
        "arrows show the effect of a temperature fitted only on source data",
        fontsize=11.5,
    )
    axis.grid(alpha=0.3, linestyle=":")
    figure.tight_layout()
    return figure


def generate_calibration_figures(
    experiment_ids: dict[str, str],
    outputs_dir: Path | str,
    *,
    domains: Sequence[str] = tuple(DOMAIN_ORDER),
    suffix: str = "",
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Write both cross-domain calibration figures.

    ``experiment_ids`` maps domain -> experiment id. Domains whose runs are
    missing are rendered as NOT RUN panels rather than omitted, so the figure
    always shows the full four-domain matrix it claims to.
    """
    apply_style()
    outputs = Path(outputs_dir)
    runs = [
        load_target_predictions(experiment_ids[d], d, outputs_dir=outputs)
        if d in experiment_ids else None
        for d in domains
    ]
    found = sum(r is not None for r in runs)
    log.info("calibration figures: %d/%d domains available", found, len(domains))

    figures_dir = outputs / "figures"
    written: list[Path] = []
    written.extend(save_figure(
        figure_reliability_grid(runs, domains=domains),
        f"calibration_reliability_grid{suffix}", figures_dir, formats=formats,
    ))
    written.extend(save_figure(
        figure_discrimination_vs_calibration(runs),
        f"calibration_qwk_vs_ece{suffix}", figures_dir, formats=formats,
    ))
    return written
