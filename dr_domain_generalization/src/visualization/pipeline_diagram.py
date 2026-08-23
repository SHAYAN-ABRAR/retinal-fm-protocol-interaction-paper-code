"""Programmatically drawn pipeline and protocol diagrams.

Two figures, drawn in matplotlib so they regenerate with the code rather than
drifting out of date in a separate drawing tool:

:func:`figure_pipeline`
    The end-to-end flow, from the four source datasets to a confidence-bearing
    prediction on an unseen target domain. For a leave-one-domain-out run the
    held-out dataset is greyed out and struck through on the input side, which
    makes the central protocol constraint visible at a glance.

:func:`figure_protocol`
    The three evaluation protocols side by side, and -- more importantly -- what
    each one is and is not allowed to touch. The "target is sacred" rule is easy
    to state and easy to violate; a diagram that shows the target column touching
    only the final evaluation box is worth a paragraph of prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..utils.logging import get_logger
from .style import DOMAIN_LABELS, DOMAIN_ORDER, apply_style, domain_color, save_figure

log = get_logger("viz.pipeline")

__all__ = ["figure_pipeline", "figure_protocol", "generate_diagram_figures"]

_BOX = {"boxstyle": "round,pad=0.45", "linewidth": 1.4}


def _box(axis, x, y, text, *, facecolor, edgecolor="black", fontsize=9, weight="normal",
         width=None, alpha=1.0):
    return axis.text(
        x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight,
        bbox={**_BOX, "facecolor": facecolor, "edgecolor": edgecolor, "alpha": alpha},
        zorder=3, wrap=True,
    )


def _arrow(axis, start, end, *, style="-|>", color="#333333", linewidth=1.4, linestyle="-"):
    axis.annotate(
        "", xy=end, xytext=start,
        arrowprops={
            "arrowstyle": style, "color": color, "linewidth": linewidth,
            "linestyle": linestyle, "shrinkA": 6, "shrinkB": 6,
        },
        zorder=2,
    )


def figure_pipeline(
    *,
    held_out: str | None = "idrid",
    sources: Sequence[str] | None = None,
    title: str | None = None,
) -> Any:
    """The full pipeline, with the held-out domain visibly excluded.

    ``held_out=None`` draws the generic pipeline with all four datasets active.
    """
    import matplotlib.pyplot as plt

    apply_style()
    sources = list(sources) if sources else [d for d in DOMAIN_ORDER if d != held_out]

    figure, axis = plt.subplots(figsize=(11.5, 6.4))
    axis.set_xlim(-0.15, 12)
    axis.set_ylim(0, 7.4)
    axis.set_axis_off()
    axis.grid(False)

    # -- left column: the four datasets ----------------------------------
    y_positions = [6.2, 5.0, 3.8, 2.6]
    for domain, y in zip(DOMAIN_ORDER, y_positions):
        excluded = domain == held_out
        label = DOMAIN_LABELS[domain]
        if excluded:
            label = f"{label}\n(HELD OUT)"
        _box(
            axis, 1.15, y, label,
            facecolor=domain_color(domain) if not excluded else "#DDDDDD",
            edgecolor="#B22222" if excluded else "black",
            fontsize=9, weight="bold" if excluded else "normal",
            alpha=0.45 if excluded else 0.85,
        )
        if not excluded:
            _arrow(axis, (1.95, y), (3.05, 4.4))

    # -- the pipeline spine ----------------------------------------------
    stages = [
        (3.85, 4.4, "Source domains\n" + " + ".join(DOMAIN_LABELS[d] for d in sources), "#F2F2F2"),
        (3.85, 2.9, "Harmonised preprocessing\nretina crop -> square pad -> resize", "#F2F2F2"),
        (6.35, 2.9, "Backbone\nDenseNet121 / ConvNeXt-T / DINOv2", "#DCE9F5"),
        (6.35, 4.4, "Domain-generalized representation\nDeep CORAL | MixStyle", "#DCE9F5"),
        (8.85, 4.4, "DR head\n5-class CE  or  CORAL ordinal", "#E6F2E6"),
        (8.85, 2.9, "Temperature scaling\nfitted on SOURCE validation only", "#FFF2CC"),
        (8.85, 1.5, "Calibrated confidence\n+ selective prediction", "#FFF2CC"),
    ]
    for x, y, text, colour in stages:
        _box(axis, x, y, text, facecolor=colour, fontsize=8.5)

    _arrow(axis, (3.85, 4.05), (3.85, 3.3))
    _arrow(axis, (4.75, 2.9), (5.35, 2.9))
    _arrow(axis, (6.35, 3.3), (6.35, 4.0))
    _arrow(axis, (7.5, 4.4), (8.0, 4.4))
    _arrow(axis, (8.85, 4.0), (8.85, 3.3))
    _arrow(axis, (8.85, 2.5), (8.85, 1.9))

    # -- the target evaluation box ---------------------------------------
    target_label = DOMAIN_LABELS.get(held_out, "held-out domain") if held_out else "unseen domain"
    _box(
        axis, 8.85, 0.45,
        f"Evaluation on UNSEEN {target_label}\n"
        f"QWK | macro F1 | AUROC | ECE | NLL | risk-coverage",
        facecolor="#F8D7DA", edgecolor="#B22222", fontsize=8.5, weight="bold",
    )
    _arrow(axis, (8.85, 1.1), (8.85, 0.85), color="#B22222", linewidth=1.8)

    if held_out:
        # The one path that carries the protocol: target data enters at the very
        # end and nowhere else.
        #
        # Routed DOWN and along the bottom margin rather than diagonally across
        # the figure. A straight line from the dataset to the evaluation box
        # passes through the preprocessing and backbone boxes, which reads as the
        # held-out data flowing through the pipeline -- the exact opposite of
        # what the diagram is asserting.
        # Exit LEFT out of the held-out box, then down the outside margin. Going
        # straight down from the box would pass through whichever dataset sits
        # below it, making the line look like it starts from a source domain.
        held_out_y = y_positions[DOMAIN_ORDER.index(held_out)]
        axis.plot(
            [0.42, 0.15, 0.15, 7.55],
            [held_out_y, held_out_y, 0.45, 0.45],
            color="#B22222", linewidth=1.6, linestyle="--",
            solid_capstyle="round", zorder=2,
        )
        _arrow(axis, (7.35, 0.45), (7.75, 0.45), color="#B22222", linewidth=1.6)
        axis.text(
            4.35, 1.35, "held-out data enters ONLY here\n"
            "never training, validation, early stopping,\n"
            "model selection or temperature fitting",
            fontsize=8, color="#B22222", style="italic", ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#B22222", "linewidth": 0.9, "alpha": 0.95},
            zorder=4,
        )

    axis.set_title(
        title or (
            f"Leave-one-domain-out protocol -- holding out {target_label}"
            if held_out else "Study pipeline"
        ),
        fontsize=12.5, fontweight="bold", pad=12,
    )
    figure.tight_layout()
    return figure


def figure_protocol() -> Any:
    """The three evaluation protocols, and what each may touch."""
    import matplotlib.pyplot as plt

    apply_style()
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))

    protocols = [
        (
            "In-domain",
            [("DDR train", "#0072B2"), ("DDR val", "#0072B2"), ("DDR test", "#0072B2")],
            "Train, tune and test inside one dataset.\n"
            "The standard reported setting -- and the\noptimistic one.",
        ),
        (
            "Single-source external",
            [("DDR train", "#0072B2"), ("DDR val", "#0072B2"), ("APTOS all", "#E69F00")],
            "Train on one dataset, test on another\ndataset entirely.\n"
            "No target data is seen before evaluation.",
        ),
        (
            "Leave-one-domain-out",
            [("DDR+APTOS+EyePACS train", "#0072B2"),
             ("DDR+APTOS+EyePACS val", "#0072B2"),
             ("IDRiD all", "#009E73")],
            "Train on three datasets, test on the fourth.\n"
            "The strongest protocol here, and the one\nthe headline results use.",
        ),
    ]

    for axis, (name, rows, note) in zip(axes, protocols):
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 10)
        axis.set_axis_off()
        axis.grid(False)

        labels = ["TRAIN", "VALIDATION\n(model selection,\nearly stopping,\ntemperature)", "TEST"]
        for index, ((text, colour), role) in enumerate(zip(rows, labels)):
            y = 8.0 - index * 2.6
            is_test = index == 2
            _box(
                axis, 5.0, y, f"{role}\n{text}",
                facecolor=colour, alpha=0.30 if not is_test else 0.55,
                edgecolor="#B22222" if is_test else "black",
                fontsize=7.5, weight="bold" if is_test else "normal",
            )
            if index < 2:
                _arrow(axis, (5.0, y - 0.85), (5.0, y - 1.75))

        axis.text(5.0, 0.9, note, ha="center", va="center", fontsize=7.5, style="italic")
        axis.set_title(name, fontsize=11, fontweight="bold")

    figure.suptitle(
        "Evaluation protocols. In every case the TEST rows are evaluated once, "
        "after the method is fixed.",
        fontsize=10.5,
    )
    figure.tight_layout()
    return figure


def generate_diagram_figures(
    figures_dir: Path | str,
    *,
    held_out_domains: Sequence[str] = ("idrid",),
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """Write the generic pipeline, one LODO variant per domain, and the protocols."""
    apply_style()
    written: list[Path] = []

    written.extend(save_figure(
        figure_pipeline(held_out=None), "diagram_pipeline", figures_dir, formats=formats
    ))
    for domain in held_out_domains:
        written.extend(save_figure(
            figure_pipeline(held_out=domain),
            f"diagram_lodo_{domain}", figures_dir, formats=formats,
        ))
    written.extend(save_figure(
        figure_protocol(), "diagram_protocols", figures_dir, formats=formats
    ))
    return written
