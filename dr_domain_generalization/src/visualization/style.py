"""Shared publication figure style and saving helpers.

One place defines fonts, sizes, colours and DPI so every figure in the paper
looks like it belongs to the same document.

Colour choices
--------------
* **Domains** use Okabe-Ito colours, which stay distinguishable under the common
  forms of colour-vision deficiency and survive greyscale printing.
* **Grades** use a sequential blue-to-red ramp, because DR severity is ordinal:
  a categorical palette would hide the fact that grade 3 sits between 2 and 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger("viz.style")

__all__ = [
    "DOMAIN_COLORS",
    "GRADE_COLORS",
    "DOMAIN_ORDER",
    "GRADE_LABELS",
    "apply_style",
    "save_figure",
    "domain_color",
    "grade_color",
]

DOMAIN_ORDER = ["ddr", "aptos", "idrid", "eyepacs"]
DOMAIN_LABELS = {
    "ddr": "DDR",
    "aptos": "APTOS 2019",
    "idrid": "IDRiD",
    "eyepacs": "EyePACS",
}

# Okabe-Ito: colour-blind safe.
DOMAIN_COLORS = {
    "ddr": "#0072B2",      # blue
    "aptos": "#E69F00",    # orange
    "idrid": "#009E73",    # green
    "eyepacs": "#CC79A7",  # reddish purple
}

# Ordinal severity ramp: light (healthy) -> dark red (proliferative).
GRADE_COLORS = {
    0: "#4575B4",
    1: "#91BFDB",
    2: "#FEE090",
    3: "#FC8D59",
    4: "#D73027",
}
GRADE_LABELS = {
    0: "0 No DR",
    1: "1 Mild",
    2: "2 Moderate",
    3: "3 Severe",
    4: "4 Proliferative",
}

FIGURE_DPI = 300


def apply_style() -> None:
    """Apply the project-wide matplotlib style. Safe to call repeatedly."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,          # on-screen; saving overrides with FIGURE_DPI
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",     # never clip labels
            "savefig.pad_inches": 0.05,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,      # grid behind the data
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
            "image.interpolation": "nearest",
        }
    )


def domain_color(domain: str) -> str:
    return DOMAIN_COLORS.get(domain, "#777777")


def grade_color(grade: int) -> str:
    return GRADE_COLORS.get(int(grade), "#777777")


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain)


def save_figure(
    figure: Any,
    name: str,
    outputs_dir: Path | str,
    *,
    formats: Iterable[str] = ("png",),
    close: bool = True,
) -> list[Path]:
    """Save a figure at publication resolution and return the written paths.

    ``name`` should be descriptive and stable -- it becomes the filename the
    paper references.
    """
    import matplotlib.pyplot as plt

    directory = ensure_dir(outputs_dir)
    written: list[Path] = []
    for extension in formats:
        path = directory / f"{name}.{extension}"
        figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        written.append(path)
    if close:
        plt.close(figure)
    log.info("saved figure %s (%s)", name, ", ".join(f.suffix.lstrip('.') for f in written))
    return written
