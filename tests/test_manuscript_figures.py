"""Every manuscript figure must ship as both a 300+ dpi PNG and a vector PDF.

A figure that exists only as a PNG cannot go into an IEEE submission, and one
that exists only as a PDF cannot be previewed in a review document. Both have
happened in this project's figure directories before, so the requirement is a
test rather than a habit.

These tests skip rather than fail when the figure directory has not been
generated, so a fresh clone without `outputs/` still passes the suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "outputs" / "figures" / "jbhi_final"
PROVENANCE = FIGURES / "FIGURE_PROVENANCE.csv"

MAIN = ["fig1_study_design", "fig2_adaptation_depth", "fig3_primary_interaction",
        "fig4_seed_interaction_slopes", "fig5_full_ft_paired_qwk"]
SUPPLEMENT = ["s1_source_validation", "s2_domain_profile", "s3_configuration",
              "s4_calibration", "s5_risk_coverage", "s6_error_pattern"]
ALL = MAIN + SUPPLEMENT


def _require_directory():
    if not FIGURES.exists():
        pytest.skip("figure directory not generated")


@pytest.mark.parametrize("name", ALL)
def test_figure_has_png_and_pdf(name: str) -> None:
    _require_directory()
    png = FIGURES / f"{name}.png"
    pdf = FIGURES / f"{name}.pdf"
    assert png.exists(), f"{name}: PNG missing"
    assert pdf.exists(), f"{name}: vector PDF missing"
    assert png.stat().st_size > 10_000, f"{name}: PNG suspiciously small"
    assert pdf.stat().st_size > 5_000, f"{name}: PDF suspiciously small"


@pytest.mark.parametrize("name", ALL)
def test_png_resolution_is_at_least_300_dpi(name: str) -> None:
    _require_directory()
    png = FIGURES / f"{name}.png"
    if not png.exists():
        pytest.skip(f"{name} not generated")
    pillow = pytest.importorskip("PIL.Image")
    with pillow.open(png) as image:
        dpi = image.info.get("dpi", (0, 0))[0]
    # Saved at 400; allow for the integer rounding Pillow reports.
    assert dpi >= 299, f"{name}: {dpi} dpi is below print resolution"


def test_pdf_is_vector_not_a_wrapped_bitmap() -> None:
    """A PDF holding one giant image defeats the point of shipping vector."""
    _require_directory()
    for name in ALL:
        pdf = FIGURES / f"{name}.pdf"
        if not pdf.exists():
            continue
        png = FIGURES / f"{name}.png"
        # A vector PDF of a plot is far smaller than its own raster; a PDF that
        # merely wraps the bitmap is not.
        assert pdf.stat().st_size < png.stat().st_size, (
            f"{name}: PDF is not smaller than the PNG, so it probably wraps a "
            f"raster instead of holding vector drawing commands")


def test_provenance_covers_every_figure() -> None:
    _require_directory()
    if not PROVENANCE.exists():
        pytest.skip("provenance not generated")
    with PROVENANCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    covered = {row["figure"] for row in rows}
    missing = sorted(set(ALL) - covered)
    assert not missing, f"no provenance row for: {missing}"
    for row in rows:
        assert row["output_png"] != "MISSING", f"{row['figure']}: PNG missing"
        assert row["output_pdf"] != "MISSING", f"{row['figure']}: PDF missing"
        assert row["source_csv"], f"{row['figure']}: no source declared"
        assert row["generator_function"], f"{row['figure']}: no generator named"
