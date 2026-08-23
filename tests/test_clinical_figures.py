"""Tests for the two clinically load-bearing figures.

The risky part here is not the drawing, it is the arithmetic the drawing
implies. Two things must hold or the figures assert something the results do
not support:

* An unestablished difference must be **labelled** as unestablished. A bar
  chart makes every gap look like a finding, and two of the four domains here
  have intervals that span zero.
* Risk-coverage curves from different seeds cover different image counts, so
  they must be resampled onto a shared coverage grid before averaging.
  Averaging them point-by-point compares different coverage levels and bends
  the mean curve in a direction no seed took.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
matplotlib = pytest.importorskip("matplotlib")

# Headless: without this the figure calls try to open a window.
matplotlib.use("Agg")

from src.visualization.clinical_figures import (  # noqa: E402
    REFERENCE_COVERAGE,
    figure_risk_coverage,
    figure_severe_error_deployment,
)


def _comparison(**overrides):
    rows = [
        {"target": "ddr", "n_test": 1862, "in_domain_severe": 0.0714,
         "lodo_severe_mean": 0.1631, "lodo_severe_sd": 0.0054,
         "delta_severe": 0.0917, "relative_increase": 1.2832,
         "ci_lower": 0.0806, "ci_upper": 0.1155, "verdict": "REAL (both bars)"},
        {"target": "aptos", "n_test": 354, "in_domain_severe": 0.0395,
         "lodo_severe_mean": 0.0669, "lodo_severe_sd": 0.0091,
         "delta_severe": 0.0273, "relative_increase": 0.6905,
         "ci_lower": -0.0085, "ci_upper": 0.0424, "verdict": "CI spans zero"},
    ]
    frame = pd.DataFrame(rows)
    for key, value in overrides.items():
        frame[key] = value
    return frame


def _texts(figure):
    return [t.get_text() for axis in figure.axes for t in axis.texts]


# ---------------------------------------------------------------------------
# Severe-error figure


def test_unestablished_domains_are_labelled_as_such() -> None:
    """The whole point of the two-bar criterion is that it reaches the figure."""
    figure = figure_severe_error_deployment(_comparison())
    texts = _texts(figure)
    assert "not established" in texts, texts
    assert "spans zero" in texts, texts


def test_established_domains_show_the_relative_increase() -> None:
    figure = figure_severe_error_deployment(_comparison())
    assert "+128%" in _texts(figure)


def test_no_percentage_is_shown_for_an_unestablished_domain() -> None:
    """+69% for APTOS would be read as a result; it is inside the noise."""
    assert "+69%" not in _texts(figure_severe_error_deployment(_comparison()))


def test_every_interval_is_inside_the_drawn_axis() -> None:
    """The 'spans zero' labels sit right of each bar and used to be clipped."""
    frame = _comparison()
    figure = figure_severe_error_deployment(frame)
    low, high = figure.axes[1].get_xlim()
    assert low <= frame["ci_lower"].min()
    assert high >= frame["ci_upper"].max()


def test_empty_comparison_renders_not_run_rather_than_raising() -> None:
    empty = _comparison().iloc[0:0]
    assert "NOT RUN" in _texts(figure_severe_error_deployment(empty))


# ---------------------------------------------------------------------------
# Risk-coverage figure


def _curves(n_per_seed=(200, 500), risk_at_full=0.40):
    """Two seeds whose curves have deliberately different lengths."""
    frames = []
    for seed, n in zip((42, 1), n_per_seed):
        coverage = np.linspace(1.0 / n, 1.0, n)
        # Linear in coverage, so the mean over seeds has a known closed form.
        frames.append(pd.DataFrame({
            "target": "eyepacs", "seed": seed,
            "coverage": coverage, "risk": risk_at_full * coverage,
        }))
    return pd.concat(frames, ignore_index=True)


def test_seeds_of_different_length_are_averaged_on_a_shared_grid() -> None:
    """Both seeds trace risk = 0.4 * coverage, so the mean must too."""
    figure = figure_risk_coverage(_curves())
    line = figure.axes[0].lines[0]
    x, y = line.get_xdata(), line.get_ydata()
    assert len(x) == len(y)
    assert np.allclose(y, 0.4 * np.asarray(x), atol=2e-3)


def test_reference_coverage_is_marked() -> None:
    figure = figure_risk_coverage(_curves())
    verticals = [ln.get_xdata()[0] for ln in figure.axes[0].lines
                 if len(set(ln.get_xdata())) == 1]
    assert any(abs(v - REFERENCE_COVERAGE) < 1e-9 for v in verticals)


def test_summary_annotations_report_the_baseline_alongside_the_reduction() -> None:
    """A reduction with no baseline rate is unreadable; both must appear."""
    summary = pd.DataFrame([{
        "target": "eyepacs", "risk@1.0_mean": 0.4074,
        f"risk@{REFERENCE_COVERAGE}_mean": 0.3370,
    }])
    text = " ".join(_texts(figure_risk_coverage(_curves(), summary)))
    assert "0.407" in text and "0.337" in text
    assert "17%" in text


def test_missing_curves_render_not_run() -> None:
    empty = _curves().iloc[0:0]
    assert any("NOT RUN" in t for t in _texts(figure_risk_coverage(empty)))
