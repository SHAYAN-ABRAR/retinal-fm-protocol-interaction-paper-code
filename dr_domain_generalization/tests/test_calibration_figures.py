"""Tests for the cross-domain calibration figures.

The reconstruction of temperature-scaled probabilities is the risky part: the
saved prediction files hold probabilities rather than logits, so the figure
recovers the scaled distribution as ``softmax(log(p)/T)``. If that identity were
wrong, the figures would show a calibration improvement that the reported ECE
numbers do not support, and nothing else in the pipeline would catch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")

# Headless: without this the figure calls try to open a Tk window and fail on
# machines with no display or no working tkinter.
matplotlib.use("Agg")

from src.visualization.calibration_figures import (  # noqa: E402
    figure_discrimination_vs_calibration,
    figure_reliability_grid,
    scaled_probabilities,
)


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# The reconstruction identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.384, 1.632, 2.5])
def test_scaling_probabilities_matches_scaling_logits(temperature: float) -> None:
    """softmax(log(p)/T) must equal softmax(z/T) for p = softmax(z)."""
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(400, 5)) * 3.0

    from_logits = _softmax(logits / temperature)
    from_probabilities = scaled_probabilities(_softmax(logits), temperature)

    assert np.allclose(from_logits, from_probabilities, atol=1e-10)


def test_temperature_one_is_the_identity() -> None:
    rng = np.random.default_rng(11)
    probabilities = _softmax(rng.normal(size=(200, 5)))

    assert np.allclose(scaled_probabilities(probabilities, 1.0), probabilities, atol=1e-12)


def test_scaled_probabilities_stay_a_distribution() -> None:
    rng = np.random.default_rng(3)
    probabilities = _softmax(rng.normal(size=(300, 5)) * 5)
    scaled = scaled_probabilities(probabilities, 1.9)

    assert np.allclose(scaled.sum(axis=1), 1.0, atol=1e-12)
    assert (scaled >= 0).all()


def test_temperature_above_one_reduces_confidence() -> None:
    """T > 1 softens the distribution; that is the whole point of the fix."""
    rng = np.random.default_rng(5)
    probabilities = _softmax(rng.normal(size=(500, 5)) * 4)
    scaled = scaled_probabilities(probabilities, 2.0)

    assert scaled.max(axis=1).mean() < probabilities.max(axis=1).mean()


def test_scaling_never_changes_the_prediction() -> None:
    """Temperature scaling is monotonic, so argmax is invariant.

    This is why the QWK/ECE figure draws arrows straight down: the point cannot
    move sideways. If this ever failed, that figure would be misleading.
    """
    rng = np.random.default_rng(13)
    probabilities = _softmax(rng.normal(size=(1000, 5)) * 3)

    for temperature in (0.4, 1.5, 3.0):
        scaled = scaled_probabilities(probabilities, temperature)
        assert (scaled.argmax(axis=1) == probabilities.argmax(axis=1)).all()


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.5])
def test_non_positive_temperature_is_rejected(bad: float) -> None:
    """A negative T would invert the distribution rather than scale it."""
    probabilities = _softmax(np.zeros((4, 5)))
    with pytest.raises(ValueError, match="temperature must be positive"):
        scaled_probabilities(probabilities, bad)


# ---------------------------------------------------------------------------
# Figures render, including with runs missing
# ---------------------------------------------------------------------------
def _fake_run(target: str, *, n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    probabilities = _softmax(rng.normal(size=(n, 5)) * 3)
    return {
        "target": target,
        "y_true": rng.integers(0, 5, size=n),
        "y_pred": probabilities.argmax(axis=1),
        "probabilities": probabilities,
        "scaled": scaled_probabilities(probabilities, 1.6),
        "temperature": 1.6,
        "qwk": 0.7,
        "ece": 0.2,
        "ece_scaled": 0.08,
        "n": n,
    }


def test_reliability_grid_renders_all_four_domains() -> None:
    import matplotlib.pyplot as plt

    runs = [_fake_run(d, seed=i) for i, d in enumerate(["ddr", "aptos", "idrid", "eyepacs"])]
    figure = figure_reliability_grid(runs)

    assert len(figure.axes) == 8
    plt.close(figure)


def test_missing_runs_render_as_not_run_panels_not_dropped() -> None:
    """A missing domain must stay in the grid, or the figure silently shrinks."""
    import matplotlib.pyplot as plt

    runs = [_fake_run("ddr"), None, None, _fake_run("eyepacs", seed=2)]
    figure = figure_reliability_grid(runs)

    assert len(figure.axes) == 8
    texts = [t.get_text() for axis in figure.axes for t in axis.texts]
    # Two panels per missing domain (uncalibrated and scaled), so two absent
    # domains produce four NOT RUN labels.
    assert texts.count("NOT RUN") == 4
    plt.close(figure)


def test_scatter_handles_no_runs_at_all() -> None:
    import matplotlib.pyplot as plt

    figure = figure_discrimination_vs_calibration([None, None, None, None])
    texts = [t.get_text() for axis in figure.axes for t in axis.texts]

    assert any("NOT RUN" in t for t in texts)
    plt.close(figure)


def test_scatter_renders_with_partial_results() -> None:
    import matplotlib.pyplot as plt

    figure = figure_discrimination_vs_calibration([_fake_run("ddr"), None, None, None])
    plt.close(figure)
