"""Tests for bootstrap confidence intervals, paired comparison and selective prediction.

These carry the statistical claims of the paper, so each one has an explicit
**null case**: a comparison of a model with itself must not be significant, and
uninformative confidence must score at chance. A test suite that only checks the
positive direction cannot distinguish a working method from a broken one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.bootstrap import (  # noqa: E402
    bootstrap_metric,
    mcnemar_test,
    paired_bootstrap_difference,
)
from src.evaluation.selective_prediction import (  # noqa: E402
    error_detection_auroc,
    evaluate_selective_prediction,
    metrics_at_coverage,
    risk_coverage_curve,
)


def _synthetic_predictions(n: int = 400, noise: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 5, n)
    y_pred = np.clip(y_true + rng.integers(-noise, noise + 1, n), 0, 4)
    probabilities = np.eye(5)[y_pred] * 0.7 + 0.075
    return y_true, y_pred, probabilities


def _confidence_correlated_with_correctness(n: int = 800, accuracy: float = 0.75, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 5, n)
    correct = rng.random(n) < accuracy
    y_pred = y_true.copy()
    y_pred[~correct] = (y_true[~correct] + rng.integers(1, 3, int((~correct).sum()))) % 5
    confidence = np.where(correct, rng.uniform(0.7, 1.0, n), rng.uniform(0.3, 0.8, n))
    return y_true, y_pred, confidence


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_the_point_estimate() -> None:
    y_true, y_pred, probabilities = _synthetic_predictions()
    result = bootstrap_metric(y_true, y_pred, probabilities, metric="qwk", n_bootstrap=300)
    assert result.lower <= result.point_estimate <= result.upper
    assert result.n_failed == 0


def test_bootstrap_is_reproducible_from_the_seed() -> None:
    y_true, y_pred, probabilities = _synthetic_predictions()
    a = bootstrap_metric(y_true, y_pred, probabilities, metric="qwk", n_bootstrap=200, seed=7)
    b = bootstrap_metric(y_true, y_pred, probabilities, metric="qwk", n_bootstrap=200, seed=7)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_bootstrap_interval_narrows_with_more_data() -> None:
    """The CI must reflect sample size, not merely decorate the number."""
    small = bootstrap_metric(*_synthetic_predictions(n=100, seed=1), metric="qwk", n_bootstrap=300)
    large = bootstrap_metric(*_synthetic_predictions(n=2000, seed=1), metric="qwk", n_bootstrap=300)
    assert (large.upper - large.lower) < (small.upper - small.lower)


def test_bootstrap_rejects_an_unknown_metric() -> None:
    y_true, y_pred, probabilities = _synthetic_predictions(n=50)
    with pytest.raises(KeyError):
        bootstrap_metric(y_true, y_pred, probabilities, metric="not_a_metric")


# ---------------------------------------------------------------------------
# paired comparison -- null case first
# ---------------------------------------------------------------------------

def test_paired_difference_of_a_model_with_itself_is_zero_and_not_significant() -> None:
    """If this ever reports significance, every comparison in the paper is suspect."""
    y_true, y_pred, probabilities = _synthetic_predictions()
    result = paired_bootstrap_difference(
        y_true, y_pred, y_pred, probabilities, probabilities, metric="qwk", n_bootstrap=300
    )
    assert result["difference"] == pytest.approx(0.0, abs=1e-12)
    assert result["significant"] is False


def test_paired_difference_detects_a_clearly_worse_model() -> None:
    # The worse model predicts independently of the truth. Note that simply
    # adding *wider* noise is not reliably worse in QWK: clipping to [0, 4]
    # changes the predicted marginal, which moves QWK's expected-agreement
    # denominator and can raise the score. An independent predictor is
    # unambiguous.
    y_true, good, good_probabilities = _synthetic_predictions(noise=1, seed=0)
    # A DIFFERENT seed: seed 0 would replay the same draw that produced y_true
    # inside _synthetic_predictions, making the "bad" model perfect.
    bad = np.random.default_rng(12345).integers(0, 5, len(y_true))
    bad_probabilities = np.eye(5)[bad] * 0.7 + 0.075

    result = paired_bootstrap_difference(
        y_true, good, bad, good_probabilities, bad_probabilities,
        metric="qwk", n_bootstrap=400,
    )
    assert result["difference"] < 0
    assert result["ci_upper"] < 0
    assert result["significant"] is True
    assert "uncorrected" in result["note"]


def test_paired_comparison_requires_matched_samples() -> None:
    with pytest.raises(ValueError, match="identical samples"):
        paired_bootstrap_difference([0, 1, 2], [0, 1, 2], [0, 1])


def test_mcnemar_uses_the_exact_test_for_few_discordant_pairs() -> None:
    """IDRiD's 507-image test split makes small discordant counts likely."""
    y_true = np.zeros(100, dtype=int)
    a = np.zeros(100, dtype=int)
    b = np.array([1] * 5 + [0] * 95)
    result = mcnemar_test(y_true, a, b)
    assert result["method"] == "exact binomial"
    assert result["n_discordant"] == 5


def test_mcnemar_on_identical_models_reports_no_discordant_pairs() -> None:
    y_true, y_pred, _ = _synthetic_predictions(n=200)
    result = mcnemar_test(y_true, y_pred, y_pred)
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0
    assert result["significant_at_0.05"] is False


# ---------------------------------------------------------------------------
# selective prediction
# ---------------------------------------------------------------------------

def test_risk_decreases_as_low_confidence_samples_are_deferred() -> None:
    """The core claim of selective prediction, tested rather than assumed."""
    result = evaluate_selective_prediction(*_confidence_correlated_with_correctness())
    risks = [row["risk"] for row in result.at_coverage]      # coverage 1.0 -> 0.5
    assert risks == sorted(risks, reverse=True)
    assert result.error_detection_auroc > 0.9


def test_uninformative_confidence_gives_chance_error_detection() -> None:
    """Random confidence must score ~0.5, not something flattering."""
    rng = np.random.default_rng(3)
    y_true, y_pred, _ = _confidence_correlated_with_correctness(n=4000)
    score = error_detection_auroc(y_true, y_pred, rng.random(4000))
    assert 0.45 < score < 0.55


def test_error_detection_auroc_is_nan_when_undefined() -> None:
    """No errors at all means AUROC is undefined; it must not invent a value."""
    y_true = np.array([0, 1, 2, 3, 4])
    assert np.isnan(error_detection_auroc(y_true, y_true, np.linspace(0.5, 1.0, 5)))


def test_aurc_is_lower_for_better_confidence_ranking() -> None:
    y_true, y_pred, good_confidence = _confidence_correlated_with_correctness()
    rng = np.random.default_rng(4)
    informative = risk_coverage_curve(y_true, y_pred, good_confidence)[2]
    random_ranking = risk_coverage_curve(y_true, y_pred, rng.random(len(y_true)))[2]
    assert informative < random_ranking


def test_coverage_accounting_is_exact() -> None:
    y_true, y_pred, confidence = _confidence_correlated_with_correctness(n=1000)
    rows = metrics_at_coverage(y_true, y_pred, confidence, coverages=(1.0, 0.9, 0.5))
    assert [r["n_retained"] for r in rows] == [1000, 900, 500]
    assert [r["n_deferred"] for r in rows] == [0, 100, 500]
    assert rows[0]["accuracy"] + rows[0]["risk"] == pytest.approx(1.0)


def test_risk_coverage_curve_is_deterministic_under_ties() -> None:
    """Constant confidence must not make the curve depend on sort instability."""
    y_true, y_pred, _ = _confidence_correlated_with_correctness(n=200)
    flat = np.full(200, 0.5)
    assert risk_coverage_curve(y_true, y_pred, flat)[2] == risk_coverage_curve(
        y_true, y_pred, flat
    )[2]
