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
    holm_bonferroni,
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


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
#
# The null case matters most here: a family of uniformly-distributed p-values,
# which is what pure noise produces, must yield almost no survivors. A
# correction that lets noise through is worse than none, because it carries
# the authority of having been "corrected".
# ---------------------------------------------------------------------------
def test_holm_rejects_in_step_down_order() -> None:
    result = holm_bonferroni([0.001, 0.02, 0.03, 0.4], ["a", "b", "c", "d"])
    survives = {r["label"]: r["survives"] for r in result}

    assert survives["a"] is True          # 0.001 <= 0.05/4
    assert survives["b"] is False         # 0.020 >  0.05/3
    assert survives["c"] is False         # blocked by step-down
    assert survives["d"] is False


def test_holm_is_more_powerful_than_bonferroni() -> None:
    """Holm's first threshold equals Bonferroni's; later ones are looser."""
    p_values = [0.01, 0.012, 0.013, 0.014]
    result = holm_bonferroni(p_values)
    thresholds = sorted(r["threshold"] for r in result)

    assert thresholds[0] == pytest.approx(0.05 / 4)
    assert thresholds[-1] == pytest.approx(0.05 / 1)
    assert sum(r["survives"] for r in result) >= sum(p <= 0.05 / 4 for p in p_values)


def test_holm_lets_almost_nothing_through_on_pure_noise() -> None:
    """Uniform p-values are what a family of true nulls looks like."""
    rng = np.random.default_rng(0)
    false_positives = 0
    trials = 200
    for _ in range(trials):
        p_values = rng.uniform(size=20)
        if any(r["survives"] for r in holm_bonferroni(p_values)):
            false_positives += 1

    # Family-wise error rate is controlled at alpha; allow sampling slack.
    assert false_positives / trials <= 0.12, f"{false_positives}/{trials} families leaked"


def test_holm_adjusted_p_values_are_monotone() -> None:
    result = holm_bonferroni([0.04, 0.001, 0.5, 0.02])
    ordered = sorted(result, key=lambda r: r["rank"])
    adjusted = [r["p_adjusted"] for r in ordered]

    assert adjusted == sorted(adjusted), "adjusted p-values must not decrease with rank"
    assert all(a <= 1.0 for a in adjusted)


def test_holm_single_comparison_is_uncorrected() -> None:
    result = holm_bonferroni([0.04])
    assert result[0]["threshold"] == pytest.approx(0.05)
    assert result[0]["survives"] is True


def test_holm_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="3 p-values but 2 labels"):
        holm_bonferroni([0.1, 0.2, 0.3], ["a", "b"])


def test_holm_handles_an_empty_family() -> None:
    assert holm_bonferroni([]) == []


def test_paired_bootstrap_reports_a_p_value() -> None:
    """Holm ranks by p-value, so the comparison function must supply one."""
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 5, size=300)
    good = np.where(rng.random(300) < 0.8, y_true, rng.integers(0, 5, size=300))
    bad = rng.integers(0, 5, size=300)

    result = paired_bootstrap_difference(y_true, bad, good, metric="qwk", n_bootstrap=500)
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["p_value"] < 0.05, "a clearly better model should be detected"


def test_identical_predictions_give_a_non_significant_p_value() -> None:
    rng = np.random.default_rng(9)
    y_true = rng.integers(0, 5, size=200)
    same = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, 5, size=200))

    result = paired_bootstrap_difference(y_true, same, same, metric="qwk", n_bootstrap=500)
    assert result["p_value"] > 0.05


# --------------------------------------------------------------------------
# severe_error_rate / within_1_grade in the bootstrap registry
#
# These were added 2026-08-22. The resolution comparison needed a confidence
# interval on the severe-error rate -- the metric the clinical framing rests on
# -- and it was the one headline metric the registry could not bootstrap.
# --------------------------------------------------------------------------


def test_severe_error_rate_matches_the_canonical_implementation() -> None:
    """A bootstrap CI and a point estimate must measure the same thing."""
    import numpy as np

    from src.evaluation.bootstrap import METRIC_FUNCTIONS
    from src.evaluation.metrics import compute_all_metrics

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 2000)
    y_pred = rng.integers(0, 5, 2000)
    reference = compute_all_metrics(y_true, y_pred)

    for name in ("severe_error_rate", "within_1_grade"):
        assert METRIC_FUNCTIONS[name](y_true, y_pred, None) == pytest.approx(
            reference[name], abs=1e-12
        ), name


def test_severe_error_rate_counts_two_step_misses_only() -> None:
    import numpy as np

    from src.evaluation.bootstrap import METRIC_FUNCTIONS

    severe = METRIC_FUNCTIONS["severe_error_rate"]
    y_true = np.array([0, 0, 0, 0])
    # errors of 0, 1, 2, 4 -> the last two are severe
    assert severe(y_true, np.array([0, 1, 2, 4]), None) == pytest.approx(0.5)
    # a perfect model has no severe errors and is entirely within one grade
    assert severe(y_true, y_true, None) == pytest.approx(0.0)
    assert METRIC_FUNCTIONS["within_1_grade"](y_true, y_true, None) == pytest.approx(1.0)


def test_severe_error_and_within_1_are_complementary() -> None:
    """|err|>=2 and |err|<=1 partition every prediction, so they sum to one."""
    import numpy as np

    from src.evaluation.bootstrap import METRIC_FUNCTIONS

    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 5, 500)
    y_pred = rng.integers(0, 5, 500)
    total = (METRIC_FUNCTIONS["severe_error_rate"](y_true, y_pred, None)
             + METRIC_FUNCTIONS["within_1_grade"](y_true, y_pred, None))
    assert total == pytest.approx(1.0)


def test_paired_bootstrap_accepts_severe_error_rate() -> None:
    import numpy as np

    from src.evaluation.bootstrap import paired_bootstrap_difference

    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 5, 600)
    good = np.clip(y_true + rng.integers(-1, 2, 600), 0, 4)   # never a severe miss
    bad = rng.integers(0, 5, 600)                              # many severe misses

    result = paired_bootstrap_difference(
        y_true, bad, good, metric="severe_error_rate", n_bootstrap=400, seed=1
    )
    # b (good) minus a (bad): the better model has the lower rate, so negative.
    assert result["difference"] < 0
    assert result["ci_upper"] < 0
    assert result["significant"] is True
