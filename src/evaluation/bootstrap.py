"""Bootstrap confidence intervals and paired model comparison.

Why this is not optional here
-----------------------------
IDRiD's test split has **507 images, 23 of them grade 1**.  A QWK computed on
that is a noisy estimate, and reporting it as a bare number invites false
precision -- the difference between two methods can easily be smaller than the
sampling error.  Every headline number in the paper therefore carries a 95%
bootstrap interval, and every method comparison is done **paired**.

Paired comparison matters
-------------------------
Two models evaluated on the *same* test images share the sampling noise from
those images.  Comparing their independent confidence intervals throws that
information away and is badly under-powered: the intervals can overlap
substantially while the paired difference is unambiguous.  So
:func:`paired_bootstrap_difference` resamples **image indices once** and scores
both models on the same resample.

The interval reported is percentile bootstrap.  It is simple, makes no normality
assumption, and is appropriate for metrics like QWK whose sampling distribution
is skewed near the boundaries. Its known weakness is slight under-coverage for
small n, which argues for reporting the interval rather than pretending a point
estimate is exact -- exactly the point.

Reproducibility
---------------
Every function takes a seed and uses it to build its own generator, so a reported
interval can be regenerated exactly from the saved predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..utils.logging import get_logger
from .calibration import expected_calibration_error
from .metrics import quadratic_weighted_kappa

log = get_logger("evaluation.bootstrap")

__all__ = [
    "BootstrapResult",
    "METRIC_FUNCTIONS",
    "bootstrap_metric",
    "bootstrap_all_metrics",
    "paired_bootstrap_difference",
    "mcnemar_test",
]

DEFAULT_N_BOOTSTRAP = 2000


@dataclass
class BootstrapResult:
    metric: str
    point_estimate: float
    lower: float
    upper: float
    standard_error: float
    n_bootstrap: int
    n_samples: int
    n_failed: int = 0
    confidence_level: float = 0.95
    seed: int = 42

    def format(self, decimals: int = 4) -> str:
        return (
            f"{self.point_estimate:.{decimals}f} "
            f"[{self.lower:.{decimals}f}, {self.upper:.{decimals}f}]"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "estimate": self.point_estimate,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "standard_error": self.standard_error,
            "n_bootstrap": self.n_bootstrap,
            "n_samples": self.n_samples,
            "n_failed_resamples": self.n_failed,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
        }


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> float:
    return quadratic_weighted_kappa(y_true, y_pred, num_classes=5)


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(5)), zero_division=0))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _mae(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def _auroc_macro(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    scores = []
    for grade in range(probabilities.shape[1]):
        positive = (y_true == grade).astype(int)
        if 0 < positive.sum() < len(positive):
            scores.append(float(roc_auc_score(positive, probabilities[:, grade])))
    return float(np.mean(scores)) if scores else float("nan")


def _ece(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> float:
    return expected_calibration_error(probabilities, y_true)


def _nll(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.log(clipped[np.arange(len(y_true)), y_true]).mean())


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "qwk": _qwk,
    "f1_macro": _f1_macro,
    "balanced_accuracy": _balanced_accuracy,
    "accuracy": _accuracy,
    "mae_grade": _mae,
    "auroc_macro": _auroc_macro,
    "ece": _ece,
    "nll": _nll,
}


def _resample_indices(n: int, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_bootstrap, n))


def bootstrap_metric(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    metric: str = "qwk",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for one metric.

    Resamples that make a metric undefined -- for example a draw containing a
    single class, which makes QWK NaN -- are counted and excluded rather than
    propagated. A large ``n_failed`` is itself a finding: it means the split is
    too small or too imbalanced for that metric to be reported confidently.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)

    if metric not in METRIC_FUNCTIONS:
        raise KeyError(f"unknown metric {metric!r}; choose from {sorted(METRIC_FUNCTIONS)}")
    function = METRIC_FUNCTIONS[metric]

    point = function(y_true, y_pred, probabilities)
    samples: list[float] = []
    n_failed = 0

    for indices in _resample_indices(len(y_true), n_bootstrap, seed):
        try:
            value = function(
                y_true[indices], y_pred[indices],
                probabilities[indices] if probabilities is not None else None,
            )
        except Exception:  # noqa: BLE001 - a degenerate resample, not a bug
            n_failed += 1
            continue
        if value == value:
            samples.append(value)
        else:
            n_failed += 1

    if not samples:
        return BootstrapResult(
            metric=metric, point_estimate=point, lower=float("nan"), upper=float("nan"),
            standard_error=float("nan"), n_bootstrap=n_bootstrap, n_samples=len(y_true),
            n_failed=n_failed, confidence_level=confidence_level, seed=seed,
        )

    array = np.asarray(samples)
    alpha = (1.0 - confidence_level) / 2
    return BootstrapResult(
        metric=metric,
        point_estimate=float(point),
        lower=float(np.percentile(array, 100 * alpha)),
        upper=float(np.percentile(array, 100 * (1 - alpha))),
        standard_error=float(array.std(ddof=1)),
        n_bootstrap=n_bootstrap,
        n_samples=len(y_true),
        n_failed=n_failed,
        confidence_level=confidence_level,
        seed=seed,
    )


def bootstrap_all_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    *,
    metrics: Sequence[str] = ("qwk", "f1_macro", "auroc_macro", "ece"),
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> dict[str, BootstrapResult]:
    """CIs for the headline metrics of one evaluation."""
    results = {
        name: bootstrap_metric(
            y_true, y_pred, probabilities,
            metric=name, n_bootstrap=n_bootstrap, seed=seed,
        )
        for name in metrics
    }
    for name, result in results.items():
        log.info("%s: %s (n=%d, %d resamples)", name, result.format(), result.n_samples, n_bootstrap)
    return results


def paired_bootstrap_difference(
    y_true: Sequence[int] | np.ndarray,
    y_pred_a: Sequence[int] | np.ndarray,
    y_pred_b: Sequence[int] | np.ndarray,
    probabilities_a: np.ndarray | None = None,
    probabilities_b: np.ndarray | None = None,
    *,
    metric: str = "qwk",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap of ``metric(B) - metric(A)`` on the same test images.

    Both models are scored on the **same** resample, which removes the shared
    sampling noise and is far better powered than comparing two independent
    intervals.

    ``significant`` is True when the interval excludes zero. That is a
    descriptive statement about this one comparison, not a multiple-testing
    corrected claim: comparing many methods across four target domains needs a
    correction before any of it is called significant in the paper.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_a = np.asarray(y_pred_a).astype(int)
    y_pred_b = np.asarray(y_pred_b).astype(int)
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("paired comparison requires predictions on identical samples")

    function = METRIC_FUNCTIONS[metric]
    observed = function(y_true, y_pred_b, probabilities_b) - function(
        y_true, y_pred_a, probabilities_a
    )

    differences: list[float] = []
    n_failed = 0
    for indices in _resample_indices(len(y_true), n_bootstrap, seed):
        try:
            value_a = function(
                y_true[indices], y_pred_a[indices],
                probabilities_a[indices] if probabilities_a is not None else None,
            )
            value_b = function(
                y_true[indices], y_pred_b[indices],
                probabilities_b[indices] if probabilities_b is not None else None,
            )
        except Exception:  # noqa: BLE001
            n_failed += 1
            continue
        if value_a == value_a and value_b == value_b:
            differences.append(value_b - value_a)
        else:
            n_failed += 1

    if not differences:
        return {
            "metric": metric, "difference": float(observed), "ci_lower": float("nan"),
            "ci_upper": float("nan"), "significant": False, "n_failed_resamples": n_failed,
            "n_samples": int(len(y_true)), "n_bootstrap": n_bootstrap, "seed": seed,
        }

    array = np.asarray(differences)
    alpha = (1.0 - confidence_level) / 2
    lower = float(np.percentile(array, 100 * alpha))
    upper = float(np.percentile(array, 100 * (1 - alpha)))

    return {
        "metric": metric,
        "difference": float(observed),
        "ci_lower": lower,
        "ci_upper": upper,
        "standard_error": float(array.std(ddof=1)),
        "significant": bool(lower > 0 or upper < 0),
        "confidence_level": confidence_level,
        "n_samples": int(len(y_true)),
        "n_bootstrap": n_bootstrap,
        "n_failed_resamples": n_failed,
        "seed": seed,
        "note": "uncorrected for multiple comparisons",
    }


def mcnemar_test(
    y_true: Sequence[int] | np.ndarray,
    y_pred_a: Sequence[int] | np.ndarray,
    y_pred_b: Sequence[int] | np.ndarray,
    *,
    exact_threshold: int = 25,
) -> dict[str, Any]:
    """McNemar's test on the paired correct/incorrect outcomes of two models.

    Uses only the **discordant** pairs -- images where exactly one model is right
    -- which is precisely the information a paired comparison should use.

    The exact binomial test is used when the discordant count is small
    (``n01 + n10 <= exact_threshold``); the chi-square approximation is
    unreliable there, and IDRiD's 507-image test split makes small discordant
    counts likely.

    Note this tests *exact-match accuracy* only. It ignores how far wrong a
    prediction was, so it is a secondary check for an ordinal task -- the paired
    bootstrap on QWK is the primary comparison.
    """
    from scipy import stats

    y_true = np.asarray(y_true).astype(int)
    correct_a = np.asarray(y_pred_a).astype(int) == y_true
    correct_b = np.asarray(y_pred_b).astype(int) == y_true

    n01 = int((~correct_a & correct_b).sum())     # only B right
    n10 = int((correct_a & ~correct_b).sum())     # only A right
    discordant = n01 + n10

    if discordant == 0:
        return {
            "n01_only_b_correct": 0, "n10_only_a_correct": 0, "n_discordant": 0,
            "statistic": float("nan"), "p_value": 1.0, "method": "no discordant pairs",
            "significant_at_0.05": False,
        }

    if discordant <= exact_threshold:
        p_value = float(stats.binomtest(n01, discordant, 0.5).pvalue)
        method, statistic = "exact binomial", float(min(n01, n10))
    else:
        # Continuity-corrected chi-square with 1 degree of freedom.
        statistic = (abs(n01 - n10) - 1) ** 2 / discordant
        p_value = float(stats.chi2.sf(statistic, df=1))
        method = "chi-square with continuity correction"

    return {
        "n01_only_b_correct": n01,
        "n10_only_a_correct": n10,
        "n_discordant": discordant,
        "statistic": float(statistic),
        "p_value": p_value,
        "method": method,
        "significant_at_0.05": bool(p_value < 0.05),
        "note": "tests exact-match accuracy only; ignores ordinal distance",
    }
