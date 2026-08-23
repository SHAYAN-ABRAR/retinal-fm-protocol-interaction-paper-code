"""Selective prediction: performance as a function of coverage.

A screening model does not have to answer every case.  If it can identify the
cases it is likely to get wrong and defer them to a clinician, the cases it
*does* answer can be far more reliable.  This module measures whether the
model's confidence actually supports that.

Definitions
-----------
**Coverage** -- the fraction of images the model answers, after rejecting the
least-confident ones.
**Risk** -- the error rate among the answered images.
**AURC** -- area under the risk-coverage curve; lower is better. It summarises
the whole curve in one number, so two methods can be compared without picking a
coverage level after the fact.

The honest framing
------------------
A risk-coverage curve shows that confidence *ranks* errors, which is a property
of the model. It does **not** show that the system is safe to deploy: the
deferred cases still need a clinician, the abstention threshold would have to be
fixed prospectively rather than chosen on the test set, and nothing here
addresses the failure modes that matter most clinically (a confidently wrong
proliferative case). The brief is explicit that this must not be presented as
evidence of clinical readiness, and the docstrings here say so because the
figures get lifted into papers.

Error detection as a ranking problem
------------------------------------
:func:`error_detection_auroc` asks a cleaner question: treating "this prediction
is wrong" as the positive class and negative confidence as the score, how well
does confidence separate errors from correct predictions? 0.5 means confidence
carries no information about correctness at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..utils.logging import get_logger
from .metrics import quadratic_weighted_kappa

log = get_logger("evaluation.selective")

__all__ = [
    "SelectivePredictionResult",
    "risk_coverage_curve",
    "metrics_at_coverage",
    "error_detection_auroc",
    "evaluate_selective_prediction",
]

DEFAULT_COVERAGES = (1.0, 0.95, 0.90, 0.80, 0.70, 0.50)


@dataclass
class SelectivePredictionResult:
    coverages: np.ndarray = field(default_factory=lambda: np.empty(0))
    risks: np.ndarray = field(default_factory=lambda: np.empty(0))
    aurc: float = float("nan")
    error_detection_auroc: float = float("nan")
    at_coverage: list[dict[str, Any]] = field(default_factory=list)
    n_samples: int = 0
    confidence_source: str = "max_softmax"

    def as_dict(self) -> dict[str, Any]:
        return {
            "aurc": self.aurc,
            "error_detection_auroc": self.error_detection_auroc,
            "n_samples": self.n_samples,
            "confidence_source": self.confidence_source,
            "at_coverage": self.at_coverage,
        }


def _sorted_by_confidence(
    confidence: np.ndarray,
) -> np.ndarray:
    """Indices ordered most- to least-confident, ties broken deterministically."""
    # Negate so argsort gives descending order; the stable kind keeps ties in
    # input order, which makes the whole curve reproducible.
    return np.argsort(-confidence, kind="stable")


def risk_coverage_curve(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Full risk-coverage curve and its area (AURC).

    Risk is 0/1 error rate among the retained samples. Returns
    ``(coverages, risks, aurc)`` with one point per possible coverage level.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    confidence = np.asarray(confidence, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return np.empty(0), np.empty(0), float("nan")

    order = _sorted_by_confidence(confidence)
    errors = (y_true[order] != y_pred[order]).astype(np.float64)

    cumulative_errors = np.cumsum(errors)
    counts = np.arange(1, n + 1)
    risks = cumulative_errors / counts
    coverages = counts / n
    aurc = float(np.trapezoid(risks, coverages)) if n > 1 else float(risks[0])
    return coverages, risks, aurc


def metrics_at_coverage(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    num_classes: int = 5,
) -> list[dict[str, Any]]:
    """Accuracy, macro F1, QWK and MAE at each requested coverage level."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    confidence = np.asarray(confidence, dtype=np.float64)
    order = _sorted_by_confidence(confidence)
    n = len(y_true)

    rows: list[dict[str, Any]] = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * n)))
        selected = order[:keep]
        true_subset, pred_subset = y_true[selected], y_pred[selected]

        rows.append({
            "coverage": float(coverage),
            "n_retained": int(keep),
            "n_deferred": int(n - keep),
            "confidence_threshold": float(confidence[selected].min()),
            "accuracy": float((true_subset == pred_subset).mean()),
            "risk": float((true_subset != pred_subset).mean()),
            "f1_macro": float(
                f1_score(true_subset, pred_subset, average="macro",
                         labels=list(range(num_classes)), zero_division=0)
            ),
            "qwk": quadratic_weighted_kappa(true_subset, pred_subset, num_classes=num_classes),
            "mae_grade": float(np.abs(true_subset - pred_subset).mean()),
            # The clinically important one: deferring should preferentially
            # remove the large-distance errors, not just the marginal ones.
            "severe_error_rate": float((np.abs(true_subset - pred_subset) >= 2).mean()),
        })
    return rows


def error_detection_auroc(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
) -> float:
    """AUROC for detecting incorrect predictions from (negative) confidence.

    0.5 means confidence tells you nothing about whether the prediction is right.
    Returns NaN when every prediction is correct or every one is wrong, since
    AUROC is undefined without both outcomes.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    confidence = np.asarray(confidence, dtype=np.float64)

    is_error = (y_true != y_pred).astype(int)
    if is_error.sum() == 0 or is_error.sum() == len(is_error):
        return float("nan")
    return float(roc_auc_score(is_error, -confidence))


def evaluate_selective_prediction(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    num_classes: int = 5,
    confidence_source: str = "max_softmax",
) -> SelectivePredictionResult:
    """Full selective-prediction analysis for one set of predictions."""
    coverage_grid, risks, aurc = risk_coverage_curve(y_true, y_pred, confidence)
    result = SelectivePredictionResult(
        coverages=coverage_grid,
        risks=risks,
        aurc=aurc,
        error_detection_auroc=error_detection_auroc(y_true, y_pred, confidence),
        at_coverage=metrics_at_coverage(
            y_true, y_pred, confidence, coverages=coverages, num_classes=num_classes
        ),
        n_samples=int(len(np.asarray(y_true))),
        confidence_source=confidence_source,
    )
    log.info(
        "selective prediction (n=%d): AURC %.4f, error-detection AUROC %.4f",
        result.n_samples, result.aurc, result.error_detection_auroc,
    )
    return result
