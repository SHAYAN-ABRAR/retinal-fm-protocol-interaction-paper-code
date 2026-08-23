"""Classification, ordinal and per-class metrics.

Everything here operates on plain NumPy arrays of predictions and probabilities,
never on a model.  That keeps evaluation independent of how a prediction was
produced, and means the same code scores an ERM baseline, a CORAL ordinal model
and a temperature-scaled model without special cases.

Metric choice
-------------
**Quadratic Weighted Kappa** is the primary metric.  DR grading is ordinal, and
QWK penalises a 0-vs-4 error 16x more than a 0-vs-1 error, which matches the
clinical cost. It is also the standard metric in the DR literature, so results
are comparable to prior work.

**Macro F1 and balanced accuracy** are reported alongside because QWK is
sensitive to the marginal distribution: a domain with different grade prevalence
can shift QWK without any change in the model's discriminative ability. Since
this study compares across domains with very different prevalence (no-DR is
32.9% in IDRiD, 73.5% in EyePACS), reporting QWK alone would be misleading.

**Ordinal error metrics** -- mean absolute grade error, the fraction within one
grade, and the severe-error rate -- express the same predictions in units a
clinician can act on.

A deliberate omission: accuracy is reported but never emphasised. Predicting
"no DR" for everything scores 73.5% accuracy on EyePACS and is useless.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger("evaluation.metrics")

__all__ = [
    "N_GRADES",
    "quadratic_weighted_kappa",
    "ordinal_metrics",
    "per_class_metrics",
    "auroc_metrics",
    "referable_dr_metrics",
    "compute_all_metrics",
    "metrics_to_frame",
]

N_GRADES = 5

# Referable DR = moderate or worse. This is the standard screening threshold
# (ICDR grade >= 2); it is reported as a secondary analysis and never replaces
# the 5-class results.
REFERABLE_THRESHOLD = 2


def _as_int_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D array of labels, got shape {array.shape}")
    return array.astype(int)


def quadratic_weighted_kappa(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int = N_GRADES,
) -> float:
    """Cohen's kappa with quadratic weights, computed explicitly.

    Implemented directly rather than delegated so that the class support is
    fixed at ``num_classes``. sklearn infers the label set from the data, so a
    test split that happens to contain no grade-3 cases would silently change the
    weight matrix and make two runs incomparable. IDRiD's test split has only 5
    grade-1 images, so this is a live concern, not a hypothetical one.
    """
    true = _as_int_array(y_true)
    pred = _as_int_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if len(true) == 0:
        return float("nan")

    observed = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(observed, (true, pred), 1.0)

    indices = np.arange(num_classes)
    weights = (indices[:, None] - indices[None, :]) ** 2 / (num_classes - 1) ** 2

    true_marginal = observed.sum(axis=1)
    pred_marginal = observed.sum(axis=0)
    expected = np.outer(true_marginal, pred_marginal) / observed.sum()

    denominator = float((weights * expected).sum())
    if denominator == 0:
        # Happens only when one marginal is degenerate; kappa is undefined.
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)


def ordinal_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Grade-distance metrics: how far wrong, not just whether wrong."""
    true = _as_int_array(y_true)
    pred = _as_int_array(y_pred)
    distance = np.abs(true - pred)
    return {
        "mae_grade": float(distance.mean()),
        "within_1_grade": float((distance <= 1).mean()),
        "exact_match": float((distance == 0).mean()),
        # |error| >= 2 is the clinically dangerous band: it spans the referral
        # boundary in at least one direction.
        "severe_error_rate": float((distance >= 2).mean()),
        "max_grade_error": int(distance.max()) if len(distance) else 0,
    }


def per_class_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int = N_GRADES,
) -> dict[str, Any]:
    """Precision, recall/sensitivity, specificity, F1 and support per class.

    Specificity is included because sensitivity alone is not interpretable for a
    screening task: a model that flags everything has perfect sensitivity.
    """
    true = _as_int_array(y_true)
    pred = _as_int_array(y_pred)

    rows: list[dict[str, Any]] = []
    for grade in range(num_classes):
        true_positive = int(((true == grade) & (pred == grade)).sum())
        false_positive = int(((true != grade) & (pred == grade)).sum())
        false_negative = int(((true == grade) & (pred != grade)).sum())
        true_negative = int(((true != grade) & (pred != grade)).sum())

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else float("nan")
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else float("nan")
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = float("nan") if np.isnan(precision) or np.isnan(recall) else 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        rows.append(
            {
                "grade": grade,
                "support": int((true == grade).sum()),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
            }
        )
    return {"per_class": rows}


def auroc_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    *,
    num_classes: int = N_GRADES,
) -> dict[str, float]:
    """One-vs-rest AUROC per class, plus macro and weighted averages.

    A class absent from ``y_true`` yields NaN rather than 0: AUROC is undefined
    without both positives and negatives, and scoring it as 0 would silently
    drag the macro average down. IDRiD's small test split makes this common.
    """
    from sklearn.metrics import roc_auc_score

    true = _as_int_array(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(true), num_classes):
        raise ValueError(
            f"probabilities must have shape ({len(true)}, {num_classes}), "
            f"got {probabilities.shape}"
        )

    per_class: dict[str, float] = {}
    values: list[float] = []
    supports: list[int] = []
    for grade in range(num_classes):
        positive = (true == grade).astype(int)
        support = int(positive.sum())
        if support == 0 or support == len(true):
            per_class[f"auroc_grade_{grade}"] = float("nan")
            continue
        score = float(roc_auc_score(positive, probabilities[:, grade]))
        per_class[f"auroc_grade_{grade}"] = score
        values.append(score)
        supports.append(support)

    result = dict(per_class)
    result["auroc_macro"] = float(np.mean(values)) if values else float("nan")
    result["auroc_weighted"] = (
        float(np.average(values, weights=supports)) if values else float("nan")
    )
    result["auroc_n_classes_scored"] = len(values)
    return result


def referable_dr_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    threshold: int = REFERABLE_THRESHOLD,
) -> dict[str, float]:
    """Binary referable-DR analysis (grade >= threshold).

    Secondary to the 5-class results, never a replacement. The threshold is
    stated explicitly and recorded in the output so it cannot drift between runs.
    """
    from sklearn.metrics import roc_auc_score

    true = _as_int_array(y_true) >= threshold
    pred = _as_int_array(y_pred) >= threshold

    true_positive = int((true & pred).sum())
    false_positive = int((~true & pred).sum())
    false_negative = int((true & ~pred).sum())
    true_negative = int((~true & ~pred).sum())

    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else float("nan")
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else float("nan")
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")

    result = {
        "referable_threshold": threshold,
        "referable_accuracy": float((true == pred).mean()),
        "referable_sensitivity": sensitivity,
        "referable_specificity": specificity,
        "referable_precision": precision,
        "referable_prevalence": float(true.mean()),
    }
    if precision and sensitivity and not (np.isnan(precision) or np.isnan(sensitivity)):
        result["referable_f1"] = 2 * precision * sensitivity / (precision + sensitivity)

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        referable_probability = probabilities[:, threshold:].sum(axis=1)
        if 0 < true.sum() < len(true):
            result["referable_auroc"] = float(roc_auc_score(true.astype(int), referable_probability))
        else:
            result["referable_auroc"] = float("nan")
    return result


def compute_all_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    num_classes: int = N_GRADES,
    include_per_class: bool = True,
    include_referable: bool = True,
) -> dict[str, Any]:
    """Every headline metric for one set of predictions."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
    )

    true = _as_int_array(y_true)
    pred = _as_int_array(y_pred)
    labels = list(range(num_classes))

    result: dict[str, Any] = {
        "n_samples": int(len(true)),
        "qwk": quadratic_weighted_kappa(true, pred, num_classes=num_classes),
        "kappa_linear": float(
            cohen_kappa_score(true, pred, weights="linear", labels=labels)
        ),
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "f1_macro": float(f1_score(true, pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(
            f1_score(true, pred, average="weighted", labels=labels, zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(true, pred)) if len(np.unique(true)) > 1 else float("nan"),
    }
    result.update(ordinal_metrics(true, pred))
    result["confusion_matrix"] = confusion_matrix(true, pred, labels=labels).tolist()

    if probabilities is not None:
        result.update(auroc_metrics(true, probabilities, num_classes=num_classes))

    if include_per_class:
        result.update(per_class_metrics(true, pred, num_classes=num_classes))
    if include_referable:
        result.update(referable_dr_metrics(true, pred, probabilities))

    return result


def metrics_to_frame(metrics: dict[str, Any]) -> Any:
    """Flatten a metrics dict to a one-row DataFrame (nested fields dropped)."""
    import pandas as pd

    scalar = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.floating, np.integer))
    }
    return pd.DataFrame([scalar])
