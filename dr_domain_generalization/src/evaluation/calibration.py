"""Calibration metrics and temperature scaling.

Calibration is a headline outcome of this study, not a footnote: a model that is
confidently wrong on a new clinic's images is more dangerous than one that knows
it is uncertain.

Metrics
-------
``ECE``   Expected Calibration Error with equal-width confidence bins. The
          conventional choice, and the one most papers report -- but it is
          sensitive to bin count and degrades badly when confidences cluster,
          which they do for an over-confident network.
``AECE``  Adaptive ECE with equal-*mass* bins. Every bin holds the same number of
          samples, so it does not collapse when 90% of predictions sit above 0.9.
          Reported alongside ECE because they can disagree, and quietly picking
          the flattering one would be cherry-picking.
``NLL``   Negative log-likelihood. A strictly proper scoring rule -- unlike ECE,
          it cannot be gamed by a model that is uninformative but well-calibrated.
``Brier`` Multi-class Brier score. Also proper, and decomposable into calibration
          and refinement.

The rule that must not be broken
--------------------------------
**Temperature is fitted on source-domain validation data only.** Fitting it on
the held-out target would be fitting a parameter to the test set -- the external
evaluation would no longer be external. :func:`fit_temperature` takes validation
logits and returns a scalar; applying it to a target domain is a separate,
explicit step, and the split it was fitted on is recorded in the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger("evaluation.calibration")

__all__ = [
    "expected_calibration_error",
    "adaptive_calibration_error",
    "calibration_metrics",
    "TemperatureScaler",
    "fit_temperature",
    "reliability_curve",
]


def _check(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {probabilities.shape}")
    if len(probabilities) != len(labels):
        raise ValueError(f"length mismatch: {len(probabilities)} vs {len(labels)}")
    return probabilities, labels


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """ECE over equal-width confidence bins.

    Empty bins contribute nothing (rather than counting as perfectly calibrated),
    which matters when confidences are concentrated.
    """
    probabilities, labels = _check(probabilities, labels)
    if len(labels) == 0:
        return float("nan")

    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        # Include the left edge on the first bin so confidence == 0 is not lost.
        in_bin = (confidence > low) & (confidence <= high) if low > 0 else (confidence <= high)
        count = int(in_bin.sum())
        if count == 0:
            continue
        error += (count / len(labels)) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def adaptive_calibration_error(
    probabilities: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Adaptive ECE with equal-mass bins.

    Robust to the confidence clustering that makes plain ECE unstable: an
    over-confident network puts nearly everything in the top equal-width bin,
    where a single bin then dominates the estimate.
    """
    probabilities, labels = _check(probabilities, labels)
    if len(labels) == 0:
        return float("nan")

    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)

    order = np.argsort(confidence)
    confidence, correct = confidence[order], correct[order]
    splits = np.array_split(np.arange(len(labels)), min(n_bins, len(labels)))

    error = 0.0
    for indices in splits:
        if len(indices) == 0:
            continue
        error += (len(indices) / len(labels)) * abs(
            correct[indices].mean() - confidence[indices].mean()
        )
    return float(error)


def calibration_metrics(
    probabilities: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 15,
    num_classes: int = 5,
) -> dict[str, float]:
    """ECE, adaptive ECE, NLL, Brier and the confidence diagnostics."""
    probabilities, labels = _check(probabilities, labels)
    if len(labels) == 0:
        return {}

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels

    # Clip before the log so a zero probability does not produce inf NLL.
    clipped = np.clip(probabilities, 1e-12, 1.0)
    nll = float(-np.log(clipped[np.arange(len(labels)), labels]).mean())

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = float(((probabilities - one_hot) ** 2).sum(axis=1).mean())

    return {
        "ece": expected_calibration_error(probabilities, labels, n_bins=n_bins),
        "adaptive_ece": adaptive_calibration_error(probabilities, labels, n_bins=n_bins),
        "nll": nll,
        "brier": brier,
        "mean_confidence": float(confidence.mean()),
        "max_confidence": float(confidence.max()),
        "accuracy": float(correct.mean()),
        # Positive => over-confident, the usual direction for a deep net.
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
        "confidence_correct": float(confidence[correct].mean()) if correct.any() else float("nan"),
        "confidence_incorrect": (
            float(confidence[~correct].mean()) if (~correct).any() else float("nan")
        ),
        "n_bins": n_bins,
    }


@dataclass
class TemperatureScaler:
    """Single-parameter post-hoc calibration: ``softmax(logits / T)``.

    Temperature scaling cannot change which class is predicted -- dividing every
    logit by a positive scalar preserves the argmax -- so accuracy, QWK and F1
    are unchanged by construction. Only the confidences move. That is exactly
    what makes it a clean intervention for this study: any change in ECE is
    attributable to calibration alone.
    """

    temperature: float = 1.0
    fitted_on: str = "NOT FITTED"
    n_fit_samples: int = 0
    nll_before: float = float("nan")
    nll_after: float = float("nan")
    # Head type this temperature was fitted for. Recorded because a scalar fitted
    # against a softmax likelihood is not interchangeable with one fitted against
    # a cumulative-link likelihood.
    head: str = "softmax"

    def apply(
        self, logits: np.ndarray, to_probabilities: Any | None = None
    ) -> np.ndarray:
        """Return temperature-scaled probabilities from raw logits.

        ``to_probabilities`` must be supplied for a CORAL ordinal head: its
        output is K-1 cumulative logits, and softmaxing those would produce a
        K-1 vector rather than a distribution over the K grades. Scaling still
        happens on the logits, before the conversion, which is what preserves
        the rank consistency that makes the conversion valid.
        """
        logits = np.asarray(logits, dtype=np.float64)
        scaled = logits / self.temperature
        if to_probabilities is not None:
            import torch

            return to_probabilities(torch.from_numpy(scaled)).numpy().astype(np.float64)
        scaled = scaled - scaled.max(axis=1, keepdims=True)     # stable softmax
        exponentiated = np.exp(scaled)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def describe(self) -> dict[str, Any]:
        return {
            "temperature": round(self.temperature, 4),
            "fitted_on": self.fitted_on,
            "n_fit_samples": self.n_fit_samples,
            "fit_nll_before": round(self.nll_before, 5),
            "fit_nll_after": round(self.nll_after, 5),
            "head": self.head,
        }


def fit_temperature(
    logits: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    fitted_on: str = "source_validation",
    max_iter: int = 200,
    bounds: tuple[float, float] = (0.05, 10.0),
    to_probabilities: Any | None = None,
) -> TemperatureScaler:
    """Fit temperature by minimising NLL on the given data.

    ``fitted_on`` is a free-text label recorded in the result. Pass something
    that identifies the split, so a reviewer can confirm from the artefact alone
    that the temperature was not fitted on the target domain.

    Optimised with a bounded scalar minimiser rather than LBFGS on a torch
    parameter: the objective is one-dimensional and strictly convex in log-T, so
    this is both faster and more reliable, and it needs no GPU.
    """
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D, got {logits.shape}")
    if len(logits) != len(labels):
        raise ValueError(f"length mismatch: {len(logits)} vs {len(labels)}")
    if len(labels) == 0:
        raise ValueError("cannot fit temperature on an empty set")

    rows = np.arange(len(labels))

    if to_probabilities is not None:
        # Ordinal head: the likelihood is the cumulative-link one, so the NLL has
        # to be built from the converted class probabilities, not from a softmax
        # over K-1 threshold logits.
        import torch

        def _nll(temperature: float) -> float:
            probabilities = to_probabilities(
                torch.from_numpy(logits / temperature)
            ).numpy().astype(np.float64)
            picked = np.clip(probabilities[rows, labels], 1e-12, 1.0)
            return float(-np.log(picked).mean())
    else:
        def _nll(temperature: float) -> float:
            scaled = logits / temperature
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            log_partition = np.log(np.exp(scaled).sum(axis=1))
            return float(-(scaled[rows, labels] - log_partition).mean())

    result = minimize_scalar(
        _nll, bounds=bounds, method="bounded", options={"maxiter": max_iter}
    )
    temperature = float(result.x)

    scaler = TemperatureScaler(
        temperature=temperature,
        fitted_on=fitted_on,
        n_fit_samples=len(labels),
        nll_before=_nll(1.0),
        nll_after=_nll(temperature),
        head="ordinal_coral" if to_probabilities is not None else "softmax",
    )
    log.info(
        "temperature fitted on %s (n=%d): T=%.4f, NLL %.5f -> %.5f%s",
        fitted_on, len(labels), temperature, scaler.nll_before, scaler.nll_after,
        "  (T>1: model was over-confident)" if temperature > 1 else
        "  (T<1: model was under-confident)",
    )
    return scaler


def reliability_curve(
    probabilities: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 15,
    adaptive: bool = False,
) -> dict[str, np.ndarray]:
    """Bin centres, accuracy and confidence per bin, for the reliability diagram."""
    probabilities, labels = _check(probabilities, labels)
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)

    if adaptive:
        order = np.argsort(confidence)
        groups = np.array_split(order, min(n_bins, max(1, len(labels))))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        groups = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (confidence > low) & (confidence <= high) if low > 0 else (confidence <= high)
            groups.append(np.flatnonzero(mask))

    bin_confidence, bin_accuracy, bin_count = [], [], []
    for indices in groups:
        if len(indices) == 0:
            bin_confidence.append(np.nan)
            bin_accuracy.append(np.nan)
            bin_count.append(0)
            continue
        bin_confidence.append(float(confidence[indices].mean()))
        bin_accuracy.append(float(correct[indices].mean()))
        bin_count.append(int(len(indices)))

    return {
        "confidence": np.array(bin_confidence),
        "accuracy": np.array(bin_accuracy),
        "count": np.array(bin_count),
    }
