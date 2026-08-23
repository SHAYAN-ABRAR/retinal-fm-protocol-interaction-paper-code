"""CORAL **ordinal regression** (Cao, Mirjalili & Raschka, 2020).

    !! NAME COLLISION WARNING !!

    There are two unrelated methods called CORAL in the literature:

    * **THIS FILE** -- CORAL *ordinal regression*: "Rank-consistent Ordinal
      Regression for Neural Networks". It exploits the fact that DR grades are
      ordered (0 < 1 < 2 < 3 < 4) by turning the 5-class problem into 4 binary
      "is the grade greater than k?" questions with a shared weight vector.
      It has NOTHING to do with domains.

    * ``deep_coral_alignment.py`` -- **Deep CORAL** (Sun & Saenko, 2016),
      "CORrelation ALignment". A domain-generalization loss that matches the
      second-order feature statistics of different source domains. It has
      NOTHING to do with ordinality.

    They solve different problems, live in different files, and are never
    interchangeable. In the ablation table they are separate axes: "ordinal"
    and "DG".

Why ordinal at all
------------------
Plain 5-class cross-entropy treats grade 0 and grade 4 as merely *different*,
so predicting 4 for a healthy eye costs exactly as much as predicting 1.
Clinically it does not: the primary metric (QWK) penalises that error 16x more.
An ordinal objective encodes the ordering in the loss rather than hoping the
network infers it.

How CORAL works
---------------
For K classes the head emits **one** scalar projection ``g(x)`` shared across
thresholds, plus K-1 independent bias terms ``b_k``:

    P(y > k | x) = sigmoid(g(x) + b_k),  k = 0 .. K-2

The shared weight vector is what guarantees **rank consistency**: because
``b_0 >= b_1 >= ... `` is enforced implicitly by the shared projection, the
predicted probabilities are automatically monotonically non-increasing in k, so
the model can never claim P(y>2) > P(y>1). Independent per-threshold weights
(the naive "ordinal binary decomposition") lose that guarantee and can produce
incoherent predictions.

The target for threshold k is the binary indicator ``1[y > k]``, so a sample
with grade 3 has targets ``[1, 1, 1, 0]``.

Prediction: count how many thresholds are passed, ``y_hat = sum_k 1[p_k > 0.5]``.

Converting to class probabilities
---------------------------------
Calibration and AUROC need a probability *distribution* over the 5 grades, not
cumulative probabilities. :func:`coral_probabilities` differences the cumulative
form:

    P(y = 0)   = 1 - P(y > 0)
    P(y = k)   = P(y > k-1) - P(y > k)
    P(y = K-1) = P(y > K-2)

Rank consistency is what makes this valid -- without it the differences could go
negative. Tiny negatives from floating point are clamped and renormalised, and
that clamping is the only approximation in the conversion.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.logging import get_logger

log = get_logger("losses.ordinal_coral")

__all__ = [
    "CoralOrdinalLoss",
    "levels_from_labels",
    "coral_probabilities",
    "coral_predict",
    "importance_weights_from_counts",
]


def levels_from_labels(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Binary threshold targets: ``levels[i, k] = 1`` iff ``labels[i] > k``.

    Grade 3 with 5 classes becomes ``[1, 1, 1, 0]``.
    """
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def importance_weights_from_counts(
    counts: Sequence[int], *, scheme: str = "sqrt"
) -> torch.Tensor:
    """Per-threshold importance weights for imbalanced ordinal data.

    CORAL admits per-threshold weights. The useful quantity is how balanced each
    binary task is: the threshold separating grade 0 from the rest is nearly
    balanced on IDRiD but very lopsided on EyePACS, and an unweighted sum lets
    the easy thresholds dominate.
    """
    counts_tensor = torch.as_tensor(list(counts), dtype=torch.float64)
    total = counts_tensor.sum()
    weights = []
    for k in range(len(counts_tensor) - 1):
        positive = counts_tensor[k + 1:].sum()
        negative = total - positive
        balance = (positive * negative).clamp(min=1.0) / (total**2)
        weights.append(balance.sqrt() if scheme == "sqrt" else balance)
    weight_tensor = torch.stack(weights).to(torch.float32)
    return weight_tensor / weight_tensor.mean()


class CoralOrdinalLoss(nn.Module):
    """Rank-consistent ordinal loss over K-1 threshold logits.

    Expects logits of shape ``(batch, K-1)`` -- the model's ``ordinal_coral``
    head produces exactly that (a shared projection plus K-1 biases).
    """

    def __init__(
        self,
        num_classes: int = 5,
        importance_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if num_classes < 3:
            raise ValueError("ordinal regression needs at least 3 ordered classes")
        self.num_classes = num_classes
        self.reduction = reduction
        if importance_weights is not None:
            if importance_weights.numel() != num_classes - 1:
                raise ValueError(
                    f"expected {num_classes - 1} importance weights, "
                    f"got {importance_weights.numel()}"
                )
            self.register_buffer("importance_weights", importance_weights.float())
        else:
            self.importance_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes - 1:
            raise ValueError(
                f"CORAL expects logits of shape (batch, {self.num_classes - 1}); "
                f"got {tuple(logits.shape)}. A standard 5-way head will not work -- "
                "use BackboneConfig(head='ordinal_coral')."
            )
        levels = levels_from_labels(target, self.num_classes)
        # Per-threshold binary cross-entropy, numerically stable in float32.
        per_threshold = F.binary_cross_entropy_with_logits(
            logits.float(), levels, reduction="none"
        )
        if self.importance_weights is not None:
            per_threshold = per_threshold * self.importance_weights.unsqueeze(0)

        per_sample = per_threshold.sum(dim=1)
        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample


def coral_probabilities(logits: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Convert K-1 threshold logits to a proper distribution over K classes.

    Differences the cumulative probabilities. Rank consistency makes the
    differences non-negative in exact arithmetic; floating point can still
    produce values around -1e-8, so the result is clamped and renormalised.
    """
    cumulative = torch.sigmoid(logits.float())          # P(y > k), shape (B, K-1)
    ones = torch.ones(cumulative.shape[0], 1, device=cumulative.device)
    zeros = torch.zeros_like(ones)
    upper = torch.cat([ones, cumulative], dim=1)        # P(y > k-1), k = 0..K-1
    lower = torch.cat([cumulative, zeros], dim=1)       # P(y > k)
    probabilities = (upper - lower).clamp(min=0.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp(min=1e-12)


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """Predicted grade: the number of thresholds whose probability exceeds 0.5.

    This is CORAL's native rule and is *not* the same as the argmax of
    :func:`coral_probabilities`; the two agree in the vast majority of cases but
    can differ on flat distributions. The native rule is used for reported
    predictions, and the distribution is used for calibration and AUROC. Which is
    which is recorded, so results stay comparable.
    """
    return (torch.sigmoid(logits.float()) > 0.5).sum(dim=1)


def describe() -> dict[str, Any]:
    """Method description for the experiment registry."""
    return {
        "objective": "CORAL ordinal regression (Cao et al. 2020)",
        "family": "ordinal",
        "not_to_be_confused_with": "Deep CORAL domain alignment (Sun & Saenko 2016)",
        "head": "shared projection + (K-1) independent biases",
        "prediction_rule": "sum_k 1[sigmoid(logit_k) > 0.5]",
    }
