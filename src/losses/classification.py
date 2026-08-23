"""Classification objectives and class-imbalance handling.

The imbalance is severe and differs by domain -- grade 3 is 1.9% of DDR but 18.0%
of IDRiD -- so the handling is made **configurable and comparable**, not decided
in advance.  The brief is explicit that oversampling should not be assumed best,
so this module provides the alternatives on equal footing:

===================== =========================================================
Strategy              What it does
===================== =========================================================
``none``              Plain cross-entropy. The honest baseline.
``weighted_ce``       Per-class weights in the loss. Cheap, no change to the
                      data pipeline, but inflates the gradient of rare classes
                      and can hurt calibration -- which matters here, because
                      calibration is a headline outcome.
``focal``             Down-weights easy examples. Popular for imbalance; also a
                      known source of under-confidence, so it interacts with the
                      calibration analysis and must be reported alongside it.
``balanced_sampling`` Leaves the loss alone and rebalances the sampler instead
                      (see :func:`make_balanced_sampler`). Keeps the loss proper,
                      so predicted probabilities stay interpretable.
===================== =========================================================

A note that matters for this project specifically: **weighting and focal loss
both distort the predicted probability distribution away from the true
conditional.**  A model trained with heavy class weights will look worse on ECE
even if it ranks better. That is a real trade-off, not a bug, and the ablation
must report calibration and discrimination together rather than picking whichever
flatters the method.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.logging import get_logger

log = get_logger("losses.classification")

__all__ = [
    "ImbalanceStrategy",
    "FocalLoss",
    "class_weights_from_counts",
    "build_classification_loss",
    "make_balanced_sampler",
]

ImbalanceStrategy = Literal["none", "weighted_ce", "focal", "balanced_sampling"]


def class_weights_from_counts(
    counts: Sequence[int],
    *,
    scheme: str = "inverse_sqrt",
    normalize: bool = True,
) -> torch.Tensor:
    """Per-class loss weights from training-set counts.

    ``inverse``       w_k = N / (K * n_k)  -- full inverse frequency. On EyePACS,
                      where grade 4 is 2% and grade 0 is 73%, that is a ~36x
                      weight ratio, which destabilises training and badly
                      distorts confidence.
    ``inverse_sqrt``  w_k proportional to 1/sqrt(n_k) -- the default. It halves
                      the exponent, giving a ~6x ratio on the same data: enough
                      to stop the rare grades being ignored, mild enough that
                      predicted probabilities remain usable.
    ``effective``     Cui et al. class-balanced weighting with beta=0.999.
    """
    counts_tensor = torch.as_tensor(list(counts), dtype=torch.float64)
    if (counts_tensor <= 0).any():
        log.warning("class counts contain zeros: %s; clamping to 1", counts)
        counts_tensor = counts_tensor.clamp(min=1.0)

    if scheme == "inverse":
        weights = counts_tensor.sum() / (len(counts_tensor) * counts_tensor)
    elif scheme == "inverse_sqrt":
        weights = 1.0 / counts_tensor.sqrt()
    elif scheme == "effective":
        beta = 0.999
        effective = (1.0 - torch.pow(beta, counts_tensor)) / (1.0 - beta)
        weights = 1.0 / effective
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")

    if normalize:
        # Mean weight 1 keeps the loss on the same scale as unweighted CE, so
        # learning rates transfer between configurations.
        weights = weights / weights.mean()
    return weights.to(torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al.).

    ``loss = -alpha_t (1 - p_t)^gamma log p_t``

    ``gamma=0`` reduces exactly to (weighted) cross-entropy, which makes it a
    clean ablation axis rather than a separate code path.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        # Per-sample cross-entropy, honouring class weights and label smoothing.
        # NOTE: this must be F.cross_entropy, not F.nll_loss -- only the former
        # accepts label_smoothing. Passing it to nll_loss raises TypeError.
        per_sample = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        # p_t from the same logits: one softmax, not two.
        log_pt = F.log_softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        modulation = (1.0 - log_pt.exp()).pow(self.gamma)
        loss = modulation * per_sample

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_classification_loss(
    strategy: ImbalanceStrategy = "none",
    *,
    class_counts: Sequence[int] | None = None,
    weight_scheme: str = "inverse_sqrt",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Return ``(loss_module, description)`` for the chosen strategy.

    The description is written into the experiment registry, so the exact
    weights used are recoverable from the record alone.
    """
    description: dict[str, Any] = {
        "strategy": strategy,
        "label_smoothing": label_smoothing,
    }

    weights = None
    if strategy in {"weighted_ce", "focal"} and class_counts is not None:
        weights = class_weights_from_counts(class_counts, scheme=weight_scheme).to(device)
        description["weight_scheme"] = weight_scheme
        description["class_counts"] = list(class_counts)
        description["class_weights"] = [round(float(w), 4) for w in weights]

    if strategy == "focal":
        description["focal_gamma"] = focal_gamma
        loss: nn.Module = FocalLoss(
            gamma=focal_gamma, weight=weights, label_smoothing=label_smoothing
        )
    elif strategy == "weighted_ce":
        loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    else:
        # 'none' and 'balanced_sampling' both use plain CE; the latter rebalances
        # the sampler instead, which keeps the loss a proper scoring rule.
        loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    log.info("classification loss: %s", description)
    return loss, description


def make_balanced_sampler(
    labels: Sequence[int],
    *,
    num_classes: int = 5,
    scheme: str = "inverse_sqrt",
    generator: torch.Generator | None = None,
) -> Any:
    """A ``WeightedRandomSampler`` that evens out class frequency.

    Sampling with replacement means one epoch no longer equals one pass over the
    data; the epoch length is kept equal to the dataset size so that "epoch"
    remains comparable across strategies.

    Rare grades get repeated within an epoch, which raises the risk of
    memorising them. Early stopping on validation QWK is the guard.
    """
    from torch.utils.data import WeightedRandomSampler

    label_tensor = torch.as_tensor(list(labels), dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=num_classes).clamp(min=1)
    class_weight = class_weights_from_counts(counts.tolist(), scheme=scheme)
    sample_weight = class_weight[label_tensor].double()

    log.info(
        "balanced sampler: counts=%s, class weights=%s",
        counts.tolist(), [round(float(w), 3) for w in class_weight],
    )
    return WeightedRandomSampler(
        weights=sample_weight,
        num_samples=len(label_tensor),
        replacement=True,
        generator=generator,
    )
