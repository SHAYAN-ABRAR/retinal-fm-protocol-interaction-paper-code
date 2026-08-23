"""Assemble a model + loss + trainer hooks for each method under comparison.

One place defines what "ERM", "ordinal", "Deep CORAL" and "MixStyle" mean, so
the ablation compares methods rather than accidental differences in wiring.
Everything else -- optimiser, schedule, seed, splits, preprocessing, augmentation
-- is held identical across methods by construction.

The methods
-----------
======================= =============================================================
``erm``                 Plain 5-class cross-entropy. The baseline everything else
                        must beat.
``ordinal``             CORAL **ordinal regression** head + loss. Exploits the grade
                        ordering. Not a domain method.
``deep_coral``          Deep CORAL covariance alignment across source domains, added
                        to the task loss. Not an ordinal method.
``mixstyle``            MixStyle modules after the early backbone stages.
``mixstyle_ordinal``    MixStyle + ordinal, to test whether the two axes compose.
``deep_coral_ordinal``  Deep CORAL + ordinal.
======================= =============================================================

Calibration is deliberately **not** a method here. Temperature scaling is
post-hoc: it is applied after training in the evaluation step, so it can be
combined with any of the above without retraining. Treating it as a training
variant would triple the run count for nothing.

Note on the two CORALs
----------------------
``ordinal`` uses ``src/losses/ordinal_coral_loss.py`` (Cao et al., label
ordering). ``deep_coral`` uses ``src/losses/deep_coral_alignment.py``
(Sun & Saenko, domain alignment). Same acronym, unrelated methods; they are
separate axes and ``deep_coral_ordinal`` combines them without redundancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch import nn

from ..losses.classification import build_classification_loss
from ..losses.deep_coral_alignment import DeepCoralLoss
from ..losses.ordinal_coral_loss import (
    CoralOrdinalLoss,
    coral_predict,
    coral_probabilities,
    importance_weights_from_counts,
)
from ..models.backbones import BackboneConfig, build_model
from ..models.mixstyle import insert_mixstyle, mixstyle_modules, set_mixstyle_domains
from ..utils.logging import get_logger

log = get_logger("training.methods")

__all__ = ["MethodConfig", "BuiltMethod", "METHODS", "build_method"]

METHODS = (
    "erm",
    "ordinal",
    "deep_coral",
    "mixstyle",
    "mixstyle_ordinal",
    "deep_coral_ordinal",
)


@dataclass
class MethodConfig:
    """Which method to build, and its hyperparameters."""

    name: str = "erm"
    imbalance_strategy: str = "none"
    # Deep CORAL weight. 1.0 is the paper's default starting point; it is a
    # SOURCE-side hyperparameter and must be tuned on source validation only,
    # never on the held-out target.
    coral_lambda: float = 1.0
    mixstyle_p: float = 0.5
    mixstyle_alpha: float = 0.1
    mixstyle_stages: int = 2
    ordinal_importance_weights: bool = False
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in METHODS:
            raise ValueError(f"unknown method {self.name!r}; choose from {list(METHODS)}")

    @property
    def uses_ordinal(self) -> bool:
        return "ordinal" in self.name

    @property
    def uses_deep_coral(self) -> bool:
        return "deep_coral" in self.name

    @property
    def uses_mixstyle(self) -> bool:
        return "mixstyle" in self.name

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.name,
            "imbalance_strategy": self.imbalance_strategy,
            "ordinal": self.uses_ordinal,
            "domain_generalization": (
                "deep_coral" if self.uses_deep_coral
                else "mixstyle" if self.uses_mixstyle
                else "none"
            ),
        }
        if self.uses_deep_coral:
            payload["coral_lambda"] = self.coral_lambda
        if self.uses_mixstyle:
            payload.update(
                mixstyle_p=self.mixstyle_p,
                mixstyle_alpha=self.mixstyle_alpha,
                mixstyle_stages=self.mixstyle_stages,
            )
        return payload


@dataclass
class BuiltMethod:
    """A ready-to-train model plus everything the Trainer needs."""

    model: nn.Module
    loss_fn: nn.Module
    feature_loss: nn.Module | None = None
    batch_hook: Callable[[nn.Module, torch.Tensor], Any] | None = None
    to_probabilities: Callable[[torch.Tensor], torch.Tensor] | None = None
    # The head's own decision rule. None means argmax of the probabilities,
    # which is correct for a softmax head. The ordinal head supplies CORAL's
    # threshold-counting rule instead -- see predict_with_logits for why that
    # distinction is load-bearing for the calibration analysis.
    predict_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    description: dict[str, Any] = field(default_factory=dict)

    def statistics(self) -> dict[str, Any]:
        """Post-training diagnostics: did the DG components actually fire?"""
        stats: dict[str, Any] = {}
        if self.feature_loss is not None and hasattr(self.feature_loss, "statistics"):
            stats["deep_coral"] = self.feature_loss.statistics()
        modules = mixstyle_modules(self.model)
        if modules:
            stats["mixstyle"] = modules[0].statistics()
            stats["mixstyle_n_modules"] = len(modules)
        return stats


def build_method(
    method: MethodConfig,
    backbone: BackboneConfig,
    *,
    class_counts: Sequence[int] | None = None,
    device: str = "cuda",
) -> BuiltMethod:
    """Construct the model, loss and hooks for one method.

    The backbone config is copied, not mutated -- the ordinal variants need
    ``head='ordinal_coral'``, and silently editing the caller's config would make
    a later ERM run in the same session use an ordinal head.
    """
    from dataclasses import replace

    backbone = replace(backbone, head="ordinal_coral" if method.uses_ordinal else "linear")
    model = build_model(backbone)

    description: dict[str, Any] = {**method.describe(), **backbone.describe()}

    # -- task loss --------------------------------------------------------
    if method.uses_ordinal:
        weights = None
        if method.ordinal_importance_weights and class_counts is not None:
            weights = importance_weights_from_counts(class_counts)
            description["ordinal_importance_weights"] = [round(float(w), 4) for w in weights]
        loss_fn: nn.Module = CoralOrdinalLoss(
            num_classes=backbone.num_classes, importance_weights=weights
        )
        description["loss"] = "coral_ordinal"
        # The head emits K-1 cumulative logits, so evaluation must difference them
        # into a distribution rather than softmaxing K-1 numbers.
        to_probabilities: Callable[[torch.Tensor], torch.Tensor] | None = (
            lambda logits: coral_probabilities(logits, backbone.num_classes)
        )
        # CORAL's native rule, and the reason it is used: it is exactly invariant
        # to temperature scaling, so calibration cannot change accuracy or QWK.
        predict_fn: Callable[[torch.Tensor], torch.Tensor] | None = coral_predict
        description["prediction_rule"] = "coral_threshold_count"
    else:
        loss_fn, loss_description = build_classification_loss(
            method.imbalance_strategy,
            class_counts=class_counts,
            label_smoothing=method.label_smoothing,
            device=device,
        )
        description.update(loss_description)
        description["loss"] = "cross_entropy"
        to_probabilities = None
        predict_fn = None
        description["prediction_rule"] = "argmax_softmax"

    # -- domain-generalization components ---------------------------------
    feature_loss: nn.Module | None = None
    if method.uses_deep_coral:
        feature_loss = DeepCoralLoss(weight=method.coral_lambda)
        description["feature_loss"] = "deep_coral"

    batch_hook: Callable[[nn.Module, torch.Tensor], Any] | None = None
    if method.uses_mixstyle:
        inserted = insert_mixstyle(
            model,
            n_stages=method.mixstyle_stages,
            p=method.mixstyle_p,
            alpha=method.mixstyle_alpha,
        )
        if not inserted:
            # Refuse to run a "MixStyle" experiment that is silently plain ERM.
            raise RuntimeError(
                f"MixStyle could not be inserted into {backbone.name}; refusing to "
                "register a run under a method name it does not implement."
            )
        description["mixstyle_modules"] = len(inserted)
        batch_hook = lambda model_, domains: set_mixstyle_domains(model_, domains)  # noqa: E731

    log.info("built method %r: %s", method.name, description)
    return BuiltMethod(
        model=model,
        loss_fn=loss_fn,
        feature_loss=feature_loss,
        batch_hook=batch_hook,
        to_probabilities=to_probabilities,
        predict_fn=predict_fn,
        description=description,
    )
