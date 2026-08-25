"""Deep CORAL -- **CORrelation ALignment** for domain generalization
(Sun & Saenko, 2016).

    !! NAME COLLISION WARNING !!

    * **THIS FILE** -- Deep CORAL: a *domain* loss. It matches the second-order
      feature statistics (covariance matrices) of different source domains, so
      the learned representation stops encoding "which dataset is this?".
      It has NOTHING to do with ordinal labels.

    * ``ordinal_coral_loss.py`` -- CORAL *ordinal regression* (Cao et al. 2020):
      a *label* loss that exploits grade ordering. It has NOTHING to do with
      domains.

    Same acronym, unrelated papers, different problems. They are separate axes
    in the ablation and can be combined; combining them is not redundant.

The idea
--------
If a representation is domain-invariant, features extracted from DDR and from
APTOS should have similar distributions. Deep CORAL approximates "similar
distribution" by "similar covariance", which is cheap and differentiable:

    L_CORAL = (1 / (4 d^2)) * || C_source_i - C_source_j ||_F^2

where ``C`` is the feature covariance of one domain within the batch and ``d``
is the feature dimension. The ``1/(4 d^2)`` normalisation is from the paper and
keeps the loss on a scale that does not depend on feature width -- important
here, because the three backbones have d = 1024, 768 and 384.

Total objective::

    L = L_task + lambda * L_CORAL

What it can and cannot do
-------------------------
Deep CORAL only matches **second-order** statistics. Two distributions can share
a mean and covariance and still differ substantially, so alignment is necessary
but not sufficient for invariance. It is included because it is a standard,
well-cited baseline that must be beaten before any new method is proposed -- not
because it is expected to solve domain shift.

Practical requirement
---------------------
Covariance is estimated **within the batch, per domain**, so a batch must contain
at least two domains with at least two samples each. With domains of very
different size (EyePACS 35k vs IDRiD 507) ordinary shuffling produces batches
that are almost entirely EyePACS, and the loss becomes noise computed from two
or three samples. :func:`domain_balanced_batch_indices` and the guard inside
:class:`DeepCoralLoss` address this explicitly rather than letting it degrade
silently.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from ..utils.logging import get_logger

log = get_logger("losses.deep_coral")

__all__ = [
    "covariance",
    "coral_distance",
    "DeepCoralLoss",
    "domain_balanced_batch_indices",
]

# Fewer samples than this in a domain makes its covariance estimate meaningless.
MIN_SAMPLES_PER_DOMAIN = 4


def covariance(features: torch.Tensor) -> torch.Tensor:
    """Unbiased feature covariance of one domain's samples, shape ``(d, d)``."""
    n = features.shape[0]
    if n < 2:
        raise ValueError("covariance needs at least 2 samples")
    centred = features - features.mean(dim=0, keepdim=True)
    return centred.t() @ centred / (n - 1)


def coral_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius distance between two covariances, paper-normalised."""
    d = source.shape[0]
    difference = source - target
    return (difference * difference).sum() / (4.0 * d * d)


class DeepCoralLoss(nn.Module):
    """Pairwise covariance alignment across the source domains in a batch.

    Averaged over all domain pairs present, so the value does not grow with the
    number of source domains -- otherwise a 3-source LODO run would need a
    different ``lambda`` from a 2-source one.

    Returns exactly 0 (with grad) when fewer than two domains are usable, so a
    degenerate batch contributes nothing instead of injecting noise.
    """

    def __init__(
        self,
        weight: float = 1.0,
        min_samples_per_domain: int = MIN_SAMPLES_PER_DOMAIN,
        warn_on_degenerate: bool = True,
    ) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError("weight (lambda) must be non-negative")
        self.weight = weight
        self.min_samples_per_domain = min_samples_per_domain
        self.warn_on_degenerate = warn_on_degenerate
        self._degenerate_batches = 0
        self._total_batches = 0

    def forward(self, features: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must be (batch, d); got {tuple(features.shape)}")
        features = features.float()
        self._total_batches += 1

        usable = []
        for domain in torch.unique(domain_ids):
            mask = domain_ids == domain
            if int(mask.sum()) >= self.min_samples_per_domain:
                usable.append(covariance(features[mask]))

        if len(usable) < 2:
            self._degenerate_batches += 1
            if self.warn_on_degenerate and self._degenerate_batches in (1, 10, 100, 1000):
                log.warning(
                    "Deep CORAL: %d/%d batches had fewer than two domains with >= %d "
                    "samples, so the alignment term was zero for them. Use a "
                    "domain-balanced sampler -- see domain_balanced_batch_indices().",
                    self._degenerate_batches, self._total_batches,
                    self.min_samples_per_domain,
                )
            # Keep it attached to the graph so the caller can always add it.
            return features.sum() * 0.0

        distances = [
            coral_distance(usable[i], usable[j])
            for i in range(len(usable))
            for j in range(i + 1, len(usable))
        ]
        return self.weight * torch.stack(distances).mean()

    def statistics(self) -> dict[str, Any]:
        """Diagnostics worth recording: how often the term was actually active."""
        return {
            "lambda": self.weight,
            "min_samples_per_domain": self.min_samples_per_domain,
            "batches_seen": self._total_batches,
            "degenerate_batches": self._degenerate_batches,
            "degenerate_fraction": (
                round(self._degenerate_batches / self._total_batches, 4)
                if self._total_batches else 0.0
            ),
        }


def domain_balanced_batch_indices(
    domain_ids: Sequence[int],
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    drop_last: bool = True,
    epoch_length: str = "largest_domain",
) -> list[list[int]]:
    """Batch indices holding an equal number of samples from each domain.

    Deep CORAL and MixStyle both need several domains per batch. Ordinary
    shuffling cannot guarantee that when domain sizes differ by 70x (EyePACS
    35,108 vs IDRiD 507): most batches would be pure EyePACS.

    This draws ``batch_size // n_domains`` samples from each domain per batch,
    cycling through the smaller domains repeatedly within an epoch. That means
    small domains are oversampled relative to natural frequency -- a real
    trade-off, recorded in the experiment config, not hidden: it changes the
    effective class prior as well as the domain prior.

    ``epoch_length`` decides how many batches make an epoch, and the choice is
    load-bearing when this sampler is compared against ordinary shuffling:

    ``"largest_domain"``
        Enough batches to pass through the largest domain once. On the
        EyePACS-target pool that is 869 batches of 30 = 26,070 samples against a
        natural epoch of 11,841 -- so the run takes 2.2x as long **and takes
        2.2x as many optimisation steps**. A method trained that way beating a
        baseline trained the ordinary way would be confounded with having simply
        trained longer.

    ``"natural"``
        ``len(domain_ids) // batch_size`` batches, matching what ordinary
        shuffling would give. Every batch still contains every domain; the small
        domains simply cycle fewer times. This is the setting to use when the
        balanced and unbalanced arms must be comparable, which is the reason the
        option exists.
    """
    tensor = torch.as_tensor(list(domain_ids))
    domains = torch.unique(tensor).tolist()
    per_domain = batch_size // len(domains)
    if per_domain < 1:
        raise ValueError(
            f"batch_size {batch_size} cannot cover {len(domains)} domains; "
            f"use at least {len(domains)}"
        )

    pools: dict[int, list[int]] = {}
    for domain in domains:
        indices = torch.nonzero(tensor == domain, as_tuple=False).flatten()
        order = torch.randperm(len(indices), generator=generator)
        pools[domain] = indices[order].tolist()

    if epoch_length == "largest_domain":
        # No domain is truncated, at the cost of a longer epoch.
        n_batches = max(len(pool) for pool in pools.values()) // per_domain
    elif epoch_length == "natural":
        # Same number of steps an ordinary shuffled loader would take, so the
        # two samplers can be compared without a training-budget confound.
        n_batches = len(tensor) // batch_size
    else:
        raise ValueError(
            f"epoch_length must be 'largest_domain' or 'natural', got {epoch_length!r}")
    cursors = {domain: 0 for domain in domains}

    batches: list[list[int]] = []
    for _ in range(n_batches):
        batch: list[int] = []
        for domain in domains:
            pool = pools[domain]
            for _ in range(per_domain):
                if cursors[domain] >= len(pool):
                    cursors[domain] = 0          # cycle the smaller domains
                batch.append(pool[cursors[domain]])
                cursors[domain] += 1
        batches.append(batch)

    if not drop_last and n_batches == 0:
        batches.append([i for pool in pools.values() for i in pool])
    return batches


def describe() -> dict[str, Any]:
    """Method description for the experiment registry."""
    return {
        "objective": "Deep CORAL covariance alignment (Sun & Saenko 2016)",
        "family": "domain_generalization",
        "not_to_be_confused_with": "CORAL ordinal regression (Cao et al. 2020)",
        "aligns": "second-order feature statistics across source domains",
        "limitation": "matches covariance only; equal covariance does not imply equal distribution",
    }
