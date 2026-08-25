"""Group Distributionally Robust Optimisation (Sagawa et al., ICLR 2020).

ERM minimises the *average* loss over the training pool, which lets a model buy
accuracy on the largest source domain at the expense of the smallest. GroupDRO
minimises the loss of the **worst** group instead, approximated online:

    q_g  <-  q_g * exp(eta * L_g)        (then renormalised)
    L    =   sum_g q_g * L_g

Groups here are **source domains**, which is the standard reading for domain
generalization. A domain whose loss stays high accumulates weight, so gradient
steps are steered toward it.

Why this does not fit the trainer's ``feature_loss`` hook
--------------------------------------------------------
Deep CORAL is *additive*: task loss plus an alignment term. GroupDRO is not --
it **replaces** how the per-sample losses are aggregated into a scalar. Adding a
term cannot express a reweighted mean, so this is wired in through the trainer's
``objective_fn`` hook, which takes ``(logits, targets, domains)`` and returns the
training loss outright.

Deviation from the paper, and why
---------------------------------
The paper assumes every group appears in every minibatch, because it samples a
fixed number of examples per group. This project's loaders shuffle the pooled
source data, so a batch contains whatever the shuffle produced. Two consequences
are handled explicitly rather than silently:

* ``q`` is indexed by the project's **fixed** domain id (0=DDR, 1=APTOS,
  2=IDRiD, 3=EyePACS), not by position within the batch. A domain absent from
  this batch keeps the weight it had, rather than having its weight quietly
  reassigned to whichever domain happened to land in that slot.
* The weighted sum is renormalised over the domains **present**, so a batch
  missing a heavily-weighted domain does not silently shrink the loss scale.

The cost is that a domain contributing one image to a batch has its loss
estimated from that one image, and the exponential update amplifies that noise.
For the EyePACS-target pool IDRiD supplies 0.9 images per batch of 32, so this
is not hypothetical. :meth:`statistics` records per-domain participation and the
final weights so the effect is visible in the registry instead of being assumed
away. A domain-balanced sampler is the principled fix; see
``domain_balanced_batch_indices`` in ``deep_coral_alignment``.

Samples are never dropped for being in a thin group: that would change the
effective training set relative to the ERM baseline, which is a worse confound
than a noisy weight.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.logging import get_logger

log = get_logger("losses.group_dro")

__all__ = ["GroupDROObjective", "DEFAULT_ETA"]

# Sagawa et al. use 0.01 for the group-weight step size, as does DomainBed.
DEFAULT_ETA = 0.01

# The protocol fixes four domains; q is indexed by domain id directly.
N_DOMAINS = 4


class GroupDROObjective(nn.Module):
    """Worst-group loss over source domains, with online group weights.

    Call as ``objective(logits, targets, domains) -> loss``.
    """

    def __init__(
        self,
        eta: float = DEFAULT_ETA,
        n_domains: int = N_DOMAINS,
        *,
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta!r}")
        if n_domains < 2:
            raise ValueError("GroupDRO needs at least two groups to be meaningful")
        self.eta = float(eta)
        self.n_domains = int(n_domains)
        self.label_smoothing = float(label_smoothing)
        # A buffer, not a plain tensor: it moves with .to(device) and is written
        # into the checkpoint, so a resumed run continues with the weights it had
        # rather than silently restarting from uniform.
        self.register_buffer("q", torch.ones(n_domains) / n_domains)
        self.register_buffer("class_weight", class_weight
                             if class_weight is not None else torch.empty(0))
        self.register_buffer("_seen", torch.zeros(n_domains, dtype=torch.long))
        self.register_buffer("_thin", torch.zeros(n_domains, dtype=torch.long))
        self._batches = 0

    def _per_sample_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.class_weight if self.class_weight.numel() else None
        return F.cross_entropy(
            logits, targets, weight=weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, domains: torch.Tensor
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must be (batch, classes); got {tuple(logits.shape)}")
        if not (len(logits) == len(targets) == len(domains)):
            raise ValueError("logits, targets and domains must have the same length")

        self._batches += 1
        per_sample = self._per_sample_loss(logits, targets)
        domains = domains.long()

        present = torch.unique(domains)
        group_losses = []
        for domain in present:
            mask = domains == domain
            count = int(mask.sum())
            self._seen[domain] += 1
            if count < 2:
                self._thin[domain] += 1
            group_losses.append(per_sample[mask].mean())
        losses = torch.stack(group_losses)

        # The weight update is on detached losses: q is a state variable of the
        # optimisation, not something the network differentiates through.
        with torch.no_grad():
            updated = self.q[present] * torch.exp(self.eta * losses.detach())
            self.q[present] = updated
            total = self.q.sum()
            if total > 0:
                self.q /= total

        weights = self.q[present]
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
        return (weights * losses).sum()

    def statistics(self) -> dict[str, Any]:
        """Diagnostics: where the weight went, and how thin the groups were."""
        seen = self._seen.tolist()
        return {
            "eta": self.eta,
            "batches_seen": self._batches,
            "final_group_weights": [round(float(w), 4) for w in self.q.tolist()],
            "batches_containing_domain": seen,
            "batches_with_single_sample": self._thin.tolist(),
            # The fraction of a domain's appearances that were a single image.
            # High values mean its loss estimate -- and so its weight -- is noise.
            "single_sample_fraction": [
                round(thin / s, 4) if s else 0.0
                for thin, s in zip(self._thin.tolist(), seen)
            ],
        }
