"""Invariant Risk Minimisation, IRMv1 (Arjovsky et al., 2019).

The idea: a predictor is *invariant* if the same classifier is simultaneously
optimal on every training domain. IRM penalises departures from that. IRMv1 is
the tractable surrogate the paper actually recommends -- for each domain, the
gradient of that domain's loss with respect to a **dummy scalar multiplier** on
the logits is computed, and the squared norm of that gradient becomes the
penalty:

    L = mean_d NLL_d  +  lambda * mean_d || grad_w NLL_d(w * logits) ||^2   at w = 1

If one classifier were optimal for every domain, each of those gradients would
be zero. The dummy multiplier is the trick that makes this a scalar derivative
rather than a full-parameter one, which is what makes it affordable.

Why this does not fit the trainer's ``feature_loss`` hook
--------------------------------------------------------
The penalty is a function of **logits**, not of pooled features, and it needs
the targets as well. The ``feature_loss`` hook supplies ``(features, domains)``
only. IRM therefore uses the trainer's ``objective_fn`` hook, which receives
``(logits, targets, domains)`` and returns the training loss.

The annealing, and why it is not optional
-----------------------------------------
With a large ``lambda`` from step zero the penalty dominates and the network
learns nothing -- it finds a degenerate predictor whose per-domain gradients are
all zero because it predicts nothing useful. The paper, and DomainBed after it,
anneal: train with ``lambda = 1`` for the first ``anneal_iters`` steps, then
switch to the full value. After the switch the whole objective is divided by
``lambda`` to keep the gradient scale comparable to the pre-switch phase, which
otherwise jumps by two orders of magnitude and destabilises the optimiser.

Practical requirement
---------------------
The penalty splits each domain's samples into two halves and differentiates each
separately, so a domain needs **at least two samples in the batch** to
contribute. Domains with fewer are skipped for the penalty -- their samples
still count toward the NLL term, so no data is discarded. :meth:`statistics`
records how often that happened, because on this project's pools it happens a
lot: in the EyePACS-target split IDRiD supplies 0.9 images per batch of 32, so
its invariance constraint is enforced on a minority of steps. That is a real
limitation of running IRM on naturally-imbalanced source pools and is reported
rather than hidden.

What this can and cannot do
---------------------------
IRM is included because it is the most-cited invariance method and any claim
that ERM is not beaten must have faced it. It is also known to be fragile:
Rosenfeld et al. (2021) show IRMv1 can fail to recover the invariant predictor
even when one exists, and DomainBed finds it does not beat ERM under a fair
sweep. A negative result here is the expected outcome, not a surprising one.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import autograd, nn

from ..utils.logging import get_logger

log = get_logger("losses.irm")

__all__ = ["IRMObjective", "irm_penalty", "DEFAULT_LAMBDA", "DEFAULT_ANNEAL_ITERS"]

# DomainBed's defaults for the non-swept configuration.
DEFAULT_LAMBDA = 100.0
DEFAULT_ANNEAL_ITERS = 500

N_DOMAINS = 4


def irm_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """IRMv1 penalty for one domain's samples.

    The two halves are differentiated separately and their gradients multiplied,
    which is the paper's unbiased estimator of the squared gradient norm: using
    the same samples twice would square a biased estimate instead.

    **This value can be negative.** It is an inner product of two half-batch
    gradients, so a single draw carries the sign of their disagreement; only its
    expectation over orderings is a squared norm. That is not a bug and must not
    be "fixed" by taking an absolute value or clamping at zero -- doing so
    reintroduces exactly the bias the two-half construction exists to remove. It
    does mean the reported ``last_penalty`` in :meth:`IRMObjective.statistics`
    is occasionally negative, and that a batch whose halves happen to disagree
    pushes the loss down rather than up.
    """
    if len(logits) < 2:
        raise ValueError("the IRMv1 penalty needs at least two samples")
    scale = torch.ones(1, device=logits.device, dtype=logits.dtype, requires_grad=True)

    def half(start: int) -> torch.Tensor:
        return F.cross_entropy(
            logits[start::2] * scale, targets[start::2],
            weight=class_weight, label_smoothing=label_smoothing,
        )

    grad_even = autograd.grad(half(0), [scale], create_graph=True)[0]
    grad_odd = autograd.grad(half(1), [scale], create_graph=True)[0]
    return (grad_even * grad_odd).sum()


class IRMObjective(nn.Module):
    """IRMv1 objective with the standard penalty annealing.

    Call as ``objective(logits, targets, domains) -> loss``.
    """

    def __init__(
        self,
        penalty_weight: float = DEFAULT_LAMBDA,
        anneal_iters: int = DEFAULT_ANNEAL_ITERS,
        n_domains: int = N_DOMAINS,
        *,
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if penalty_weight < 0:
            raise ValueError("penalty_weight (lambda) must be non-negative")
        if anneal_iters < 0:
            raise ValueError("anneal_iters must be non-negative")
        self.penalty_weight = float(penalty_weight)
        self.anneal_iters = int(anneal_iters)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("class_weight", class_weight
                             if class_weight is not None else torch.empty(0))
        # A buffer so a resumed run continues past the anneal point rather than
        # restarting the schedule and re-entering the lambda = 1 phase.
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_skipped", torch.zeros(n_domains, dtype=torch.long))
        self.register_buffer("_seen", torch.zeros(n_domains, dtype=torch.long))
        self._last_penalty = 0.0

    @property
    def current_weight(self) -> float:
        """lambda for this step: 1.0 while annealing, the full value after."""
        return 1.0 if int(self.step) < self.anneal_iters else self.penalty_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, domains: torch.Tensor
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must be (batch, classes); got {tuple(logits.shape)}")
        if not (len(logits) == len(targets) == len(domains)):
            raise ValueError("logits, targets and domains must have the same length")

        weight = self.class_weight if self.class_weight.numel() else None
        domains = domains.long()

        nll = F.cross_entropy(
            logits, targets, weight=weight,
            label_smoothing=self.label_smoothing,
        )

        penalties = []
        for domain in torch.unique(domains):
            mask = domains == domain
            self._seen[domain] += 1
            if int(mask.sum()) < 2:
                # Its samples still count in the NLL above; only the invariance
                # constraint is skipped for this step.
                self._skipped[domain] += 1
                continue
            penalties.append(irm_penalty(
                logits[mask], targets[mask],
                class_weight=weight, label_smoothing=self.label_smoothing,
            ))

        if not penalties:
            self.step += 1
            self._last_penalty = 0.0
            return nll

        penalty = torch.stack(penalties).mean()
        self._last_penalty = float(penalty.detach())

        lambda_now = self.current_weight
        loss = nll + lambda_now * penalty
        if lambda_now > 1.0:
            # Rescale so the switch at anneal_iters does not multiply the
            # gradient magnitude by lambda overnight.
            loss = loss / lambda_now
        self.step += 1
        return loss

    def statistics(self) -> dict[str, Any]:
        seen = self._seen.tolist()
        return {
            "penalty_weight": self.penalty_weight,
            "anneal_iters": self.anneal_iters,
            "steps_taken": int(self.step),
            "annealing_finished": int(self.step) >= self.anneal_iters,
            "last_penalty": round(self._last_penalty, 6),
            "batches_containing_domain": seen,
            "batches_skipped_for_penalty": self._skipped.tolist(),
            # How often a domain was present but too thin to constrain. High
            # values mean its invariance constraint was rarely enforced.
            "skipped_fraction": [
                round(s / n, 4) if n else 0.0
                for s, n in zip(self._skipped.tolist(), seen)
            ],
        }
