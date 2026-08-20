"""MixStyle (Zhou et al., ICLR 2021) -- domain generalization by mixing feature statistics.

The idea
--------
Instance normalisation statistics -- the per-sample, per-channel mean and
standard deviation of a feature map -- carry most of what makes an image look
like it came from a particular camera.  MixStyle exploits that: during training
it swaps those statistics between samples from *different domains*, so a DDR
image is rendered with APTOS-like low-level statistics while keeping its own
content and its own label.

    x_hat = sigma_mix * (x - mu(x)) / sigma(x) + mu_mix

    mu_mix    = lam * mu(x)    + (1 - lam) * mu(x_perm)
    sigma_mix = lam * sigma(x) + (1 - lam) * sigma(x_perm)
    lam ~ Beta(alpha, alpha)

The network can no longer rely on style cues to predict the grade, because the
style it sees for a given lesion pattern varies from batch to batch.

Why it suits this project
-------------------------
Phase 2 measured exactly the kind of difference MixStyle targets: EyePACS' blue
channel is ~3x APTOS' (51.0 vs 16.6), and the domains differ sharply in
brightness and contrast.  Those are first- and second-order statistics of the
low-level features -- precisely what this mixes.

It is also nearly free: no extra parameters, no extra forward pass, negligible
compute.  On an 8 GB laptop that matters.

Where to insert it
------------------
After the **early** stages only.  Style information lives in low-level features;
deep layers encode semantics, and mixing there corrupts the lesion evidence the
label depends on.  The paper inserts after the first two or three residual
stages, and :func:`insert_mixstyle` follows that.

Two correctness requirements
----------------------------
1. **Training only.** The module is a no-op in ``eval()``. Mixing at test time
   would make predictions depend on which other images happened to share the
   batch -- non-deterministic evaluation, and the calibration numbers would be
   meaningless.
2. **Mix across domains, not within.** Swapping style between two DDR images
   teaches nothing about domain invariance. :meth:`MixStyle.set_domain_ids`
   supplies the batch's domain labels so the permutation pairs different
   domains; without them the module falls back to the paper's random shuffle and
   says so once.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..utils.logging import get_logger

log = get_logger("models.mixstyle")

__all__ = ["MixStyle", "insert_mixstyle", "set_mixstyle_domains", "mixstyle_modules"]


class MixStyle(nn.Module):
    """Mix per-sample feature statistics across domains.

    Parameters
    ----------
    p:
        Probability of applying the mix to a given batch.
    alpha:
        Beta distribution parameter. 0.1 (the paper's default) puts most mass
        near 0 and 1, so a mixed sample usually resembles one of the two styles
        rather than an average of both -- which keeps the mixed images plausible.
    eps:
        Numerical floor inside the square root.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self._beta = torch.distributions.Beta(alpha, alpha)

        self._domain_ids: torch.Tensor | None = None
        self._warned_no_domains = False
        self.n_applied = 0
        self.n_skipped = 0

    def set_domain_ids(self, domain_ids: torch.Tensor | None) -> None:
        """Supply the current batch's domain labels (call before the forward)."""
        self._domain_ids = domain_ids

    def _cross_domain_permutation(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Pair each sample with one from a *different* domain where possible.

        Falls back to a random permutation for samples whose domain is the only
        one present -- that sample then contributes nothing useful, which is
        preferable to skipping the whole batch.
        """
        domain_ids = self._domain_ids
        if domain_ids is None or len(domain_ids) != batch_size:
            if not self._warned_no_domains:
                log.warning(
                    "MixStyle has no domain ids for this batch; falling back to a "
                    "random permutation. Style will sometimes be mixed within a "
                    "domain, which teaches nothing about domain invariance. Call "
                    "set_mixstyle_domains(model, domain_ids) each step."
                )
                self._warned_no_domains = True
            return torch.randperm(batch_size, device=device)

        domain_ids = domain_ids.to(device)
        permutation = torch.arange(batch_size, device=device)
        for index in range(batch_size):
            candidates = torch.nonzero(domain_ids != domain_ids[index], as_tuple=False).flatten()
            if len(candidates):
                choice = torch.randint(len(candidates), (1,), device=device)
                permutation[index] = candidates[choice]
            else:
                permutation[index] = torch.randint(batch_size, (1,), device=device)
        return permutation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No-op at eval time: evaluation must not depend on batch composition.
        if not self.training or self.p == 0.0:
            return x
        if torch.rand(1).item() > self.p:
            self.n_skipped += 1
            return x
        if x.ndim != 4:
            raise ValueError(f"MixStyle expects NCHW features, got {tuple(x.shape)}")

        batch_size = x.shape[0]
        if batch_size < 2:
            self.n_skipped += 1
            return x

        # Instance statistics: mean/std over spatial dims, per sample per channel.
        mean = x.mean(dim=[2, 3], keepdim=True)
        variance = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        std = (variance + self.eps).sqrt()
        normalised = (x - mean) / std

        # detach(): the statistics are treated as a style *signal*, not something
        # to backpropagate through. This follows the reference implementation and
        # keeps the gradient flowing through the content path only.
        mean, std = mean.detach(), std.detach()

        lam = self._beta.sample((batch_size, 1, 1, 1)).to(x.device, x.dtype)
        permutation = self._cross_domain_permutation(batch_size, x.device)

        mean_mixed = mean * lam + mean[permutation] * (1 - lam)
        std_mixed = std * lam + std[permutation] * (1 - lam)

        self.n_applied += 1
        return normalised * std_mixed + mean_mixed

    def statistics(self) -> dict[str, Any]:
        total = self.n_applied + self.n_skipped
        return {
            "p": self.p,
            "alpha": self.alpha,
            "batches_mixed": self.n_applied,
            "batches_skipped": self.n_skipped,
            "applied_fraction": round(self.n_applied / total, 4) if total else 0.0,
        }

    def extra_repr(self) -> str:
        return f"p={self.p}, alpha={self.alpha}"


def _stage_container(backbone: nn.Module) -> tuple[nn.Module, list[str]] | None:
    """Find the sequential stage container of a timm backbone.

    Returns ``(container, ordered child names)``. Different families name this
    differently, hence the search rather than a hard-coded attribute.
    """
    for attribute in ("stages", "layers", "blocks"):
        candidate = getattr(backbone, attribute, None)
        if isinstance(candidate, (nn.Sequential, nn.ModuleList)) and len(candidate) >= 2:
            return candidate, [str(i) for i in range(len(candidate))]

    # DenseNet (timm) exposes `features` as a Sequential of denseblock/transition.
    features = getattr(backbone, "features", None)
    if isinstance(features, nn.Sequential):
        names = [n for n, _ in features.named_children()]
        blocks = [n for n in names if "denseblock" in n or "layer" in n]
        if len(blocks) >= 2:
            return features, blocks
    return None


def insert_mixstyle(
    model: nn.Module,
    *,
    n_stages: int = 2,
    p: float = 0.5,
    alpha: float = 0.1,
) -> list[MixStyle]:
    """Insert MixStyle after the first ``n_stages`` early stages of the backbone.

    Returns the inserted modules (empty if the backbone shape was not
    recognised -- reported, never silent, so a run cannot claim to use MixStyle
    while doing nothing).
    """
    backbone = getattr(model, "backbone", model)
    found = _stage_container(backbone)
    if found is None:
        log.warning(
            "could not locate a stage container in %s; MixStyle NOT inserted. "
            "The run would be plain ERM -- do not label it as MixStyle.",
            type(backbone).__name__,
        )
        return []

    container, names = found
    inserted: list[MixStyle] = []
    targets = names[:n_stages]

    for name in targets:
        module = MixStyle(p=p, alpha=alpha)
        original = getattr(container, name) if hasattr(container, name) else container[int(name)]
        wrapped = nn.Sequential(original, module)
        if hasattr(container, name):
            setattr(container, name, wrapped)
        else:
            container[int(name)] = wrapped
        inserted.append(module)

    log.info(
        "MixStyle inserted after %d early stage(s) of %s (p=%.2f, alpha=%.2f): %s",
        len(inserted), type(backbone).__name__, p, alpha, targets,
    )
    return inserted


def mixstyle_modules(model: nn.Module) -> list[MixStyle]:
    """Every MixStyle module inside a model."""
    return [m for m in model.modules() if isinstance(m, MixStyle)]


def set_mixstyle_domains(model: nn.Module, domain_ids: torch.Tensor | None) -> int:
    """Give every MixStyle module the current batch's domain ids.

    Call once per training step, before the forward pass. Returns how many
    modules were updated, so a caller can assert the wiring actually took.
    """
    modules = mixstyle_modules(model)
    for module in modules:
        module.set_domain_ids(domain_ids)
    return len(modules)
