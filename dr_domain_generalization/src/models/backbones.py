"""Backbones and classification heads, sized for an 8 GB laptop GPU.

Three models, chosen to span the design space without inflating the study:

============== ============================ =========================================
Name           Model                        Why it is here
============== ============================ =========================================
``densenet121`` DenseNet-121                The classic medical-imaging CNN. Appears in
                                            most DR grading papers, so it anchors the
                                            results to prior work.
``convnext_tiny`` ConvNeXt-Tiny             A modern convolutional design at comparable
                                            cost -- tests whether the classic baseline
                                            is simply outdated.
``dinov2_vits14`` DINOv2 ViT-S/14           A strong self-supervised representation
                                            trained on non-medical data. Interesting
                                            precisely because its features were never
                                            tuned to any fundus camera, which is the
                                            kind of bias domain generalization fights.
============== ============================ =========================================

Two practical constraints shape the code:

* **DINOv2 uses patch 14**, so its input side must be a multiple of 14. 224 works
  (16 patches); 384 does not. :func:`resolve_input_size` snaps to the nearest
  valid size and says so rather than letting the model fail at runtime.
* **8 GB VRAM.** :func:`freeze_backbone` supports the staged DINOv2 recipe from
  the brief -- train the head first, optionally unfreeze the last blocks later --
  and :func:`estimate_memory` measures real peak usage for a given batch size
  instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from ..utils.logging import get_logger

log = get_logger("models.backbones")

__all__ = [
    "BackboneConfig",
    "SUPPORTED_BACKBONES",
    "DRModel",
    "build_model",
    "resolve_input_size",
    "count_parameters",
    "freeze_backbone",
    "estimate_memory",
]

HeadType = Literal["linear", "ordinal_coral"]

# timm model names, patch constraints and the recommended starting recipe.
SUPPORTED_BACKBONES: dict[str, dict[str, Any]] = {
    "densenet121": {
        "timm_name": "densenet121.ra_in1k",
        "patch_multiple": 1,
        "notes": "classic medical-imaging CNN baseline",
    },
    "convnext_tiny": {
        "timm_name": "convnext_tiny.fb_in22k_ft_in1k",
        "patch_multiple": 32,
        "notes": "modern convolutional baseline",
    },
    "dinov2_vits14": {
        "timm_name": "vit_small_patch14_dinov2.lvd142m",
        "patch_multiple": 14,
        "notes": "self-supervised ViT-S/14; freeze most blocks initially",
    },
}


@dataclass
class BackboneConfig:
    """Model configuration. Recorded verbatim in the experiment registry."""

    name: str = "densenet121"
    pretrained: bool = True
    num_classes: int = 5
    head: HeadType = "linear"
    dropout: float = 0.2
    image_size: int = 224
    # Number of trailing blocks left trainable; None means the whole backbone.
    trainable_blocks: int | None = None
    mixstyle: bool = False          # wired in Phase 4
    extra: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "backbone": self.name,
            "pretrained": self.pretrained,
            "num_classes": self.num_classes,
            "head": self.head,
            "dropout": self.dropout,
            "image_size": self.image_size,
            "trainable_blocks": self.trainable_blocks,
            "mixstyle": self.mixstyle,
        }


def resolve_input_size(name: str, requested: int) -> int:
    """Snap ``requested`` to a size the backbone can actually accept.

    DINOv2 is patch-14, so 384 is invalid (384/14 = 27.43). Silently feeding it
    an invalid size produces a shape error deep inside the model; snapping to 378
    and logging the change is far easier to debug -- and, importantly, the change
    is announced so it can be recorded with the experiment.
    """
    if name not in SUPPORTED_BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {sorted(SUPPORTED_BACKBONES)}")
    multiple = SUPPORTED_BACKBONES[name]["patch_multiple"]
    if requested % multiple == 0:
        return requested
    snapped = max(multiple, round(requested / multiple) * multiple)
    log.warning(
        "%s requires an input size divisible by %d; %d -> %d",
        name, multiple, requested, snapped,
    )
    return snapped


class DRModel(nn.Module):
    """Backbone + pooling + head, with a feature hook for the DG methods.

    ``forward`` returns logits.  ``forward_features`` returns the pooled feature
    vector, which Deep CORAL, the embedding figures and the combined method all
    need -- exposing it here means none of them has to monkey-patch the model.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, config: BackboneConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        if config.head == "linear":
            out_features = config.num_classes
        elif config.head == "ordinal_coral":
            # CORAL ordinal regression: K-1 binary "is the grade > k?" tasks with
            # a shared weight vector and independent biases. See
            # src/losses/ordinal_coral_loss.py -- unrelated to Deep CORAL.
            out_features = 1
        else:
            raise ValueError(f"unknown head {config.head!r}")

        self.classifier = nn.Linear(feature_dim, out_features)
        if config.head == "ordinal_coral":
            self.coral_bias = nn.Parameter(torch.zeros(config.num_classes - 1))
        else:
            self.register_parameter("coral_bias", None)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim > 2:                      # safety net for pooled outputs
            features = features.flatten(1)
        return features

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        features = self.forward_features(x)
        hidden = self.dropout(features)
        logits = self.classifier(hidden)
        if self.config.head == "ordinal_coral":
            # Broadcast the shared projection across the K-1 thresholds.
            logits = logits + self.coral_bias
        if return_features:
            return logits, features
        return logits


def build_model(config: BackboneConfig) -> DRModel:
    """Instantiate a backbone via timm and attach the requested head."""
    import timm

    if config.name not in SUPPORTED_BACKBONES:
        raise KeyError(f"unknown backbone {config.name!r}; choose from {sorted(SUPPORTED_BACKBONES)}")
    spec = SUPPORTED_BACKBONES[config.name]

    kwargs: dict[str, Any] = {
        "pretrained": config.pretrained,
        "num_classes": 0,        # timm returns pooled features
    }
    if spec["patch_multiple"] == 14:
        # ViTs need to know the input size so position embeddings are interpolated.
        kwargs["img_size"] = config.image_size

    backbone = timm.create_model(spec["timm_name"], **kwargs)
    feature_dim = int(backbone.num_features)

    model = DRModel(backbone, feature_dim, config)
    if config.trainable_blocks is not None:
        freeze_backbone(model, trainable_blocks=config.trainable_blocks)

    total, trainable = count_parameters(model)
    log.info(
        "%s: %.1fM parameters (%.1fM trainable), feature_dim=%d, head=%s",
        config.name, total / 1e6, trainable / 1e6, feature_dim, config.head,
    )
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model: DRModel, *, trainable_blocks: int = 0) -> tuple[int, int]:
    """Freeze the backbone, leaving the last ``trainable_blocks`` blocks trainable.

    The head is always trainable.  With ``trainable_blocks=0`` this is linear
    probing, which is the recommended first step for DINOv2 on 8 GB: it cuts
    activation memory sharply and avoids destroying a strong pretrained
    representation with a large learning rate.

    Returns ``(total, trainable)`` parameter counts after the change.
    """
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    if trainable_blocks > 0:
        blocks = _find_blocks(model.backbone)
        if not blocks:
            log.warning(
                "could not identify sequential blocks in %s; leaving the whole "
                "backbone frozen", model.config.name,
            )
        else:
            for block in blocks[-trainable_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            log.info("unfroze the last %d of %d backbone blocks", trainable_blocks, len(blocks))

    # Normalisation layers stay trainable only if their block was unfrozen; the
    # head always trains.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if model.coral_bias is not None:
        model.coral_bias.requires_grad = True

    total, trainable = count_parameters(model)
    log.info("after freezing: %.1fM/%.1fM parameters trainable", trainable / 1e6, total / 1e6)
    return total, trainable


def _find_blocks(backbone: nn.Module) -> list[nn.Module]:
    """Locate the repeated block list of a timm backbone, if it has one."""
    for attribute in ("blocks", "stages", "layers"):
        candidate = getattr(backbone, attribute, None)
        if isinstance(candidate, (nn.ModuleList, nn.Sequential)):
            return list(candidate)
    features = getattr(backbone, "features", None)
    if isinstance(features, nn.Sequential):
        return [m for m in features if isinstance(m, nn.Module)]
    return []


@torch.no_grad()
def _forward_only_peak(model: nn.Module, batch: torch.Tensor) -> float:
    torch.cuda.reset_peak_memory_stats()
    model(batch)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


def estimate_memory(
    config: BackboneConfig,
    batch_size: int,
    *,
    device: str = "cuda",
    amp: bool = True,
    train: bool = True,
) -> dict[str, Any]:
    """Measure real peak VRAM for one forward (+backward) pass.

    Guessing batch sizes on an 8 GB card wastes time and produces confusing OOM
    crashes mid-epoch. This runs the actual step once and reports what it cost,
    so the choice recorded in the experiment registry is an observation.

    Returns a dict with ``fits`` False and the error message instead of raising,
    so a caller can sweep candidate batch sizes in a loop.
    """
    if not torch.cuda.is_available():
        return {"fits": False, "error": "CUDA unavailable", "batch_size": batch_size}

    size = resolve_input_size(config.name, config.image_size)
    model = build_model(config).to(device)
    model.train(train)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    result: dict[str, Any] = {"batch_size": batch_size, "image_size": size, "amp": amp}
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = torch.randn(batch_size, 3, size, size, device=device)
        target = torch.randint(0, config.num_classes, (batch_size,), device=device)

        if train:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(batch)
                loss = nn.functional.cross_entropy(logits.float(), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                model(batch)

        torch.cuda.synchronize()
        result["fits"] = True
        result["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        result["reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    except torch.cuda.OutOfMemoryError as exc:
        result["fits"] = False
        result["error"] = f"OOM: {str(exc).splitlines()[0]}"
    finally:
        del model, optimizer
        torch.cuda.empty_cache()

    return result
