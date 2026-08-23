"""Training augmentation and deterministic evaluation transforms.

Clinical constraint on augmentation choice
------------------------------------------
DR grading depends on small, low-contrast lesions -- microaneurysms are a few
pixels across at 384px, and the boundary between grade 1 and grade 2 can rest on
whether any are visible.  Aggressive photometric augmentation can erase exactly
that evidence, so the label becomes wrong rather than merely harder.

Therefore:

**Used** -- horizontal and vertical flip (a fundus has no canonical handedness;
left and right eyes are mirror images of each other), small rotation, mild
scale/shift, mild brightness/contrast.

**Deliberately avoided** -- heavy colour jitter and hue shift (they mimic the
inter-camera variation this study is trying to *measure*, so they would confound
the domain-shift analysis), elastic/grid distortion (changes vessel and lesion
morphology), cutout over large areas (can delete the only lesion present),
aggressive blur (destroys microaneurysms).

The vertical flip deserves a note: it is anatomically implausible on its own,
but combined with horizontal flip it gives the four dihedral orientations that a
fundus camera can plausibly produce, and it is standard in the DR literature.

Evaluation transforms are strictly deterministic -- no randomness ever touches
validation or test data, or the calibration measurements would be noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.logging import get_logger
from .preprocessing import IMAGENET_MEAN, IMAGENET_STD

log = get_logger("data.augmentations")

__all__ = ["AugmentationConfig", "build_train_transform", "build_eval_transform"]


@dataclass(frozen=True)
class AugmentationConfig:
    """Augmentation strength. Recorded with every experiment."""

    image_size: int = 224
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    rotation_degrees: int = 15          # small: the disc/macula axis is informative
    scale_limit: float = 0.10
    shift_limit: float = 0.05
    affine_probability: float = 0.5
    brightness_limit: float = 0.15      # mild: lesion contrast must survive
    contrast_limit: float = 0.15
    photometric_probability: float = 0.5
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def describe(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "horizontal_flip": self.horizontal_flip,
            "vertical_flip": self.vertical_flip,
            "rotation_degrees": self.rotation_degrees,
            "scale_limit": self.scale_limit,
            "shift_limit": self.shift_limit,
            "brightness_limit": self.brightness_limit,
            "contrast_limit": self.contrast_limit,
        }


def build_train_transform(config: AugmentationConfig | None = None) -> Any:
    """Stochastic training transform (albumentations), ending in a tensor."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    config = config or AugmentationConfig()
    return A.Compose(
        [
            A.HorizontalFlip(p=config.horizontal_flip),
            A.VerticalFlip(p=config.vertical_flip),
            A.Affine(
                rotate=(-config.rotation_degrees, config.rotation_degrees),
                scale=(1 - config.scale_limit, 1 + config.scale_limit),
                translate_percent=(-config.shift_limit, config.shift_limit),
                p=config.affine_probability,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                p=config.photometric_probability,
            ),
            A.Normalize(mean=config.mean, std=config.std),
            ToTensorV2(),
        ]
    )


def build_eval_transform(config: AugmentationConfig | None = None) -> Any:
    """Deterministic transform for validation, test and calibration.

    Contains no randomness at all: the same image always produces the same
    tensor, so repeated evaluation gives identical metrics.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    config = config or AugmentationConfig()
    return A.Compose([A.Normalize(mean=config.mean, std=config.std), ToTensorV2()])
