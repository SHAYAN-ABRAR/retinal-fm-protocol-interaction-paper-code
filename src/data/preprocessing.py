"""Harmonised retinal preprocessing, applied identically to all four domains.

The problem this solves
-----------------------
The four domains arrive in visually incompatible states:

======== ============================= ==================================
Domain   Native size(s)                Framing
======== ============================= ==================================
ddr      512x512 (many) up to 3456px   mixed; some already tightly cropped
aptos    819x614 .. 4288x2848          wide black borders on many images
idrid    4288x2848 (uniform)           wide black borders, off-centre disc
eyepacs  width 1024, varied height     pre-resized by the derivative
======== ============================= ==================================

If that heterogeneity is fed to the network unchanged, part of what the model
learns as "domain" is simply *how much black border each dataset has* -- an
artefact, not a clinical signal.  Since this project's whole subject is domain
shift, removing the trivially removable part of it is essential: what remains
should be genuine acquisition difference (camera, illumination, population),
not framing.

The pipeline
------------
1. **Retina localisation** -- threshold the image and take the bounding box of
   the fundus disc, discarding the black surround.
2. **Square framing** -- expand the tighter axis to a square around the disc
   centre, so the retina keeps its circular shape.  Aspect ratio is never
   distorted by stretching.
3. **Resize** to the working resolution (224 for development, 384 for final
   runs).
4. **Optional circular mask** -- zero the corners outside the fundus.  Off by
   default: it discards no lesion information but does remove a cue the model
   could exploit, so it is an ablation, not an assumption.

All steps are deterministic.  Nothing here is random -- augmentation lives in
``augmentations.py`` and is applied to training data only.

An honest note on how much this actually does
---------------------------------------------
Measured median retained area after cropping: ddr 95.7%, aptos 99.9%,
idrid 79.7%, eyepacs 100.0%.  Only IDRiD carries substantial black border; the
other three ship images that are already close to tightly cropped, and APTOS is
bimodal (most images need no crop, a minority need a large one).  So border
removal is *not* the main lever on cross-domain difference in this corpus -- it
removes one confound cheaply, and the residual shift is the interesting part.
Claiming it harmonises the domains would overstate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logging import get_logger

log = get_logger("data.preprocessing")

__all__ = [
    "PreprocessConfig",
    "estimate_retina_bbox",
    "crop_to_retina",
    "square_pad",
    "resize_image",
    "circular_mask",
    "preprocess_array",
    "load_and_preprocess",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

# Backbones are ImageNet/LVD-pretrained, so their normalisation is used.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    """Deterministic preprocessing settings, recorded with every experiment."""

    image_size: int = 224
    crop_to_retina: bool = True
    # Fraction of the global mean intensity below which a pixel counts as
    # background.
    #
    # 0.20 was chosen from a sweep over 20 images per domain, not by taste.
    # Median retained area (%) against threshold:
    #
    #   thr      0.05   0.10   0.15   0.20   0.25   0.30   0.40   0.50
    #   ddr      96.1   96.0   95.7   95.7   95.6   95.5   95.5   95.5
    #   aptos   100.0  100.0  100.0   99.9   99.6   99.5   99.4   99.3
    #   idrid    97.8   90.4   80.7   79.7   79.6   79.6   79.5   79.5
    #   eyepacs 100.0  100.0  100.0  100.0  100.0  100.0   99.9   99.9
    #
    # Below 0.15 the result swings by 7-10 percentage points because JPEG ringing
    # in IDRiD's black surround clears the cut. From 0.20 onwards every domain is
    # flat to within 0.3 pp per step, so 0.20 is the first point of the stable
    # plateau and has margin on both sides.
    background_threshold: float = 0.20
    # Refuse a crop that keeps less than this fraction of the frame; such a
    # result almost always means the heuristic failed on an unusual image.
    min_retina_fraction: float = 0.05
    apply_circular_mask: bool = False
    pad_value: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "crop_to_retina": self.crop_to_retina,
            "background_threshold": self.background_threshold,
            "apply_circular_mask": self.apply_circular_mask,
        }


def estimate_retina_bbox(
    array: np.ndarray,
    *,
    threshold: float = 0.10,
    min_fraction: float = 0.05,
) -> tuple[int, int, int, int] | None:
    """Bounding box ``(x0, y0, x1, y1)`` of the fundus disc, or None if unclear.

    Uses a per-image relative threshold rather than an absolute one, because the
    four domains differ by more than a factor of two in mean brightness -- a
    fixed cut-off that works on IDRiD erases dark EyePACS images entirely.

    Returning ``None`` rather than a guess is deliberate: the caller then keeps
    the full frame, which is always safe, instead of cropping to nonsense.
    """
    if array.ndim == 3:
        # Luminance weights; the green channel dominates fundus contrast.
        grey = array[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    else:
        grey = array.astype(np.float32)

    cut = grey.mean() * threshold
    mask = grey > cut
    if not mask.any():
        return None

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    height, width = grey.shape[:2]
    if (y1 - y0) * (x1 - x0) < min_fraction * height * width:
        return None
    return x0, y0, x1, y1


def crop_to_retina(array: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Crop away the black surround; return the frame unchanged if unsure."""
    bbox = estimate_retina_bbox(
        array,
        threshold=config.background_threshold,
        min_fraction=config.min_retina_fraction,
    )
    if bbox is None:
        return array
    x0, y0, x1, y1 = bbox
    return array[y0:y1, x0:x1]


def square_pad(array: np.ndarray, *, pad_value: int = 0) -> np.ndarray:
    """Pad the shorter side so the image becomes square.

    Padding rather than stretching: a squashed fundus changes lesion shape and
    the apparent size of the optic disc, both of which carry grading signal.
    """
    height, width = array.shape[:2]
    if height == width:
        return array

    side = max(height, width)
    pad_y, pad_x = side - height, side - width
    padding = [
        (pad_y // 2, pad_y - pad_y // 2),
        (pad_x // 2, pad_x - pad_x // 2),
    ]
    if array.ndim == 3:
        padding.append((0, 0))
    return np.pad(array, padding, mode="constant", constant_values=pad_value)


def resize_image(array: np.ndarray, size: int) -> np.ndarray:
    """Resize to ``size x size`` with a high-quality filter."""
    from PIL import Image

    if array.shape[0] == size and array.shape[1] == size:
        return array
    image = Image.fromarray(array)
    # LANCZOS for downscaling (the usual direction here); BILINEAR when growing.
    resample = Image.LANCZOS if max(array.shape[:2]) > size else Image.BILINEAR
    return np.asarray(image.resize((size, size), resample))


def circular_mask(array: np.ndarray) -> np.ndarray:
    """Zero the corners outside the inscribed circle."""
    height, width = array.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    centre_y, centre_x = (height - 1) / 2.0, (width - 1) / 2.0
    radius = min(height, width) / 2.0
    inside = (yy - centre_y) ** 2 + (xx - centre_x) ** 2 <= radius**2
    out = array.copy()
    out[~inside] = 0
    return out


def preprocess_array(array: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Full deterministic pipeline on an already-loaded RGB array."""
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]

    if config.crop_to_retina:
        array = crop_to_retina(array, config)
    array = square_pad(array, pad_value=config.pad_value)
    array = resize_image(array, config.image_size)
    if config.apply_circular_mask:
        array = circular_mask(array)
    return array


def load_and_preprocess(path: Path | str, config: PreprocessConfig) -> np.ndarray:
    """Read an image from disk and run the deterministic pipeline."""
    from PIL import Image

    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"))
    return preprocess_array(array, config)
