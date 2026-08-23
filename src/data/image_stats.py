"""Per-image statistics used for quality control and domain-shift analysis.

Computes geometry (width, height, aspect) from the file header and photometry
(brightness, contrast, per-channel mean and standard deviation) from the pixels.

Two performance decisions matter here, because this runs over thousands of
multi-megapixel scans:

* **Geometry is read from the header.**  ``PIL.Image.open`` is lazy, so image
  size costs no decode.
* **Photometry uses JPEG draft mode.**  ``Image.draft`` lets libjpeg decode at
  1/2, 1/4 or 1/8 scale directly in the DCT domain, which is several times
  faster than a full decode followed by a resize.  Since every statistic here is
  an average over the frame, the reduced resolution changes them only in the
  fourth decimal place.

Photometric statistics are computed **after** the same retina crop the model
sees.  Measuring brightness over the black surround would mostly measure how
much border a dataset ships, which is an artefact, not a domain property.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import ensure_dir
from ..utils.logging import get_logger
from .preprocessing import PreprocessConfig, crop_to_retina

log = get_logger("data.image_stats")

__all__ = ["STAT_COLUMNS", "compute_image_statistics", "sample_manifest", "domain_shift_summary"]

STAT_COLUMNS = [
    "image_id", "domain", "domain_id", "grade", "split",
    "width", "height", "aspect_ratio", "megapixels", "file_kb",
    "brightness", "contrast",
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "readable", "error",
]

# Photometry is computed at roughly this resolution; draft mode picks the
# nearest power-of-two downscale at or above it.
_ANALYSIS_SIZE = 256


def sample_manifest(
    manifest: "Any", *, n_per_domain: int | None = 400, seed: int = 42
) -> "Any":
    """Take a seeded, per-domain sample, stratified by grade where possible."""
    import pandas as pd

    if n_per_domain is None:
        return manifest.copy()

    frames = []
    for _domain, group in manifest.groupby("domain", sort=False):
        if len(group) <= n_per_domain:
            frames.append(group)
            continue
        # Proportional allocation across grades, with at least one per grade
        # present so rare classes are never sampled away entirely.
        picks = []
        remaining = n_per_domain
        grades = sorted(group["grade"].unique())
        for i, grade in enumerate(grades):
            subset = group[group["grade"] == grade]
            share = int(round(n_per_domain * len(subset) / len(group)))
            take = min(len(subset), max(1, share) if i < len(grades) - 1 else remaining)
            take = min(take, remaining)
            picks.append(subset.sample(take, random_state=seed))
            remaining -= take
        frames.append(pd.concat(picks))
    return pd.concat(frames, ignore_index=True)


def _photometry(path: str, config: PreprocessConfig) -> dict[str, float]:
    from PIL import Image

    with Image.open(path) as handle:
        # Fast path for JPEG: decode straight to a reduced scale.
        try:
            handle.draft("RGB", (_ANALYSIS_SIZE, _ANALYSIS_SIZE))
        except (AttributeError, ValueError):
            pass
        array = np.asarray(handle.convert("RGB"))

    if config.crop_to_retina:
        array = crop_to_retina(array, config)

    pixels = array.astype(np.float32)
    grey = pixels @ np.array([0.299, 0.587, 0.114], np.float32)
    return {
        "brightness": float(grey.mean()),
        # RMS contrast: the standard deviation of luminance. Simple, and
        # comparable across images of different size.
        "contrast": float(grey.std()),
        "r_mean": float(pixels[..., 0].mean()),
        "g_mean": float(pixels[..., 1].mean()),
        "b_mean": float(pixels[..., 2].mean()),
        "r_std": float(pixels[..., 0].std()),
        "g_std": float(pixels[..., 1].std()),
        "b_std": float(pixels[..., 2].std()),
    }


def compute_image_statistics(
    manifest: "Any",
    *,
    config: PreprocessConfig | None = None,
    cache_path: Path | str | None = None,
    show_progress: bool = True,
) -> "Any":
    """Compute geometry and photometry for every row of ``manifest``.

    Unreadable files are recorded with ``readable=False`` and the exception
    message, never dropped silently -- a corrupt-image count is itself a
    quality-control result.
    """
    import os

    import pandas as pd
    from PIL import Image

    config = config or PreprocessConfig()

    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_csv(cache_path)
        if set(cached["image_id"]) >= set(manifest["image_id"]):
            log.info("loaded cached image statistics from %s", cache_path)
            return cached[cached["image_id"].isin(set(manifest["image_id"]))].reset_index(drop=True)

    iterator: Any = manifest.itertuples()
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(manifest), desc="image stats", unit="img")
        except ImportError:
            pass

    records: list[dict[str, Any]] = []
    for row in iterator:
        record: dict[str, Any] = {
            "image_id": row.image_id,
            "domain": row.domain,
            "domain_id": int(row.domain_id),
            "grade": int(row.grade),
            "split": getattr(row, "split", pd.NA),
            "readable": True,
            "error": pd.NA,
        }
        try:
            with Image.open(row.path) as handle:
                width, height = handle.size
            record.update(
                width=width,
                height=height,
                aspect_ratio=round(width / height, 4),
                megapixels=round(width * height / 1e6, 3),
                file_kb=round(os.path.getsize(row.path) / 1024, 1),
            )
            record.update(_photometry(row.path, config))
        except Exception as exc:  # noqa: BLE001 - corrupt files are a finding
            record["readable"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"

        records.append(record)

    frame = pd.DataFrame(records)
    for column in STAT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[STAT_COLUMNS]

    n_bad = int((~frame["readable"]).sum())
    log.info("image statistics: %d images, %d unreadable", len(frame), n_bad)

    if cache_path is not None:
        ensure_dir(Path(cache_path).parent)
        frame.to_csv(cache_path, index=False)
        log.info("cached image statistics -> %s", cache_path)
    return frame


def domain_shift_summary(stats: "Any") -> "Any":
    """Per-domain medians of the appearance statistics.

    These are descriptive only.  Differences here are consistent with domain
    shift but do not establish that a model relies on them -- that needs the
    embedding analysis and the actual cross-domain results.
    """
    import pandas as pd

    usable = stats[stats["readable"]]
    columns = [
        "width", "height", "aspect_ratio", "megapixels",
        "brightness", "contrast", "r_mean", "g_mean", "b_mean",
    ]
    summary = usable.groupby("domain")[columns].median().round(2)
    summary.insert(0, "n", usable.groupby("domain").size())
    return summary.reset_index()
