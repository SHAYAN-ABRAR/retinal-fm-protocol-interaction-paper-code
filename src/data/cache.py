"""Pre-resized image cache.

Why
---
Decoding the raw corpus is the training bottleneck on this hardware, not the
GPU.  Measured single-image decode-plus-preprocess cost:

    ddr  10 ms | eyepacs  30 ms | aptos 203 ms | idrid 376 ms

A leave-one-domain-out epoch over ~36k source images therefore spends minutes in
libjpeg before the RTX 5060 does any useful work.  Preprocessing is fully
deterministic (``preprocessing.py`` contains no randomness), so it can be done
once and reused for every epoch of every experiment.

What it does
------------
Runs the deterministic pipeline -- retina crop, square pad, resize -- once per
image and writes the result to ``cache_root/<size>/<domain>/<filename>``.  Then
returns a manifest whose ``path`` column points at the cache, so nothing
downstream needs to know the cache exists.

Format
------
JPEG at quality 95 by default.  At 384px that is roughly 30-40 KB per image, so
the whole corpus fits in about 2 GB; lossless PNG would need ~13 GB and gains
little, because the downscale from 4288x2848 to 384x384 already discards far
more information than quality-95 JPEG does.  :func:`measure_cache_fidelity`
quantifies that claim on real images rather than asserting it -- run it before
trusting the default.

Correctness
-----------
The cache is keyed by image size *and* by the preprocessing settings.  Changing
either writes to a different directory, so a stale cache cannot silently feed
mismatched images into an experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import ensure_dir
from ..utils.logging import get_logger
from .preprocessing import PreprocessConfig, load_and_preprocess

log = get_logger("data.cache")

__all__ = [
    "CacheReport",
    "cache_key",
    "build_image_cache",
    "measure_cache_fidelity",
]


@dataclass
class CacheReport:
    cache_dir: str = ""
    n_total: int = 0
    n_written: int = 0
    n_reused: int = 0
    n_failed: int = 0
    failures: list[str] = field(default_factory=list)
    bytes_written: int = 0
    seconds: float = 0.0

    @property
    def megabytes(self) -> float:
        return round(self.bytes_written / 1024**2, 1)


def cache_key(config: PreprocessConfig, *, image_format: str, quality: int) -> str:
    """Stable directory name encoding every setting that changes the pixels.

    Two runs with different preprocessing must never share a cache directory, or
    an experiment could silently train on images produced by settings it did not
    declare.
    """
    payload = {
        **config.describe(),
        "min_retina_fraction": config.min_retina_fraction,
        "pad_value": config.pad_value,
        "format": image_format,
        "quality": quality if image_format.lower() in {"jpg", "jpeg"} else None,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"size{config.image_size}_{digest}"


def _cached_path(cache_dir: Path, row: Any, extension: str) -> Path:
    # image_id is '<domain>/<source_split>/<filename>' and is unique, so it maps
    # to a unique cache path without any risk of collision.
    domain, source_split, filename = row.image_id.split("/", 2)
    stem = Path(filename).stem
    return cache_dir / domain / source_split / f"{stem}{extension}"


def build_image_cache(
    manifest: "Any",
    config: PreprocessConfig,
    cache_root: Path | str,
    *,
    image_format: str = "jpeg",
    quality: int = 95,
    workers: int = 8,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple["Any", CacheReport]:
    """Materialise the preprocessed corpus and return a re-pointed manifest.

    Resumable: images already present are reused unless ``overwrite`` is set, so
    an interrupted run can simply be restarted.
    """
    import time

    import pandas as pd
    from PIL import Image

    extension = ".jpg" if image_format.lower() in {"jpg", "jpeg"} else f".{image_format.lower()}"
    cache_dir = ensure_dir(Path(cache_root) / cache_key(config, image_format=image_format, quality=quality))

    report = CacheReport(cache_dir=str(cache_dir), n_total=len(manifest))
    rows = list(manifest.itertuples())
    targets = [_cached_path(cache_dir, row, extension) for row in rows]
    for directory in {p.parent for p in targets}:
        directory.mkdir(parents=True, exist_ok=True)

    # Record the settings next to the images so a cache directory is always
    # self-describing.
    (cache_dir / "cache_config.json").write_text(
        json.dumps(
            {
                **config.describe(),
                "format": image_format,
                "quality": quality,
                "n_images_expected": len(manifest),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def _process(index: int) -> tuple[int, str | None, int, bool]:
        row, target = rows[index], targets[index]
        if target.exists() and not overwrite:
            try:
                return index, None, target.stat().st_size, False
            except OSError:
                pass
        try:
            array = load_and_preprocess(row.path, config)
            image = Image.fromarray(array)
            if extension == ".jpg":
                image.save(target, format="JPEG", quality=quality, subsampling=0)
            else:
                image.save(target)
            return index, None, target.stat().st_size, True
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            return index, f"{row.image_id}: {type(exc).__name__}: {exc}", 0, False

    start = time.perf_counter()
    iterator: Any = range(len(rows))
    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(rows), desc=f"caching @{config.image_size}px", unit="img")
        except ImportError:
            pass

    ok = np.zeros(len(rows), dtype=bool)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process, i) for i in iterator]
        for future in as_completed(futures):
            index, error, size, written = future.result()
            if error is None:
                ok[index] = True
                report.bytes_written += size
                if written:
                    report.n_written += 1
                else:
                    report.n_reused += 1
            else:
                report.n_failed += 1
                if len(report.failures) < 20:
                    report.failures.append(error)
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()

    report.seconds = round(time.perf_counter() - start, 1)
    log.info(
        "cache %s: %d written, %d reused, %d failed, %.1f MB, %.0fs",
        cache_dir.name, report.n_written, report.n_reused, report.n_failed,
        report.megabytes, report.seconds,
    )
    if report.n_failed:
        log.warning("cache failures (first few): %s", report.failures[:5])

    cached = manifest.copy().reset_index(drop=True)
    cached["path"] = pd.Series([str(p) for p in targets], dtype="string")
    cached = cached[ok].reset_index(drop=True)
    return cached, report


def measure_cache_fidelity(
    manifest: "Any",
    config: PreprocessConfig,
    *,
    n_samples: int = 40,
    quality: int = 95,
    seed: int = 42,
) -> dict[str, float]:
    """Quantify what JPEG-95 caching costs, against a lossless reference.

    Compares the preprocessed array written as quality-``quality`` JPEG and read
    back, with the same array written as lossless PNG and read back.  Reports
    mean absolute error and PSNR in 0-255 units.

    Run this before trusting the default: it converts "JPEG-95 is fine" from an
    assertion into a measurement.
    """
    import io

    from PIL import Image

    rng = np.random.default_rng(seed)
    sample = manifest.iloc[rng.choice(len(manifest), min(n_samples, len(manifest)), replace=False)]

    absolute_errors: list[float] = []
    psnrs: list[float] = []
    jpeg_bytes: list[int] = []
    png_bytes: list[int] = []

    for row in sample.itertuples():
        try:
            reference = load_and_preprocess(row.path, config)
        except Exception:  # noqa: BLE001
            continue

        buffer = io.BytesIO()
        Image.fromarray(reference).save(buffer, format="JPEG", quality=quality, subsampling=0)
        jpeg_bytes.append(buffer.tell())
        buffer.seek(0)
        decoded = np.asarray(Image.open(buffer).convert("RGB")).astype(np.float32)

        png_buffer = io.BytesIO()
        Image.fromarray(reference).save(png_buffer, format="PNG")
        png_bytes.append(png_buffer.tell())

        difference = decoded - reference.astype(np.float32)
        mse = float((difference**2).mean())
        absolute_errors.append(float(np.abs(difference).mean()))
        psnrs.append(float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf"))

    return {
        "n_compared": len(absolute_errors),
        "mean_absolute_error_0_255": round(float(np.mean(absolute_errors)), 4),
        "max_absolute_error_0_255": round(float(np.max(absolute_errors)), 4),
        "psnr_db_mean": round(float(np.mean(psnrs)), 2),
        "psnr_db_min": round(float(np.min(psnrs)), 2),
        "jpeg_kb_mean": round(float(np.mean(jpeg_bytes)) / 1024, 1),
        "png_kb_mean": round(float(np.mean(png_bytes)) / 1024, 1),
        "size_ratio_png_over_jpeg": round(float(np.mean(png_bytes) / np.mean(jpeg_bytes)), 1),
    }
