"""Build a pre-resized image cache at a given resolution.

Everything after Phase 3 reads a cache rather than decoding raw images, because
decoding is the bottleneck: ~203 ms per APTOS image and ~376 ms per IDRiD scan
against ~10 ms for DDR. Preprocessing is deterministic, so it is done once.

The cache directory name encodes the preprocessing settings, so changing the
resolution writes to a **new** directory rather than silently reusing images at
the wrong size. Both caches coexist; nothing is overwritten.

Why 512px matters for this project
----------------------------------
Diabetic retinopathy grading depends on microaneurysms roughly 50-100 microns
across. Downsampled from a ~2000 px fundus photograph to 224 px those features
are close to sub-pixel. The in-domain EyePACS ceiling measured here is 0.709
QWK against roughly 0.85 in published work on the same data, and resolution is
one of two candidate explanations -- the other being the label noise the Phase-1
audit documented. Building this cache is what lets the two be separated.

This is I/O-bound, not GPU-bound, so it can run alongside a training job.

Cost at 512px: ~5.2 GB and roughly an hour for 51,543 images.

Usage:
    python build_cache.py --size 512
    python build_cache.py --size 512 --workers 6     # fewer if the GPU job needs CPU
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

CACHE_ROOT = "D:/DB/_cache"


def main() -> None:
    from pathlib import Path

    from src.data.cache import build_image_cache, measure_cache_fidelity
    from src.data.preprocessing import PreprocessConfig
    from src.data.unified_dataset import load_manifest, save_manifest
    from src.utils.io import project_root
    from src.utils.seed import set_global_seed

    arguments = sys.argv[1:]

    def _take(flag: str, default: str) -> str:
        if flag in arguments:
            index = arguments.index(flag)
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        return default

    size = int(_take("--size", "512"))
    workers = int(_take("--workers", "6"))
    seed = int(_take("--seed", "42"))

    set_global_seed(seed)
    outputs = project_root() / "outputs"

    # The post-deduplication, split-annotated manifest with ORIGINAL paths.
    # Deliberately not manifest_cached_224.csv, whose paths already point into
    # the 224px cache -- caching a cache would bake in the downsampling this
    # run exists to avoid.
    raw_path = outputs / "reports" / "unified_manifest_split.csv"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found; it is written by research_pipeline.py Cell 12.")
    manifest = load_manifest(raw_path)
    print(f"raw manifest: {len(manifest):,} images across "
          f"{manifest['domain'].nunique()} domains")

    config = PreprocessConfig(image_size=size)
    print(f"preprocess: {config.describe()}")

    # What JPEG-95 costs, measured rather than asserted.
    fidelity = measure_cache_fidelity(manifest, config, n_samples=40, quality=95, seed=seed)
    print("\ncache fidelity vs lossless PNG:")
    for key, value in fidelity.items():
        print(f"    {key:28s}: {value}")

    print(f"\nbuilding {size}px cache with {workers} workers (resumable) ...", flush=True)
    cached, report = build_image_cache(
        manifest, config, Path(CACHE_ROOT), workers=workers, show_progress=False
    )

    destination = outputs / "reports" / f"manifest_cached_{size}.csv"
    save_manifest(cached, destination)

    print(f"\ncache: {report.n_written} written, {report.n_reused} reused, "
          f"{report.n_failed} failed, {report.megabytes} MB, {report.seconds}s")
    print(f"cache dir : {report.cache_dir}")
    print(f"manifest  -> {destination}")

    if report.n_failed:
        print(f"\n!! {report.n_failed} image(s) failed to cache. They are absent from "
              "the manifest above, so any run using it trains on fewer images than "
              "the 224px runs. Investigate before comparing resolutions.")


if __name__ == "__main__":
    main()
