"""DataLoader construction for one experiment.

Centralised so that every experiment gets identical, correct loader settings --
worker seeding, pinned memory, deterministic evaluation order -- rather than each
cell re-deriving them.

Measured on this machine, reading the pre-resized 224px cache
-------------------------------------------------------------

===============  ==================
Loader           Throughput
===============  ==================
num_workers=0     575.6 images/s
num_workers=2    1342.8 images/s
num_workers=4    2234.3 images/s
===============  ==================

===============  ==================  ==========
Model (batch 16)  GPU step (fwd+bwd)  Peak VRAM
===============  ==================  ==========
densenet121       216.8 images/s      1.13 GB
convnext_tiny     316.6 images/s      1.33 GB
===============  ==================  ==========

**Training is GPU-bound once the cache exists**, by a factor of ~2.7 even with
zero workers.  So ``num_workers`` is not a throughput lever here; it is a
RAM/robustness choice.  The default is 2 -- comfortably ahead of the GPU, half
the memory of 4, and it leaves headroom on a machine that had ~3.5 GB of free
RAM when audited.  Raising it will not make training faster.

(Before the cache, the raw corpus decoded at roughly 5 images/s for APTOS and
2.7 for IDRiD, so the loader *was* the bottleneck by two orders of magnitude.
Building the cache is what moved the bottleneck to the GPU.)

WINDOWS: the ``__main__`` guard is mandatory
--------------------------------------------
Windows has no ``fork``; DataLoader workers are created with ``spawn``, which
**re-imports the main module in each worker**.  A plain script with top-level
training code therefore re-executes itself once per worker -- the symptom is a
run that appears to hang and never reaches the first batch.

* In a VS Code interactive cell / notebook this is fine: ``__main__`` is the
  kernel, and it is not re-imported.
* In a standalone ``.py`` script, all work must sit behind
  ``if __name__ == "__main__":`` whenever ``num_workers > 0``.

:func:`build_loaders` emits a warning when it detects the risky combination.

Other settings
--------------
* ``persistent_workers=True`` when workers > 0, avoiding process startup every
  epoch at the cost of holding worker memory for the whole run.
* ``pin_memory`` only when CUDA is present -- wasted work otherwise.
* ``shuffle=False`` and no ``drop_last`` for evaluation, so every sample is
  scored exactly once and predictions align with the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

log = get_logger("data.loaders")

__all__ = ["LoaderConfig", "build_loaders", "benchmark_loader"]


@dataclass
class LoaderConfig:
    batch_size: int = 16
    eval_batch_size: int | None = None      # defaults to batch_size
    # 2, not 4: measured throughput already exceeds the GPU by ~6x at 2 workers,
    # and 2 halves the RAM cost on a 16 GB machine. See the module docstring.
    num_workers: int = 2
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last_train: bool = True
    balanced_sampling: bool = False
    seed: int = 42

    def describe(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size or self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "balanced_sampling": self.balanced_sampling,
        }


def _warn_if_spawn_unguarded(workers: int) -> None:
    """Warn about the Windows spawn trap before it manifests as a hang.

    Detects the specific dangerous case: workers requested, on Windows, from a
    plain script whose top-level code is not behind a ``__main__`` guard. An
    interactive kernel is fine and is not warned about.
    """
    import platform
    import sys

    if workers <= 0 or platform.system() != "Windows":
        return
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file is None:
        return                      # interactive kernel: spawn re-import is safe
    try:
        source = Path(main_file).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    if "__main__" not in source:
        log.warning(
            "num_workers=%d on Windows from the script %s, which has no "
            '`if __name__ == "__main__":` guard. Workers are created with spawn, '
            "which re-imports that module in every worker, so the script will "
            "re-execute itself and appear to hang. Either add the guard or set "
            "num_workers=0 (throughput is GPU-bound anyway -- see the module "
            "docstring).",
            workers, Path(main_file).name,
        )


def build_loaders(
    experiment: Any,
    *,
    loader_config: LoaderConfig,
    train_transform: Any,
    eval_transform: Any,
    preprocess: Any | None = None,
    expected_size: int | None = None,
    num_classes: int = 5,
) -> dict[str, Any]:
    """Build train/val/test loaders for an :class:`ExperimentSplit`.

    ``preprocess`` is passed straight through to :class:`RetinaDataset`: give it
    a ``PreprocessConfig`` for raw images, or ``None`` for the pre-resized cache.
    """
    import torch
    from torch.utils.data import DataLoader

    from ..utils.seed import seed_worker
    from .unified_dataset import RetinaDataset

    pin_memory = torch.cuda.is_available()
    workers = max(0, loader_config.num_workers)
    _warn_if_spawn_unguarded(workers)
    eval_batch = loader_config.eval_batch_size or loader_config.batch_size

    generator = torch.Generator()
    generator.manual_seed(loader_config.seed)

    common = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": loader_config.persistent_workers and workers > 0,
    }
    if workers > 0:
        common["prefetch_factor"] = loader_config.prefetch_factor

    def _dataset(frame: Any, transform: Any) -> RetinaDataset:
        return RetinaDataset(
            frame, transform=transform, preprocess=preprocess, expected_size=expected_size
        )

    sampler = None
    shuffle = True
    if loader_config.balanced_sampling:
        from ..losses.classification import make_balanced_sampler

        sampler = make_balanced_sampler(
            experiment.train["grade"].astype(int).tolist(),
            num_classes=num_classes,
            generator=generator,
        )
        shuffle = False        # a sampler and shuffle are mutually exclusive

    loaders = {
        "train": DataLoader(
            _dataset(experiment.train, train_transform),
            batch_size=loader_config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=loader_config.drop_last_train,
            worker_init_fn=seed_worker,
            generator=generator,
            **common,
        ),
        "val": DataLoader(
            _dataset(experiment.val, eval_transform),
            batch_size=eval_batch, shuffle=False, drop_last=False, **common,
        ),
        "test": DataLoader(
            _dataset(experiment.test, eval_transform),
            batch_size=eval_batch, shuffle=False, drop_last=False, **common,
        ),
    }

    log.info(
        "loaders for %s: train %d (%d batches), val %d, test %d | %s",
        experiment.name(), len(experiment.train), len(loaders["train"]),
        len(experiment.val), len(experiment.test), loader_config.describe(),
    )
    return loaders


def benchmark_loader(loader: Any, *, n_batches: int = 30, warmup: int = 3) -> dict[str, float]:
    """Measure achievable throughput so worker count is chosen on evidence.

    Reports images/second after a warmup, which excludes worker startup. Compare
    against the GPU's step time: if the loader is slower, the run is I/O bound
    and more workers (or the pre-resized cache) will help; if it is faster, extra
    workers only cost RAM.
    """
    import time

    iterator = iter(loader)
    for _ in range(warmup):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)

    started = time.perf_counter()
    images = 0
    for _ in range(n_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        images += len(batch[1])
    elapsed = time.perf_counter() - started

    result = {
        "batches": n_batches,
        "images": images,
        "seconds": round(elapsed, 2),
        "images_per_second": round(images / elapsed, 1) if elapsed > 0 else float("nan"),
    }
    log.info("loader throughput: %.1f images/s over %d images", result["images_per_second"], images)
    return result
