"""Global seeding and reproducibility bookkeeping.

:func:`set_global_seed` seeds Python, NumPy and (when installed) PyTorch CPU and
CUDA.  PyTorch is imported lazily so that the data-inspection phase of this
project runs in an environment where torch is not installed yet.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = ["set_global_seed", "seed_worker", "environment_fingerprint", "RunFingerprint"]


def set_global_seed(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed every RNG this project uses.

    Parameters
    ----------
    seed:
        Seed value. Recorded in the experiment registry for every run.
    deterministic:
        If True, request deterministic cuDNN / cuBLAS kernels.

        TRADE-OFF: deterministic mode disables cuDNN autotuning and forces
        deterministic algorithm selection.  On this project's convolutional
        backbones that typically costs roughly 10-30 percent throughput, and a
        few operations have no deterministic implementation at all (with
        ``warn_only=True`` they warn and fall back rather than raising).
        Recommended usage: ``deterministic=False`` with a fixed seed for the
        bulk of training -- run-to-run variation is then small but non-zero on
        GPU -- and ``deterministic=True`` for unit tests and for any result that
        must be bit-reproducible.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Needed for deterministic cuBLAS GEMMs. Must be set before the first
        # CUDA context is created to take full effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        # Input sizes are fixed in this project, so autotuning pays off.
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` giving each worker a distinct derived seed."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass
class RunFingerprint:
    """Everything needed to reproduce a run; saved alongside every experiment."""

    timestamp_utc: str
    seed: int
    deterministic: bool
    python_version: str
    platform: str
    packages: dict[str, str] = field(default_factory=dict)
    cuda: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None


_TRACKED_PACKAGES = (
    "numpy", "pandas", "sklearn", "scipy", "torch", "torchvision",
    "timm", "albumentations", "cv2", "matplotlib", "torchmetrics", "umap",
)


def environment_fingerprint(seed: int, *, deterministic: bool = False) -> RunFingerprint:
    """Collect version, hardware and git information for the experiment record."""
    packages: dict[str, str] = {}
    for mod in _TRACKED_PACKAGES:
        try:
            packages[mod] = str(getattr(__import__(mod), "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - module absent or import-time failure
            packages[mod] = "NOT INSTALLED"

    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda["torch_cuda_build"] = torch.version.cuda
        cuda["cudnn"] = torch.backends.cudnn.version()
        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            cuda.update(
                device_name=props.name,
                compute_capability=f"{props.major}.{props.minor}",
                total_memory_gb=round(props.total_memory / 1024**3, 2),
                device_count=torch.cuda.device_count(),
            )
    except ImportError:
        cuda["torch"] = "NOT INSTALLED"

    return RunFingerprint(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
        deterministic=deterministic,
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        packages=packages,
        cuda=cuda,
        git_commit=_git_commit(),
    )
