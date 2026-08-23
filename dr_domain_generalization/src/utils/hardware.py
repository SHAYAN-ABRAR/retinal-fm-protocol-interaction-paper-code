"""CUDA / CPU / RAM diagnostics.

The target machine is an RTX 5060 Laptop GPU (Blackwell, compute capability
sm_120) with 8 GB of VRAM.  Two failure modes are common enough on that card to
be worth detecting explicitly rather than letting them surface as a confusing
runtime error:

1. **A non-CUDA or too-old PyTorch wheel.**  ``pip install torch`` without an
   index URL installs a CPU-only build on Windows.  A cu126-or-older build
   *imports* fine and reports ``cuda.is_available() == True``, then fails on the
   first kernel launch with "no kernel image is available for execution on the
   device", because sm_120 support starts with the CUDA 12.8 builds.
   :func:`assert_cuda_ready` catches both by actually launching a kernel.

2. **Silently falling back to CPU.**  Never acceptable here -- the study is not
   feasible on CPU -- so the check raises instead of warning.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GpuReport", "cuda_report", "assert_cuda_ready", "system_report", "format_report"]

# sm_120 (Blackwell consumer) needs a CUDA >= 12.8 PyTorch build.
_MIN_CUDA_FOR_BLACKWELL = (12, 8)


@dataclass
class GpuReport:
    torch_version: str | None = None
    torch_cuda_build: str | None = None
    cudnn_version: int | None = None
    cuda_available: bool = False
    device_count: int = 0
    device_name: str | None = None
    compute_capability: str | None = None
    total_vram_gb: float | None = None
    driver_version: str | None = None
    driver_cuda_version: str | None = None
    matmul_smoke_test: str = "NOT RUN"
    problems: list[str] = field(default_factory=list)


def _nvidia_smi() -> dict[str, str]:
    """Query driver information. Returns an empty dict if nvidia-smi is absent."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    if not line:
        return {}
    parts = [p.strip() for p in line.split(",")]
    info = {"name": parts[0]}
    if len(parts) > 1:
        info["driver_version"] = parts[1]
    if len(parts) > 2:
        info["memory_total"] = parts[2]

    # The CUDA version in the nvidia-smi banner is the driver's maximum
    # supported runtime, which is not the version PyTorch was built against.
    try:
        banner = subprocess.run([exe], capture_output=True, text=True, timeout=20, check=False)
        for token in banner.stdout.split("\n")[:4]:
            if "CUDA Version:" in token:
                info["driver_cuda_version"] = token.split("CUDA Version:")[1].split("|")[0].strip()
                break
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def cuda_report(*, run_smoke_test: bool = True) -> GpuReport:
    """Collect GPU diagnostics, optionally launching a real CUDA kernel."""
    report = GpuReport()
    smi = _nvidia_smi()
    report.driver_version = smi.get("driver_version")
    report.driver_cuda_version = smi.get("driver_cuda_version")

    try:
        import torch
    except ImportError:
        report.problems.append(
            "PyTorch is not installed. Install the CUDA 12.8 build:\n"
            "    pip install torch==2.9.1 torchvision==0.24.1 "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
        return report

    report.torch_version = torch.__version__
    report.torch_cuda_build = torch.version.cuda
    report.cuda_available = bool(torch.cuda.is_available())

    if report.torch_cuda_build is None:
        report.problems.append(
            "This is a CPU-only PyTorch build (torch.version.cuda is None). "
            "Reinstall from the cu128 index -- see README, 'Environment setup'."
        )
        return report

    if not report.cuda_available:
        report.problems.append(
            "torch.cuda.is_available() is False. Check that the NVIDIA driver is "
            "loaded (run nvidia-smi) and that CUDA_VISIBLE_DEVICES is not set to ''."
        )
        return report

    report.cudnn_version = torch.backends.cudnn.version()
    report.device_count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report.device_name = props.name
    report.compute_capability = f"{props.major}.{props.minor}"
    report.total_vram_gb = round(props.total_memory / 1024**3, 2)

    # Blackwell / cu128 compatibility check, done before the smoke test so the
    # error message is actionable rather than a raw CUDA fault.
    if props.major >= 12:
        build = tuple(int(x) for x in (report.torch_cuda_build or "0.0").split(".")[:2])
        if build < _MIN_CUDA_FOR_BLACKWELL:
            report.problems.append(
                f"GPU is compute capability sm_{props.major}{props.minor} (Blackwell) but "
                f"PyTorch was built against CUDA {report.torch_cuda_build}. "
                f"sm_120 kernels require CUDA >= 12.8. Reinstall from the cu128 index."
            )

    if run_smoke_test:
        try:
            a = torch.randn(512, 512, device="cuda", dtype=torch.float32)
            value = float((a @ a.T).sum().item())
            torch.cuda.synchronize()
            report.matmul_smoke_test = f"PASS (512x512 matmul, checksum {value:.3e})"
            del a
        except Exception as exc:  # noqa: BLE001 - surface any CUDA fault verbatim
            report.matmul_smoke_test = f"FAIL: {type(exc).__name__}: {exc}"
            report.problems.append(
                "A CUDA kernel launch failed. This is the classic symptom of a "
                "PyTorch build that lacks kernels for this GPU architecture."
            )

    return report


def assert_cuda_ready() -> GpuReport:
    """Return the GPU report, raising if the GPU is not usable.

    Training on CPU is not a supported fallback for this project, so failures are
    loud by design.
    """
    report = cuda_report(run_smoke_test=True)
    if report.problems or not report.cuda_available:
        raise RuntimeError(
            "CUDA is not usable; refusing to fall back to CPU.\n\n"
            + "\n\n".join(report.problems or ["torch.cuda.is_available() is False"])
        )
    return report


def system_report() -> dict[str, Any]:
    """CPU / RAM / disk information, using only the standard library."""
    info: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cores": os.cpu_count(),
    }
    try:
        usage = shutil.disk_usage(os.path.abspath(os.sep))
        info["disk_free_gb"] = round(usage.free / 1024**3, 1)
        info["disk_total_gb"] = round(usage.total / 1024**3, 1)
    except OSError:
        pass

    # Total physical RAM. psutil is not a dependency, so use the Win32 API.
    try:
        if platform.system() == "Windows":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            info["ram_total_gb"] = round(status.ullTotalPhys / 1024**3, 1)
            info["ram_available_gb"] = round(status.ullAvailPhys / 1024**3, 1)
            info["ram_used_percent"] = status.dwMemoryLoad
        else:
            info["ram_total_gb"] = round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1
            )
    except Exception:  # noqa: BLE001 - diagnostics must never break the pipeline
        info["ram_total_gb"] = "unavailable"

    return info


def format_report(gpu: GpuReport, system: dict[str, Any]) -> str:
    """Render a human-readable diagnostics block for the notebook cell."""
    lines = ["=" * 68, "HARDWARE DIAGNOSTICS", "=" * 68, "", "-- System --"]
    for key, value in system.items():
        lines.append(f"  {key:22s}: {value}")

    lines += ["", "-- GPU --"]
    for key in (
        "torch_version", "torch_cuda_build", "cudnn_version", "cuda_available",
        "device_count", "device_name", "compute_capability", "total_vram_gb",
        "driver_version", "driver_cuda_version", "matmul_smoke_test",
    ):
        lines.append(f"  {key:22s}: {getattr(gpu, key)}")

    if gpu.problems:
        lines += ["", "!! PROBLEMS !!"]
        lines += [f"  * {p}" for p in gpu.problems]
    else:
        lines += ["", "  No problems detected."]
    lines.append("=" * 68)
    return "\n".join(lines)
