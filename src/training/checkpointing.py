"""Checkpoint saving, loading and resume support.

Three checkpoints are kept per run:

``last.pt``       Every epoch. The resume point after a crash, restart or power
                  loss -- it holds model, optimizer, scheduler, AMP scaler, epoch
                  and RNG state, so resuming continues the same trajectory rather
                  than starting a statistically different one.
``best_qwk.pt``   Best **validation** QWK. This is the checkpoint every reported
                  result uses. It holds **model weights only** -- plus epoch,
                  metrics and config -- because every consumer loads it as
                  ``load_checkpoint(path, model=...)`` and reads nothing else.
                  Optimizer moments belong to resuming, and resuming reads
                  ``last.pt``. For a ViT-L this is the difference between 1.2 GB
                  and 3.4 GB per run, and ten full-FT runs at 3.4 GB do not fit
                  on the disk available.
``best_loss.pt``  Best validation loss. Optional, and off for the full
                  fine-tuning runs: no analysis in this repository reads it
                  (verified by search), and it costs 1.2 GB per run.

The rule that must never be broken
----------------------------------
**Model selection uses source-domain validation data only.** Selecting on the
held-out target domain would leak it into the pipeline just as surely as training
on it. :func:`CheckpointManager.update` takes validation metrics and has no way
to see the test set at all -- the constraint is enforced by the interface, not by
remembering to be careful.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger("training.checkpointing")

__all__ = ["CheckpointManager", "save_checkpoint", "load_checkpoint"]


def _rng_state() -> dict[str, Any]:
    import random

    import numpy as np

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_byte_tensor(value: Any) -> Any:
    """A CPU uint8 tensor, which is the only thing set_rng_state accepts.

    ``load_checkpoint`` is called with ``map_location="cuda"``, so every tensor
    in the payload -- including the saved RNG state -- arrives on the device.
    ``torch.set_rng_state`` rejects that with "RNG state must be a
    torch.ByteTensor", and because the restore was wrapped in a single
    try/except the run continued with an *unrestored* data-ordering and
    augmentation stream. It resumed happily and was no longer reproducible from
    its seed, which on this machine matters: load shedding makes resumes
    routine, so every interrupted run was silently losing its stream.
    """
    tensor = value
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "to") and getattr(tensor, "dtype", None) is not torch.uint8:
        tensor = tensor.to(torch.uint8)
    return tensor


def _restore_rng_state(state: dict[str, Any]) -> list[str]:
    """Restore each stream independently; return the names that failed.

    Independently, because one try/except around all four meant a failure in
    the third skipped the fourth, and the warning could not say which stream was
    lost. "CUDA RNG unavailable on this box" and "the torch CPU stream did not
    restore" have very different consequences for reproducibility and must not
    look the same in a log.
    """
    import random

    import numpy as np

    failed: list[str] = []

    def attempt(name: str, restore) -> None:
        try:
            restore()
        except Exception as exc:  # noqa: BLE001 - a resume must not die over RNG
            failed.append(name)
            log.warning("could not restore %s RNG state: %s", name, exc)

    if "python" in state:
        attempt("python", lambda: random.setstate(state["python"]))
    if "numpy" in state:
        attempt("numpy", lambda: np.random.set_state(state["numpy"]))
    if "torch" in state:
        attempt("torch", lambda: torch.set_rng_state(_as_byte_tensor(state["torch"])))
    if "cuda" in state and torch.cuda.is_available():
        attempt("cuda", lambda: torch.cuda.set_rng_state_all(
            [_as_byte_tensor(t) for t in state["cuda"]]))

    if failed:
        log.warning(
            "resume is NOT reproducible from its seed: %s stream(s) were not "
            "restored, so data ordering and augmentation diverge from an "
            "uninterrupted run", ", ".join(failed))
    return failed


def save_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    include_rng: bool = True,
) -> Path:
    """Write a checkpoint atomically.

    Written to a temporary file and then moved into place, so an interruption
    mid-write cannot leave a corrupt checkpoint where a valid one used to be.
    """
    path = Path(path)
    ensure_dir(path.parent)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics or {},
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if include_rng:
        payload["rng"] = _rng_state()

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    shutil.move(str(temporary), str(path))
    return path


def load_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring optimizer/scheduler/scaler/RNG."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    # weights_only=False: our payload contains RNG state and config dicts, not
    # only tensors. The file is produced by this project, never downloaded.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
        if missing or unexpected:
            log.warning("state_dict mismatch: missing=%s unexpected=%s", missing, unexpected)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and "rng" in payload:
        _restore_rng_state(payload["rng"])

    log.info(
        "loaded checkpoint %s (epoch %s, metrics %s)",
        path.name, payload.get("epoch"), payload.get("metrics"),
    )
    return payload


@dataclass
class CheckpointManager:
    """Owns the three checkpoints for one experiment.

    ``update`` is deliberately given *validation* metrics only. There is no
    parameter through which target-domain performance could influence which
    checkpoint is kept.
    """

    directory: Path
    experiment_id: str
    monitor: str = "qwk"
    mode: str = "max"
    save_last: bool = True
    save_best_loss: bool = True

    best_value: float = field(init=False)
    best_epoch: int = field(default=-1, init=False)
    best_loss: float = field(default=float("inf"), init=False)
    best_loss_epoch: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.best_value = -float("inf") if self.mode == "max" else float("inf")
        self.directory = ensure_dir(Path(self.directory) / self.experiment_id)

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / f"best_{self.monitor}.pt"

    @property
    def best_loss_path(self) -> Path:
        return self.directory / "best_loss.pt"

    def restore_best_state(self) -> bool:
        """Re-learn what "best" means from the checkpoints already on disk.

        Without this, a resumed run restarts with ``best_value = -inf``, so the
        first epoch after the resume always compares as an improvement and
        overwrites ``best_<monitor>.pt`` -- with a model that may be worse than
        the one it replaces. The run then reports metrics from a checkpoint that
        was never actually the best, and the early-stopping counter restarts as
        well, so it also trains longer than it should.

        This is not hypothetical: on 2026-08-26 a MixStyle run resumed at epoch
        9 and immediately logged "new best val qwk: 0.6912 (was -inf)" when
        epoch 8 had reached 0.7113.

        The value is read back from the checkpoint's own recorded metrics rather
        than tracked in a side file, so it stays correct even if the process
        that wrote it is long gone.
        """
        restored = False
        if self.best_path.exists():
            payload = torch.load(self.best_path, map_location="cpu", weights_only=False)
            value = (payload.get("metrics") or {}).get(self.monitor)
            if value is not None and value == value:   # not NaN
                self.best_value = float(value)
                self.best_epoch = int(payload.get("epoch", -1))
                restored = True
        if self.best_loss_path.exists():
            payload = torch.load(self.best_loss_path, map_location="cpu", weights_only=False)
            loss = (payload.get("metrics") or {}).get("loss")
            if loss is not None and loss == loss:
                self.best_loss = float(loss)
                self.best_loss_epoch = int(payload.get("epoch", -1))
        return restored

    def _is_better(self, value: float) -> bool:
        if value != value:            # NaN never wins
            return False
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def update(
        self,
        *,
        epoch: int,
        val_metrics: dict[str, Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        global_step: int = 0,
        config: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """Save checkpoints for this epoch. Returns which ones were written."""
        written = {"last": False, "best": False, "best_loss": False}

        if self.save_last:
            save_checkpoint(
                self.last_path, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch, global_step=global_step,
                metrics=val_metrics, config=config,
            )
            written["last"] = True

        value = float(val_metrics.get(self.monitor, float("nan")))
        if self._is_better(value):
            previous = self.best_value
            self.best_value, self.best_epoch = value, epoch
            # Inference weights only. Every consumer of this file loads it as
            # load_checkpoint(path, model=...) and reads nothing else, but it
            # was being written with optimizer, scheduler, scaler and RNG state
            # anyway -- for a ViT-L that is 3.4 GB where 1.2 GB carries the
            # same information. The two AdamW moment tensors per parameter are
            # what a *resume* needs, and a resume reads last.pt.
            #
            # epoch, global_step, metrics and config stay: restore_best_state()
            # reads the best value back from this file's own metrics, and the
            # config is what makes the artifact reproducible.
            save_checkpoint(
                self.best_path, model=model, epoch=epoch,
                global_step=global_step, metrics=val_metrics, config=config,
                include_rng=False,
            )
            written["best"] = True
            log.info(
                "new best val %s: %.4f (was %.4f) at epoch %d",
                self.monitor, value, previous, epoch,
            )

        if self.save_best_loss:
            loss = float(val_metrics.get("loss", float("nan")))
            if loss == loss and loss < self.best_loss:
                self.best_loss, self.best_loss_epoch = loss, epoch
                save_checkpoint(
                    self.best_loss_path, model=model, epoch=epoch,
                    global_step=global_step, metrics=val_metrics, config=config,
                    include_rng=False,
                )
                written["best_loss"] = True

        return written

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.directory),
            "monitor": self.monitor,
            f"best_val_{self.monitor}": (
                None if self.best_value in (float("inf"), -float("inf")) else self.best_value
            ),
            "best_epoch": self.best_epoch,
            "best_val_loss": None if self.best_loss == float("inf") else self.best_loss,
            "best_loss_epoch": self.best_loss_epoch,
        }
