"""Checkpoint saving, loading and resume support.

Three checkpoints are kept per run:

``last.pt``       Every epoch. The resume point after a crash, restart or power
                  loss -- it holds model, optimizer, scheduler, AMP scaler, epoch
                  and RNG state, so resuming continues the same trajectory rather
                  than starting a statistically different one.
``best_qwk.pt``   Best **validation** QWK. This is the checkpoint every reported
                  result uses.
``best_loss.pt``  Best validation loss. Kept for the calibration analysis, where
                  the lowest-NLL model is sometimes not the highest-QWK one.

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


def _restore_rng_state(state: dict[str, Any]) -> None:
    import random

    import numpy as np

    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # noqa: BLE001 - a resume must not die over RNG state
        log.warning("could not fully restore RNG state: %s", exc)


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
            save_checkpoint(
                self.best_path, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch, global_step=global_step,
                metrics=val_metrics, config=config,
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
