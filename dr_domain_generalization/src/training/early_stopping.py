"""Early stopping on a source-domain validation metric.

Like the checkpoint manager, this only ever sees **validation** metrics.  Early
stopping on the held-out target domain is one of the subtler ways to leak it into
the pipeline -- the stopping epoch becomes a hyperparameter fitted to the test
set -- so the interface makes it impossible rather than discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..utils.logging import get_logger

log = get_logger("training.early_stopping")

__all__ = ["EarlyStopping"]


@dataclass
class EarlyStopping:
    """Stop when the monitored validation metric stops improving.

    Parameters
    ----------
    patience:
        Epochs without improvement before stopping.
    min_delta:
        Minimum change that counts as an improvement. Guards against declaring
        victory on numerical noise -- QWK moves by ~0.002 between epochs on a
        converged model, so a delta below that would never trigger.
    mode:
        ``"max"`` for QWK/F1, ``"min"`` for loss.
    """

    patience: int = 8
    min_delta: float = 1e-4
    mode: Literal["max", "min"] = "max"
    monitor: str = "qwk"

    best: float = field(init=False)
    best_epoch: int = field(default=-1, init=False)
    epochs_without_improvement: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)
    history: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        self.best = -float("inf") if self.mode == "max" else float("inf")

    def _improved(self, value: float) -> bool:
        if value != value:                      # NaN
            return False
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Record an epoch's validation metric. Returns True if training should stop."""
        self.history.append(float(value))

        if self._improved(value):
            self.best, self.best_epoch = float(value), epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
                log.info(
                    "early stopping at epoch %d: no improvement in val %s for %d epochs "
                    "(best %.4f at epoch %d)",
                    epoch, self.monitor, self.patience, self.best, self.best_epoch,
                )
        return self.should_stop

    def state(self) -> dict[str, object]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": None if self.best in (float("inf"), -float("inf")) else self.best,
            "best_epoch": self.best_epoch,
            "stopped_early": self.should_stop,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
