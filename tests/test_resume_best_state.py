"""Resuming must not forget what "best" meant.

The CheckpointManager tracked best_value in memory only. A resumed run started
again at -inf, so the first epoch after the resume always compared as an
improvement and overwrote best_<monitor>.pt -- possibly with a worse model. The
run then reported metrics from a checkpoint that had never been the best, and
the early-stopping counter restarted too.

Observed on 2026-08-26: a MixStyle run resumed at epoch 9 and logged
"new best val qwk: 0.6912 (was -inf)" when epoch 8 had already reached 0.7113.
"""

from __future__ import annotations

import torch
from torch import nn

from src.training.checkpointing import CheckpointManager


def _manager(tmp_path):
    return CheckpointManager(directory=tmp_path, experiment_id="exp", monitor="qwk")


def _model():
    torch.manual_seed(0)
    return nn.Linear(4, 2)


def test_a_worse_epoch_after_resume_does_not_overwrite_the_best(tmp_path) -> None:
    first = _manager(tmp_path)
    first.update(epoch=8, val_metrics={"qwk": 0.7113, "loss": 0.62}, model=_model())
    assert first.best_value == 0.7113

    # A new process for the same experiment: fresh manager, same directory.
    resumed = _manager(tmp_path)
    assert resumed.restore_best_state() is True
    assert resumed.best_value == 0.7113
    assert resumed.best_epoch == 8

    written = resumed.update(epoch=9, val_metrics={"qwk": 0.6912, "loss": 0.64},
                             model=_model())
    assert written["best"] is False, "a worse epoch overwrote the best checkpoint"
    assert resumed.best_value == 0.7113
    assert resumed.best_epoch == 8


def test_without_restore_the_worse_epoch_would_have_won(tmp_path) -> None:
    """Pin the bug itself, so the fix cannot be quietly removed."""
    first = _manager(tmp_path)
    first.update(epoch=8, val_metrics={"qwk": 0.7113, "loss": 0.62}, model=_model())

    naive = _manager(tmp_path)          # no restore_best_state() call
    assert naive.best_value == -float("inf")
    written = naive.update(epoch=9, val_metrics={"qwk": 0.6912, "loss": 0.64},
                           model=_model())
    assert written["best"] is True


def test_a_better_epoch_after_resume_still_wins(tmp_path) -> None:
    first = _manager(tmp_path)
    first.update(epoch=8, val_metrics={"qwk": 0.7113, "loss": 0.62}, model=_model())

    resumed = _manager(tmp_path)
    resumed.restore_best_state()
    written = resumed.update(epoch=9, val_metrics={"qwk": 0.7500, "loss": 0.60},
                             model=_model())
    assert written["best"] is True
    assert resumed.best_value == 0.7500
    assert resumed.best_epoch == 9


def test_restore_on_a_fresh_directory_is_a_no_op(tmp_path) -> None:
    fresh = _manager(tmp_path)
    assert fresh.restore_best_state() is False
    assert fresh.best_value == -float("inf")
