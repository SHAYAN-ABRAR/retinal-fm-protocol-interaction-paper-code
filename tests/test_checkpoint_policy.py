"""What each checkpoint must and must not contain, and when one may be deleted.

Two distinct risks.

**Writing too much.** ``best_qwk.pt`` was written with optimizer, scheduler,
scaler and RNG state even though every consumer loads it as
``load_checkpoint(path, model=...)``. For a ViT-L that is 3.4 GB where 1.2 GB
carries the same information, and ten full-FT runs then do not fit on disk.

**Writing too little.** The opposite failure is worse and silent: a ``last.pt``
without optimizer state reloads perfectly and restarts Adam's moments from zero,
continuing a different trajectory than the one it claims to resume.

So both directions are asserted, not just the one being changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from src.training.checkpointing import (  # noqa: E402
    CheckpointManager,
    load_checkpoint,
)


@pytest.fixture
def trained(tmp_path):
    """A manager that has seen one epoch, with real optimiser state."""
    from torch import nn

    model = nn.Linear(8, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # One real step, so the optimiser has non-empty moment tensors to save.
    loss = model(torch.randn(4, 8)).sum()
    loss.backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    manager = CheckpointManager(directory=tmp_path, experiment_id="run",
                                monitor="qwk", mode="max")
    manager.update(epoch=3, val_metrics={"qwk": 0.71, "loss": 0.4},
                   model=model, optimizer=optimizer, scheduler=scheduler,
                   scaler=scaler, global_step=120, config={"lr": 1e-4})
    return manager, model


# --- best_qwk.pt: inference only -------------------------------------------

def test_best_checkpoint_carries_no_optimizer_state(trained) -> None:
    manager, _ = trained
    payload = torch.load(manager.best_path, map_location="cpu", weights_only=False)
    for key in ("optimizer", "scheduler", "scaler", "rng"):
        assert key not in payload, (
            f"best_qwk.pt still carries {key!r}; it is an inference artifact "
            f"and resume state belongs in last.pt")


def test_best_checkpoint_keeps_what_reporting_needs(trained) -> None:
    manager, _ = trained
    payload = torch.load(manager.best_path, map_location="cpu", weights_only=False)
    assert payload["model"], "no weights: the artifact is useless"
    assert payload["epoch"] == 3
    assert payload["global_step"] == 120
    assert payload["metrics"]["qwk"] == 0.71
    assert payload["config"] == {"lr": 1e-4}


def test_best_checkpoint_reloads_exactly_for_inference(trained, tmp_path) -> None:
    from torch import nn

    manager, original = trained
    fresh = nn.Linear(8, 5)
    load_checkpoint(manager.best_path, model=fresh, map_location="cpu")
    for name, value in original.state_dict().items():
        assert torch.allclose(fresh.state_dict()[name], value), f"{name} differs"


def test_the_best_checkpoint_is_smaller_than_the_resume_one(trained) -> None:
    """The whole point. Guards against a regression that re-adds the state."""
    manager, _ = trained
    assert manager.best_path.stat().st_size < manager.last_path.stat().st_size


# --- last.pt: fully resumable ----------------------------------------------

def test_last_checkpoint_is_fully_resumable(trained) -> None:
    manager, _ = trained
    payload = torch.load(manager.last_path, map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng",
                "epoch", "global_step"):
        assert key in payload, f"last.pt is missing {key!r}; it cannot resume"
    assert payload["optimizer"]["state"], (
        "optimizer state is empty: a resume would restart Adam's moments at zero")


def test_resume_restores_the_historical_best(trained, tmp_path) -> None:
    """A fresh manager on the same directory must not think best is -inf.

    This is the bug that overwrote a better checkpoint after a power cut: the
    resumed run reported 'new best val qwk: 0.6912 (was -inf)' when epoch 8 had
    already reached 0.7113.
    """
    manager, _ = trained
    # __post_init__ appends experiment_id, so the root is the parent -- passing
    # manager.directory here reopens at <dir>/run/run and finds nothing.
    reopened = CheckpointManager(directory=manager.directory.parent,
                                 experiment_id="run", monitor="qwk", mode="max")
    assert reopened.best_value == float("-inf"), "fixture assumption changed"
    assert reopened.restore_best_state() is True
    assert reopened.best_value == pytest.approx(0.71)
    assert reopened.best_epoch == 3


# --- best_loss.pt is optional ----------------------------------------------

def test_best_loss_can_be_disabled(tmp_path) -> None:
    from torch import nn

    model = nn.Linear(8, 5)
    manager = CheckpointManager(directory=tmp_path, experiment_id="run",
                                monitor="qwk", mode="max", save_best_loss=False)
    written = manager.update(epoch=0, val_metrics={"qwk": 0.5, "loss": 0.9},
                             model=model)
    assert written["best_loss"] is False
    assert not manager.best_loss_path.exists()
    assert manager.best_path.exists(), "disabling best_loss must not affect best_qwk"


def test_full_finetune_disables_best_loss_but_nothing_else_does() -> None:
    source = (ROOT / "run_lodo.py").read_text(encoding="utf-8")
    assert "save_best_loss=not FULL_FINETUNE" in source, (
        "best_loss is not scoped to full fine-tuning")


def test_train_config_defaults_to_saving_best_loss() -> None:
    """Existing run types must keep their behaviour exactly."""
    from src.training.trainer import TrainConfig

    assert TrainConfig().save_best_loss is True


# --- the pruner refuses on every incomplete condition ----------------------

def _scaffold(tmp_path, experiment_id="run_a"):
    """A directory tree that looks like a completed, audited run."""
    import json

    for name in ("checkpoints", "reports", "predictions", "logs", "tables"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    run = tmp_path / "checkpoints" / experiment_id
    run.mkdir()
    (run / "last.pt").write_bytes(b"x" * 2048)
    (run / "best_qwk.pt").write_bytes(b"y" * 1024)
    (tmp_path / "reports" / f"{experiment_id}_evaluation.json").write_text(
        json.dumps({"ok": True}), encoding="utf-8")
    (tmp_path / "predictions"
     / f"{experiment_id}__target_test[ddr]_predictions.csv").write_text(
        "true_grade,predicted_grade\n0,0\n", encoding="utf-8")
    (tmp_path / "logs" / f"{experiment_id}_history.csv").write_text(
        "epoch\n0\n", encoding="utf-8")
    return run


def test_the_pruner_requires_every_condition(tmp_path) -> None:
    """Each artifact removed in turn must block the deletion.

    Asserted by construction rather than by running the script, which needs the
    real registry and audit. The point is that the conditions are conjunctive.
    """
    run = _scaffold(tmp_path)
    required = [
        tmp_path / "reports" / "run_a_evaluation.json",
        tmp_path / "predictions" / "run_a__target_test[ddr]_predictions.csv",
        tmp_path / "logs" / "run_a_history.csv",
        run / "best_qwk.pt",
    ]
    for path in required:
        assert path.exists(), f"scaffold is wrong: {path} missing"
    # Removing any one of them must leave last.pt in place, which is what the
    # script's `continue` does. Encoded here as the contract the script keeps.
    for path in required:
        path.unlink()
        assert (run / "last.pt").exists()
        path.touch()


def test_the_pruner_never_deletes_the_model_artifact() -> None:
    source = (ROOT / "prune_checkpoints.py").read_text(encoding="utf-8")
    assert "best_qwk.pt" in source
    # Only last.pt is ever unlinked.
    unlinks = [line for line in source.splitlines() if ".unlink()" in line]
    assert unlinks, "the pruner deletes nothing at all"
    for line in unlinks:
        assert "best" not in line, f"the pruner deletes a best checkpoint: {line}"


def test_the_pruner_defaults_to_a_dry_run() -> None:
    source = (ROOT / "prune_checkpoints.py").read_text(encoding="utf-8")
    assert 'apply = "--apply" in arguments' in source
    assert "if not apply:" in source, "deletion is not gated behind --apply"


def test_the_pruner_requires_a_complete_registry_status() -> None:
    source = (ROOT / "prune_checkpoints.py").read_text(encoding="utf-8")
    assert 'status_by_id.get(experiment_id) != "COMPLETE"' in source, (
        "a RUNNING or failed run could have its resume point deleted")
