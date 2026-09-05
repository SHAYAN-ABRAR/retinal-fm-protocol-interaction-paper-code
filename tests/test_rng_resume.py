"""A resumed run must continue the same random stream, or it is a different run.

The bug this guards: ``load_checkpoint`` is called with ``map_location="cuda"``,
so the saved RNG tensors arrive on the device, and ``torch.set_rng_state``
rejects a CUDA tensor with "RNG state must be a torch.ByteTensor". The restore
sat inside one try/except, so the failure logged a warning and the run
continued with an *unrestored* stream -- resuming happily while no longer being
reproducible from its seed.

It surfaced on the seed-2 RETFound full fine-tuning run after a deliberate
pause. On this machine load shedding makes resumes routine, so every
interrupted run had been quietly losing its data ordering and augmentation
stream.

The tests below assert the property that actually matters -- the numbers drawn
after a resume are the numbers an uninterrupted run would have drawn -- rather
than merely that no exception was raised.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from src.training.checkpointing import (  # noqa: E402
    _as_byte_tensor,
    _restore_rng_state,
    _rng_state,
)


def _draw():
    """One sample from each stream the training loop actually uses."""
    return (random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()))


def test_a_restored_stream_produces_the_same_numbers() -> None:
    """The property the whole mechanism exists for."""
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    state = _rng_state()
    expected = [_draw() for _ in range(5)]

    # Disturb every stream, as a few epochs of training would.
    for _ in range(50):
        _draw()

    assert _restore_rng_state(state) == [], "a stream failed to restore"
    assert [_draw() for _ in range(5)] == expected


def test_a_cuda_resident_state_is_accepted() -> None:
    """The exact failure: map_location='cuda' puts the state on the device.

    Simulated on CPU by round-tripping through the coercion, so the test runs
    on a machine without a GPU as well as on one with.
    """
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    state = _rng_state()
    expected = [_draw() for _ in range(3)]

    moved = dict(state)
    if torch.cuda.is_available():
        moved["torch"] = state["torch"].to("cuda")
    else:
        # Same rejection path without a GPU: a non-uint8 tensor.
        moved["torch"] = state["torch"].to(torch.int64)

    for _ in range(20):
        _draw()

    assert _restore_rng_state(moved) == [], (
        "a device-resident or wrong-dtype RNG state was not coerced")
    assert [_draw() for _ in range(3)] == expected


def test_the_coercion_returns_cpu_uint8() -> None:
    original = torch.get_rng_state()
    for candidate in (original, original.to(torch.int64), original.clone()):
        coerced = _as_byte_tensor(candidate)
        assert coerced.dtype is torch.uint8
        assert coerced.device.type == "cpu"


def test_each_stream_is_restored_independently() -> None:
    """One broken stream must not silently skip the ones after it."""
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    state = _rng_state()

    torch.manual_seed(999)
    expected_torch = float(torch.rand(1).item())
    torch.manual_seed(3)
    _rng_state()

    broken = dict(state)
    broken["numpy"] = "not a numpy state at all"

    failed = _restore_rng_state(broken)
    assert failed == ["numpy"], f"expected only numpy to fail, got {failed}"

    # torch is restored despite numpy failing first -- the old single
    # try/except would have skipped it.
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    reference = float(torch.rand(1).item())
    _restore_rng_state(broken)
    assert float(torch.rand(1).item()) == reference


def test_a_failure_is_reported_not_swallowed(caplog) -> None:
    """Silence after a failed restore is how this went unnoticed for a run."""
    import logging

    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("dr_dg.training.checkpointing")
    handler = Collect()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        _restore_rng_state({"numpy": "broken"})
    finally:
        logger.removeHandler(handler)

    assert any("numpy" in m for m in records), "the failing stream is not named"
    assert any("NOT reproducible" in m for m in records), (
        "the consequence for reproducibility is not stated")


def test_a_checkpoint_round_trip_restores_the_stream(tmp_path) -> None:
    """End to end through save_checkpoint / load_checkpoint."""
    from torch import nn

    from src.training.checkpointing import load_checkpoint, save_checkpoint

    random.seed(21)
    np.random.seed(21)
    torch.manual_seed(21)

    model = nn.Linear(4, 2)
    path = save_checkpoint(tmp_path / "last.pt", model=model, epoch=1)
    expected = [_draw() for _ in range(4)]

    for _ in range(30):
        _draw()

    payload = load_checkpoint(path, model=model, map_location="cpu")
    assert "rng" in payload, "last.pt carries no RNG state; resume cannot be exact"
    assert _restore_rng_state(payload["rng"]) == []
    assert [_draw() for _ in range(4)] == expected


# --- the scheduler ---------------------------------------------------------
# A second, worse resume bug found the same way. Trainer.resume() restored
# model, optimizer, scaler and RNG but not the scheduler -- it could not,
# because the scheduler needs steps_per_epoch and is built in fit(), which runs
# AFTER resume(). So the learning-rate schedule restarted from step zero at
# every resume: a run interrupted at epoch 2 of 20 re-ran warmup and then
# followed a cosine two epochs behind the intended one. It completed,
# registered COMPLETE, and only the per-epoch lr in the log gave it away.

def test_resume_stashes_scheduler_state_for_fit() -> None:
    """The state must survive resume() even though the scheduler is None then."""
    source = (ROOT / "src" / "training" / "trainer.py").read_text(encoding="utf-8")
    assert "_pending_scheduler_state" in source, (
        "resume() drops the scheduler state entirely")
    assert 'self._pending_scheduler_state = payload.get("scheduler")' in source, (
        "resume() does not stash the saved scheduler state")
    assert "self.scheduler.load_state_dict(self._pending_scheduler_state)" in source, (
        "fit() never applies the stashed scheduler state")


def test_a_restored_scheduler_continues_the_schedule() -> None:
    """The property: lr after a resume equals lr in an uninterrupted run."""
    from src.training.trainer import TrainConfig, _build_optimizer, _build_scheduler
    from torch import nn

    config = TrainConfig(epochs=20, batch_size=16, learning_rate=1e-4,
                         warmup_epochs=1, scheduler="cosine")
    steps = 100

    def fresh():
        model = nn.Linear(4, 2)
        optimizer = _build_optimizer(model, config)
        return optimizer, _build_scheduler(optimizer, config, steps)

    # Uninterrupted: step through two epochs' worth.
    opt_a, sched_a = fresh()
    for _ in range(2 * steps):
        opt_a.step()
        sched_a.step()
    uninterrupted_lr = opt_a.param_groups[0]["lr"]

    # Interrupted at the same point, state saved and restored into a new one.
    opt_b, sched_b = fresh()
    for _ in range(2 * steps):
        opt_b.step()
        sched_b.step()
    saved = sched_b.state_dict()

    opt_c, sched_c = fresh()
    sched_c.load_state_dict(saved)
    # load_state_dict restores last_epoch but leaves the optimizer's lr alone --
    # torch recomputes it on the next step(). fit() therefore writes _last_lr
    # back explicitly, and this mirrors that.
    for group, lr in zip(opt_c.param_groups, sched_c._last_lr):
        group["lr"] = lr
    resumed_lr = opt_c.param_groups[0]["lr"]

    assert resumed_lr == pytest.approx(uninterrupted_lr), (
        f"resumed lr {resumed_lr:.3e} != uninterrupted {uninterrupted_lr:.3e}")

    # And the bug it replaces: a scheduler that was never restored sits at the
    # start of warmup, two orders of magnitude away.
    opt_d, _ = fresh()
    assert opt_d.param_groups[0]["lr"] != pytest.approx(uninterrupted_lr), (
        "the test cannot tell a restored scheduler from a fresh one")


def test_fit_writes_the_restored_lr_back_to_the_optimizer() -> None:
    source = (ROOT / "src" / "training" / "trainer.py").read_text(encoding="utf-8")
    assert '_last_lr' in source and 'group["lr"] = lr' in source, (
        "fit() restores the scheduler but never applies its lr, so the first "
        "step after a resume runs at the fresh warmup rate")


# --- optimizer, scaler, and end-to-end equivalence -------------------------
# The tests above assert the keys are present in last.pt. Presence is not
# restoration: a checkpoint can carry an optimizer state dict that is never
# loaded, or loaded into the wrong object, and still pass a key check. These
# assert the values actually come back.

def test_optimizer_moments_are_restored_by_value(tmp_path) -> None:
    from torch import nn

    from src.training.checkpointing import load_checkpoint, save_checkpoint

    torch.manual_seed(5)
    model = nn.Linear(6, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(3):                       # build non-trivial moments
        model(torch.randn(8, 6)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    expected = {i: {k: v.clone() for k, v in s.items() if torch.is_tensor(v)}
                for i, s in optimizer.state_dict()["state"].items()}
    assert expected, "fixture built no optimizer state"

    path = save_checkpoint(tmp_path / "last.pt", model=model,
                           optimizer=optimizer, epoch=2)

    fresh_model = nn.Linear(6, 3)
    fresh = torch.optim.AdamW(fresh_model.parameters(), lr=1e-4)
    load_checkpoint(path, model=fresh_model, optimizer=fresh, map_location="cpu")

    restored = fresh.state_dict()["state"]
    assert set(restored) == set(expected), "optimizer state keys differ"
    for index, tensors in expected.items():
        for name, value in tensors.items():
            assert torch.allclose(restored[index][name], value), (
                f"optimizer moment {name} for param {index} was not restored")


def test_amp_scaler_scale_is_restored_by_value(tmp_path) -> None:
    """A scaler reset to its default restarts loss scaling mid-run."""
    from torch import nn

    from src.training.checkpointing import load_checkpoint, save_checkpoint

    # A *disabled* scaler has an empty state_dict, so it cannot demonstrate
    # anything. Training enables AMP whenever CUDA is present, which is the
    # configuration that matters.
    if not torch.cuda.is_available():
        pytest.skip("AMP scaler state only exists with CUDA")

    scaler = torch.amp.GradScaler("cuda", enabled=True)
    state = scaler.state_dict()
    default_scale = state["scale"]
    state["scale"] = default_scale / 8.0     # distinct from the default
    scaler.load_state_dict(state)

    path = save_checkpoint(tmp_path / "last.pt", model=nn.Linear(2, 2),
                           scaler=scaler, epoch=1)

    fresh = torch.amp.GradScaler("cuda", enabled=True)
    assert fresh.state_dict()["scale"] == default_scale, "fixture assumption"
    load_checkpoint(path, model=nn.Linear(2, 2), scaler=fresh,
                    map_location="cpu")
    assert fresh.state_dict()["scale"] == default_scale / 8.0, (
        "scaler scale was not restored; a resumed run would restart loss "
        "scaling from the default mid-training")


def test_resume_equivalence_end_to_end(tmp_path) -> None:
    """An interrupted-and-resumed run must land where an uninterrupted one does.

    The whole point, exercised through save_checkpoint/load_checkpoint plus the
    RNG and scheduler restores, rather than through any single component.
    """
    from torch import nn

    from src.training.checkpointing import load_checkpoint, save_checkpoint
    from src.training.trainer import TrainConfig, _build_optimizer, _build_scheduler

    config = TrainConfig(epochs=6, batch_size=4, learning_rate=1e-3,
                         warmup_epochs=1, scheduler="cosine")
    steps = 10

    def build(seed):
        torch.manual_seed(seed)
        model = nn.Linear(4, 2)
        optimizer = _build_optimizer(model, config)
        return model, optimizer, _build_scheduler(optimizer, config, steps)

    def train_steps(model, optimizer, scheduler, n):
        for _ in range(n):
            x = torch.randn(4, 4)
            model(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    # Uninterrupted: 30 steps straight through.
    random.seed(1); np.random.seed(1)
    model_a, opt_a, sched_a = build(1)
    train_steps(model_a, opt_a, sched_a, 30)

    # Interrupted after 12, checkpointed, restored, then the remaining 18.
    random.seed(1); np.random.seed(1)
    model_b, opt_b, sched_b = build(1)
    train_steps(model_b, opt_b, sched_b, 12)
    path = save_checkpoint(tmp_path / "last.pt", model=model_b,
                           optimizer=opt_b, scheduler=sched_b, epoch=1)

    model_c, opt_c, sched_c = build(999)      # deliberately different init
    payload = load_checkpoint(path, model=model_c, optimizer=opt_c,
                              scheduler=sched_c, map_location="cpu",
                              restore_rng=True)
    for group, lr in zip(opt_c.param_groups, sched_c._last_lr):
        group["lr"] = lr
    train_steps(model_c, opt_c, sched_c, 18)

    assert payload["epoch"] == 1
    for name, value in model_a.state_dict().items():
        assert torch.allclose(model_c.state_dict()[name], value, atol=1e-6), (
            f"{name} diverged between the uninterrupted and resumed runs")
