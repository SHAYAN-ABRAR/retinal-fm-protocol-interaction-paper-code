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
