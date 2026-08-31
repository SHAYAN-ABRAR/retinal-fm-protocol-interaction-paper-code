"""Gradient accumulation must preserve the effective batch, visibly.

The trainer has implemented accumulation for some time, but ``run_lodo.py``
hardcoded ``accumulation_steps: 1`` and ``effective_batch_size: BATCH_SIZE``
into the registry row. A run using accumulation would therefore have been
recorded as if it had not -- and effective batch size is precisely the quantity
that has to match across a comparison for it to be a comparison.

Full fine-tuning a ViT-L in 8 GB needs a physical batch far below 16, so this is
the mechanism that lets a full-FT arm keep the same optimisation as the
partial-FT arm it is compared against. If it is silently wrong, the comparison
is silently wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")


# --- the arithmetic --------------------------------------------------------

def test_effective_batch_size_is_the_product() -> None:
    from src.training.trainer import TrainConfig

    config = TrainConfig(batch_size=4, accumulation_steps=4)
    assert config.effective_batch_size == 16
    assert TrainConfig(batch_size=16).effective_batch_size == 16
    assert TrainConfig(batch_size=1, accumulation_steps=16).effective_batch_size == 16


@pytest.mark.parametrize("batch,accum", [(16, 1), (4, 4), (2, 8), (1, 16)])
def test_the_documented_fallback_ladder_holds_effective_batch_at_16(batch, accum):
    """The sequence a full-FT run steps down through when VRAM will not fit."""
    from src.training.trainer import TrainConfig

    assert TrainConfig(batch_size=batch,
                       accumulation_steps=accum).effective_batch_size == 16


# --- the optimisation ------------------------------------------------------

def _train_one_epoch(batch_size, accumulation, n_samples=16, seed=0):
    """Train a tiny linear model one epoch and return its weights."""
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    x = torch.randn(n_samples, 4)
    y = (x.sum(dim=1) > 0).long()
    model = nn.Linear(4, 2)
    torch.manual_seed(seed)          # identical initial weights either way
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    optimizer.zero_grad(set_to_none=True)
    for step, (features, target) in enumerate(loader):
        loss = nn.functional.cross_entropy(model(features), target)
        (loss / accumulation).backward()
        if (step + 1) % accumulation == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return model.weight.detach().clone()


def test_accumulation_reproduces_the_large_batch_update() -> None:
    """batch 4 x accum 4 must land where batch 16 x accum 1 lands.

    This is the property the whole mechanism exists for. Mean-reduced
    cross-entropy over four micro-batches of equal size, each scaled by 1/4,
    sums to the mean over all sixteen.
    """
    big = _train_one_epoch(batch_size=16, accumulation=1)
    small = _train_one_epoch(batch_size=4, accumulation=4)
    assert torch.allclose(big, small, atol=1e-6), (
        "accumulated update diverged from the equivalent large-batch update")


def test_without_scaling_the_update_would_be_wrong() -> None:
    """Guards the loss division itself, not just the stepping schedule."""
    correct = _train_one_epoch(batch_size=4, accumulation=4)
    # accumulation=4 for the stepping schedule, but no 1/4 on the loss.
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    x = torch.randn(16, 4)
    y = (x.sum(dim=1) > 0).long()
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    optimizer.zero_grad(set_to_none=True)
    for step, (features, target) in enumerate(loader):
        nn.functional.cross_entropy(model(features), target).backward()
        if (step + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    unscaled = model.weight.detach().clone()
    assert not torch.allclose(correct, unscaled, atol=1e-6), (
        "the test cannot tell scaled from unscaled; it proves nothing")


# --- the wiring ------------------------------------------------------------

def test_run_lodo_exposes_the_flag_and_records_what_it_did() -> None:
    source = (ROOT / "run_lodo.py").read_text(encoding="utf-8")
    assert "--accumulation-steps" in source, "no CLI flag"
    assert "ACCUMULATION_STEPS" in source
    # The registry row must carry all three, not a hardcoded 1.
    for field in ('"physical_batch_size"', '"accumulation_steps"',
                  '"effective_batch_size"'):
        assert field in source, f"{field} is not recorded"
    assert '"accumulation_steps": 1,' not in source, (
        "accumulation_steps is still hardcoded to 1 in a recorded row")
    assert "BATCH_SIZE * ACCUMULATION_STEPS" in source, (
        "effective_batch_size is not derived from the two factors")


def test_accumulation_enters_the_experiment_id() -> None:
    """batch 4 x accum 4 and batch 4 alone must not share an id."""
    import run_lodo

    run_lodo.BATCH_SIZE = 4
    run_lodo.IMAGE_SIZE = 224
    run_lodo.TRAINABLE_BLOCKS = None
    run_lodo.FULL_FINETUNE = False
    run_lodo.IRM_ANNEAL_ITERS = None
    run_lodo.LEARNING_RATE = 3e-4
    try:
        run_lodo.ACCUMULATION_STEPS = 1
        plain = run_lodo._method_tag("erm")
        run_lodo.ACCUMULATION_STEPS = 4
        accumulated = run_lodo._method_tag("erm")
    finally:
        run_lodo.ACCUMULATION_STEPS = 1
        run_lodo.BATCH_SIZE = 32
    assert plain != accumulated, (
        "two runs with different effective batch sizes share one id")
    assert accumulated == plain + "-ga4"


def test_accumulation_is_part_of_the_results_key() -> None:
    """Otherwise the two runs merge into one row and the newer wins."""
    import pandas as pd

    from src.utils.registry import merge_results_table

    key = ["target", "method", "seed", "batch_size", "accumulation_steps"]

    def row(accum, qwk):
        return pd.DataFrame([{"target": "ddr", "method": "erm", "seed": 42,
                              "batch_size": 4, "accumulation_steps": accum,
                              "target_qwk": qwk}])

    merged = merge_results_table(row(1, 0.71), row(4, 0.75), key)
    assert len(merged) == 2, "the accumulated run overwrote the un-accumulated one"


def test_a_blank_accumulation_is_backfilled_to_one() -> None:
    """A table predating the column must not gain a phantom second row."""
    import pandas as pd

    from src.utils.registry import merge_results_table

    key = ["target", "method", "seed", "batch_size", "accumulation_steps"]
    legacy = pd.DataFrame([{"target": "ddr", "method": "erm", "seed": 42,
                            "batch_size": 4, "target_qwk": 0.71}])
    new = pd.DataFrame([{"target": "ddr", "method": "erm", "seed": 42,
                         "batch_size": 4, "accumulation_steps": 1,
                         "target_qwk": 0.72}])
    merged = merge_results_table(legacy, new, key)
    assert len(merged) == 1
    assert merged["target_qwk"].iloc[0] == 0.72


def test_zero_or_negative_accumulation_is_refused() -> None:
    """max(1, ...) in the trainer would silently accept 0 and train normally."""
    source = (ROOT / "run_lodo.py").read_text(encoding="utf-8")
    assert "ACCUMULATION_STEPS < 1" in source, (
        "an invalid accumulation count is not rejected at the CLI")


# --- resume ----------------------------------------------------------------

def test_accumulation_is_carried_in_the_saved_config(tmp_path) -> None:
    """Resume must not restart a run at a different effective batch size."""
    from src.training.trainer import TrainConfig

    config = TrainConfig(batch_size=2, accumulation_steps=8, epochs=1)
    described = config.describe() if hasattr(config, "describe") else vars(config)
    assert described.get("accumulation_steps") == 8, (
        "accumulation is absent from the persisted config, so a resumed run "
        "could silently continue at a different effective batch size")
    assert config.effective_batch_size == 16


# --- the audit and run_lodo must agree on what an id looks like ------------
# These drifted twice in one session: -ftfull and -e{N} were added to the id
# without being added to the summary row, so the audit rebuilt
# "erm-b16-lr0.0001" for a run actually named "erm-b16-ftfull-lr0.0001-e2" and
# reported a completed run as missing. -ftfull is the subtle one: it cannot be
# derived from trainable_blocks, because a full fine-tune and an unfrozen CNN
# both record -1.

import pytest as _pytest


@_pytest.mark.parametrize("settings,expected", [
    (dict(batch=16, accum=1, epochs=20, full=False, blocks=None, lr=3e-4),
     "erm-b16"),
    (dict(batch=16, accum=1, epochs=20, full=True, blocks=None, lr=1e-4),
     "erm-b16-ftfull-lr0.0001"),
    (dict(batch=4, accum=4, epochs=2, full=True, blocks=None, lr=1e-4),
     "erm-b4-ftfull-lr0.0001-ga4-e2"),
    (dict(batch=16, accum=1, epochs=20, full=False, blocks=4, lr=1e-4),
     "erm-b16-tb4-lr0.0001"),
])
def test_the_audit_rebuilds_the_id_run_lodo_writes(settings, expected):
    import audit_consistency
    import run_lodo

    saved = (run_lodo.BATCH_SIZE, run_lodo.ACCUMULATION_STEPS, run_lodo.EPOCHS,
             run_lodo.FULL_FINETUNE, run_lodo.TRAINABLE_BLOCKS,
             run_lodo.LEARNING_RATE, run_lodo.IMAGE_SIZE,
             run_lodo.IRM_ANNEAL_ITERS)
    try:
        run_lodo.BATCH_SIZE = settings["batch"]
        run_lodo.ACCUMULATION_STEPS = settings["accum"]
        run_lodo.EPOCHS = settings["epochs"]
        run_lodo.FULL_FINETUNE = settings["full"]
        run_lodo.TRAINABLE_BLOCKS = settings["blocks"]
        run_lodo.LEARNING_RATE = settings["lr"]
        run_lodo.IMAGE_SIZE = 224
        run_lodo.IRM_ANNEAL_ITERS = None
        written = run_lodo._method_tag("erm")
    finally:
        (run_lodo.BATCH_SIZE, run_lodo.ACCUMULATION_STEPS, run_lodo.EPOCHS,
         run_lodo.FULL_FINETUNE, run_lodo.TRAINABLE_BLOCKS,
         run_lodo.LEARNING_RATE, run_lodo.IMAGE_SIZE,
         run_lodo.IRM_ANNEAL_ITERS) = saved

    assert written == expected, "run_lodo built an unexpected tag"

    # The row the run would record, then the audit's reconstruction from it.
    row = {
        "method": "erm", "batch_size": settings["batch"], "image_size": 224,
        "domain_balanced": False, "irm_anneal_iters": 500,
        "trainable_blocks": -1 if settings["blocks"] is None else settings["blocks"],
        "learning_rate": settings["lr"],
        "accumulation_steps": settings["accum"],
        "full_finetune": settings["full"], "epochs": settings["epochs"],
    }
    spec = next(s for s in audit_consistency.specifications()
                if s["file"].startswith("lodo_results"))
    rebuilt = spec["experiment_id"]({**row, "target": "ddr",
                                     "backbone": "densenet121", "seed": 42})
    assert rebuilt.endswith(f"_{written}_s42"), (
        f"audit rebuilt {rebuilt!r}, which does not carry run_lodo's tag "
        f"{written!r}; a completed run would be reported as missing")


def test_full_finetune_is_not_inferable_from_trainable_blocks():
    """Why full_finetune has to be its own recorded column."""
    import run_lodo

    saved = (run_lodo.FULL_FINETUNE, run_lodo.TRAINABLE_BLOCKS)
    try:
        run_lodo.TRAINABLE_BLOCKS = None
        run_lodo.FULL_FINETUNE = False
        assert run_lodo.recorded_trainable_blocks() == -1
        run_lodo.FULL_FINETUNE = True
        assert run_lodo.recorded_trainable_blocks() == -1, (
            "if these ever differ, this test is obsolete -- but while they are "
            "both -1, the flag must be recorded separately")
    finally:
        run_lodo.FULL_FINETUNE, run_lodo.TRAINABLE_BLOCKS = saved
