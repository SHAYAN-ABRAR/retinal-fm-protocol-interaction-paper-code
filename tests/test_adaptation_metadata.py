"""The experiment id and the registry record must not disagree.

run_lodo wrote trainable_blocks to the summary row but not to the registry
record, so 27 of the 33 runs whose id says "tb4" carried NaN in the registry's
trainable_blocks column. The registry is the authoritative record, so an id and
its own record disagreed about what had been trained, and nothing checked it.

These tests pin the two directions that matter: the tag must encode the mode,
and the mode must be recoverable from the tag.
"""

from __future__ import annotations

import re

import pytest

import run_lodo


@pytest.fixture(autouse=True)
def _restore():
    saved = (run_lodo.TRAINABLE_BLOCKS, run_lodo.FULL_FINETUNE,
             run_lodo.LEARNING_RATE, run_lodo.GRADIENT_CHECKPOINTING,
             run_lodo.BATCH_SIZE)
    yield
    (run_lodo.TRAINABLE_BLOCKS, run_lodo.FULL_FINETUNE,
     run_lodo.LEARNING_RATE, run_lodo.GRADIENT_CHECKPOINTING,
     run_lodo.BATCH_SIZE) = saved


def test_adaptation_mode_names_each_protocol() -> None:
    run_lodo.FULL_FINETUNE = False
    run_lodo.TRAINABLE_BLOCKS = None
    assert run_lodo.adaptation_mode() == "full_network"
    run_lodo.TRAINABLE_BLOCKS = 0
    assert run_lodo.adaptation_mode() == "linear_probe"
    run_lodo.TRAINABLE_BLOCKS = 4
    assert run_lodo.adaptation_mode() == "partial_finetune_4"
    run_lodo.TRAINABLE_BLOCKS = None
    run_lodo.FULL_FINETUNE = True
    assert run_lodo.adaptation_mode() == "full_finetune"


def test_full_finetune_tag_cannot_collide_with_partial() -> None:
    run_lodo.FULL_FINETUNE = False
    run_lodo.TRAINABLE_BLOCKS = 24
    partial = run_lodo._method_tag("erm")
    run_lodo.TRAINABLE_BLOCKS = None
    run_lodo.FULL_FINETUNE = True
    full = run_lodo._method_tag("erm")
    assert partial != full
    assert "ftfull" in full and "tb" not in full
    assert "tb24" in partial


def test_every_setting_that_changes_the_id_is_recoverable_from_it() -> None:
    """The id is the fallback source when a record is incomplete."""
    run_lodo.FULL_FINETUNE = False
    run_lodo.TRAINABLE_BLOCKS = 4
    run_lodo.LEARNING_RATE = 1e-4
    run_lodo.BATCH_SIZE = 16
    tag = run_lodo._method_tag("erm")
    assert re.search(r"-b16\b", tag)
    assert re.search(r"-tb4\b", tag)
    assert "lr0.0001" in tag


def test_assert_full_finetune_rejects_a_frozen_tensor() -> None:
    import torch
    from torch import nn

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    model[0].weight.requires_grad = False
    with pytest.raises(RuntimeError, match="frozen"):
        run_lodo.assert_full_finetune(model)

    for p in model.parameters():
        p.requires_grad = True
    total, trainable = run_lodo.assert_full_finetune(model)
    assert total == trainable


def test_gradient_checkpointing_refuses_a_backbone_without_support() -> None:
    from torch import nn

    class NoSupport(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="set_grad_checkpointing"):
        run_lodo.enable_gradient_checkpointing(NoSupport())


def test_registry_ids_agree_with_their_recorded_adaptation_mode() -> None:
    """Guards the bug itself against the real registry."""
    from pathlib import Path

    import pandas as pd

    path = Path("outputs/experiment_registry.csv")
    if not path.exists():
        pytest.skip("registry not present")
    frame = pd.read_csv(path)
    if "adaptation_mode" not in frame.columns:
        pytest.fail("registry has no adaptation_mode column")
    tb4 = frame[frame.experiment_id.astype(str).str.contains("-tb4")]
    assert not tb4.empty
    assert (tb4.adaptation_mode == "partial_finetune_4").all()
    assert (tb4.trainable_blocks == 4).all()
    assert frame.trainable_blocks.isna().sum() == 0
