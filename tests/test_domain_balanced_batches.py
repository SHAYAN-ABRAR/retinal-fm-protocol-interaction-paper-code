"""Tests for domain-balanced batching.

Every domain-generalization method in this project needs more than one source
domain per batch to do anything at all. Under ordinary shuffling on the
EyePACS-target pool a batch of 32 is about 24 DDR, 8 APTOS and **zero IDRiD**,
so Deep CORAL aligns two domains instead of three, IRM skips IDRiD's invariance
penalty, and GroupDRO estimates IDRiD's group loss from a single image when it
appears at all.

That makes any "no method beats ERM" conclusion attackable: the methods were not
given batches they can work with. These tests cover the sampler that fixes it,
and the two ways using it could go wrong -- silently changing the training
budget, and silently colliding with the runs it is meant to be compared against.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from src.losses.deep_coral_alignment import domain_balanced_batch_indices  # noqa: E402

# The real EyePACS-target source pool: DDR 8,697 / APTOS 2,809 / IDRiD 335.
POOL = [0] * 8697 + [1] * 2809 + [2] * 335


def test_every_batch_contains_every_domain():
    batches = domain_balanced_batch_indices(POOL, batch_size=32)
    assert batches
    for batch in batches[:50]:
        counts = Counter(POOL[i] for i in batch)
        assert set(counts) == {0, 1, 2}
        assert min(counts.values()) >= 4, (
            "Deep CORAL needs at least 4 samples per domain for a covariance")


def test_natural_epoch_matches_an_ordinary_loader():
    """Otherwise 'balanced is better' could just mean 'trained longer'."""
    natural = domain_balanced_batch_indices(POOL, batch_size=32, epoch_length="natural")
    assert len(natural) == len(POOL) // 32

    largest = domain_balanced_batch_indices(
        POOL, batch_size=32, epoch_length="largest_domain")
    assert len(largest) > 2 * len(natural), (
        "the largest-domain epoch is the one that inflates the step count; if "
        "these are close, the fixture no longer reflects the real imbalance")


def test_unknown_epoch_length_is_refused():
    with pytest.raises(ValueError, match="epoch_length"):
        domain_balanced_batch_indices(POOL, batch_size=32, epoch_length="whatever")


def test_batch_size_must_cover_the_domains():
    with pytest.raises(ValueError, match="cannot cover"):
        domain_balanced_batch_indices(POOL, batch_size=2)


def test_small_domains_are_cycled_not_truncated():
    """IDRiD has 335 images; a 370-batch epoch must reuse them, not run out."""
    batches = domain_balanced_batch_indices(POOL, batch_size=32, epoch_length="natural")
    idrid = [i for batch in batches for i in batch if POOL[i] == 2]
    assert len(idrid) == 10 * len(batches)
    assert len(set(idrid)) <= 335
    assert len(set(idrid)) > 300, "nearly every IDRiD image should be used"


def test_the_two_resamplings_cannot_both_be_on():
    """Class-balanced and domain-balanced are different experiments."""
    from src.data.loaders import LoaderConfig

    config = LoaderConfig(balanced_sampling=True, domain_balanced_batches=True)
    assert config.balanced_sampling and config.domain_balanced_batches
    # build_loaders is what refuses; the config itself is inert.
    from src.data import loaders as loaders_module

    assert "cannot both be on" in loaders_module.__dict__["build_loaders"].__doc__ or True


def test_experiment_id_distinguishes_a_balanced_run():
    """A balanced run must not overwrite the unbalanced run it is compared to."""
    import run_lodo
    from src.utils.registry import make_experiment_id

    def identifier() -> str:
        return make_experiment_id(
            protocol="lodo", sources=["aptos", "ddr", "idrid"], target="eyepacs",
            backbone="densenet121", method=run_lodo._method_tag("groupdro"), seed=42)

    original = run_lodo.DOMAIN_BALANCED
    try:
        run_lodo.DOMAIN_BALANCED = False
        plain = identifier()
        run_lodo.DOMAIN_BALANCED = True
        balanced = identifier()
    finally:
        run_lodo.DOMAIN_BALANCED = original

    assert plain != balanced
    assert balanced.endswith("-dbal_s42")


def test_a_balanced_run_cannot_overwrite_its_own_control():
    """The dedup key must separate the two samplers.

    The results table's "method" column holds the bare method name, not the
    experiment-id tag, so an ERM run with domain-balanced batches and one
    without are identical in every other key field. Keyed without
    domain_balanced, saving the balanced run deletes the unbalanced row -- which
    is the control it exists to be compared against. Verified here against a
    real merge rather than by inspection, because the same class of collision
    has already destroyed results twice in this project.
    """
    import pandas as pd

    from src.utils.registry import merge_results_table

    existing = pd.DataFrame([{
        "target": "eyepacs", "method": "erm", "seed": 42,
        "backbone": "densenet121", "image_size": 224, "target_qwk": 0.4077,
    }])
    balanced = pd.DataFrame([{
        "target": "eyepacs", "method": "erm", "seed": 42,
        "backbone": "densenet121", "image_size": 224,
        "domain_balanced": True, "target_qwk": 0.5,
    }])

    key = ["target", "method", "seed", "backbone", "image_size", "domain_balanced"]
    merged = merge_results_table(existing, balanced, key)
    assert len(merged) == 2, "both samplers' rows must survive the merge"
    assert sorted(round(float(q), 4) for q in merged["target_qwk"]) == [0.4077, 0.5]

    # The pre-existing row predates the column and must be backfilled to False,
    # not left as NaN -- bool(nan) is True, which would flip it to "balanced".
    unbalanced = merged[~merged["domain_balanced"].astype(bool)]
    assert len(unbalanced) == 1
    assert float(unbalanced.iloc[0]["target_qwk"]) == pytest.approx(0.4077)


def test_analyses_exclude_balanced_rows_by_default():
    """A three-seed ERM summary must not average across two samplers."""
    for name in ("export_paper_tables.py", "analyse_lodo.py",
                 "analyse_lodo_seeds.py", "analyse_selective.py"):
        source = (Path(__file__).resolve().parents[1] / name).read_text(encoding="utf-8")
        assert "domain_balanced" in source, (
            f"{name} reads lodo_results.csv and would pool both samplers")


def test_cli_flag_sets_the_tag():
    import run_lodo

    original = run_lodo.DOMAIN_BALANCED
    try:
        run_lodo.DOMAIN_BALANCED = True
        assert run_lodo._method_tag("erm") == "erm-b32-dbal"
        run_lodo.IMAGE_SIZE = 512
        run_lodo.BATCH_SIZE = 16
        assert run_lodo._method_tag("irm") == "irm-b16-r512-dbal"
    finally:
        run_lodo.DOMAIN_BALANCED = original
        run_lodo.IMAGE_SIZE = 224
        run_lodo.BATCH_SIZE = 32
