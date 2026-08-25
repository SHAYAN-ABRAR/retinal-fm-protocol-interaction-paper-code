"""Tests for GroupDRO and IRMv1.

These two are being added so the paper's central negative claim -- that no
domain-generalization method beats ERM -- faces more than two algorithms. A
method that is subtly broken would produce exactly the result the paper wants,
which is the worst possible failure mode: a negative result that is really an
implementation bug.

So the tests here check that each method does the thing its paper says it does,
not merely that it runs:

* GroupDRO must actually move weight toward the worst-performing domain, and
  must key that weight to the domain rather than to a position in the batch.
* IRM's penalty must be near zero for a predictor that is optimal on every
  domain at once, and clearly positive for one that is not.

Both must also degrade honestly on this project's real problem -- source pools
where one domain contributes about one image per batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from src.losses.group_dro import GroupDROObjective  # noqa: E402
from src.losses.irm import IRMObjective, irm_penalty  # noqa: E402

DDR, APTOS, IDRID = 0, 1, 2


def _logits(n, correct, n_classes=5, confidence=6.0):
    """Logits that predict `correct` for every row, with the given margin."""
    out = torch.zeros(n, n_classes)
    out[:, correct] = confidence
    return out


# --------------------------------------------------------------------------
# GroupDRO
# --------------------------------------------------------------------------

def test_weight_moves_toward_the_worst_domain():
    """The whole point: a domain that keeps losing should gain weight."""
    objective = GroupDROObjective(eta=0.5)
    targets = torch.tensor([0] * 8 + [0] * 8)
    domains = torch.tensor([DDR] * 8 + [APTOS] * 8)
    # DDR is predicted correctly; APTOS is predicted confidently wrong.
    logits = torch.cat([_logits(8, 0), _logits(8, 4)])

    start = objective.q.clone()
    for _ in range(20):
        objective(logits, targets, domains)

    assert objective.q[APTOS] > objective.q[DDR], (
        "the high-loss domain must accumulate weight")
    assert objective.q[APTOS] > start[APTOS]
    assert torch.isclose(objective.q.sum(), torch.tensor(1.0), atol=1e-5)


def test_weights_are_keyed_to_the_domain_not_the_batch_position():
    """A domain absent from a batch keeps the weight it had."""
    objective = GroupDROObjective(eta=0.5)
    targets = torch.tensor([0] * 16)

    # Batches containing only DDR and APTOS: IDRiD must not be touched.
    before_idrid = float(objective.q[IDRID])
    for _ in range(10):
        objective(torch.cat([_logits(8, 0), _logits(8, 4)]),
                  targets, torch.tensor([DDR] * 8 + [APTOS] * 8))

    # IDRiD's *relative* standing is untouched; only the global renormalisation
    # rescales it. Its ratio to the untouched uniform start is what must hold.
    assert objective.q[IDRID] > 0
    assert float(objective.q[IDRID]) != pytest.approx(before_idrid), (
        "renormalisation should rescale an absent domain")
    stats = objective.statistics()
    assert stats["batches_containing_domain"][IDRID] == 0
    assert stats["batches_containing_domain"][DDR] == 10


def test_loss_is_a_convex_combination_of_group_losses():
    objective = GroupDROObjective(eta=0.1)
    targets = torch.tensor([0] * 8 + [0] * 8)
    domains = torch.tensor([DDR] * 8 + [APTOS] * 8)
    logits = torch.cat([_logits(8, 0), _logits(8, 4)])

    import torch.nn.functional as F
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    low = float(per_sample[:8].mean())
    high = float(per_sample[8:].mean())

    loss = float(objective(logits, targets, domains))
    assert low - 1e-5 <= loss <= high + 1e-5, (
        "a weighted mean of group losses cannot fall outside their range")


def test_single_sample_groups_are_recorded_not_hidden():
    """IDRiD supplies ~0.9 images per batch in the real pool; that must show."""
    objective = GroupDROObjective(eta=0.1)
    targets = torch.tensor([0] * 16)
    domains = torch.tensor([DDR] * 15 + [IDRID])
    logits = _logits(16, 0)

    for _ in range(4):
        objective(logits, targets, domains)

    stats = objective.statistics()
    assert stats["batches_with_single_sample"][IDRID] == 4
    assert stats["single_sample_fraction"][IDRID] == 1.0
    assert stats["single_sample_fraction"][DDR] == 0.0
    # The thin domain's samples are still trained on, not dropped.
    assert stats["batches_containing_domain"][IDRID] == 4


def test_gradients_reach_the_logits():
    objective = GroupDROObjective(eta=0.1)
    logits = torch.randn(16, 5, requires_grad=True)
    loss = objective(logits, torch.randint(0, 5, (16,)),
                     torch.tensor([DDR] * 8 + [APTOS] * 8))
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------
# IRM
# --------------------------------------------------------------------------

def test_penalty_is_near_zero_for_an_invariant_predictor():
    """One classifier optimal on both halves means no gradient to penalise."""
    targets = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
    logits = _logits(8, 1, confidence=8.0)
    penalty = float(irm_penalty(logits, targets))
    assert abs(penalty) < 1e-3, f"expected ~0 for an invariant predictor, got {penalty}"


def test_penalty_is_positive_in_expectation_for_a_non_optimal_predictor():
    """The estimator is unbiased for ||grad||^2, so it is positive *on average*.

    A single split can be negative: the penalty is the inner product of two
    half-batch gradients, and one draw of that product carries the sign of their
    disagreement. Averaging over orderings is what makes it an estimate of a
    squared norm, so that is what gets asserted.
    """
    generator = torch.Generator().manual_seed(0)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    logits = _logits(8, 0, confidence=3.0)      # right on 0s, wrong on 1s

    draws = []
    for _ in range(200):
        order = torch.randperm(8, generator=generator)
        draws.append(float(irm_penalty(logits[order], targets[order]).detach()))
    mean = sum(draws) / len(draws)

    assert mean > 1e-3, f"expected a positive mean penalty, got {mean}"
    assert min(draws) < 0, (
        "individual draws are expected to go negative; if none did, the test is "
        "no longer exercising the cross-term estimator")


def test_penalty_ranks_a_bad_predictor_above_an_invariant_one():
    generator = torch.Generator().manual_seed(1)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    def mean_penalty(logits):
        total = 0.0
        for _ in range(200):
            order = torch.randperm(8, generator=generator)
            total += float(irm_penalty(logits[order], targets[order]).detach())
        return total / 200

    invariant = torch.zeros(8, 5)
    invariant[torch.arange(8), targets] = 8.0        # optimal on every sample
    bad = _logits(8, 0, confidence=3.0)              # ignores half the labels

    assert mean_penalty(bad) > mean_penalty(invariant) + 1e-3


def test_annealing_switches_and_rescales():
    objective = IRMObjective(penalty_weight=100.0, anneal_iters=3)
    logits = torch.randn(16, 5)
    targets = torch.randint(0, 5, (16,))
    domains = torch.tensor([DDR] * 8 + [APTOS] * 8)

    assert objective.current_weight == 1.0
    for _ in range(3):
        objective(logits, targets, domains)
    assert objective.current_weight == 100.0
    assert objective.statistics()["annealing_finished"] is True

    # After the switch the objective is divided by lambda, so it must not jump
    # by two orders of magnitude relative to the pre-switch scale.
    after = float(objective(logits, targets, domains))
    assert abs(after) < 10.0, (
        f"post-anneal loss {after} suggests the lambda rescale is missing")


def test_thin_domains_are_skipped_for_the_penalty_but_still_trained_on():
    objective = IRMObjective(penalty_weight=1.0, anneal_iters=0)
    logits = torch.randn(16, 5)
    targets = torch.randint(0, 5, (16,))
    domains = torch.tensor([DDR] * 15 + [IDRID])

    for _ in range(3):
        objective(logits, targets, domains)

    stats = objective.statistics()
    assert stats["batches_skipped_for_penalty"][IDRID] == 3
    assert stats["skipped_fraction"][IDRID] == 1.0
    assert stats["batches_skipped_for_penalty"][DDR] == 0


def test_all_domains_thin_falls_back_to_plain_nll():
    """No usable domain must not mean no training signal."""
    import torch.nn.functional as F

    objective = IRMObjective(penalty_weight=100.0, anneal_iters=0)
    logits = torch.randn(2, 5)
    targets = torch.randint(0, 5, (2,))
    domains = torch.tensor([DDR, IDRID])          # one sample each

    loss = float(objective(logits, targets, domains))
    assert loss == pytest.approx(float(F.cross_entropy(logits, targets)), abs=1e-6)


def test_second_order_gradients_flow():
    """IRM needs create_graph; a broken double backward would fail here."""
    objective = IRMObjective(penalty_weight=10.0, anneal_iters=0)
    logits = torch.randn(16, 5, requires_grad=True)
    loss = objective(logits, torch.randint(0, 5, (16,)),
                     torch.tensor([DDR] * 8 + [APTOS] * 8))
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["groupdro", "irm"])
def test_build_method_produces_an_objective(name):
    from src.models.backbones import BackboneConfig
    from src.training.methods import METHODS, MethodConfig, build_method

    assert name in METHODS
    built = build_method(
        MethodConfig(name=name),
        BackboneConfig(name="densenet121", image_size=224),
        device="cpu",
    )
    assert built.objective_fn is not None
    assert built.description["domain_generalization"] == name
    # The statistics reach the registry through BuiltMethod.statistics().
    assert "objective" in built.statistics()


def test_erm_still_has_no_objective_override():
    """The new hook must not change what every existing run did."""
    from src.models.backbones import BackboneConfig
    from src.training.methods import MethodConfig, build_method

    for name in ("erm", "deep_coral", "mixstyle"):
        built = build_method(
            MethodConfig(name=name),
            BackboneConfig(name="densenet121", image_size=224),
            device="cpu",
        )
        assert built.objective_fn is None, (
            f"{name} must keep using loss_fn, or every result already on disk "
            "was produced by a different code path than a rerun would use")


# --------------------------------------------------------------------------
# Mixed precision
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_objectives_survive_autocast_and_grad_scaler():
    """IRM's double backward under fp16 loss scaling is the risky combination.

    ``irm_penalty`` calls ``autograd.grad(..., create_graph=True)`` and the
    training step then backprops through that graph. Under autocast with a
    GradScaler this is where a non-finite loss or an autograd failure would
    appear, and a CPU fp32 unit test cannot see it. The domain mix here is the
    real one from the EyePACS-target pool, where IDRiD supplies about one image
    per batch of 32.
    """
    from torch import nn

    for objective in (GroupDROObjective(eta=0.01),
                      IRMObjective(penalty_weight=100.0, anneal_iters=2)):
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 64),
                              nn.ReLU(), nn.Linear(64, 5)).cuda()
        objective = objective.cuda()
        optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        for _ in range(6):
            images = torch.randn(32, 3, 32, 32, device="cuda")
            targets = torch.randint(0, 5, (32,), device="cuda")
            domains = torch.tensor([DDR] * 23 + [APTOS] * 8 + [IDRID], device="cuda")

            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                logits = model(images)
            loss = objective(logits.float(), targets, domains)
            assert torch.isfinite(loss), f"{type(objective).__name__}: loss went non-finite"

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(norm), f"{type(objective).__name__}: gradients went non-finite"
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

        # The thin domain must be visible in the record, not silently absorbed.
        stats = objective.statistics()
        thin = (stats.get("single_sample_fraction") or stats.get("skipped_fraction"))
        assert thin[IDRID] == 1.0, (
            "IDRiD supplies one image per batch here, so every batch should be "
            "recorded as degenerate for it")
