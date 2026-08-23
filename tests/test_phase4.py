"""Tests for Phase 4: MixStyle and the method-assembly layer.

The important property tested here is that a method's *name* matches what it
actually does. A run registered as "mixstyle" whose MixStyle modules never fired
is plain ERM under a misleading label, and would silently corrupt the ablation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="Phase 4 requires PyTorch")

from src.models.backbones import BackboneConfig  # noqa: E402
from src.models.mixstyle import (  # noqa: E402
    MixStyle,
    insert_mixstyle,
    mixstyle_modules,
    set_mixstyle_domains,
)
from src.training.methods import METHODS, MethodConfig, build_method  # noqa: E402

# Untrained weights keep these tests fast and offline.
BACKBONE = BackboneConfig(name="densenet121", image_size=224, pretrained=False)
CLASS_COUNTS = [5744, 717, 3881, 309, 855]


# ---------------------------------------------------------------------------
# MixStyle
# ---------------------------------------------------------------------------

def test_mixstyle_is_a_noop_in_eval_mode() -> None:
    """Mixing at test time would make predictions depend on batch composition."""
    module = MixStyle(p=1.0)
    module.eval()
    x = torch.randn(8, 16, 8, 8)
    assert torch.equal(module(x), x)


def test_mixstyle_changes_features_in_train_mode() -> None:
    module = MixStyle(p=1.0)
    module.train()
    module.set_domain_ids(torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    x = torch.randn(8, 16, 8, 8)
    assert not torch.equal(module(x), x)


def test_mixstyle_preserves_shape_and_is_finite() -> None:
    module = MixStyle(p=1.0)
    module.train()
    module.set_domain_ids(torch.tensor([0, 1] * 4))
    x = torch.randn(8, 16, 8, 8)
    out = module(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mixstyle_p_zero_never_applies() -> None:
    module = MixStyle(p=0.0)
    module.train()
    x = torch.randn(8, 16, 8, 8)
    assert torch.equal(module(x), x)


def test_mixstyle_skips_a_single_sample_batch() -> None:
    """Nothing to mix with; must pass through rather than error."""
    module = MixStyle(p=1.0)
    module.train()
    x = torch.randn(1, 16, 8, 8)
    assert torch.equal(module(x), x)
    assert module.n_skipped == 1


def test_mixstyle_pairs_across_domains() -> None:
    """Mixing within one domain teaches nothing about domain invariance."""
    module = MixStyle(p=1.0)
    domains = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    module.set_domain_ids(domains)
    permutation = module._cross_domain_permutation(8, torch.device("cpu"))
    assert (domains[permutation] != domains).all()


def test_mixstyle_rejects_non_spatial_features() -> None:
    module = MixStyle(p=1.0)
    module.train()
    module.set_domain_ids(torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="NCHW"):
        module(torch.randn(2, 16))


def test_mixstyle_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError):
        MixStyle(p=1.5)
    with pytest.raises(ValueError):
        MixStyle(alpha=0.0)


def test_insert_mixstyle_finds_the_early_stages() -> None:
    from src.models.backbones import build_model

    model = build_model(BACKBONE)
    inserted = insert_mixstyle(model, n_stages=2, p=0.5)
    assert len(inserted) == 2
    assert len(mixstyle_modules(model)) == 2


def test_set_mixstyle_domains_reaches_every_module() -> None:
    from src.models.backbones import build_model

    model = build_model(BACKBONE)
    insert_mixstyle(model, n_stages=2)
    domains = torch.tensor([0, 1, 0, 1])
    assert set_mixstyle_domains(model, domains) == 2
    for module in mixstyle_modules(model):
        assert torch.equal(module._domain_ids, domains)


def test_model_with_mixstyle_is_deterministic_in_eval() -> None:
    """Calibration numbers would be meaningless if evaluation were stochastic."""
    from src.models.backbones import build_model

    model = build_model(BACKBONE)
    insert_mixstyle(model, n_stages=2, p=1.0)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.inference_mode():
        assert torch.allclose(model(x), model(x))


# ---------------------------------------------------------------------------
# method assembly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", METHODS)
def test_every_method_builds_and_produces_a_valid_distribution(name: str) -> None:
    built = build_method(
        MethodConfig(name=name), BACKBONE, class_counts=CLASS_COUNTS, device="cpu"
    )
    built.model.train()
    x = torch.randn(8, 3, 224, 224)
    domains = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])

    if built.batch_hook is not None:
        built.batch_hook(built.model, domains)

    if built.feature_loss is not None:
        logits, features = built.model(x, return_features=True)
        auxiliary = built.feature_loss(features.float(), domains)
        assert torch.isfinite(auxiliary)
    else:
        logits = built.model(x)

    loss = built.loss_fn(logits.float(), targets)
    assert torch.isfinite(loss)

    probabilities = (
        built.to_probabilities(logits) if built.to_probabilities is not None
        else torch.softmax(logits, dim=-1)
    )
    assert probabilities.shape == (8, 5)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(8), atol=1e-5)


def test_ordinal_methods_use_the_ordinal_head() -> None:
    """K-1 threshold logits, not K class logits."""
    for name in ("ordinal", "mixstyle_ordinal", "deep_coral_ordinal"):
        built = build_method(MethodConfig(name=name), BACKBONE, device="cpu")
        assert built.model(torch.randn(2, 3, 224, 224)).shape == (2, 4)
        assert built.to_probabilities is not None


def test_non_ordinal_methods_use_the_linear_head() -> None:
    for name in ("erm", "deep_coral", "mixstyle"):
        built = build_method(MethodConfig(name=name), BACKBONE, device="cpu")
        assert built.model(torch.randn(2, 3, 224, 224)).shape == (2, 5)
        assert built.to_probabilities is None


def test_only_deep_coral_methods_get_a_feature_loss() -> None:
    for name in METHODS:
        built = build_method(MethodConfig(name=name), BACKBONE, device="cpu")
        expected = "deep_coral" in name
        assert (built.feature_loss is not None) is expected, name


def test_only_mixstyle_methods_get_mixstyle_modules() -> None:
    """The label must match the mechanism, or the ablation is meaningless."""
    for name in METHODS:
        built = build_method(MethodConfig(name=name), BACKBONE, device="cpu")
        expected = "mixstyle" in name
        assert (len(mixstyle_modules(built.model)) > 0) is expected, name
        assert (built.batch_hook is not None) is expected, name


def test_build_method_does_not_mutate_the_caller_backbone_config() -> None:
    """An ordinal build must not leave a later ERM build with an ordinal head."""
    config = BackboneConfig(name="densenet121", image_size=224, pretrained=False)
    assert config.head == "linear"
    build_method(MethodConfig(name="ordinal"), config, device="cpu")
    assert config.head == "linear"
    assert build_method(MethodConfig(name="erm"), config, device="cpu").model(
        torch.randn(2, 3, 224, 224)
    ).shape == (2, 5)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        MethodConfig(name="magic")


def test_method_description_records_both_axes_separately() -> None:
    """Ordinal and DG are distinct ablation axes and must be recorded as such."""
    combined = MethodConfig(name="deep_coral_ordinal").describe()
    assert combined["ordinal"] is True
    assert combined["domain_generalization"] == "deep_coral"

    plain = MethodConfig(name="erm").describe()
    assert plain["ordinal"] is False
    assert plain["domain_generalization"] == "none"


def test_built_method_statistics_expose_whether_components_fired() -> None:
    """A DG run whose component never fired must be detectable after the fact."""
    built = build_method(MethodConfig(name="mixstyle_ordinal"), BACKBONE, device="cpu")
    built.model.train()
    set_mixstyle_domains(built.model, torch.tensor([0, 1, 0, 1]))
    built.model(torch.randn(4, 3, 224, 224))
    stats = built.statistics()
    assert stats["mixstyle_n_modules"] == 2
    assert stats["mixstyle"]["batches_mixed"] + stats["mixstyle"]["batches_skipped"] > 0


# ---------------------------------------------------------------------------
# the ordinal evaluation path -- regression tests for a bug that reached a run
# ---------------------------------------------------------------------------

def test_ordinal_head_needs_a_probability_converter() -> None:
    """Regression: the ordinal sweep crashed because K-1 logits were softmaxed.

    Without a converter the evaluation produced a (N, 4) array where every metric
    expects (N, 5). The error must be explicit and name the fix.
    """
    import numpy as np

    from src.evaluation.evaluate import predict_with_logits

    class _FourLogitModel(torch.nn.Module):
        def forward(self, x):  # noqa: D102
            return torch.randn(len(x), 4)

    loader = [(torch.randn(6, 3, 8, 8), torch.zeros(6, dtype=torch.long),
               torch.zeros(6, dtype=torch.long), torch.arange(6))]

    with pytest.raises(ValueError, match="to_probabilities"):
        predict_with_logits(_FourLogitModel(), loader, device="cpu", amp=False)


def test_coral_native_rule_is_invariant_to_temperature() -> None:
    """The property that makes temperature scaling a calibration-only step.

    sigmoid(x / T) > 0.5 iff x > 0 for any T > 0, so the threshold count cannot
    change. The argmax of the differenced distribution is NOT invariant, which is
    why the native rule is used for predictions.
    """
    from src.losses.ordinal_coral_loss import coral_predict, coral_probabilities

    rng = torch.Generator().manual_seed(0)
    logits = torch.randn(500, 4, generator=rng) * 2

    for temperature in (0.5, 1.0, 2.0, 5.0):
        assert torch.equal(coral_predict(logits), coral_predict(logits / temperature))

    # And the contrast: argmax of the distribution does move.
    argmax_at_1 = coral_probabilities(logits).argmax(dim=1)
    argmax_at_half = coral_probabilities(logits / 0.5).argmax(dim=1)
    assert not torch.equal(argmax_at_1, argmax_at_half)


def test_ordinal_methods_supply_the_native_prediction_rule() -> None:
    for name in ("ordinal", "mixstyle_ordinal", "deep_coral_ordinal"):
        built = build_method(MethodConfig(name=name), BACKBONE, device="cpu")
        assert built.predict_fn is not None, name
        assert built.description["prediction_rule"] == "coral_threshold_count"

    for name in ("erm", "deep_coral", "mixstyle"):
        built = build_method(MethodConfig(name=name), BACKBONE, device="cpu")
        assert built.predict_fn is None, name
        assert built.description["prediction_rule"] == "argmax_softmax"


def test_confidence_is_the_probability_of_the_predicted_class() -> None:
    """Not the maximum probability -- they differ when the rules disagree."""
    import numpy as np

    from src.evaluation.evaluate import build_prediction_table

    probabilities = np.array([[0.1, 0.6, 0.1, 0.1, 0.1], [0.5, 0.2, 0.1, 0.1, 0.1]])
    outputs = {
        "probabilities": probabilities,
        "y_true": np.array([1, 0]),
        "y_pred": np.array([2, 0]),          # first row: NOT the argmax
        "index": np.array([0, 1]),
    }
    manifest = __import__("pandas").DataFrame({
        "image_id": ["a", "b"], "domain": ["ddr", "ddr"], "path": ["a.jpg", "b.jpg"],
    })
    table = build_prediction_table(outputs, manifest)

    assert table["predicted_grade"].tolist() == [2, 0]
    assert table["confidence"].tolist() == pytest.approx([0.1, 0.5])


def test_ordinal_temperature_scaling_produces_a_valid_distribution() -> None:
    import numpy as np

    from src.evaluation.calibration import fit_temperature
    from src.losses.ordinal_coral_loss import coral_probabilities

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 400)
    projection = (labels - 2.0)[:, None] * 1.5 + rng.normal(0, 0.8, (400, 1))
    logits = projection + np.array([1.5, 0.5, -0.5, -1.5])
    convert = lambda t: coral_probabilities(t, 5)  # noqa: E731

    scaler = fit_temperature(logits, labels, to_probabilities=convert)
    assert scaler.head == "ordinal_coral"
    assert scaler.nll_after <= scaler.nll_before

    scaled = scaler.apply(logits, convert)
    assert scaled.shape == (400, 5)
    assert np.allclose(scaled.sum(axis=1), 1.0, atol=1e-6)
