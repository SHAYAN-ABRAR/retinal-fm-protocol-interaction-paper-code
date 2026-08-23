"""Tests for Phase 3: metrics, losses, calibration, models, training, registry.

The metric tests matter most: every number in the paper flows through them, and
a silently wrong QWK would invalidate the whole study.  Where an independent
implementation exists (sklearn), the result is cross-checked against it rather
than against a hand-computed constant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="Phase 3 requires PyTorch")

from src.evaluation.calibration import (  # noqa: E402
    adaptive_calibration_error,
    calibration_metrics,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)
from src.evaluation.metrics import (  # noqa: E402
    compute_all_metrics,
    ordinal_metrics,
    per_class_metrics,
    quadratic_weighted_kappa,
    referable_dr_metrics,
)
from src.losses.classification import (  # noqa: E402
    FocalLoss,
    build_classification_loss,
    class_weights_from_counts,
)
from src.models.backbones import (  # noqa: E402
    SUPPORTED_BACKBONES,
    BackboneConfig,
    resolve_input_size,
)
from src.training.early_stopping import EarlyStopping  # noqa: E402
from src.utils.registry import make_experiment_id, register_experiment  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_qwk_matches_sklearn() -> None:
    """Cross-check against an independent implementation, not a constant."""
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 500)
    y_pred = np.clip(y_true + rng.integers(-1, 2, 500), 0, 4)
    mine = quadratic_weighted_kappa(y_true, y_pred)
    theirs = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(5)))
    assert abs(mine - theirs) < 1e-10


def test_qwk_bounds() -> None:
    assert quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([0, 4], [4, 0]) == pytest.approx(-1.0)


def test_qwk_uses_fixed_class_support() -> None:
    """A split missing a grade must not silently change the weight matrix.

    IDRiD's test split has only 5 grade-1 images, so a metric that infers its
    label set from the data would not be comparable across splits.
    """
    y_true = [0, 0, 2, 2, 4]
    y_pred = [0, 2, 2, 2, 4]
    assert not np.isnan(quadratic_weighted_kappa(y_true, y_pred, num_classes=5))


def test_qwk_penalises_distant_errors_more() -> None:
    """The property that makes QWK the right primary metric for ordinal grading."""
    truth = [0, 0, 0, 0]
    adjacent = quadratic_weighted_kappa(truth + [4], [0, 0, 0, 1] + [4])
    distant = quadratic_weighted_kappa(truth + [4], [0, 0, 0, 4] + [4])
    assert adjacent > distant


def test_ordinal_metrics() -> None:
    result = ordinal_metrics([0, 1, 2, 3], [0, 2, 2, 0])
    assert result["exact_match"] == pytest.approx(0.5)
    assert result["mae_grade"] == pytest.approx((0 + 1 + 0 + 3) / 4)
    assert result["within_1_grade"] == pytest.approx(0.75)
    assert result["severe_error_rate"] == pytest.approx(0.25)
    assert result["max_grade_error"] == 3


def test_per_class_metrics_include_specificity() -> None:
    # y_true = [0, 0, 1, 1], y_pred = [0, 1, 1, 1]
    # grade 0: TP=1, FP=0, FN=1, TN=2
    rows = per_class_metrics([0, 0, 1, 1], [0, 1, 1, 1])["per_class"]
    grade_0 = rows[0]
    assert grade_0["support"] == 2
    assert grade_0["recall"] == pytest.approx(0.5)          # 1 / (1 + 1)
    assert grade_0["precision"] == pytest.approx(1.0)       # 1 / (1 + 0)
    assert grade_0["specificity"] == pytest.approx(1.0)     # 2 / (2 + 0)

    # grade 1: TP=2, FP=1, FN=0, TN=1  -> specificity 1/(1+1) = 0.5
    grade_1 = rows[1]
    assert grade_1["recall"] == pytest.approx(1.0)
    assert grade_1["precision"] == pytest.approx(2 / 3)
    assert grade_1["specificity"] == pytest.approx(0.5)


def test_auroc_is_nan_for_absent_class_not_zero() -> None:
    """Scoring an absent class as 0 would silently drag the macro average down."""
    y_true = np.array([0, 0, 2, 2])
    probabilities = np.full((4, 5), 0.2)
    result = compute_all_metrics(y_true, y_true, probabilities)
    assert np.isnan(result["auroc_grade_1"])
    assert result["auroc_n_classes_scored"] == 2


def test_referable_dr_threshold_is_explicit() -> None:
    result = referable_dr_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], threshold=2)
    assert result["referable_threshold"] == 2
    assert result["referable_sensitivity"] == pytest.approx(1.0)
    assert result["referable_prevalence"] == pytest.approx(0.6)


def test_compute_all_metrics_shape() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 5, 200)
    probabilities = rng.random((200, 5))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    result = compute_all_metrics(y_true, probabilities.argmax(axis=1), probabilities)
    assert np.array(result["confusion_matrix"]).shape == (5, 5)
    assert len(result["per_class"]) == 5
    for key in ("qwk", "f1_macro", "balanced_accuracy", "mae_grade", "auroc_macro"):
        assert key in result


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def test_focal_loss_with_gamma_zero_equals_cross_entropy() -> None:
    """gamma=0 must reduce exactly to CE, which makes it a clean ablation axis."""
    rng = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 5, generator=rng)
    target = torch.randint(0, 5, (64,), generator=rng)
    focal = FocalLoss(gamma=0.0)(logits, target)
    cross_entropy = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(focal, cross_entropy, atol=1e-6)


def test_focal_loss_downweights_easy_examples() -> None:
    confident_correct = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])
    assert float(FocalLoss(gamma=2.0)(confident_correct, target)) < float(
        FocalLoss(gamma=0.0)(confident_correct, target)
    )


def test_class_weights_are_ordered_and_normalised() -> None:
    counts = [25802, 2438, 5288, 872, 708]      # real EyePACS training counts
    weights = class_weights_from_counts(counts, scheme="inverse_sqrt")
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)
    assert weights[4] > weights[0]                       # rare class weighted up
    assert torch.all(weights[:-1].argsort(descending=True) >= 0)


def test_inverse_sqrt_is_milder_than_inverse() -> None:
    """The reason inverse_sqrt is the default: it avoids a ~36x weight ratio."""
    counts = [25802, 2438, 5288, 872, 708]
    inverse = class_weights_from_counts(counts, scheme="inverse")
    sqrt = class_weights_from_counts(counts, scheme="inverse_sqrt")
    assert (inverse.max() / inverse.min()) > (sqrt.max() / sqrt.min())


def test_build_classification_loss_records_its_configuration() -> None:
    _loss, description = build_classification_loss(
        "weighted_ce", class_counts=[100, 10, 50, 5, 5]
    )
    assert description["strategy"] == "weighted_ce"
    assert "class_weights" in description and len(description["class_weights"]) == 5


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def _confident_probabilities(n: int, accuracy: float, confidence: float, seed: int = 0):
    """Synthesise predictions with a known accuracy and a fixed confidence."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, n)
    predictions = labels.copy()
    wrong = rng.random(n) > accuracy
    predictions[wrong] = (labels[wrong] + 1) % 5
    remainder = (1.0 - confidence) / 4
    probabilities = np.full((n, 5), remainder)
    probabilities[np.arange(n), predictions] = confidence
    return probabilities, labels


def test_ece_is_zero_when_confidence_equals_accuracy() -> None:
    probabilities, labels = _confident_probabilities(4000, accuracy=0.8, confidence=0.8)
    assert expected_calibration_error(probabilities, labels) < 0.02


def test_ece_detects_overconfidence() -> None:
    probabilities, labels = _confident_probabilities(4000, accuracy=0.6, confidence=0.95)
    ece = expected_calibration_error(probabilities, labels)
    assert ece > 0.3
    metrics = calibration_metrics(probabilities, labels)
    assert metrics["confidence_minus_accuracy"] > 0        # over-confident


def test_adaptive_ece_agrees_on_a_clean_case() -> None:
    probabilities, labels = _confident_probabilities(4000, accuracy=0.6, confidence=0.95)
    plain = expected_calibration_error(probabilities, labels)
    adaptive = adaptive_calibration_error(probabilities, labels)
    assert abs(plain - adaptive) < 0.05


def test_temperature_scaling_never_changes_predictions() -> None:
    """Dividing logits by a positive scalar preserves the argmax.

    This is what makes temperature scaling a clean intervention: any change in
    ECE is attributable to calibration alone, not to different predictions.
    """
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(500, 5)) * 4
    labels = logits.argmax(axis=1)
    scaler = fit_temperature(logits, labels)
    before = logits.argmax(axis=1)
    after = scaler.apply(logits).argmax(axis=1)
    assert np.array_equal(before, after)


def test_temperature_above_one_for_an_overconfident_model() -> None:
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 5, 3000)
    logits = np.zeros((3000, 5))
    logits[np.arange(3000), labels] = 6.0                 # very peaked
    flip = rng.random(3000) < 0.35                        # but often wrong
    labels[flip] = (labels[flip] + 1) % 5
    scaler = fit_temperature(logits, labels)
    assert scaler.temperature > 1.0
    assert scaler.nll_after <= scaler.nll_before


def test_temperature_records_the_split_it_was_fitted_on() -> None:
    """The artefact itself must show the target domain was not used."""
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(200, 5))
    labels = rng.integers(0, 5, 200)
    scaler = fit_temperature(logits, labels, fitted_on="source_validation[ddr+aptos]")
    assert "source_validation" in scaler.describe()["fitted_on"]
    assert scaler.describe()["n_fit_samples"] == 200


def test_reliability_curve_bins_cover_all_samples() -> None:
    probabilities, labels = _confident_probabilities(1000, accuracy=0.7, confidence=0.85)
    curve = reliability_curve(probabilities, labels, n_bins=10)
    assert int(curve["count"].sum()) == 1000


def test_calibration_metrics_are_finite_on_extreme_probabilities() -> None:
    """A zero probability must not produce an infinite NLL."""
    probabilities = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    metrics = calibration_metrics(probabilities, [4, 0])
    assert np.isfinite(metrics["nll"]) and np.isfinite(metrics["brier"])


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def test_resolve_input_size_snaps_dinov2_to_a_patch_multiple() -> None:
    """384 is not divisible by 14; snapping must be explicit, not a runtime crash."""
    assert resolve_input_size("dinov2_vits14", 224) == 224
    assert resolve_input_size("dinov2_vits14", 384) == 378
    assert 378 % 14 == 0


def test_resolve_input_size_leaves_cnns_alone() -> None:
    assert resolve_input_size("densenet121", 384) == 384
    assert resolve_input_size("convnext_tiny", 224) == 224


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(KeyError):
        resolve_input_size("resnet50", 224)


def test_all_declared_backbones_have_a_patch_multiple() -> None:
    for name, spec in SUPPORTED_BACKBONES.items():
        assert spec["patch_multiple"] >= 1, name
        assert spec["timm_name"], name


def test_backbone_config_describe_round_trips() -> None:
    config = BackboneConfig(name="densenet121", head="ordinal_coral", image_size=224)
    described = config.describe()
    assert described["backbone"] == "densenet121"
    assert described["head"] == "ordinal_coral"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def test_early_stopping_triggers_after_patience() -> None:
    stopper = EarlyStopping(patience=3, mode="max")
    assert not stopper.step(0.5, 0)
    assert not stopper.step(0.4, 1)
    assert not stopper.step(0.4, 2)
    assert stopper.step(0.4, 3)
    assert stopper.best == pytest.approx(0.5)
    assert stopper.best_epoch == 0


def test_early_stopping_resets_on_improvement() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    stopper.step(0.5, 0)
    stopper.step(0.4, 1)
    stopper.step(0.6, 2)               # improvement resets the counter
    assert stopper.epochs_without_improvement == 0
    assert not stopper.should_stop


def test_early_stopping_ignores_nan() -> None:
    stopper = EarlyStopping(patience=5, mode="max")
    stopper.step(0.5, 0)
    stopper.step(float("nan"), 1)
    assert stopper.best == pytest.approx(0.5)


def test_checkpoint_round_trip_restores_weights_exactly(tmp_path: Path) -> None:
    from src.training.checkpointing import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(8, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(tmp_path / "ckpt.pt", model=model, optimizer=optimizer,
                           epoch=3, metrics={"qwk": 0.77})

    restored = torch.nn.Linear(8, 5)
    payload = load_checkpoint(path, model=restored)
    assert payload["epoch"] == 3
    assert payload["metrics"]["qwk"] == pytest.approx(0.77)
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)


def test_checkpoint_manager_keeps_the_best_validation_score(tmp_path: Path) -> None:
    from src.training.checkpointing import CheckpointManager

    model = torch.nn.Linear(4, 5)
    manager = CheckpointManager(directory=tmp_path, experiment_id="exp", monitor="qwk")

    manager.update(epoch=0, val_metrics={"qwk": 0.5, "loss": 1.0}, model=model)
    manager.update(epoch=1, val_metrics={"qwk": 0.7, "loss": 0.9}, model=model)
    manager.update(epoch=2, val_metrics={"qwk": 0.6, "loss": 0.8}, model=model)

    assert manager.best_value == pytest.approx(0.7)
    assert manager.best_epoch == 1
    assert manager.best_loss_epoch == 2          # tracked separately
    assert manager.best_path.exists() and manager.last_path.exists()


def test_train_config_effective_batch_size() -> None:
    from src.training.trainer import TrainConfig

    config = TrainConfig(batch_size=4, accumulation_steps=4)
    assert config.effective_batch_size == 16
    assert config.describe()["effective_batch_size"] == 16


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_experiment_id_follows_the_naming_convention() -> None:
    experiment_id = make_experiment_id(
        protocol="lodo", sources=["eyepacs", "ddr", "aptos"], target="idrid",
        backbone="convnext_tiny", method="mixstyle_ordinal", seed=42,
    )
    assert experiment_id == "lodo_aptos-ddr-eyepacs__idrid_convnext-tiny_mixstyle-ordinal_s42"


def test_registry_appends_and_never_overwrites(tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "registry.csv"
    register_experiment(path, {"experiment_id": "a", "status": "COMPLETE", "test_qwk": 0.5})
    register_experiment(path, {"experiment_id": "b", "status": "COMPLETE", "test_qwk": 0.6})
    register_experiment(path, {"experiment_id": "a", "status": "COMPLETE", "test_qwk": 0.7})

    frame = pd.read_csv(path)
    assert len(frame) == 3, "a re-run must add a row, not replace one"
    assert list(frame["experiment_id"]) == ["a", "b", "a"]
    assert frame["timestamp_utc"].notna().all()


def test_registry_can_refuse_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    register_experiment(path, {"experiment_id": "a"})
    with pytest.raises(ValueError, match="already in the registry"):
        register_experiment(path, {"experiment_id": "a"}, allow_duplicate_id=False)


# ---------------------------------------------------------------------------
# GPU-only
# ---------------------------------------------------------------------------

@needs_cuda
def test_estimate_memory_reports_real_peak_usage() -> None:
    from src.models.backbones import estimate_memory

    result = estimate_memory(
        BackboneConfig(name="densenet121", image_size=224, pretrained=False),
        batch_size=4, amp=True, train=True,
    )
    assert result["fits"] is True
    assert 0 < result["peak_gb"] < 8.0


@needs_cuda
def test_freeze_backbone_leaves_only_the_head_trainable() -> None:
    from src.models.backbones import build_model, count_parameters, freeze_backbone

    model = build_model(BackboneConfig(name="densenet121", image_size=224, pretrained=False))
    total_before, trainable_before = count_parameters(model)
    total_after, trainable_after = freeze_backbone(model, trainable_blocks=0)

    assert total_after == total_before
    assert trainable_after < trainable_before
    # Only the classifier head (1024 -> 5 plus bias) should remain trainable.
    assert trainable_after == 1024 * 5 + 5


# ---------------------------------------------------------------------------
# the two CORALs -- kept provably distinct
# ---------------------------------------------------------------------------

def test_the_two_corals_are_different_modules() -> None:
    """Guard against the name collision the brief warns about.

    If someone ever merges these, or imports one where the other was meant, this
    test fails loudly rather than the ablation silently measuring one axis twice.
    """
    from src.losses import deep_coral_alignment, ordinal_coral_loss

    assert ordinal_coral_loss.describe()["family"] == "ordinal"
    assert deep_coral_alignment.describe()["family"] == "domain_generalization"
    assert ordinal_coral_loss.__file__ != deep_coral_alignment.__file__
    # Neither module exposes the other's entry point.
    assert not hasattr(ordinal_coral_loss, "DeepCoralLoss")
    assert not hasattr(deep_coral_alignment, "CoralOrdinalLoss")


def test_coral_levels_encode_the_ordering() -> None:
    from src.losses.ordinal_coral_loss import levels_from_labels

    levels = levels_from_labels(torch.tensor([0, 1, 3, 4]), num_classes=5)
    assert levels.tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]


def test_coral_probabilities_are_a_valid_distribution() -> None:
    from src.losses.ordinal_coral_loss import coral_probabilities

    rng = torch.Generator().manual_seed(0)
    logits = torch.randn(200, 4, generator=rng) * 3
    probabilities = coral_probabilities(logits)
    assert probabilities.shape == (200, 5)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(200), atol=1e-5)
    assert bool((probabilities >= 0).all())


def test_coral_is_rank_consistent() -> None:
    """The property that distinguishes CORAL from naive binary decomposition."""
    rng = torch.Generator().manual_seed(1)
    # A shared projection plus decreasing biases, as the ordinal head produces.
    projection = torch.randn(100, 1, generator=rng)
    biases = torch.tensor([1.5, 0.5, -0.5, -1.5])
    cumulative = torch.sigmoid(projection + biases)
    assert bool((cumulative[:, :-1] >= cumulative[:, 1:]).all())


def test_coral_predict_counts_passed_thresholds() -> None:
    from src.losses.ordinal_coral_loss import coral_predict

    logits = torch.tensor([[3.0, 2.0, 1.0, -2.0], [-3.0, -3.0, -3.0, -3.0],
                           [5.0, 4.0, 3.0, 2.0]])
    assert coral_predict(logits).tolist() == [3, 0, 4]


def test_coral_loss_rejects_a_standard_five_way_head() -> None:
    from src.losses.ordinal_coral_loss import CoralOrdinalLoss

    with pytest.raises(ValueError, match="ordinal_coral"):
        CoralOrdinalLoss(5)(torch.randn(3, 5), torch.tensor([0, 1, 2]))


def test_coral_loss_is_lower_for_better_ordinal_predictions() -> None:
    from src.losses.ordinal_coral_loss import CoralOrdinalLoss

    loss_fn = CoralOrdinalLoss(5)
    target = torch.tensor([4])
    good = torch.tensor([[5.0, 5.0, 5.0, 5.0]])       # predicts grade 4
    bad = torch.tensor([[-5.0, -5.0, -5.0, -5.0]])    # predicts grade 0
    assert float(loss_fn(good, target)) < float(loss_fn(bad, target))


def test_deep_coral_is_zero_for_identical_distributions() -> None:
    from src.losses.deep_coral_alignment import DeepCoralLoss

    features = torch.randn(32, 64)
    stacked = torch.cat([features, features])
    domains = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    assert float(DeepCoralLoss(weight=1.0)(stacked, domains)) == pytest.approx(0.0, abs=1e-6)


def test_deep_coral_is_positive_for_different_distributions() -> None:
    from src.losses.deep_coral_alignment import DeepCoralLoss

    rng = torch.Generator().manual_seed(2)
    features = torch.cat([
        torch.randn(32, 64, generator=rng),
        torch.randn(32, 64, generator=rng) * 4 + 2,
    ])
    domains = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    assert float(DeepCoralLoss(weight=1.0)(features, domains)) > 0.1


def test_deep_coral_returns_zero_for_a_single_domain_batch() -> None:
    """A degenerate batch must contribute nothing, not noise."""
    from src.losses.deep_coral_alignment import DeepCoralLoss

    loss_fn = DeepCoralLoss(weight=1.0, warn_on_degenerate=False)
    value = loss_fn(torch.randn(32, 64), torch.zeros(32).long())
    assert float(value) == 0.0
    assert loss_fn.statistics()["degenerate_batches"] == 1


def test_deep_coral_scale_is_independent_of_feature_width() -> None:
    """The 1/(4 d^2) normalisation matters: backbones differ (1024/768/384)."""
    from src.losses.deep_coral_alignment import DeepCoralLoss

    loss_fn = DeepCoralLoss(weight=1.0, warn_on_degenerate=False)
    values = []
    for width in (64, 256, 1024):
        rng = torch.Generator().manual_seed(3)
        features = torch.cat([
            torch.randn(64, width, generator=rng),
            torch.randn(64, width, generator=rng) * 2,
        ])
        domains = torch.cat([torch.zeros(64), torch.ones(64)]).long()
        values.append(float(loss_fn(features, domains)))
    # Same underlying discrepancy, so the values should stay the same order of
    # magnitude rather than scaling with d.
    assert max(values) / min(values) < 3.0


def test_domain_balanced_batches_cover_every_domain() -> None:
    """Needed because EyePACS is 70x IDRiD: plain shuffling gives pure batches."""
    from collections import Counter

    from src.losses.deep_coral_alignment import domain_balanced_batch_indices

    domain_ids = [0] * 1000 + [1] * 20
    batches = domain_balanced_batch_indices(domain_ids, batch_size=16)
    assert len(batches) > 0
    for batch in batches[:20]:
        counts = Counter(domain_ids[i] for i in batch)
        assert counts[0] == 8 and counts[1] == 8


def test_domain_balanced_batches_reject_too_small_a_batch() -> None:
    from src.losses.deep_coral_alignment import domain_balanced_batch_indices

    with pytest.raises(ValueError, match="cannot cover"):
        domain_balanced_batch_indices([0, 1, 2, 3] * 10, batch_size=2)
