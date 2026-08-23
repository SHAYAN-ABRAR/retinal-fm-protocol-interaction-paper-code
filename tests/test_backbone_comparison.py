"""Tests for the ConvNeXt-vs-DenseNet comparison.

The original Phase 8 table was produced without a script, and two of its
properties were invisible to anyone reading it:

1. Its significance bar was DenseNet's across-seed SD, borrowed because
   ConvNeXt had exactly one seed. Nothing in the table said so, so a ratio of
   4.5 read as "four and a half times its own seed noise" when ConvNeXt's own
   seed noise had never been measured.
2. Its calibration column compared two backbones by sign alone, with no
   interval, on one seed.

Both are guarded here. The third test guards the ordinary way a paired analysis
goes wrong -- differencing scores that were computed on different images.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import analyse_backbone  # noqa: E402
from src.utils.registry import make_experiment_id  # noqa: E402

TARGET = "ddr"
ALL_TARGETS = analyse_backbone.ALL_TARGETS


def _write_run(outputs, backbone, seed, y_true, y_pred, *, temperature=1.2):
    """One run's prediction CSV and evaluation report, as the trainer writes them."""
    experiment_id = make_experiment_id(
        protocol="lodo", sources=[d for d in ALL_TARGETS if d != TARGET],
        target=TARGET, backbone=backbone,
        method=f"erm-b{analyse_backbone.BATCH_SIZE}", seed=seed,
    )
    frame = pd.DataFrame({
        "image_id": [f"{TARGET}/test_{i}.jpg" for i in range(len(y_true))],
        "true_grade": y_true,
        "predicted_grade": y_pred,
    })
    for grade in range(5):
        frame[f"probability_grade_{grade}"] = np.where(
            frame["predicted_grade"] == grade, 0.9, 0.025)
    frame.to_csv(
        outputs / "predictions" / f"{experiment_id}__target_test[{TARGET}]_predictions.csv",
        index=False)

    (outputs / "reports" / f"{experiment_id}_evaluation.json").write_text(
        json.dumps({
            "temperature": {"temperature": temperature, "fitted_on": "source_validation"},
            "results": {"target_test": {
                "metrics": {"qwk": 0.0},
                "calibration": {"ece": 0.0},
                "calibration_scaled": {"ece": 0.0},
            }},
        }), encoding="utf-8")
    return experiment_id


def _workspace(tmp_path, *, densenet_seeds, convnext_seeds, n=400):
    """A workspace where ConvNeXt is deliberately a little better than DenseNet.

    Both backbones are given the same labels; they differ only in how many of
    them they get right, and by how much that varies from seed to seed.
    """
    outputs = tmp_path / "outputs"
    for name in ("tables", "predictions", "reports"):
        (outputs / name).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=n)

    def predictions(wrong_count, seed):
        """Correct everywhere except `wrong_count` images, chosen per seed."""
        y_pred = y_true.copy()
        chosen = np.random.default_rng(seed).choice(n, size=wrong_count, replace=False)
        y_pred[chosen] = (y_pred[chosen] + 2) % 5
        return y_pred

    for seed in densenet_seeds:
        _write_run(outputs, "densenet121", seed, y_true, predictions(80 + seed % 5, seed))
    for seed in convnext_seeds:
        _write_run(outputs, "convnext_tiny", seed, y_true, predictions(40 + seed % 5, seed))
    return outputs, y_true


def _run(monkeypatch, tmp_path, outputs):
    import src.utils.io as io

    monkeypatch.setattr(io, "project_root", lambda: tmp_path)
    # ALL_TARGETS is deliberately left alone. It is not just the loop range --
    # the source list, and so the experiment id, is derived from it, so
    # shortening it here would make main() look for files under names the
    # fixture never wrote. The other three targets simply report no pair.
    monkeypatch.setattr(analyse_backbone, "N_BOOTSTRAP", 200)
    monkeypatch.setattr(sys, "argv", ["analyse_backbone.py"])
    analyse_backbone.main()
    return pd.read_csv(outputs / "tables" / "backbone_comparison.csv")


def test_single_candidate_seed_labels_the_bar_as_borrowed(monkeypatch, tmp_path):
    """One ConvNeXt seed means no ConvNeXt SD -- and the table must say so."""
    outputs, _ = _workspace(tmp_path, densenet_seeds=[42, 1, 2], convnext_seeds=[42])
    table = _run(monkeypatch, tmp_path, outputs)

    row = table.iloc[0]
    assert row["n_paired_seeds"] == 1
    assert row["sd_source"].startswith("proxy: densenet121")
    # The bar it used is DenseNet's, not something computed from one number.
    assert np.isfinite(row["delta_qwk_sd"])
    # And the calibration verdict does not pretend to a bar it does not have.
    assert np.isnan(row["delta_ece_scaled_sd"])
    assert row["ece_verdict"] == "no SD available"


def test_paired_seeds_use_the_sd_of_the_differences(monkeypatch, tmp_path):
    """Bar 1 is the SD of per-seed differences, not a difference of SDs."""
    outputs, y_true = _workspace(tmp_path, densenet_seeds=[42, 1, 2],
                                 convnext_seeds=[42, 1, 2])
    table = _run(monkeypatch, tmp_path, outputs)

    row = table.iloc[0]
    assert row["n_paired_seeds"] == 3
    assert row["sd_source"] == "paired (3 seeds)"

    from src.evaluation.bootstrap import METRIC_FUNCTIONS

    qwk = METRIC_FUNCTIONS["qwk"]
    differences = []
    for seed in (1, 2, 42):
        pair = []
        for backbone in ("densenet121", "convnext_tiny"):
            experiment_id = make_experiment_id(
                protocol="lodo", sources=[d for d in ALL_TARGETS if d != TARGET],
                target=TARGET, backbone=backbone,
                method=f"erm-b{analyse_backbone.BATCH_SIZE}", seed=seed)
            frame = pd.read_csv(
                outputs / "predictions"
                / f"{experiment_id}__target_test[{TARGET}]_predictions.csv")
            pair.append(qwk(frame["true_grade"].to_numpy(),
                            frame["predicted_grade"].to_numpy(), None))
        differences.append(pair[1] - pair[0])

    assert row["delta_qwk"] == pytest.approx(float(np.mean(differences)), abs=1e-9)
    assert row["delta_qwk_sd"] == pytest.approx(
        float(np.std(differences, ddof=1)), abs=1e-9)

    # Differencing the two across-seed SDs would give something quite different;
    # confirm the test would actually notice.
    naive = abs(row["densenet_qwk_sd"] - row["convnext_qwk_sd"])
    assert not np.isclose(row["delta_qwk_sd"], naive, atol=1e-6)


def test_mismatched_labels_are_refused(monkeypatch, tmp_path):
    """Two runs whose labels disagree must raise, not be silently differenced."""
    outputs, y_true = _workspace(tmp_path, densenet_seeds=[42], convnext_seeds=[])
    corrupted = (y_true + 1) % 5
    _write_run(outputs, "convnext_tiny", 42, corrupted, corrupted)

    with pytest.raises(AssertionError, match="labels disagree"):
        _run(monkeypatch, tmp_path, outputs)


def test_lower_ece_counts_as_an_improvement(monkeypatch, tmp_path):
    """ECE runs the opposite way to QWK; the verdict must respect that."""
    outputs, _ = _workspace(tmp_path, densenet_seeds=[42, 1, 2],
                            convnext_seeds=[42, 1, 2])
    table = _run(monkeypatch, tmp_path, outputs)
    row = table.iloc[0]

    if row["ece_verdict"].startswith("REAL"):
        improved = row["ece_verdict"] == "REAL (better)"
        assert improved == (row["delta_ece_scaled"] < 0), (
            "a negative ECE difference means ConvNeXt is better calibrated")

    # QWK runs the usual way, and here ConvNeXt is built to be more accurate.
    assert row["delta_qwk"] > 0
    assert row["qwk_verdict"] == "REAL (better)"
