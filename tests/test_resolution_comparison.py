"""Tests for the 224 px vs 512 px comparison.

The comparison is confounded on purpose and unavoidably: 512 px at batch 32
does not fit in 8 GB, so the 512 px arm runs at batch 16. That makes the two
arms differ in two settings at once. The script is allowed to report the
comparison -- it is the only one this hardware can run -- but it is not allowed
to present it as an isolated resolution effect, and the first test holds it to
that.

The rest guard the same failure modes as every other paired analysis here: a
borrowed significance bar passing as a measured one, and metrics whose better
direction is downwards being scored as though it were upwards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import analyse_resolution  # noqa: E402
from src.utils.registry import make_experiment_id  # noqa: E402

TARGET = "eyepacs"
ALL_TARGETS = analyse_resolution.ALL_TARGETS


def _write_run(outputs, protocol, image_size, batch_size, seed, y_true, y_pred):
    sources = ([TARGET] if protocol == "in_domain"
               else [d for d in ALL_TARGETS if d != TARGET])
    tag = f"erm-b{batch_size}" + (f"-r{image_size}" if image_size != 224 else "")
    experiment_id = make_experiment_id(
        protocol=protocol, sources=sources, target=TARGET,
        backbone=analyse_resolution.BACKBONE, method=tag, seed=seed,
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
        outputs / "predictions"
        / f"{experiment_id}__target_test[{TARGET}]_predictions.csv", index=False)
    (outputs / "reports" / f"{experiment_id}_evaluation.json").write_text(
        json.dumps({
            "temperature": {"temperature": 1.1, "fitted_on": "source_validation"},
            "results": {"target_test": {
                "metrics": {"qwk": 0.0},
                "calibration": {"ece": 0.0},
                "calibration_scaled": {"ece": 0.0},
            }},
        }), encoding="utf-8")


def _workspace(tmp_path, *, seeds_224, seeds_512, protocol="lodo", n=400):
    """512 px is built to be genuinely better: fewer wrong, fewer of them severe."""
    outputs = tmp_path / "outputs"
    for name in ("tables", "predictions", "reports"):
        (outputs / name).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 5, size=n)

    def predictions(wrong_count, step, seed):
        y_pred = y_true.copy()
        chosen = np.random.default_rng(seed).choice(n, size=wrong_count, replace=False)
        y_pred[chosen] = (y_pred[chosen] + step) % 5
        return y_pred

    for seed in seeds_224:
        _write_run(outputs, protocol, 224, 32, seed,
                   y_true, predictions(120 + seed % 7, 2, seed))
    for seed in seeds_512:
        _write_run(outputs, protocol, 512, 16, seed,
                   y_true, predictions(60 + seed % 7, 1, seed + 100))
    return outputs, y_true


def _run(monkeypatch, tmp_path, outputs):
    import src.utils.io as io

    monkeypatch.setattr(io, "project_root", lambda: tmp_path)
    monkeypatch.setattr(analyse_resolution, "N_BOOTSTRAP", 200)
    monkeypatch.setattr(sys, "argv", ["analyse_resolution.py"])
    analyse_resolution.main()
    return pd.read_csv(outputs / "tables" / "resolution_comparison.csv")


def test_every_row_records_the_batch_size_confound(monkeypatch, tmp_path):
    """The comparison changes two settings, and each row has to say which."""
    outputs, _ = _workspace(tmp_path, seeds_224=[42, 1, 2], seeds_512=[42])
    table = _run(monkeypatch, tmp_path, outputs)

    assert len(table) >= 1
    for configuration in table["configuration"]:
        assert "224px b32" in configuration and "512px b16" in configuration, (
            "a row that names only the resolution invites the result being read "
            "as an isolated resolution effect")


def test_single_512_seed_labels_the_bar_as_borrowed(monkeypatch, tmp_path):
    outputs, _ = _workspace(tmp_path, seeds_224=[42, 1, 2], seeds_512=[42])
    row = _run(monkeypatch, tmp_path, outputs).iloc[0]

    assert row["n_paired_seeds"] == 1
    assert row["sd_source"].startswith("proxy: 224px")
    assert np.isfinite(row["delta_qwk_sd"])


def test_paired_seeds_replace_the_borrowed_bar(monkeypatch, tmp_path):
    outputs, _ = _workspace(tmp_path, seeds_224=[42, 1, 2], seeds_512=[42, 1, 2])
    row = _run(monkeypatch, tmp_path, outputs).iloc[0]

    assert row["n_paired_seeds"] == 3
    assert row["sd_source"] == "paired (3 seeds)"

    from src.evaluation.bootstrap import METRIC_FUNCTIONS

    qwk = METRIC_FUNCTIONS["qwk"]
    differences = []
    for seed in (1, 2, 42):
        pair = []
        for image_size, batch_size in ((224, 32), (512, 16)):
            tag = f"erm-b{batch_size}" + (f"-r{image_size}" if image_size != 224 else "")
            experiment_id = make_experiment_id(
                protocol="lodo", sources=[d for d in ALL_TARGETS if d != TARGET],
                target=TARGET, backbone=analyse_resolution.BACKBONE,
                method=tag, seed=seed)
            frame = pd.read_csv(
                outputs / "predictions"
                / f"{experiment_id}__target_test[{TARGET}]_predictions.csv")
            pair.append(qwk(frame["true_grade"].to_numpy(),
                            frame["predicted_grade"].to_numpy(), None))
        differences.append(pair[1] - pair[0])

    assert row["delta_qwk_sd"] == pytest.approx(
        float(np.std(differences, ddof=1)), abs=1e-9)


def test_fewer_severe_errors_counts_as_an_improvement(monkeypatch, tmp_path):
    """severe_error_rate is better when it falls; the verdict must agree."""
    outputs, _ = _workspace(tmp_path, seeds_224=[42, 1, 2], seeds_512=[42, 1, 2])
    row = _run(monkeypatch, tmp_path, outputs).iloc[0]

    assert row["delta_severe"] < 0, "the fixture makes 512 px the better arm"
    assert row["severe_verdict"] != "REAL (worse)"
    if row["severe_verdict"].startswith("REAL"):
        assert row["severe_verdict"] == "REAL (better)"


def test_both_protocols_are_reported_separately(monkeypatch, tmp_path):
    """in-domain and LODO are different comparisons and must not be pooled."""
    outputs, _ = _workspace(tmp_path, seeds_224=[42], seeds_512=[42], protocol="lodo")
    # Called for its side effect: it writes the in-domain arm into the same
    # workspace, so main() sees both protocols for the same target.
    _workspace(tmp_path, seeds_224=[42], seeds_512=[42], protocol="in_domain")
    table = _run(monkeypatch, tmp_path, outputs)

    assert set(table["protocol"]) == {"lodo", "in_domain"}
    assert len(table[(table["protocol"] == "lodo") & (table["target"] == TARGET)]) == 1
    assert len(table[(table["protocol"] == "in_domain")
                     & (table["target"] == TARGET)]) == 1
