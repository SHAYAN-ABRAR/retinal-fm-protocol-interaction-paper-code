"""Tests for the DG method comparison.

This script generates the evidence for the paper's central negative claim, so
the ways it could quietly overstate that claim are what get guarded:

* A verdict of "no method beats ERM" must require both bars, not just one.
* The point estimate printed beside a confidence interval must be the quantity
  that interval describes. The seed-42 bootstrap and the three-seed mean are
  different numbers and can disagree in sign -- MixStyle is +0.021 on seed 42
  and -0.006 averaged -- and a row showing the mean beside the anchor's interval
  reads as though the interval endorsed the mean.
* A method that was never run must report NOT RUN rather than being dropped,
  which would make the comparison look complete when it is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import analyse_methods  # noqa: E402
from src.utils.registry import make_experiment_id  # noqa: E402

TARGET = "eyepacs"


def _write(outputs, method, balanced, seed, y_true, y_pred):
    tag = f"{method}-b32" + ("-dbal" if balanced else "")
    experiment_id = make_experiment_id(
        protocol="lodo",
        sources=[d for d in analyse_methods.ALL_DOMAINS if d != TARGET],
        target=TARGET, backbone="densenet121", method=tag, seed=seed)
    frame = pd.DataFrame({
        "image_id": [f"{TARGET}/{i}.jpg" for i in range(len(y_true))],
        "true_grade": y_true, "predicted_grade": y_pred,
    })
    for grade in range(5):
        frame[f"probability_grade_{grade}"] = np.where(
            frame["predicted_grade"] == grade, 0.9, 0.025)
    frame.to_csv(outputs / "predictions"
                 / f"{experiment_id}__target_test[{TARGET}]_predictions.csv", index=False)
    (outputs / "reports" / f"{experiment_id}_evaluation.json").write_text(json.dumps({
        "temperature": {"temperature": 1.2, "fitted_on": "source_validation"},
        "component_statistics": {},
        "results": {"target_test": {"metrics": {"qwk": 0.0},
                                    "calibration": {"ece": 0.0},
                                    "calibration_scaled": {"ece": 0.0}}},
    }), encoding="utf-8")


def _workspace(tmp_path, method_wrong, n=400):
    """ERM gets 100 wrong; the method gets `method_wrong` wrong, per seed."""
    outputs = tmp_path / "outputs"
    for name in ("tables", "predictions", "reports"):
        (outputs / name).mkdir(parents=True, exist_ok=True)
    y_true = np.random.default_rng(0).integers(0, 5, size=n)

    def predictions(wrong, seed):
        y = y_true.copy()
        chosen = np.random.default_rng(seed).choice(n, size=wrong, replace=False)
        y[chosen] = (y[chosen] + 2) % 5
        return y

    for seed in analyse_methods.SEEDS:
        _write(outputs, "erm", False, seed, y_true, predictions(100, seed))
        _write(outputs, "deep_coral", False, seed, y_true,
               predictions(method_wrong(seed), seed + 50))
    return outputs


def _run(monkeypatch, tmp_path, outputs):
    import src.utils.io as io

    monkeypatch.setattr(io, "project_root", lambda: tmp_path)
    monkeypatch.setattr(analyse_methods, "METHODS", ["deep_coral"])
    monkeypatch.setattr(analyse_methods, "N_BOOTSTRAP", 200)
    monkeypatch.setattr(sys, "argv", ["analyse_methods.py"])
    analyse_methods.main()
    return pd.read_csv(outputs / "tables" / f"method_comparison_lodo_{TARGET}.csv")


def test_a_clearly_better_method_is_reported_as_beating_erm(monkeypatch, tmp_path):
    """If the script cannot detect a real win, its null result means nothing."""
    outputs = _workspace(tmp_path, lambda seed: 30)     # far fewer errors
    row = _run(monkeypatch, tmp_path, outputs).iloc[0]
    assert row["delta_qwk"] > 0
    assert row["verdict"] == "BEATS ERM"


def test_a_noisy_tie_is_within_seed_noise(monkeypatch, tmp_path):
    """A difference smaller than its own seed spread must not be called real."""
    outputs = _workspace(tmp_path, lambda seed: 100 + (seed % 3) * 20)
    row = _run(monkeypatch, tmp_path, outputs).iloc[0]
    assert abs(row["delta_qwk"]) <= row["delta_qwk_sd"]
    assert row["verdict"] == "within seed noise"


def test_the_interval_matches_the_estimate_it_describes(monkeypatch, tmp_path):
    """delta_qwk_seed42 must be the anchor seed's own difference."""
    from src.evaluation.bootstrap import METRIC_FUNCTIONS

    outputs = _workspace(tmp_path, lambda seed: 60)
    row = _run(monkeypatch, tmp_path, outputs).iloc[0]

    qwk = METRIC_FUNCTIONS["qwk"]
    pair = []
    for method in ("erm", "deep_coral"):
        experiment_id = make_experiment_id(
            protocol="lodo",
            sources=[d for d in analyse_methods.ALL_DOMAINS if d != TARGET],
            target=TARGET, backbone="densenet121",
            method=f"{method}-b32", seed=analyse_methods.ANCHOR_SEED)
        frame = pd.read_csv(outputs / "predictions"
                            / f"{experiment_id}__target_test[{TARGET}]_predictions.csv")
        pair.append(qwk(frame["true_grade"].to_numpy(),
                        frame["predicted_grade"].to_numpy(), None))

    assert row["delta_qwk_seed42"] == pytest.approx(pair[1] - pair[0], abs=1e-9)
    # The interval brackets the anchor's difference, not necessarily the mean.
    assert row["ci_lower"] <= row["delta_qwk_seed42"] <= row["ci_upper"]


def test_a_missing_method_is_reported_not_dropped(monkeypatch, tmp_path, capsys):
    outputs = _workspace(tmp_path, lambda seed: 60)
    import src.utils.io as io

    monkeypatch.setattr(io, "project_root", lambda: tmp_path)
    monkeypatch.setattr(analyse_methods, "METHODS", ["deep_coral", "groupdro"])
    monkeypatch.setattr(analyse_methods, "N_BOOTSTRAP", 200)
    monkeypatch.setattr(sys, "argv", ["analyse_methods.py"])
    analyse_methods.main()

    printed = capsys.readouterr().out
    assert "groupdro" in printed and "NOT RUN" in printed
    table = pd.read_csv(outputs / "tables" / f"method_comparison_lodo_{TARGET}.csv")
    assert set(table["method"]) == {"deep_coral"}
