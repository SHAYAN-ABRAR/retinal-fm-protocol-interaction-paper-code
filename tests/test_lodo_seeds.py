"""Tests for the multi-seed LODO aggregation.

Two failure modes are guarded here, both of which produce output that looks
entirely reasonable:

1. Claiming "within seed noise" when only one seed exists. That asserts variance
   was measured and found small; in fact nothing was measured.
2. Differencing scores computed on different test sets. The LODO runs are
   evaluated on the whole target domain while the in-domain model can only be
   tested on that domain's held-out split, so a comparison that forgets to
   restrict them credits the LODO model for images the in-domain model trained
   on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import analyse_lodo_seeds  # noqa: E402

ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]


def _predictions(image_ids, true, predicted):
    frame = pd.DataFrame({
        "image_id": image_ids,
        "true_grade": true,
        "predicted_grade": predicted,
    })
    for grade in range(5):
        frame[f"probability_grade_{grade}"] = np.where(frame["predicted_grade"] == grade, 0.9, 0.025)
    return frame


def _build_workspace(tmp_path, seeds, *, target="ddr", n_in_domain=200, n_extra=300):
    """A LODO run tested on n_in_domain + n_extra images; in-domain on n_in_domain."""
    outputs = tmp_path / "outputs"
    (outputs / "tables").mkdir(parents=True)
    (outputs / "predictions").mkdir(parents=True)

    rng = np.random.default_rng(0)
    shared_ids = [f"{target}/test_{i}.jpg" for i in range(n_in_domain)]
    extra_ids = [f"{target}/train_{i}.jpg" for i in range(n_extra)]
    shared_truth = rng.integers(0, 5, size=n_in_domain)
    extra_truth = rng.integers(0, 5, size=n_extra)

    # In-domain model: perfect on its own test split.
    in_id = (f"in_domain_{target}__{target}_densenet121_erm-b32_s42")
    _predictions(shared_ids, shared_truth, shared_truth).to_csv(
        outputs / "predictions" / f"{in_id}__target_test[{target}]_predictions.csv", index=False)

    sources = "-".join(sorted(d for d in ALL_TARGETS if d != target))
    rows = []
    for seed in seeds:
        seed_rng = np.random.default_rng(seed)
        # Wrong on the shared split, but perfect on the extra images the
        # in-domain model trained on. A comparison that fails to restrict to the
        # shared ids will therefore look far too good.
        shared_predicted = (shared_truth + seed_rng.integers(1, 3, size=n_in_domain)) % 5
        lodo_id = f"lodo_{sources}__{target}_densenet121_erm-b32_s{seed}"
        _predictions(
            list(shared_ids) + list(extra_ids),
            np.concatenate([shared_truth, extra_truth]),
            np.concatenate([shared_predicted, extra_truth]),
        ).to_csv(
            outputs / "predictions" / f"{lodo_id}__target_test[{target}]_predictions.csv",
            index=False)
        rows.append({
            "target": target, "method": "erm", "seed": seed,
            "n_train": 27718, "n_test": n_in_domain + n_extra,
            "source_qwk": 0.8, "target_qwk": 0.95,   # inflated by the extra images
            "source_f1": 0.5, "target_f1": 0.5,
            "source_ece": 0.05, "target_ece": 0.2,
            "target_ece_scaled": 0.1, "target_severe": 0.1, "seconds": 100.0,
        })
    pd.DataFrame(rows).to_csv(outputs / "tables" / "lodo_results.csv", index=False)

    pd.DataFrame([{
        "domain": target, "n_test": n_in_domain,
        "in_domain_qwk": 1.0, "lodo_qwk": 0.5, "delta_qwk": -0.5,
        "ci_lower": -0.6, "ci_upper": -0.4, "ci_excludes_zero": True,
    }]).to_csv(outputs / "tables" / "in_domain_vs_lodo_erm_s42.csv", index=False)
    return outputs


def _run(tmp_path, monkeypatch, capsys, seeds):
    _build_workspace(tmp_path, seeds)
    monkeypatch.setattr("src.utils.io.project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["analyse_lodo_seeds.py"])
    analyse_lodo_seeds.main()
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# One seed cannot establish anything about variance
# ---------------------------------------------------------------------------
def test_single_seed_reports_cannot_assess(tmp_path, monkeypatch, capsys) -> None:
    output = _run(tmp_path, monkeypatch, capsys, [42])

    assert "CANNOT ASSESS (1 seed)" in output
    assert "within seed noise" not in output


def test_single_seed_shows_no_standard_deviation(tmp_path, monkeypatch, capsys) -> None:
    """An SD printed from one sample would be a fabricated number."""
    output = _run(tmp_path, monkeypatch, capsys, [42])

    assert "n/a" in output


def test_three_seeds_produce_a_verdict(tmp_path, monkeypatch, capsys) -> None:
    output = _run(tmp_path, monkeypatch, capsys, [42, 1, 2])

    assert "CANNOT ASSESS" not in output
    assert any(v in output for v in ("REAL (both bars)", "within seed noise", "CI spans zero"))


# ---------------------------------------------------------------------------
# The comparison must use only the in-domain test images
# ---------------------------------------------------------------------------
def test_lodo_mean_is_restricted_to_the_in_domain_test_split(
    tmp_path, monkeypatch, capsys
) -> None:
    """The fixture's LODO run is perfect on images the in-domain model trained on.

    lodo_results.csv therefore reports 0.95. Restricted to the shared test
    images the model is deliberately wrong, so the reported LODO mean must be
    far below 0.95 -- if it is not, the restriction was skipped.
    """
    output = _run(tmp_path, monkeypatch, capsys, [42, 1, 2])

    # Take the ddr row of the deployment-cost table specifically; "ddr" also
    # appears in the completeness list further up.
    lines = output.splitlines()
    start = next(i for i, l in enumerate(lines) if "cost of cross-domain deployment" in l)
    line = next(l for l in lines[start:] if l.strip().startswith("ddr"))
    # columns: target, in-domain, LODO mean, ...
    lodo_mean = float(line.split()[2])
    assert lodo_mean < 0.5, f"LODO mean {lodo_mean} looks unrestricted (csv says 0.95)"


def test_mismatched_ground_truth_is_refused(tmp_path, monkeypatch, capsys) -> None:
    """Stale prediction files must raise, not silently compare wrong pairs."""
    outputs = _build_workspace(tmp_path, [42])
    path = next((outputs / "predictions").glob("in_domain_*target_test*.csv"))
    frame = pd.read_csv(path)
    frame["true_grade"] = (frame["true_grade"] + 1) % 5
    frame.to_csv(path, index=False)

    monkeypatch.setattr("src.utils.io.project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["analyse_lodo_seeds.py"])
    with pytest.raises(AssertionError, match="disagree on ground truth"):
        analyse_lodo_seeds.main()


def test_missing_targets_are_reported_not_dropped(tmp_path, monkeypatch, capsys) -> None:
    output = _run(tmp_path, monkeypatch, capsys, [42])

    for absent in ("aptos", "idrid", "eyepacs"):
        assert absent in output


def test_experiment_ids_match_the_runner(tmp_path) -> None:
    import run_lodo
    from src.utils.registry import make_experiment_id

    for target in ALL_TARGETS:
        sources = [d for d in run_lodo.ALL_DOMAINS if d != target]
        expected = make_experiment_id(
            protocol="lodo", sources=sources, target=target,
            backbone=run_lodo.BACKBONE, method=f"erm-b{run_lodo.BATCH_SIZE}", seed=1,
        )
        assert analyse_lodo_seeds._experiment_id(target, "erm", 1) == expected
