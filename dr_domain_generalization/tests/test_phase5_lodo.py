"""Tests for Phase 5: the full leave-one-domain-out matrix and its analysis.

The failure this file guards against is a partial matrix reported as a complete
one. Three of four targets finishing, and a mean quietly taken over whichever
rows happen to exist, would produce a headline number that looks exactly like
the one the paper claims to report but is not it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pd = pytest.importorskip("pandas", reason="Phase 5 analysis requires pandas")
np = pytest.importorskip("numpy")

import analyse_lodo  # noqa: E402
from src.data.splits import build_experiment_split  # noqa: E402

ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]


def _results_frame(targets, *, method="erm", seed=42):
    """A minimal lodo_results.csv covering the named targets."""
    return pd.DataFrame([
        {"target": t, "method": method, "seed": seed,
         "n_train": 27718, "n_test": 507,
         "source_qwk": 0.80, "target_qwk": 0.70,
         "source_f1": 0.50, "target_f1": 0.40,
         "source_ece": 0.05, "target_ece": 0.25,
         "target_ece_scaled": 0.10, "target_severe": 0.11, "seconds": 100.0}
        for t in targets
    ])


def _run_analysis(tmp_path, monkeypatch, capsys, targets):
    outputs = tmp_path / "outputs"
    (outputs / "tables").mkdir(parents=True)
    (outputs / "predictions").mkdir(parents=True)
    (outputs / "figures").mkdir(parents=True)
    _results_frame(targets).to_csv(outputs / "tables" / "lodo_results.csv", index=False)

    monkeypatch.setattr("src.utils.io.project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["analyse_lodo.py", "--n-bootstrap", "50"])
    analyse_lodo.main()
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# A partial matrix must never be averaged
# ---------------------------------------------------------------------------
def test_missing_targets_are_named_not_dropped(tmp_path, monkeypatch, capsys) -> None:
    output = _run_analysis(tmp_path, monkeypatch, capsys, ["ddr", "aptos"])

    assert "NOT RUN" in output
    for absent in ("idrid", "eyepacs"):
        assert absent in output, f"{absent} vanished from the table instead of being flagged"


def test_no_mean_is_reported_over_a_partial_matrix(tmp_path, monkeypatch, capsys) -> None:
    """The mean QWK drop is the paper's headline; a 2-of-4 mean is not that number."""
    output = _run_analysis(tmp_path, monkeypatch, capsys, ["ddr", "aptos"])

    assert "mean QWK change" not in output
    assert "2 of 4 targets NOT RUN" in output


def test_mean_is_reported_once_all_four_targets_exist(tmp_path, monkeypatch, capsys) -> None:
    output = _run_analysis(tmp_path, monkeypatch, capsys, ALL_DOMAINS)

    assert "mean QWK change" in output
    assert "NOT RUN" not in output


def test_missing_predictions_are_reported_rather_than_skipped(
    tmp_path, monkeypatch, capsys
) -> None:
    """A result row without its predictions file must say so, not silently omit CIs."""
    output = _run_analysis(tmp_path, monkeypatch, capsys, ["ddr"])

    assert "predictions file is missing" in output


# ---------------------------------------------------------------------------
# Bootstrap metric names must exist
# ---------------------------------------------------------------------------
def test_requested_bootstrap_metrics_are_all_known() -> None:
    """A typo'd metric name would raise only after a 3-hour run had finished."""
    from src.evaluation.bootstrap import METRIC_FUNCTIONS

    unknown = set(analyse_lodo.BOOTSTRAP_METRICS) - set(METRIC_FUNCTIONS)
    assert not unknown, f"unknown bootstrap metric(s) {sorted(unknown)}"


# ---------------------------------------------------------------------------
# The experiment ids the analyser looks for must match the ones the runner writes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target", ALL_DOMAINS)
def test_analyser_and_runner_agree_on_experiment_ids(target: str) -> None:
    """If these drift, the analyser reports every run as NOT RUN and nobody notices."""
    import run_lodo
    from src.utils.registry import make_experiment_id

    sources = [d for d in run_lodo.ALL_DOMAINS if d != target]
    expected = make_experiment_id(
        protocol="lodo", sources=sources, target=target,
        backbone=run_lodo.BACKBONE,
        method=f"erm-b{run_lodo.BATCH_SIZE}", seed=42,
    )
    assert analyse_lodo._experiment_id(target, "erm", 42) == expected


def test_runner_and_analyser_share_the_same_domain_set() -> None:
    import run_lodo

    assert sorted(run_lodo.ALL_DOMAINS) == sorted(analyse_lodo.ALL_TARGETS)


def test_domain_indices_match_the_fixed_numbering() -> None:
    """Domain numbering is fixed by the brief: 0=DDR, 1=APTOS, 2=IDRiD, 3=EyePACS."""
    assert analyse_lodo.DOMAIN_INDEX == {"ddr": 0, "aptos": 1, "idrid": 2, "eyepacs": 3}


# ---------------------------------------------------------------------------
# Target isolation, restated at the LODO level
# ---------------------------------------------------------------------------
def _toy_manifest() -> "pd.DataFrame":
    """A four-domain manifest with within-domain splits already assigned."""
    from src.data.splits import assign_within_domain_splits

    rows = []
    for index, domain in enumerate(ALL_DOMAINS):
        for i in range(40):
            rows.append({
                "image_id": f"{domain}/{i}.jpg",
                "domain": domain,
                "domain_id": index,
                "path": f"D:/cache/{domain}/{i}.jpg",
                "grade": i % 5,
                "patient_id": f"{domain}_p{i // 2}",
                "eye": "left" if i % 2 == 0 else "right",
                "source_split": "train",
            })
    return assign_within_domain_splits(pd.DataFrame(rows))


@pytest.mark.parametrize("target", ALL_DOMAINS)
def test_no_target_image_reaches_train_or_val(target: str) -> None:
    manifest = _toy_manifest()
    sources = [d for d in ALL_DOMAINS if d != target]
    experiment = build_experiment_split(
        manifest, protocol="lodo", sources=sources, target=target
    )

    assert target not in set(experiment.train["domain"])
    assert target not in set(experiment.val["domain"])
    assert set(experiment.test["domain"]) == {target}

    target_ids = set(manifest.loc[manifest["domain"] == target, "image_id"])
    assert not target_ids & set(experiment.train["image_id"])
    assert not target_ids & set(experiment.val["image_id"])


@pytest.mark.parametrize("target", ALL_DOMAINS)
def test_lodo_uses_exactly_three_source_domains(target: str) -> None:
    manifest = _toy_manifest()
    sources = [d for d in ALL_DOMAINS if d != target]
    experiment = build_experiment_split(
        manifest, protocol="lodo", sources=sources, target=target
    )

    assert set(experiment.train["domain"]) == set(sources)
    assert len(sources) == 3
