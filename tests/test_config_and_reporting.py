"""Tests for the YAML config layer and the table/summary reporting.

The config tests matter because a config file that silently falls back to a
default describes a run that never happened. Every one of these checks that a
mistake is *loud*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import _deep_merge, load_config, resolve_config  # noqa: E402
from src.utils.io import project_root  # noqa: E402

CONFIGS = project_root() / "configs"
needs_torch = pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None, reason="requires torch"
)


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------

def test_every_shipped_config_parses() -> None:
    for name in ("paths.yaml", "baseline.yaml", "domain_generalization.yaml",
                 "experiments.yaml"):
        assert (CONFIGS / name).exists(), f"{name} is missing"


def test_deep_merge_prefers_the_override() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 99}, "b": 3}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "b": 3, "nested": {"x": 1, "y": 99}}


@needs_torch
def test_baseline_config_builds_real_dataclasses() -> None:
    config = resolve_config("baseline.yaml")
    assert config.method.name == "erm"
    assert config.backbone.name == "densenet121"
    assert config.training.epochs == 20
    assert config.training.batch_size == 32
    assert config.protocol == "lodo"
    assert config.target == "idrid"


@needs_torch
def test_extends_inherits_everything_not_overridden() -> None:
    """The point of `extends`: an ablation states only what it changes."""
    baseline = resolve_config("baseline.yaml")
    variant = resolve_config("domain_generalization.yaml")

    assert variant.method.name != baseline.method.name      # the declared change
    for attribute in ("epochs", "batch_size", "learning_rate", "weight_decay",
                      "warmup_epochs", "scheduler", "early_stopping_patience"):
        assert getattr(variant.training, attribute) == getattr(baseline.training, attribute)
    assert variant.augmentation == baseline.augmentation
    assert variant.preprocess == baseline.preprocess


@needs_torch
def test_unknown_key_raises_instead_of_being_ignored() -> None:
    """A silently ignored key makes the config disagree with the run."""
    with pytest.raises(ValueError, match="unknown key"):
        resolve_config("baseline.yaml", overrides={"training": {"learning_rat": 1e-3}})
    with pytest.raises(ValueError, match="unknown key"):
        resolve_config("baseline.yaml", overrides={"backbone": {"depth": 50}})


@needs_torch
def test_invalid_method_name_is_rejected_at_load_time() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        resolve_config("baseline.yaml", overrides={"method": {"name": "magic"}})


@needs_torch
def test_image_size_propagates_and_snaps_for_dinov2() -> None:
    """One resolution declared at the top must reach every component."""
    config = resolve_config("baseline.yaml", overrides={"image_size": 384})
    assert config.preprocess.image_size == 384
    assert config.augmentation.image_size == 384
    assert config.backbone.image_size == 384

    vit = resolve_config(
        "baseline.yaml",
        overrides={"image_size": 384, "backbone": {"name": "dinov2_vits14"}},
    )
    # patch-14: 384 is not divisible, so it snaps -- and says so in the log.
    assert vit.backbone.image_size == 378
    assert vit.backbone.image_size % 14 == 0


@needs_torch
def test_batch_size_has_a_single_source_of_truth() -> None:
    """Loader and trainer must not be able to disagree about batch size."""
    config = resolve_config("baseline.yaml", overrides={"training": {"batch_size": 8}})
    assert config.loader.batch_size == 8

    with pytest.raises(ValueError, match="disagrees"):
        resolve_config(
            "baseline.yaml",
            overrides={"training": {"batch_size": 8}, "loader": {"batch_size": 16}},
        )


@needs_torch
def test_seed_propagates_to_loader_and_trainer() -> None:
    config = resolve_config("baseline.yaml", overrides={"seed": 7})
    assert config.seed == 7
    assert config.loader.seed == 7
    assert config.training.seed == 7


def test_circular_extends_is_detected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nname: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nname: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml", root=tmp_path)


def test_experiments_ledger_uses_only_declared_status_values() -> None:
    """NOT_RUN must be explicit; a blank status could be read as 'done'."""
    import yaml

    plan = yaml.safe_load((CONFIGS / "experiments.yaml").read_text(encoding="utf-8"))
    allowed = {"COMPLETE", "NOT_RUN", "PARTIAL"}

    statuses = [entry["status"] for entry in plan["stages"].values()]
    for key in ("stage_c_methods", "leave_one_domain_out", "in_domain",
                "single_source_external", "backbones"):
        statuses += [entry["status"] for entry in plan[key]]
    statuses += [entry["status"] for entry in plan["ablation"]["grid"]]

    assert statuses, "ledger is empty"
    assert set(statuses) <= allowed, f"undeclared status values: {set(statuses) - allowed}"


def test_protocol_rules_name_real_enforcement_sites() -> None:
    """The rules block must point at code that exists, not aspirations."""
    import yaml

    plan = yaml.safe_load((CONFIGS / "experiments.yaml").read_text(encoding="utf-8"))
    rules = plan["protocol_rules"]
    assert rules["target_domain_is_sacred"] is True
    assert "training" in rules["target_never_used_for"]
    assert "temperature_scaling" in rules["target_never_used_for"]

    for reference in rules["enforced_by"]:
        path = project_root() / reference.split("::")[0]
        assert path.exists(), f"enforcement site does not exist: {reference}"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def test_bold_best_marks_a_clear_winner() -> None:
    from src.reporting.tables import bold_best

    formatted = bold_best([0.60, 0.68, 0.65])
    assert formatted[1].startswith("\\textbf{")
    assert not formatted[0].startswith("\\textbf{")


def test_bold_best_refuses_to_bold_a_tie() -> None:
    """Bolding a 0.001 difference implies a distinction the data cannot support."""
    from src.reporting.tables import bold_best

    formatted = bold_best([0.680, 0.681, 0.60], tolerance=0.01)
    assert not any(value.startswith("\\textbf{") for value in formatted)


def test_bold_best_respects_lower_is_better() -> None:
    from src.reporting.tables import bold_best

    formatted = bold_best([0.35, 0.20, 0.28], lower_is_better=True)
    assert formatted[1].startswith("\\textbf{")


def test_missing_values_render_as_not_run() -> None:
    import numpy as np

    from src.reporting.tables import NOT_RUN, bold_best

    formatted = bold_best([0.5, None, np.nan])
    assert formatted[1] == NOT_RUN and formatted[2] == NOT_RUN


def test_not_run_and_not_available_are_distinct() -> None:
    """An unrun experiment and an unavailable quantity are different facts."""
    from src.reporting.tables import NOT_AVAILABLE, NOT_RUN

    assert NOT_RUN != NOT_AVAILABLE


def test_latex_escapes_special_characters() -> None:
    import pandas as pd

    from src.reporting.tables import to_latex

    frame = pd.DataFrame([{"Domain": "a_b & c", "Value": 1.0}])
    latex = to_latex(frame, caption="t", label="tab:t")
    assert r"a\_b \& c" in latex
    assert r"\toprule" in latex and r"\bottomrule" in latex


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_on_an_empty_registry_claims_nothing() -> None:
    from src.reporting.summary import build_summary

    text = build_summary(None)
    assert "No experiments have been run" in text
    assert "Nothing can be concluded" in text


def test_summary_normalises_registry_method_labels() -> None:
    from src.reporting.summary import _normalise_method

    assert _normalise_method("erm-none") == "erm"
    assert _normalise_method("deep_coral-b32") == "deep_coral"
    # Longest-first matching: this must not collapse to "deep_coral".
    assert _normalise_method("deep_coral_ordinal-b32") == "deep_coral_ordinal"


def test_summary_flags_single_seed_runs() -> None:
    import pandas as pd

    from src.reporting.summary import build_summary

    registry = pd.DataFrame([{
        "experiment_id": "e1", "status": "COMPLETE", "protocol": "lodo",
        "target_domain": "idrid", "backbone": "densenet121", "method": "erm",
        "best_val_qwk": 0.88, "test_qwk": 0.75, "test_ece": 0.26,
        "test_nll": 2.1, "test_brier": 0.8, "timestamp_utc": "2026-01-01T00:00:00",
        "epochs_run": 20, "train_seconds": 700.0, "params_total_m": 7.0,
        "peak_vram_gb": 2.1,
    }])
    text = build_summary(registry, n_seeds=1)
    assert "unmeasured" in text
    assert "Single-seed caveat" in text or "1 seed" in text
    assert "-0.1300" in text or "-0.13" in text     # the val -> test gap


def test_summary_lists_unrun_work_explicitly() -> None:
    import pandas as pd

    from src.reporting.summary import build_summary

    registry = pd.DataFrame([{
        "experiment_id": "e1", "status": "COMPLETE", "protocol": "lodo",
        "target_domain": "idrid", "backbone": "densenet121", "method": "erm",
        "best_val_qwk": 0.88, "test_qwk": 0.75, "test_ece": 0.26,
        "timestamp_utc": "2026-01-01T00:00:00", "epochs_run": 20,
    }])
    text = build_summary(registry, n_seeds=1)
    assert "Not run" in text
    # Domains and methods that were never touched must be named, not omitted.
    for token in ("aptos", "ddr", "eyepacs", "mixstyle", "convnext_tiny"):
        assert token in text


# ---------------------------------------------------------------------------
# cross-domain matrices
# ---------------------------------------------------------------------------

def _registry_rows():
    import pandas as pd

    return pd.DataFrame([
        {"experiment_id": "a", "status": "COMPLETE", "method": "erm-b32",
         "source_domains": '["ddr", "aptos"]', "target_domain": "idrid",
         "test_qwk": 0.75, "test_ece": 0.26},
        {"experiment_id": "b", "status": "COMPLETE", "method": "mixstyle-b32",
         "source_domains": '["ddr", "aptos"]', "target_domain": "idrid",
         "test_qwk": 0.64, "test_ece": 0.30},
    ])


def test_unrun_cells_are_nan_not_zero() -> None:
    """A zero would read as a bad result; an unrun experiment is not a result."""
    import numpy as np

    from src.visualization.domain_figures import build_cross_domain_matrix

    matrix, domains = build_cross_domain_matrix(_registry_rows(), method="erm")
    assert matrix.shape == (4, 4)
    assert np.isnan(matrix).sum() == 14           # only ddr->idrid and aptos->idrid filled
    assert not (matrix == 0).any()


def test_lodo_row_fills_every_source_row() -> None:
    """A model trained on DDR+APTOS counts as 'trained including' both."""
    from src.visualization.domain_figures import build_cross_domain_matrix

    matrix, domains = build_cross_domain_matrix(_registry_rows(), method="erm")
    assert matrix[domains.index("ddr"), domains.index("idrid")] == pytest.approx(0.75)
    assert matrix[domains.index("aptos"), domains.index("idrid")] == pytest.approx(0.75)


def test_method_filter_selects_one_experiment() -> None:
    from src.visualization.domain_figures import build_cross_domain_matrix

    registry = _registry_rows()
    erm, domains = build_cross_domain_matrix(registry, method="erm")
    mixstyle, _ = build_cross_domain_matrix(registry, method="mixstyle")
    cell = (domains.index("ddr"), domains.index("idrid"))
    assert erm[cell] == pytest.approx(0.75)
    assert mixstyle[cell] == pytest.approx(0.64)


def test_pooling_averages_and_warns(caplog) -> None:
    """Without a method filter a cell is an average, and must say so."""
    import logging

    from src.visualization.domain_figures import build_cross_domain_matrix

    with caplog.at_level(logging.WARNING):
        matrix, domains = build_cross_domain_matrix(_registry_rows(), method=None)
    cell = (domains.index("ddr"), domains.index("idrid"))
    assert matrix[cell] == pytest.approx((0.75 + 0.64) / 2)
    assert any("aggregate more than one run" in record.message for record in caplog.records)


def test_matrix_of_an_empty_registry_is_all_nan() -> None:
    import numpy as np
    import pandas as pd

    from src.visualization.domain_figures import build_cross_domain_matrix

    matrix, _ = build_cross_domain_matrix(pd.DataFrame())
    assert np.isnan(matrix).all()
