"""Tests for the consistency audit.

An audit that passes is worthless unless it fails on the thing it was written
for. These tests reconstruct both August-22 bugs in miniature and assert that
each is caught, then assert the audit stays quiet on a clean table -- a checker
that cries wolf gets switched off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pd = pytest.importorskip("pandas")

from audit_consistency import Audit, _registry_backbone, _with_defaults  # noqa: E402


# ---------------------------------------------------------------------------
# The Audit accumulator


def test_audit_collects_every_failure_rather_than_stopping_at_the_first() -> None:
    audit = Audit()
    audit.check(False, "a", "first")
    audit.check(True, "a", "")
    audit.check(False, "b", "second")
    assert audit.checks == 3
    assert len(audit.failures) == 2
    assert audit.report() == 1


def test_a_clean_audit_exits_zero() -> None:
    audit = Audit()
    audit.check(True, "a", "")
    assert audit.report() == 0


def test_skips_do_not_count_as_failures() -> None:
    """A NOT RUN experiment must not fail the audit -- absence is not error."""
    audit = Audit()
    audit.check(True, "a", "")
    audit.skip("in_domain_results.csv: NOT RUN")
    assert audit.report() == 0
    assert len(audit.skipped) == 1


# ---------------------------------------------------------------------------
# Backfilling the late-added key columns


def test_missing_key_columns_are_backfilled_to_the_project_defaults() -> None:
    frame = _with_defaults(pd.DataFrame([{"target": "ddr", "target_qwk": 0.73}]))
    assert frame["backbone"].iloc[0] == "densenet121"
    assert frame["image_size"].iloc[0] == 224
    assert frame["batch_size"].iloc[0] == 32


def test_existing_key_values_are_not_overwritten_by_the_defaults() -> None:
    frame = _with_defaults(pd.DataFrame([{
        "target": "ddr", "backbone": "convnext-tiny",
        "image_size": 512, "batch_size": 16,
    }]))
    assert frame["backbone"].iloc[0] == "convnext-tiny"
    assert frame["image_size"].iloc[0] == 512
    assert frame["batch_size"].iloc[0] == 16


def test_registry_backbone_names_map_onto_experiment_id_spelling() -> None:
    """The registry writes convnext_tiny; ids use convnext-tiny."""
    assert _registry_backbone("convnext_tiny") == "convnext-tiny"
    assert _registry_backbone("densenet121") == "densenet121"


# ---------------------------------------------------------------------------
# The two bugs, in miniature.
#
# audit_tables() reads from the real outputs directory, so rather than faking a
# whole tree these tests exercise the two properties the audit turns on: an
# orphaned registry run, and a metric that disagrees with the registry.
# ---------------------------------------------------------------------------


def _detect_orphans(table_ids, registry_ids):
    audit = Audit()
    for experiment_id in registry_ids:
        audit.check(experiment_id in table_ids, "orphan runs",
                    f"{experiment_id} missing from its summary table")
    return audit


def test_an_overwritten_row_shows_up_as_an_orphaned_registry_run() -> None:
    """Bug 1: the ConvNeXt run replaced the DenseNet rows instead of joining."""
    registry_ids = [
        "lodo_aptos-eyepacs-idrid__ddr_densenet121_erm-b32_s42",
        "lodo_aptos-eyepacs-idrid__ddr_convnext-tiny_erm-b32_s42",
    ]
    # Keyed without backbone, the table keeps only one of the two runs.
    table_ids = {"lodo_aptos-eyepacs-idrid__ddr_densenet121_erm-b32_s42"}
    audit = _detect_orphans(table_ids, registry_ids)
    assert audit.report() == 1
    assert "convnext-tiny" in audit.failures[0][1]


def test_a_resolution_overwrite_also_shows_up_as_an_orphan() -> None:
    """Bug 2: the 512px run replaced the 224px EyePACS row."""
    registry_ids = [
        "in_domain_eyepacs__eyepacs_densenet121_erm-b32_s42",
        "in_domain_eyepacs__eyepacs_densenet121_erm-b16-r512_s42",
    ]
    table_ids = {"in_domain_eyepacs__eyepacs_densenet121_erm-b16-r512_s42"}
    audit = _detect_orphans(table_ids, registry_ids)
    assert audit.report() == 1
    assert "erm-b32_s42" in audit.failures[0][1]


def test_both_runs_present_produces_no_orphan() -> None:
    registry_ids = [
        "lodo_aptos-eyepacs-idrid__ddr_densenet121_erm-b32_s42",
        "lodo_aptos-eyepacs-idrid__ddr_convnext-tiny_erm-b32_s42",
    ]
    audit = _detect_orphans(set(registry_ids), registry_ids)
    assert audit.report() == 0


def test_a_contaminated_metric_disagrees_with_the_registry() -> None:
    """The other signature of bug 1: the row survives but holds wrong numbers."""
    from audit_consistency import TOLERANCE

    audit = Audit()
    # The DenseNet row carrying the ConvNeXt value.
    audit.check(abs(0.757960 - 0.733372) < TOLERANCE, "registry agreement",
                "target_qwk disagrees with the registry")
    assert audit.report() == 1


def test_an_uncontaminated_metric_agrees() -> None:
    from audit_consistency import TOLERANCE

    audit = Audit()
    audit.check(abs(0.733372 - 0.733372) < TOLERANCE, "registry agreement", "")
    assert audit.report() == 0


# ---------------------------------------------------------------------------
# End to end, against the real repository


@pytest.mark.slow
def test_the_repository_currently_passes_its_own_audit() -> None:
    """The audit must pass on the checked-in artifacts, or it will be ignored."""
    from src.utils.io import project_root

    if not (project_root() / "outputs" / "experiment_registry.csv").exists():
        pytest.skip("no experiment registry in this checkout")

    import audit_consistency

    assert audit_consistency.main() == 0
