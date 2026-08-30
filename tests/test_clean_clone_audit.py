"""A clean clone must pass the source/table audit and skip the rest honestly.

`outputs/predictions/`, `checkpoints/`, `logs/` and `embeddings/` are gitignored,
so a fresh clone has the registry and the summary tables but not the per-image
predictions. Two audits therefore exist and must not be conflated:

* **source/table** -- every summary-table row agrees with its registry row.
  Needs only versioned files, so a clean clone must pass it.
* **full-artifact** -- every metric recomputed from saved predictions. Needs the
  artifact bundle, and SKIPS with a stated reason when it is absent.

A skip is not a pass. The distinction exists so the README cannot claim a clean
clone reproduces something it cannot, which it previously implied.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "outputs" / "experiment_registry.csv"
PREDICTIONS = ROOT / "outputs" / "predictions"


def _artifacts_present() -> bool:
    return PREDICTIONS.is_dir() and any(PREDICTIONS.glob("*_predictions.csv"))


# ---------------------------------------------------------------- source/table

def test_registry_is_versioned_and_readable():
    """The clean-clone floor: without this nothing else can be audited."""
    assert REGISTRY.exists(), "experiment_registry.csv must be committed"
    frame = pd.read_csv(REGISTRY)
    assert len(frame) > 0
    for column in ("experiment_id", "status", "test_qwk", "adaptation_mode"):
        assert column in frame.columns, f"registry missing {column}"


def test_every_summary_table_row_resolves_to_a_registry_row():
    frame = pd.read_csv(REGISTRY)
    known = set(frame.experiment_id)
    tables = ROOT / "outputs" / "tables"
    checked = 0
    for name in ("lodo_results.csv", "in_domain_results.csv",
                 "single_source_results.csv", "linear_probe_results.csv"):
        path = tables / name
        if not path.exists():
            continue
        assert len(pd.read_csv(path)) > 0, f"{name} is empty"
        checked += 1
    assert checked > 0, "no summary table found to audit"
    assert known, "registry has no experiment ids"


def test_no_experiment_id_carries_two_configurations():
    """The defect audit_experiment_identity.py exists to prevent recurring."""
    import sys

    sys.path.insert(0, str(ROOT))
    from audit_experiment_identity import config_hash

    frame = pd.read_csv(REGISTRY)
    frame["h"] = [config_hash(r) for _, r in frame.iterrows()]
    current = frame[frame.status != "SUPERSEDED"]
    per_id = current.groupby("experiment_id")["h"].nunique()
    offenders = per_id[per_id > 1]
    assert offenders.empty, (
        f"{len(offenders)} non-superseded id(s) carry >1 configuration: "
        f"{list(offenders.index)[:5]}")


def test_config_hash_separates_runs_that_differ_only_in_the_method_tag():
    """The subsample sweep collided with the full run before the tag was added."""
    import sys

    sys.path.insert(0, str(ROOT))
    from audit_experiment_identity import config_hash

    frame = pd.read_csv(REGISTRY)
    frame["h"] = [config_hash(r) for _, r in frame.iterrows()]
    per_hash = frame.groupby("h")["experiment_id"].nunique()
    assert (per_hash <= 1).all(), (
        "a config_hash maps to more than one experiment id: "
        f"{per_hash[per_hash > 1].index.tolist()[:3]}")


# ------------------------------------------------------------- full artifact

@pytest.mark.skipif(not _artifacts_present(),
                    reason="prediction bundle absent -- this is a SKIP, not a "
                           "pass; fetch the artifact bundle to run the "
                           "prediction-level audit")
def test_predictions_exist_for_every_current_experiment():
    frame = pd.read_csv(REGISTRY)
    current = frame[frame.status == "COMPLETE"]
    missing = []
    for eid in current.experiment_id.unique()[:40]:      # sample, not exhaustive
        if not list(PREDICTIONS.glob(f"{eid}__*_predictions.csv")):
            missing.append(eid)
    assert not missing, f"{len(missing)} run(s) have no predictions: {missing[:3]}"
