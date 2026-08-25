"""Regression tests for the results-table merge.

The bug these exist to prevent: ``in_domain_results.csv`` was keyed on
(domain, method, seed) with no resolution, so a 512 px run silently replaced
the 224 px EyePACS row. The file still held four well-formed rows afterwards,
which is why nothing caught it -- there is no shape or type check that fails.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.registry import merge_results_table

KEY = ["domain", "method", "seed", "image_size"]


def _row(domain="eyepacs", size=224, qwk=0.7090, method="erm", seed=42):
    return pd.DataFrame([{"domain": domain, "method": method, "seed": seed,
                          "image_size": size, "test_qwk": qwk}])


def test_different_resolution_does_not_overwrite() -> None:
    merged = merge_results_table(_row(size=224, qwk=0.7090),
                                 _row(size=512, qwk=0.8004), KEY)
    assert len(merged) == 2
    assert set(merged["test_qwk"]) == {0.7090, 0.8004}


def test_same_resolution_is_replaced_by_the_rerun() -> None:
    """A genuine re-run at identical settings should supersede, not duplicate."""
    merged = merge_results_table(_row(size=224, qwk=0.7090),
                                 _row(size=224, qwk=0.7123), KEY)
    assert len(merged) == 1
    assert merged["test_qwk"].iloc[0] == 0.7123


def test_legacy_rows_without_image_size_are_backfilled_to_224() -> None:
    legacy = _row(size=224).drop(columns=["image_size"])
    merged = merge_results_table(legacy, _row(size=512, qwk=0.8004), KEY)
    assert len(merged) == 2
    assert sorted(merged["image_size"]) == [224, 512]


def test_missing_key_column_raises_rather_than_overwriting() -> None:
    new = _row(size=512, qwk=0.8004).drop(columns=["image_size"])
    with pytest.raises(ValueError, match="image_size"):
        merge_results_table(_row(size=224), new, KEY)


def test_other_domains_are_untouched() -> None:
    previous = pd.concat([_row("ddr", 224, 0.8757), _row("aptos", 224, 0.9091),
                          _row("eyepacs", 224, 0.7090)], ignore_index=True)
    merged = merge_results_table(previous, _row("eyepacs", 512, 0.8004), KEY)
    assert len(merged) == 4
    ddr = merged[merged["domain"] == "ddr"]["test_qwk"]
    assert len(ddr) == 1 and ddr.iloc[0] == 0.8757


def test_single_source_key_shape() -> None:
    key = ["source", "target", "method", "seed", "image_size"]
    def row(size, qwk):
        return pd.DataFrame([{"source": "eyepacs", "target": "ddr", "method": "erm",
                              "seed": 42, "image_size": size, "target_qwk": qwk}])
    merged = merge_results_table(row(224, 0.731), row(512, 0.790), key)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# Backbone, the second column that went missing from a key.
#
# lodo_results.csv was keyed on (target, method, seed). The ConvNeXt-Tiny
# seed-42 run replaced the DenseNet121 seed-42 rows, so the three-seed means
# read off the file mixed two architectures and reported DDR's across-seed SD
# as 0.0105 rather than 0.0055 -- nearly doubling a bar the two-bar criterion
# depends on. The phase documents were unaffected because analyse_lodo_seeds.py
# loads predictions by experiment id, which does encode the backbone.
# ---------------------------------------------------------------------------

LODO_KEY = ["target", "method", "seed", "backbone", "image_size"]


def _lodo_row(backbone="densenet121", qwk=0.7334, target="ddr", seed=42):
    return pd.DataFrame([{"target": target, "method": "erm", "seed": seed,
                          "backbone": backbone, "image_size": 224,
                          "target_qwk": qwk}])


def test_a_second_backbone_does_not_overwrite_the_first() -> None:
    merged = merge_results_table(_lodo_row("densenet121", 0.7334),
                                 _lodo_row("convnext-tiny", 0.7580), LODO_KEY)
    assert len(merged) == 2
    assert set(merged["target_qwk"]) == {0.7334, 0.7580}


def test_seed_sd_is_computed_within_a_backbone_not_across() -> None:
    """The regression itself: pooling backbones inflates the across-seed SD."""
    previous = pd.concat(
        [_lodo_row("densenet121", qwk, seed=seed)
         for qwk, seed in [(0.7334, 42), (0.7373, 1), (0.7443, 2)]],
        ignore_index=True,
    )
    merged = merge_results_table(previous, _lodo_row("convnext-tiny", 0.7580),
                                 LODO_KEY)
    assert len(merged) == 4

    densenet = merged[merged["backbone"] == "densenet121"]["target_qwk"]
    assert len(densenet) == 3
    assert densenet.std(ddof=1) == pytest.approx(0.0055, abs=5e-5)
    # Pooled, the SD nearly doubles and is no longer a seed SD at all.
    assert merged["target_qwk"].std(ddof=1) > 0.009


def test_rerunning_the_same_backbone_and_seed_still_supersedes() -> None:
    merged = merge_results_table(_lodo_row("densenet121", 0.7334),
                                 _lodo_row("densenet121", 0.7350), LODO_KEY)
    assert len(merged) == 1
    assert merged["target_qwk"].iloc[0] == 0.7350


def test_missing_backbone_column_in_new_rows_raises() -> None:
    new = _lodo_row("convnext-tiny", 0.7580).drop(columns=["backbone"])
    with pytest.raises(ValueError, match="backbone"):
        merge_results_table(_lodo_row(), new, LODO_KEY)


# A key column that exists but is blank
# ------------------------------------------------------------------
# The backfill originally fired only when a key column was absent from the
# file. But a column is created the instant the first run that varies it is
# written, and every row already in the file gets NaN rather than the old
# default. Those rows are in exactly the same position as rows in a file with
# no such column, and must be treated the same way -- NaN never compares equal
# to False, so without this a natural re-run of an existing configuration
# appends a second row instead of replacing it, and the seed mean is then taken
# over a value and its own replacement.

BALANCED_KEY = ["target", "method", "seed", "backbone", "image_size", "domain_balanced"]


def _balanced_row(balanced, qwk: float = 0.4077):
    return pd.DataFrame([{
        "target": "eyepacs", "method": "erm", "seed": 42,
        "backbone": "densenet121", "image_size": 224,
        "domain_balanced": balanced, "qwk": qwk,
    }])


def test_blank_domain_balanced_is_backfilled_not_treated_as_a_new_setting() -> None:
    merged = merge_results_table(
        _balanced_row(float("nan"), 0.4077), _balanced_row(False, 0.4099), BALANCED_KEY)
    assert len(merged) == 1, "a natural re-run duplicated the row it should have replaced"
    assert merged.iloc[0]["qwk"] == pytest.approx(0.4099)
    assert not bool(merged.iloc[0]["domain_balanced"])


def test_a_balanced_run_still_does_not_overwrite_a_blank_natural_row() -> None:
    merged = merge_results_table(
        _balanced_row(float("nan"), 0.4077), _balanced_row(True, 0.4371), BALANCED_KEY)
    assert len(merged) == 2, "the balanced run overwrote the natural result"
    assert sorted(round(v, 4) for v in merged["qwk"]) == [0.4077, 0.4371]
