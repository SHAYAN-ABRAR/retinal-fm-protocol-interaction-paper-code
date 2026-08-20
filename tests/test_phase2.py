"""Tests for Phase 2: manifest, deduplication, splits, leakage, preprocessing.

The important tests here are the ones that try to *break* the protocol -- they
inject leakage and assert that the audit catches it.  A leakage check that has
never been shown to fail is not evidence of anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.deduplicate import deduplicate_manifest  # noqa: E402
from src.data.leakage import audit_split_leakage  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    PreprocessConfig,
    circular_mask,
    estimate_retina_bbox,
    preprocess_array,
    square_pad,
)
from src.data.schema import make_image_id, validate_manifest  # noqa: E402
from src.data.splits import (  # noqa: E402
    SplitConfig,
    assign_within_domain_splits,
    build_experiment_split,
    leave_one_domain_out_plan,
)
from src.utils.io import load_paths_config, project_root  # noqa: E402


def _datasets_available() -> bool:
    try:
        load_paths_config()
    except (FileNotFoundError, OSError):
        return False
    return True


needs_data = pytest.mark.skipif(
    not _datasets_available(), reason="raw datasets not present on this machine"
)

MANIFEST_PATH = project_root() / "outputs" / "reports" / "unified_manifest_split.csv"
needs_manifest = pytest.mark.skipif(
    not MANIFEST_PATH.exists(), reason="manifest not built yet (run Cell 9)"
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _toy_manifest(tmp_path: Path, *, n_patients: int = 40) -> pd.DataFrame:
    """A small synthetic manifest with real files on disk."""
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    for patient in range(n_patients):
        grade = patient % 5
        for eye in ("left", "right"):
            name = f"{patient}_{eye}.png"
            path = tmp_path / name
            # Every file must have distinct bytes. Identical images would be
            # (correctly) collapsed by the deduplicator, which would then be
            # testing the fixture rather than the behaviour under test.
            Image.fromarray(
                rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
            ).save(path)
            rows.append(
                {
                    "image_id": make_image_id("eyepacs", "pooled", name),
                    "domain": "eyepacs",
                    "domain_id": 3,
                    "path": str(path),
                    "grade": grade,
                    "patient_id": str(patient),
                    "eye": eye,
                    "source_split": "pooled",
                }
            )
    return validate_manifest(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_validate_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    frame = _toy_manifest(tmp_path, n_patients=3)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate image_id"):
        validate_manifest(doubled)


def test_validate_manifest_rejects_out_of_range_grade(tmp_path: Path) -> None:
    frame = _toy_manifest(tmp_path, n_patients=3)
    frame.loc[0, "grade"] = 7
    with pytest.raises(ValueError, match="grades outside"):
        validate_manifest(frame)


def test_validate_manifest_rejects_eye_without_patient(tmp_path: Path) -> None:
    """An eye label with no patient id would silently degrade patient splitting."""
    frame = _toy_manifest(tmp_path, n_patients=3)
    frame.loc[0, "patient_id"] = pd.NA
    with pytest.raises(ValueError, match="eye but no patient_id"):
        validate_manifest(frame)


def test_image_id_namespacing_separates_colliding_filenames() -> None:
    """IDRiD reuses filenames across splits; the id must keep them distinct."""
    train = make_image_id("idrid", "train", "IDRiD_001.jpg")
    test = make_image_id("idrid", "test", "IDRiD_001.jpg")
    assert train != test


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def test_patient_level_split_keeps_both_eyes_together(tmp_path: Path) -> None:
    """The core guarantee for EyePACS: no patient may straddle two splits."""
    frame = assign_within_domain_splits(_toy_manifest(tmp_path), SplitConfig(seed=0))
    per_patient = frame.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all(), "a patient's eyes landed in different splits"


def test_split_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    frame = _toy_manifest(tmp_path)
    a = assign_within_domain_splits(frame, SplitConfig(seed=7))["split"].tolist()
    b = assign_within_domain_splits(frame, SplitConfig(seed=7))["split"].tolist()
    assert a == b


def test_different_seeds_give_different_splits(tmp_path: Path) -> None:
    frame = _toy_manifest(tmp_path)
    a = assign_within_domain_splits(frame, SplitConfig(seed=1))["split"].tolist()
    b = assign_within_domain_splits(frame, SplitConfig(seed=2))["split"].tolist()
    assert a != b


def test_every_split_receives_every_grade(tmp_path: Path) -> None:
    """Stratification must not exile a rare grade into a single split."""
    frame = assign_within_domain_splits(_toy_manifest(tmp_path), SplitConfig(seed=0))
    for split in ("train", "val", "test"):
        grades = set(frame[frame["split"] == split]["grade"])
        assert grades == {0, 1, 2, 3, 4}, f"{split} is missing grades {set(range(5)) - grades}"


def test_split_config_rejects_impossible_fractions() -> None:
    with pytest.raises(ValueError):
        SplitConfig(val_fraction=0.6, test_fraction=0.5)


# ---------------------------------------------------------------------------
# experiment protocols -- the sacred-target rule
# ---------------------------------------------------------------------------

def _two_domain_manifest(tmp_path: Path) -> pd.DataFrame:
    frame = _toy_manifest(tmp_path, n_patients=30)
    other = frame.copy()
    other["domain"] = "ddr"
    other["domain_id"] = 0
    other["patient_id"] = pd.NA
    other["eye"] = pd.NA
    other["image_id"] = other["image_id"].str.replace("eyepacs/", "ddr/", regex=False)
    combined = pd.concat([frame, other], ignore_index=True)
    combined["path"] = combined["path"] + "#" + combined["domain"]  # keep paths unique
    return validate_manifest(combined)


def test_single_source_target_never_appears_in_train_or_val(tmp_path: Path) -> None:
    frame = assign_within_domain_splits(_two_domain_manifest(tmp_path), SplitConfig(seed=0))
    split = build_experiment_split(
        frame, protocol="single_source", sources=["ddr"], target="eyepacs"
    )
    assert set(split.train["domain"]) == {"ddr"}
    assert set(split.val["domain"]) == {"ddr"}
    assert set(split.test["domain"]) == {"eyepacs"}
    # The ENTIRE target domain is available as test data.
    assert len(split.test) == int((frame["domain"] == "eyepacs").sum())


def test_target_listed_as_source_is_refused(tmp_path: Path) -> None:
    """The most damaging protocol error must raise, not warn."""
    frame = assign_within_domain_splits(_two_domain_manifest(tmp_path), SplitConfig(seed=0))
    with pytest.raises(ValueError, match="also listed as a source"):
        build_experiment_split(
            frame, protocol="single_source", sources=["eyepacs"], target="eyepacs"
        )


def test_lodo_requires_at_least_two_sources(tmp_path: Path) -> None:
    frame = assign_within_domain_splits(_two_domain_manifest(tmp_path), SplitConfig(seed=0))
    with pytest.raises(ValueError, match="at least two source"):
        build_experiment_split(frame, protocol="lodo", sources=["ddr"], target="eyepacs")


def test_lodo_plan_holds_each_domain_out_once() -> None:
    plan = leave_one_domain_out_plan(["ddr", "aptos", "idrid", "eyepacs"])
    assert len(plan) == 4
    assert sorted(p["target"] for p in plan) == ["aptos", "ddr", "eyepacs", "idrid"]
    for entry in plan:
        assert entry["target"] not in entry["sources"]
        assert len(entry["sources"]) == 3


# ---------------------------------------------------------------------------
# leakage audit -- must actually catch injected leakage
# ---------------------------------------------------------------------------

def test_leakage_audit_passes_on_a_clean_split(tmp_path: Path) -> None:
    frame = assign_within_domain_splits(_toy_manifest(tmp_path), SplitConfig(seed=0))
    report = audit_split_leakage(frame, check_content_hashes=True)
    assert report.passed, report.problems


def test_leakage_audit_detects_duplicated_image_id(tmp_path: Path) -> None:
    frame = assign_within_domain_splits(_toy_manifest(tmp_path), SplitConfig(seed=0))
    train_row = frame[frame["split"] == "train"].iloc[0].copy()
    train_row["split"] = "test"
    contaminated = pd.concat([frame, pd.DataFrame([train_row])], ignore_index=True)
    report = audit_split_leakage(contaminated, check_content_hashes=False)
    assert not report.passed
    assert any("image_ids shared" in p for p in report.problems)


def test_leakage_audit_detects_patient_spanning_splits(tmp_path: Path) -> None:
    frame = assign_within_domain_splits(_toy_manifest(tmp_path), SplitConfig(seed=0))
    # Move one eye of a train patient into test.
    patient = frame[frame["split"] == "train"]["patient_id"].iloc[0]
    mask = (frame["patient_id"] == patient) & (frame["eye"] == "right")
    frame.loc[mask, "split"] = "test"
    report = audit_split_leakage(frame, check_content_hashes=False)
    assert not report.passed
    assert any("patients shared" in p for p in report.problems)


def test_leakage_audit_detects_byte_identical_copy_across_splits(tmp_path: Path) -> None:
    """A copied file under a new name is exactly the DDR/APTOS failure mode."""
    from PIL import Image

    frame = assign_within_domain_splits(_toy_manifest(tmp_path), SplitConfig(seed=0))
    source = frame[frame["split"] == "train"].iloc[0]
    clone_path = tmp_path / "cloned_copy.png"
    clone_path.write_bytes(Path(source["path"]).read_bytes())

    clone = source.copy()
    clone["image_id"] = "eyepacs/pooled/cloned_copy.png"
    clone["path"] = str(clone_path)
    clone["patient_id"] = "9999"
    clone["split"] = "test"
    contaminated = pd.concat([frame, pd.DataFrame([clone])], ignore_index=True)

    report = audit_split_leakage(contaminated, check_content_hashes=True)
    assert not report.passed
    assert any("byte-identical" in p for p in report.problems)


# ---------------------------------------------------------------------------
# deduplication policy
# ---------------------------------------------------------------------------

def test_deduplication_keeps_one_of_a_consistent_group(tmp_path: Path) -> None:
    frame = _toy_manifest(tmp_path, n_patients=5)
    source = frame.iloc[0]
    clone_path = tmp_path / "same_bytes.png"
    clone_path.write_bytes(Path(source["path"]).read_bytes())
    clone = source.copy()
    clone["image_id"] = "eyepacs/pooled/same_bytes.png"
    clone["path"] = str(clone_path)
    clone["patient_id"] = "555"
    frame = pd.concat([frame, pd.DataFrame([clone])], ignore_index=True)

    cleaned, report = deduplicate_manifest(frame)
    assert report.n_consistent_groups == 1
    assert report.n_conflicting_groups == 0
    assert report.n_dropped_redundant == 1
    assert len(cleaned) == len(frame) - 1


def test_deduplication_drops_the_whole_conflicting_group(tmp_path: Path) -> None:
    """Identical bytes with different grades: neither label can be trusted."""
    frame = _toy_manifest(tmp_path, n_patients=5)
    source = frame.iloc[0]
    clone_path = tmp_path / "conflict.png"
    clone_path.write_bytes(Path(source["path"]).read_bytes())
    clone = source.copy()
    clone["image_id"] = "eyepacs/pooled/conflict.png"
    clone["path"] = str(clone_path)
    clone["patient_id"] = "556"
    clone["grade"] = (int(source["grade"]) + 1) % 5
    frame = pd.concat([frame, pd.DataFrame([clone])], ignore_index=True)

    cleaned, report = deduplicate_manifest(frame)
    assert report.n_conflicting_groups == 1
    assert report.n_dropped_conflicting == 2
    assert source["image_id"] not in set(cleaned["image_id"])
    assert clone["image_id"] not in set(cleaned["image_id"])


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def test_square_pad_does_not_distort_aspect() -> None:
    array = np.zeros((40, 100, 3), dtype=np.uint8)
    array[:, 30:70] = 255
    padded = square_pad(array)
    assert padded.shape[0] == padded.shape[1] == 100
    # The original content keeps its own proportions; only borders were added.
    assert (padded == 255).sum() == (array == 255).sum()


def test_estimate_retina_bbox_finds_a_disc_on_black() -> None:
    array = np.zeros((200, 300, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:200, :300]
    disc = (yy - 100) ** 2 + (xx - 150) ** 2 <= 80**2
    array[disc] = 180
    bbox = estimate_retina_bbox(array, threshold=0.2)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert 60 <= x0 <= 75 and 215 <= x1 <= 235
    assert 15 <= y0 <= 25 and 175 <= y1 <= 185


def test_estimate_retina_bbox_returns_none_on_an_all_black_image() -> None:
    assert estimate_retina_bbox(np.zeros((50, 50, 3), dtype=np.uint8)) is None


def test_estimate_retina_bbox_declines_a_degenerate_crop() -> None:
    """A tiny bright speck must not be mistaken for the fundus."""
    array = np.zeros((200, 200, 3), dtype=np.uint8)
    array[100:103, 100:103] = 255
    assert estimate_retina_bbox(array, threshold=0.2, min_fraction=0.05) is None


def test_preprocess_array_output_contract() -> None:
    rng = np.random.default_rng(0)
    array = (rng.random((300, 450, 3)) * 255).astype(np.uint8)
    out = preprocess_array(array, PreprocessConfig(image_size=224))
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_preprocess_array_handles_grayscale_and_rgba() -> None:
    grey = np.full((100, 100), 120, dtype=np.uint8)
    assert preprocess_array(grey, PreprocessConfig(image_size=64)).shape == (64, 64, 3)
    rgba = np.full((100, 100, 4), 120, dtype=np.uint8)
    assert preprocess_array(rgba, PreprocessConfig(image_size=64)).shape == (64, 64, 3)


def test_circular_mask_clears_corners_and_keeps_centre() -> None:
    array = np.full((64, 64, 3), 200, dtype=np.uint8)
    masked = circular_mask(array)
    assert masked[0, 0].sum() == 0
    assert masked[32, 32].sum() > 0


def test_preprocessing_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    array = (rng.random((200, 260, 3)) * 255).astype(np.uint8)
    config = PreprocessConfig(image_size=128)
    assert np.array_equal(preprocess_array(array, config), preprocess_array(array, config))


# ---------------------------------------------------------------------------
# the real manifest
# ---------------------------------------------------------------------------

@needs_data
@needs_manifest
def test_real_manifest_satisfies_the_schema() -> None:
    from src.data.unified_dataset import load_manifest

    manifest = load_manifest(MANIFEST_PATH)
    assert set(manifest["domain"].unique()) == {"ddr", "aptos", "idrid", "eyepacs"}
    assert manifest["image_id"].is_unique
    assert manifest["path"].is_unique


@needs_data
@needs_manifest
def test_real_manifest_has_no_aptos_image_inside_eyepacs() -> None:
    from src.data.unified_dataset import load_manifest

    manifest = load_manifest(MANIFEST_PATH)
    eyepacs_paths = set(manifest[manifest["domain"] == "eyepacs"]["path"])
    aptos_paths = set(manifest[manifest["domain"] == "aptos"]["path"])
    assert not (eyepacs_paths & aptos_paths)
    # Every EyePACS id must look like <patient>_<eye>, never a 12-hex APTOS id.
    names = manifest[manifest["domain"] == "eyepacs"]["image_id"].str.split("/").str[-1]
    assert names.str.match(r"^\d+_(left|right)\.jpg$").all()


@needs_data
@needs_manifest
def test_real_manifest_never_reads_the_forbidden_augmented_pool() -> None:
    from src.data.unified_dataset import load_manifest

    manifest = load_manifest(MANIFEST_PATH)
    assert not manifest["path"].str.contains("augmented_resized_V2", regex=False).any()


@needs_data
@needs_manifest
def test_real_eyepacs_patients_do_not_span_splits() -> None:
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"patient_id": "string", "split": "string"})
    eyepacs = manifest[manifest["domain"] == "eyepacs"]
    assert (eyepacs.groupby("patient_id")["split"].nunique() == 1).all()
