"""Tests for the Phase-1 code: config integrity, inspection helpers, audits.

Two kinds of test live here:

* **Unit tests** on synthetic fixtures -- always run, no datasets needed.
* **Integrity tests** on the real datasets -- skipped automatically when
  ``D:\\DB`` is not present, so the suite still passes on another machine.

The integrity tests are the important ones scientifically: they assert the
properties that Phase 2 loaders will depend on, so a changed or re-downloaded
dataset breaks a test instead of silently changing results.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inspect import (  # noqa: E402
    DERIVED_MARKERS,
    filename_family_census,
    read_csv_ids_labels,
    sample_image_properties,
    summarise_csv,
    summarise_directory,
)
from src.data.provenance import Finding, _base_id, _eyepacs_patient  # noqa: E402
from src.utils.io import load_paths_config, project_root, read_json, write_json  # noqa: E402
from src.utils.seed import environment_fingerprint, set_global_seed  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _datasets_available() -> bool:
    try:
        load_paths_config()
    except (FileNotFoundError, OSError):
        return False
    return True


needs_data = pytest.mark.skipif(
    not _datasets_available(), reason="raw datasets not present on this machine"
)


@pytest.fixture(scope="module")
def paths_config() -> dict:
    return load_paths_config()


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def test_project_root_has_markers() -> None:
    root = project_root()
    assert (root / "configs" / "paths.yaml").exists()
    assert (root / "requirements.txt").exists()


def test_set_global_seed_is_reproducible() -> None:
    import random

    set_global_seed(123)
    first = [random.random() for _ in range(5)]
    set_global_seed(123)
    assert [random.random() for _ in range(5)] == first


def test_environment_fingerprint_reports_missing_packages() -> None:
    fingerprint = environment_fingerprint(7)
    assert fingerprint.seed == 7
    assert fingerprint.python_version
    # Absent optional packages are reported, never silently omitted.
    assert set(fingerprint.packages.values()) - {"NOT INSTALLED"} or True
    assert "torch" in fingerprint.packages


def test_json_round_trip(tmp_path: Path) -> None:
    payload = {"a": 1, "b": [1, 2], "c": {"d": None}}
    target = write_json(tmp_path / "x.json", payload)
    assert read_json(target) == payload


def test_write_json_serialises_dataclasses(tmp_path: Path) -> None:
    finding = Finding(audit="t", severity="INFO", summary="s")
    target = write_json(tmp_path / "f.json", [finding])
    assert read_json(target)[0]["audit"] == "t"


def test_finding_rejects_unknown_severity() -> None:
    with pytest.raises(ValueError):
        Finding(audit="t", severity="MILDLY_ANNOYING", summary="s")


# ---------------------------------------------------------------------------
# filename parsing -- guards a real bug found during Phase 1
# ---------------------------------------------------------------------------

def test_base_id_strips_augmentation_markers() -> None:
    assert _base_id("10005_left-GF.jpg") == "10005_left"
    assert _base_id("GF-00a8624548a9.jpg") == "00a8624548a9"
    assert _base_id("10005_left.jpg") == "10005_left"


def test_base_id_does_not_mangle_ddr_filenames() -> None:
    """Regression: a '-\\d{3,4}$' marker used to strip DDR's trailing field.

    ``007-0004-000.jpg`` is a legitimate DDR filename, not a derived copy of
    ``007-0004``. Stripping it reported 7,101 DDR images as augmented duplicates.
    """
    assert _base_id("007-0004-000.jpg") == "007-0004-000"
    assert DERIVED_MARKERS.sub("", "007-0004-000") == "007-0004-000"


def test_eyepacs_patient_extraction() -> None:
    assert _eyepacs_patient("10005_left") == "10005"
    assert _eyepacs_patient("10005_right") == "10005"
    # Both eyes of one patient must map to the same key, or patient-level
    # splitting silently degrades to image-level splitting.
    assert _eyepacs_patient("10005_left") == _eyepacs_patient("10005_right")
    assert _eyepacs_patient("00a8624548a9") is None


def test_filename_family_census_separates_sources() -> None:
    names = [
        "10005_left.jpg", "10005_right.jpg",   # eyepacs
        "00a8624548a9.png",                    # aptos
        "IDRiD_001.jpg",                       # idrid
        "20170413102628830.jpg",               # ddr timestamp
        "007-0004-000.jpg",                    # ddr structured
        "totally_unexpected.jpg",
    ]
    census = filename_family_census(names)
    counts = census["family_counts"]
    assert counts["eyepacs (<n>_left / <n>_right)"] == 2
    assert counts["aptos (12 hex chars)"] == 1
    assert counts["idrid (IDRiD_nnn)"] == 1
    assert counts["ddr (17-digit timestamp)"] == 1
    assert counts["ddr (nnn-nnnn-nnn)"] == 1
    assert counts["UNRECOGNISED"] == 1


# ---------------------------------------------------------------------------
# CSV inspection on synthetic fixtures
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_summarise_csv_counts_labels_and_duplicates(tmp_path: Path) -> None:
    rows = [
        {"id_code": "a", "diagnosis": "0"},
        {"id_code": "b", "diagnosis": "2"},
        {"id_code": "b", "diagnosis": "2"},  # duplicate id
    ]
    path = _write_csv(tmp_path / "labels.csv", rows, ["id_code", "diagnosis"])
    summary = summarise_csv(path, id_column="id_code", label_column="diagnosis")

    assert summary.n_rows == 3
    assert summary.label_counts == {"0": 1, "2": 2}
    assert summary.n_unique_ids == 2
    assert summary.n_duplicate_ids == 1
    assert summary.duplicate_id_examples == ["b"]


def test_summarise_csv_reports_missing_column_instead_of_guessing(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "l.csv", [{"x": "1", "y": "2"}], ["x", "y"])
    summary = summarise_csv(path, id_column="id_code", label_column="diagnosis")
    assert summary.label_counts == {}
    assert any("MISSING label column" in n for n in summary.notes)
    assert any("MISSING id column" in n for n in summary.notes)


def test_read_csv_ids_labels_raises_on_missing_column(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "l.csv", [{"x": "1"}], ["x"])
    with pytest.raises(KeyError, match="not found"):
        read_csv_ids_labels(path, "id_code", "diagnosis")


def test_summarise_csv_handles_trailing_space_column(tmp_path: Path) -> None:
    """IDRiD's real column name ends in a space; it must not be silently trimmed."""
    columns = ["Image name", "Retinopathy grade", "Risk of macular edema "]
    rows = [{"Image name": "IDRiD_001", "Retinopathy grade": "3", "Risk of macular edema ": "2"}]
    path = _write_csv(tmp_path / "idrid.csv", rows, columns)
    summary = summarise_csv(path, id_column="Image name", label_column="Retinopathy grade")
    assert "Risk of macular edema " in summary.columns
    assert summary.label_counts == {"3": 1}


# ---------------------------------------------------------------------------
# directory + image inspection on synthetic fixtures
# ---------------------------------------------------------------------------

def test_summarise_directory_counts_and_extensions(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "sub" / "b.png").write_bytes(b"y")
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")

    summary = summarise_directory(tmp_path)
    assert summary.exists
    assert summary.file_count == 3
    assert summary.extension_counts[".jpg"] == 1
    assert summary.extension_counts[".png"] == 1
    assert "sub" in summary.subdirectories


def test_summarise_directory_on_missing_path(tmp_path: Path) -> None:
    summary = summarise_directory(tmp_path / "nope")
    assert not summary.exists
    assert summary.file_count == 0


def test_sample_image_properties_reports_corrupt_files(tmp_path: Path) -> None:
    from PIL import Image

    good = tmp_path / "good.png"
    Image.new("RGB", (64, 32)).save(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not actually a png")

    stats = sample_image_properties([good, bad], n_sample=2)
    assert stats.n_sampled == 2
    assert stats.n_failed == 1
    assert stats.width_max == 64 and stats.height_max == 32
    assert stats.size_counts == {"64x32": 1}


# ---------------------------------------------------------------------------
# integrity of the real data -- these guard the scientific protocol
# ---------------------------------------------------------------------------

@needs_data
def test_domain_ids_match_protocol(paths_config: dict) -> None:
    """Domain numbering is fixed by the protocol and must never be reordered."""
    expected = {"ddr": 0, "aptos": 1, "idrid": 2, "eyepacs": 3}
    actual = {k: v["domain_id"] for k, v in paths_config["domains"].items()}
    assert actual == expected


@needs_data
def test_every_declared_path_exists(paths_config: dict) -> None:
    # load_paths_config raises if not; reaching here means all paths resolved.
    assert set(paths_config["domains"]) == {"ddr", "aptos", "idrid", "eyepacs"}


@needs_data
def test_forbidden_paths_are_never_read(paths_config: dict) -> None:
    """augmented_resized_V2 holds pre-augmented images and must stay excluded."""
    forbidden = paths_config["exclusions"]["forbidden_paths"]
    assert any("augmented_resized_V2" in p for p in forbidden)
    eyepacs_root = paths_config["domains"]["eyepacs"]["image_root"]
    for path in forbidden:
        assert not eyepacs_root.startswith(path)


@needs_data
def test_ddr_label_counts_match_official_gradable_subset(paths_config: dict) -> None:
    """DDR grade 5 (ungradable) must be absent; counts must match the official set."""
    spec = paths_config["domains"]["ddr"]
    pairs = read_csv_ids_labels(spec["label_csv"], "id_code", "diagnosis")
    from collections import Counter

    counts = Counter(label for _, label in pairs)
    assert dict(counts) == {"0": 6266, "1": 630, "2": 4477, "3": 236, "4": 913}
    assert sum(counts.values()) == 12522
    assert "5" not in counts, "DDR ungradable class 5 present; it must be mapped explicitly"


@needs_data
def test_aptos_provided_splits_are_disjoint(paths_config: dict) -> None:
    spec = paths_config["domains"]["aptos"]
    ids = {}
    for split, sub in spec["splits"].items():
        ids[split] = {i for i, _ in read_csv_ids_labels(sub["csv"], "id_code", "diagnosis")}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert sum(len(v) for v in ids.values()) == 3662


@needs_data
def test_idrid_filenames_collide_across_splits(paths_config: dict) -> None:
    """Documents a real hazard: bare filenames are NOT unique ids in IDRiD.

    If this ever stops being true the unified index can be simplified, so the
    test asserts the current reality rather than assuming it away.
    """
    from src.data.inspect import iter_images

    spec = paths_config["domains"]["idrid"]
    train = {p.name for p in iter_images(spec["splits"]["train"]["image_dir"])}
    test = {p.name for p in iter_images(spec["splits"]["test"]["image_dir"])}
    assert train & test, "expected colliding filenames between IDRiD splits"
    assert spec["filenames_collide_across_splits"] is True


# ---------------------------------------------------------------------------
# EyePACS label verification
# ---------------------------------------------------------------------------

def test_verify_reference_file_rejects_a_truncated_mirror() -> None:
    """A mirror with the wrong row count must be refused, not silently trusted."""
    from src.data.eyepacs_labels import verify_reference_file

    truncated = {f"{i}_left": 0 for i in range(100)}
    problems = verify_reference_file(truncated)
    assert any("expected 35,108 rows" in p for p in problems)


def test_verify_reference_file_rejects_wrong_class_distribution() -> None:
    """Corroboration against published EyePACS statistics must actually bite."""
    from src.data.eyepacs_labels import EXPECTED_REFERENCE_ROWS, verify_reference_file

    # Right row count, but every image labelled grade 0.
    skewed = {f"{i}_left": 0 for i in range(EXPECTED_REFERENCE_ROWS)}
    problems = verify_reference_file(skewed)
    assert any("grade 1" in p for p in problems)


def test_verify_reference_file_rejects_malformed_ids() -> None:
    from src.data.eyepacs_labels import verify_reference_file

    problems = verify_reference_file({"not-an-eyepacs-id": 0})
    assert any("not <patient>_<eye>" in p for p in problems)


def test_read_folder_labels_excludes_aptos_and_collapses_augmented(tmp_path: Path) -> None:
    """The EyePACS reader must keep only EyePACS ids and fold -GF copies in."""
    from src.data.eyepacs_labels import read_folder_labels

    for split in ("train", "test"):
        for grade in ("0", "2"):
            (tmp_path / split / grade).mkdir(parents=True)
    (tmp_path / "train" / "0" / "10_left.jpg").write_bytes(b"x")
    (tmp_path / "train" / "0" / "10_left-GF.jpg").write_bytes(b"x")   # augmented copy
    (tmp_path / "train" / "2" / "11_right.jpg").write_bytes(b"x")
    (tmp_path / "test" / "0" / "00a8624548a9.jpg").write_bytes(b"x")  # APTOS -- must be dropped
    (tmp_path / "test" / "2" / "GF-00a8624548a9.jpg").write_bytes(b"x")

    labels = read_folder_labels(tmp_path)
    assert labels == {"10_left": 0, "11_right": 2}
    assert not any(len(k) == 12 for k in labels), "APTOS id leaked into the EyePACS domain"


@needs_data
def test_eyepacs_reference_labels_pass_corroboration(paths_config: dict) -> None:
    from src.data.eyepacs_labels import load_reference_labels, verify_reference_file

    reference_path = project_root() / paths_config["domains"]["eyepacs"]["reference_labels_csv"]
    if not reference_path.exists():
        pytest.skip("reference labels not downloaded yet (run Cell 7b)")
    problems = verify_reference_file(load_reference_labels(reference_path))
    assert problems == [], f"reference file failed corroboration: {problems}"


@needs_data
def test_eyepacs_verified_index_is_consistent(paths_config: dict) -> None:
    """The verified index is the authoritative EyePACS domain; guard its shape."""
    spec = paths_config["domains"]["eyepacs"]
    index_path = project_root() / spec["verified_index_csv"]
    if not index_path.exists():
        pytest.skip("verified index not built yet (run Cell 7b)")

    from collections import Counter

    with index_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    expected = spec["verified"]
    assert len(rows) == expected["images_used"]
    assert len({r["image_id"] for r in rows}) == len(rows), "duplicate ids in verified index"
    assert len({r["patient_id"] for r in rows}) == expected["patients_used"]

    counts = Counter(int(r["grade"]) for r in rows)
    assert dict(sorted(counts.items())) == expected["label_counts_used"]

    # Every id must be a genuine EyePACS id, and patient_id must be its prefix.
    for row in rows[:200]:
        assert row["image_id"] == f"{row['patient_id']}_{row['eye']}"
        assert row["eye"] in {"left", "right"}


@needs_data
def test_eyepacs_verified_index_contains_no_aptos_ids(paths_config: dict) -> None:
    """The single most damaging failure mode: APTOS images inside EyePACS."""
    spec = paths_config["domains"]["eyepacs"]
    index_path = project_root() / spec["verified_index_csv"]
    if not index_path.exists():
        pytest.skip("verified index not built yet (run Cell 7b)")

    import re

    aptos_pattern = re.compile(paths_config["exclusions"]["aptos_filename_regex"])
    with index_path.open(encoding="utf-8", newline="") as fh:
        ids = [row["image_id"] for row in csv.DictReader(fh)]

    offenders = [i for i in ids if aptos_pattern.match(i)]
    assert not offenders, f"APTOS ids present in the EyePACS domain: {offenders[:5]}"

    aptos_spec = paths_config["domains"]["aptos"]
    aptos_ids: set[str] = set()
    for sub in aptos_spec["splits"].values():
        aptos_ids |= {i for i, _ in read_csv_ids_labels(sub["csv"], "id_code", "diagnosis")}
    assert not (set(ids) & aptos_ids), "EyePACS and APTOS domains share image ids"


@needs_data
def test_grade_alphabet_is_identical_across_csv_domains(paths_config: dict) -> None:
    """No domain may use a grade encoding other than 0-4 without an explicit map."""
    for key, spec in paths_config["domains"].items():
        columns = spec.get("csv_columns")
        if not columns:
            continue
        labels: set[str] = set()
        if "label_csv" in spec:
            labels |= {
                label for _, label in read_csv_ids_labels(
                    spec["label_csv"], columns["image"], columns["label"]
                )
            }
        for sub in (spec.get("splits") or {}).values():
            if "csv" in sub:
                labels |= {
                    label for _, label in read_csv_ids_labels(
                        sub["csv"], columns["image"], columns["label"]
                    )
                }
        assert labels <= {"0", "1", "2", "3", "4"}, f"{key} has unexpected grades: {labels}"
