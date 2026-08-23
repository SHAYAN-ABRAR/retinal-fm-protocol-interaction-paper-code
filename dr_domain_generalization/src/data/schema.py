"""The unified manifest schema shared by all four domain loaders.

Every domain loader returns a DataFrame with exactly these columns, so that
downstream code (splits, leakage audit, datasets, figures) never needs to know
which dataset a row came from.

Key design decision -- ``image_id`` is namespaced
-------------------------------------------------
IDRiD reuses 103 bare filenames across its train and test folders (both start at
``IDRiD_001.jpg``).  A manifest keyed on bare filename would silently merge two
different images and mislabel one of them.  ``image_id`` is therefore always
``<domain>/<source_split>/<filename>`` and is asserted unique.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "MANIFEST_COLUMNS",
    "OPTIONAL_COLUMNS",
    "GRADE_NAMES",
    "N_GRADES",
    "make_image_id",
    "validate_manifest",
]

MANIFEST_COLUMNS: dict[str, str] = {
    "image_id": "string",      # unique, namespaced: '<domain>/<source_split>/<filename>'
    "domain": "string",        # 'ddr' | 'aptos' | 'idrid' | 'eyepacs'
    "domain_id": "int16",      # 0 | 1 | 2 | 3 -- fixed by the protocol
    "path": "string",          # absolute path to the image on disk
    "grade": "int8",           # 0..4, ICDR scale
    "patient_id": "string",    # <NA> where the dataset does not expose it
    "eye": "string",           # 'left' | 'right' | <NA>
    "source_split": "string",  # the vendor's split label; informational only
}

# Columns added by later stages. They are optional -- a freshly built manifest
# has none of them -- but must keep their dtype once present, and must survive a
# save/load round trip. ``split`` in particular is written by
# ``splits.assign_within_domain_splits`` and read back by every Phase-3 cell.
OPTIONAL_COLUMNS: dict[str, str] = {
    "split": "string",         # 'train' | 'val' | 'test', assigned by splits.py
}

GRADE_NAMES: dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}
N_GRADES = 5


def make_image_id(domain: str, source_split: str, filename: str) -> str:
    """Build the namespaced, globally unique image id."""
    return f"{domain}/{source_split}/{filename}"


def validate_manifest(manifest: "pd.DataFrame", *, name: str = "manifest") -> "pd.DataFrame":
    """Assert the manifest contract, then return it with canonical dtypes.

    Raises rather than warns: a malformed manifest silently corrupts every
    downstream result, so it must stop the pipeline.
    """
    import pandas as pd

    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")

    known = {**MANIFEST_COLUMNS, **OPTIONAL_COLUMNS}
    extra = [c for c in manifest.columns if c not in known]
    if extra:
        raise ValueError(
            f"{name}: unexpected columns {extra}. "
            f"Known columns: {sorted(known)}"
        )

    ordered = list(MANIFEST_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in manifest.columns]
    manifest = manifest[ordered].copy()
    for column in ordered:
        manifest[column] = manifest[column].astype(known[column])

    if "split" in manifest.columns:
        bad_splits = sorted(set(manifest["split"].dropna().unique()) - {"train", "val", "test"})
        if bad_splits:
            raise ValueError(f"{name}: unexpected split labels {bad_splits}")

    if manifest["image_id"].duplicated().any():
        dupes = manifest.loc[manifest["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise ValueError(f"{name}: duplicate image_id values, e.g. {dupes}")

    bad_grades = sorted(set(manifest["grade"].unique()) - set(GRADE_NAMES))
    if bad_grades:
        raise ValueError(f"{name}: grades outside 0..4: {bad_grades}")

    if manifest["path"].isna().any():
        raise ValueError(f"{name}: rows with a null path")

    # An eye label without a patient id cannot be used for patient-level splitting.
    orphan_eyes = manifest["eye"].notna() & manifest["patient_id"].isna()
    if orphan_eyes.any():
        raise ValueError(f"{name}: {int(orphan_eyes.sum())} rows have an eye but no patient_id")

    return manifest
