"""IDRiD loader -- Domain 2.

Layout on disk (verified in Phase 1)::

    D:/DB/Dataset 3/B. Disease Grading/
        1. Original Images/a. Training Set/*.jpg   413
        1. Original Images/b. Testing Set/*.jpg    103
        2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv
        2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv

Three traps this loader handles explicitly:

1. **Filenames collide across splits.**  Both folders begin at ``IDRiD_001.jpg``;
   103 names are reused.  This is exactly why ``image_id`` is namespaced by
   split -- a bare-filename key would merge two different eyes.
2. **The training CSV has nine empty padding columns** from its export, and a
   ``"Risk of macular edema "`` column **whose name ends in a space**.  Column
   access is by exact name, so neither surprises the parser.
3. **The macular-edema column is not the DR grade.**  Only
   ``Retinopathy grade`` is used as the target.

Only ``B. Disease Grading`` is read.  ``A. Segmentation`` and ``C. Localization``
cover a subset of the same eyes with pixel-level annotations; including them
would duplicate images under different ids.

No patient ids exist. The official train/test split is preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from .inspect import read_csv_ids_labels
from .schema import make_image_id, validate_manifest

log = get_logger("data.idrid")

__all__ = ["build_idrid_manifest"]

DOMAIN = "idrid"


def build_idrid_manifest(spec: dict[str, Any]) -> "Any":
    """Build the IDRiD manifest from the disease-grading subtask."""
    import pandas as pd

    columns = spec["csv_columns"]
    extension = spec["image_ext"]
    rows: list[dict[str, Any]] = []
    names_by_split: dict[str, set[str]] = {}

    for split, sub in spec["splits"].items():
        image_dir = Path(sub["image_dir"])
        pairs = read_csv_ids_labels(sub["csv"], columns["image"], columns["label"])
        # Some exported rows are entirely blank padding; drop them by requiring an id.
        pairs = [(i, g) for i, g in pairs if i]

        on_disk = {p.stem for p in image_dir.iterdir() if p.is_file()}
        missing = sorted({i for i, _ in pairs} - on_disk)
        if missing:
            raise FileNotFoundError(
                f"IDRiD[{split}]: {len(missing)} labelled images absent from "
                f"{image_dir}, e.g. {missing[:5]}"
            )
        names_by_split[split] = {i for i, _ in pairs}

        for image_id, grade in pairs:
            rows.append(
                {
                    "image_id": make_image_id(DOMAIN, split, f"{image_id}{extension}"),
                    "domain": DOMAIN,
                    "domain_id": spec["domain_id"],
                    "path": str(image_dir / f"{image_id}{extension}"),
                    "grade": int(grade),
                    "patient_id": pd.NA,
                    "eye": pd.NA,
                    "source_split": split,
                }
            )

    # Document the collision rather than assuming it away: if it ever stops being
    # true the namespacing is still correct, just no longer strictly necessary.
    if len(names_by_split) == 2:
        a, b = names_by_split.values()
        overlap = a & b
        if overlap:
            log.info(
                "IDRiD: %d filenames are reused across splits (expected); "
                "image_id namespacing keeps them distinct",
                len(overlap),
            )

    manifest = validate_manifest(pd.DataFrame(rows), name="idrid")
    log.info(
        "IDRiD: %d images %s, grades %s",
        len(manifest),
        manifest["source_split"].value_counts().to_dict(),
        manifest["grade"].value_counts().sort_index().to_dict(),
    )
    return manifest
