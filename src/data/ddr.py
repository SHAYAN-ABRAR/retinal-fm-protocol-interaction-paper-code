"""DDR loader -- Domain 0.

Layout on disk (verified in Phase 1)::

    D:/DB/Dataset 4/
        DR_grading.csv               12,522 rows, columns: id_code, diagnosis
        DR_grading/DR_grading/*.jpg  12,524 files

Two facts that shape this loader:

* ``id_code`` **includes** the ``.jpg`` extension, unlike every other domain.
* Two files on disk have no CSV row (``007-7449-601.jpg``, ``007-7447-601.jpg``).
  They are dropped explicitly and the drop is reported, never silent.

Patient ids are **not recoverable**.  Phase 1 tested the obvious hypothesis --
that ``NNN-NNNN-NNN`` encodes ``site-patient-image`` -- and found 7,101 distinct
``(site, patient)`` groups containing exactly one image each, so the middle field
is an image counter.  The remaining 5,423 files are bare 17-digit timestamps.
``patient_id`` is therefore left null and DDR is split at image level, which is
recorded as a limitation rather than hidden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from .inspect import read_csv_ids_labels
from .schema import make_image_id, validate_manifest

log = get_logger("data.ddr")

__all__ = ["build_ddr_manifest"]

DOMAIN = "ddr"


def build_ddr_manifest(spec: dict[str, Any]) -> "Any":
    """Build the DDR manifest from ``DR_grading.csv`` plus the image folder."""
    import pandas as pd

    image_dir = Path(spec["image_dir"])
    columns = spec["csv_columns"]
    pairs = read_csv_ids_labels(spec["label_csv"], columns["image"], columns["label"])

    on_disk = {p.name for p in image_dir.iterdir() if p.is_file()}
    labelled = {image_id for image_id, _ in pairs}

    missing = sorted(labelled - on_disk)
    orphans = sorted(on_disk - labelled)
    if missing:
        raise FileNotFoundError(
            f"DDR: {len(missing)} labelled images are absent from {image_dir}, "
            f"e.g. {missing[:5]}"
        )
    if orphans:
        log.info("DDR: dropping %d file(s) with no label row: %s", len(orphans), orphans)

    rows = [
        {
            "image_id": make_image_id(DOMAIN, "all", filename),
            "domain": DOMAIN,
            "domain_id": spec["domain_id"],
            "path": str(image_dir / filename),
            "grade": int(grade),
            "patient_id": pd.NA,   # not recoverable -- see module docstring
            "eye": pd.NA,
            "source_split": "all",  # DDR's official split is not preserved here
        }
        for filename, grade in pairs
    ]

    manifest = validate_manifest(pd.DataFrame(rows), name="ddr")
    log.info(
        "DDR: %d images, grades %s",
        len(manifest),
        manifest["grade"].value_counts().sort_index().to_dict(),
    )
    return manifest
