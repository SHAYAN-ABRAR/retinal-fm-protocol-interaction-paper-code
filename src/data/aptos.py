"""APTOS 2019 loader -- Domain 1.

Layout on disk (verified in Phase 1)::

    D:/DB/Dataset 2/
        train_1.csv  + train_images/train_images/*.png   2,930
        valid.csv    + val_images/val_images/*.png         366
        test.csv     + test_images/test_images/*.png       366

The vendor split is **kept**: Phase 1 proved the three id sets are disjoint
(zero overlap) and that every CSV row has an image and vice versa.  Keeping it
means in-domain APTOS results are comparable to other work using the same split.

``id_code`` does *not* include the file extension here (it does in DDR), so the
``.png`` suffix is appended from ``image_ext``.

Patient ids do not exist: APTOS publishes anonymous 12-hex ids with no eye or
patient linkage.  APTOS may well contain both eyes of some patients, but nothing
in the release lets us detect that, so image-level splitting is the honest
ceiling and is recorded as a limitation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from .inspect import read_csv_ids_labels
from .schema import make_image_id, validate_manifest

log = get_logger("data.aptos")

__all__ = ["build_aptos_manifest"]

DOMAIN = "aptos"


def build_aptos_manifest(spec: dict[str, Any]) -> "Any":
    """Build the APTOS manifest, preserving the vendor's disjoint split."""
    import pandas as pd

    columns = spec["csv_columns"]
    extension = spec["image_ext"]
    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}

    for split, sub in spec["splits"].items():
        image_dir = Path(sub["image_dir"])
        pairs = read_csv_ids_labels(sub["csv"], columns["image"], columns["label"])
        on_disk = {p.stem for p in image_dir.iterdir() if p.is_file()}

        missing = sorted({i for i, _ in pairs} - on_disk)
        if missing:
            raise FileNotFoundError(
                f"APTOS[{split}]: {len(missing)} labelled images absent from "
                f"{image_dir}, e.g. {missing[:5]}"
            )

        for image_id, grade in pairs:
            # The vendor split is trusted only because Phase 1 measured it as
            # disjoint. Re-assert that here so a changed CSV cannot slip through.
            if image_id in seen:
                raise ValueError(
                    f"APTOS: id {image_id!r} appears in both {seen[image_id]!r} and "
                    f"{split!r} -- the provided split is no longer disjoint"
                )
            seen[image_id] = split
            rows.append(
                {
                    "image_id": make_image_id(DOMAIN, split, f"{image_id}{extension}"),
                    "domain": DOMAIN,
                    "domain_id": spec["domain_id"],
                    "path": str(image_dir / f"{image_id}{extension}"),
                    "grade": int(grade),
                    "patient_id": pd.NA,   # anonymous ids -- see module docstring
                    "eye": pd.NA,
                    "source_split": split,
                }
            )

    manifest = validate_manifest(pd.DataFrame(rows), name="aptos")
    log.info(
        "APTOS: %d images %s, grades %s",
        len(manifest),
        manifest["source_split"].value_counts().to_dict(),
        manifest["grade"].value_counts().sort_index().to_dict(),
    )
    return manifest
