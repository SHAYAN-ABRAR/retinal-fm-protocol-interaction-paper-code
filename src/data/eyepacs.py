"""EyePACS loader -- Domain 3.

This is the only domain reconstructed from a third-party derivative, so it is
the only loader that enforces exclusions rather than merely reading files.
Everything it refuses is refused because Phase 1 measured a concrete problem:

============================  ==========================================================
Rule                          Why
============================  ==========================================================
Read the verified index only  50,070 of the 85,178 images on disk have corrupted labels
                              (one grade-3 image where ~1,247 are expected). The index at
                              ``outputs/reports/eyepacs_verified_index.csv`` holds the
                              35,108 whose grades were confirmed 100% against the
                              reference ``trainLabels.csv``.
Reject APTOS filenames        The folder physically contains all 3,662 APTOS images. Left
                              in, a leave-one-domain-out run holding out APTOS would
                              train on its entire test set.
Drop ``-GF`` copies           Augmented duplicates baked onto disk. Augmentation belongs
                              in the training transform where it stays out of val/test.
Ignore the split folders      They leak: 1,231 photographs and 14,244 patients appear in
                              more than one split.
Never touch ``augmented_``    ``augmented_resized_V2`` is a further augmented 600x600
``resized_V2``                pool where minority classes are inflated ~4x.
============================  ==========================================================

Unlike the other three domains, EyePACS **does** expose patients: the filename is
``<patient>_<eye>``.  87.3% of its patients carry the same grade in both eyes, so
image-level splitting would leak a near-duplicate label between train and test.
Patient-level splitting is mandatory here and is enforced by ``splits.py``.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from .schema import make_image_id, validate_manifest

log = get_logger("data.eyepacs")

__all__ = ["build_eyepacs_manifest", "locate_verified_images"]

DOMAIN = "eyepacs"
EYEPACS_ID = re.compile(r"^\d+_(left|right)$")
DERIVED_COPY = re.compile(r"(^GF-|-GF$)")


def _base_id(name: str) -> str:
    stem = Path(name).stem
    previous = None
    while previous != stem:
        previous = stem
        stem = DERIVED_COPY.sub("", stem)
    return stem


def locate_verified_images(
    image_root: Path | str,
    wanted: set[str],
    *,
    forbidden_paths: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Find one canonical file path for each wanted EyePACS base id.

    The derivative stores the same photograph in several places (different split
    folders, plus ``-GF`` augmented copies).  Any original is equivalent, so a
    deterministic choice is made -- the lexicographically smallest path among
    non-augmented files -- to keep manifest construction reproducible.

    Returns ``(base_id -> path, statistics)``.
    """
    image_root = Path(image_root)
    forbidden = [Path(p).resolve() for p in (forbidden_paths or [])]
    resolved_root = image_root.resolve()
    for bad in forbidden:
        if resolved_root == bad or bad in resolved_root.parents:
            raise ValueError(f"EyePACS image_root {image_root} is inside forbidden path {bad}")

    candidates: dict[str, list[str]] = defaultdict(list)
    stats = {"files_scanned": 0, "augmented_skipped": 0, "aptos_skipped": 0, "other_skipped": 0}

    for dirpath, _dirnames, filenames in os.walk(image_root):
        for name in filenames:
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stats["files_scanned"] += 1
            stem = Path(name).stem
            base = _base_id(name)
            if base != stem:
                stats["augmented_skipped"] += 1
                continue
            if not EYEPACS_ID.match(base):
                # 12-hex APTOS ids land here, as does anything unrecognised.
                if re.fullmatch(r"[0-9a-f]{12}", base):
                    stats["aptos_skipped"] += 1
                else:
                    stats["other_skipped"] += 1
                continue
            if base in wanted:
                candidates[base].append(os.path.join(dirpath, name))

    located = {base: min(paths) for base, paths in candidates.items()}
    stats["located"] = len(located)
    stats["ids_with_multiple_copies"] = sum(1 for v in candidates.values() if len(v) > 1)
    return located, stats


def build_eyepacs_manifest(
    spec: dict[str, Any],
    *,
    project_root: Path,
    exclusions: dict[str, Any] | None = None,
) -> "Any":
    """Build the EyePACS manifest from the verified index only."""
    import pandas as pd

    if not spec.get("restrict_to_verified_ids", False):
        raise ValueError(
            "EyePACS: restrict_to_verified_ids is not set in configs/paths.yaml. "
            "Refusing to build the domain from unverified folder labels -- 50,070 "
            "of them are corrupted. Run Cell 7b to produce the verified index."
        )

    index_path = project_root / spec["verified_index_csv"]
    if not index_path.exists():
        raise FileNotFoundError(
            f"EyePACS verified index not found at {index_path}.\n"
            "Run Cell 7b (src/data/eyepacs_labels.py) to create it."
        )

    with index_path.open("r", encoding="utf-8", newline="") as fh:
        index_rows = list(csv.DictReader(fh))
    if not index_rows:
        raise ValueError(f"EyePACS verified index {index_path} is empty")

    wanted = {row["image_id"] for row in index_rows}

    # Belt and braces: the index itself must not contain an APTOS id.
    aptos_pattern = re.compile((exclusions or {}).get("aptos_filename_regex", r"^[0-9a-f]{12}$"))
    offenders = sorted(i for i in wanted if aptos_pattern.match(i))
    if offenders:
        raise ValueError(
            f"EyePACS verified index contains {len(offenders)} APTOS ids "
            f"(e.g. {offenders[:5]}); these belong to Domain 1 only."
        )

    log.info("EyePACS: locating %d verified images under %s ...", len(wanted), spec["image_root"])
    located, stats = locate_verified_images(
        spec["image_root"], wanted, forbidden_paths=(exclusions or {}).get("forbidden_paths")
    )
    log.info(
        "EyePACS: scanned %d files -> located %d/%d "
        "(skipped %d augmented, %d APTOS, %d other; %d ids had multiple copies)",
        stats["files_scanned"], stats["located"], len(wanted),
        stats["augmented_skipped"], stats["aptos_skipped"], stats["other_skipped"],
        stats["ids_with_multiple_copies"],
    )

    absent = sorted(wanted - set(located))
    if absent:
        raise FileNotFoundError(
            f"EyePACS: {len(absent)} verified ids have no image file under "
            f"{spec['image_root']}, e.g. {absent[:5]}"
        )

    rows = [
        {
            "image_id": make_image_id(DOMAIN, "pooled", f"{row['image_id']}.jpg"),
            "domain": DOMAIN,
            "domain_id": spec["domain_id"],
            "path": located[row["image_id"]],
            "grade": int(row["grade"]),
            "patient_id": row["patient_id"],
            "eye": row["eye"],
            # 'pooled': the vendor's splits leak and are deliberately discarded.
            "source_split": "pooled",
        }
        for row in index_rows
    ]

    manifest = validate_manifest(pd.DataFrame(rows), name="eyepacs")
    log.info(
        "EyePACS: %d images, %d patients, grades %s",
        len(manifest),
        manifest["patient_id"].nunique(),
        manifest["grade"].value_counts().sort_index().to_dict(),
    )
    return manifest
