"""Leakage audit over the *constructed* train/val/test splits.

Distinct from ``provenance.py``, which audits the raw folders before anything is
built.  This module answers the question that decides whether a result is
publishable: **could any test image, or anything derived from it, have been seen
during training?**

Checks performed
----------------
1. ``image_id`` overlap between every pair of splits.
2. **Content-hash** overlap -- byte-identical files under different names, both
   within and across domains.
3. **Patient** overlap between splits, wherever patient ids exist.
4. Duplicate rows / duplicate file paths in the manifest.
5. Repeated bare filenames (the IDRiD collision class) that a weaker id scheme
   would have merged.
6. Cross-domain duplication -- the same photograph claimed by two domains.

Cost control
------------
Hashing 51,808 images is I/O bound.  Files whose *size* is unique cannot be
byte-identical to anything else, so only files sharing a size are hashed.  In
practice that reduces hashing to a few percent of the corpus.

Honest limitation
-----------------
Content hashing finds byte-identical duplicates.  It does **not** find the same
photograph re-encoded, resized or re-compressed -- those produce different bytes.
Near-duplicate detection would need perceptual hashing, which trades exactness
for recall; this module reports what it can prove and says so.  The strongest
guarantee in this project comes from the id- and patient-level checks, which are
exact.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from ..utils.logging import get_logger
from .inspect import file_hash

log = get_logger("data.leakage")

__all__ = ["LeakageReport", "audit_split_leakage", "format_leakage_report"]


@dataclass
class LeakageReport:
    """Result of the split-level leakage audit."""

    n_images: int = 0
    split_sizes: dict[str, int] = field(default_factory=dict)

    id_overlap: dict[str, int] = field(default_factory=dict)
    id_overlap_examples: dict[str, list[str]] = field(default_factory=dict)

    patient_overlap: dict[str, int] = field(default_factory=dict)
    patient_overlap_examples: dict[str, list[str]] = field(default_factory=dict)
    domains_with_patient_ids: list[str] = field(default_factory=list)
    domains_without_patient_ids: list[str] = field(default_factory=list)

    hashing_performed: bool = False
    n_files_hashed: int = 0
    n_duplicate_content_groups: int = 0
    duplicate_content_across_splits: dict[str, int] = field(default_factory=dict)
    duplicate_content_across_domains: int = 0
    duplicate_content_examples: list[dict[str, Any]] = field(default_factory=list)

    duplicate_rows: int = 0
    duplicate_paths: int = 0
    repeated_bare_filenames: int = 0
    repeated_bare_filename_examples: list[str] = field(default_factory=list)

    limitations: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems


def _pairs(names: Iterable[str]) -> list[tuple[str, str]]:
    return list(combinations(sorted(names), 2))


def audit_split_leakage(
    manifest: "Any",
    *,
    check_content_hashes: bool = True,
    hash_algorithm: str = "md5",
) -> LeakageReport:
    """Audit a split-annotated manifest for every form of leakage we can detect."""
    if "split" not in manifest.columns:
        raise ValueError("manifest has no 'split' column; call assign_within_domain_splits first")

    report = LeakageReport(n_images=len(manifest))
    report.split_sizes = manifest["split"].value_counts().to_dict()

    # -- 1. duplicate bookkeeping -------------------------------------------
    report.duplicate_rows = int(manifest.duplicated().sum())
    report.duplicate_paths = int(manifest["path"].duplicated().sum())
    if report.duplicate_rows:
        report.problems.append(f"{report.duplicate_rows} exactly duplicated manifest rows")
    if report.duplicate_paths:
        report.problems.append(f"{report.duplicate_paths} rows share a file path")

    bare = manifest["path"].map(lambda p: Path(p).name)
    repeated = Counter(bare)
    repeated_names = [name for name, count in repeated.items() if count > 1]
    report.repeated_bare_filenames = len(repeated_names)
    report.repeated_bare_filename_examples = sorted(repeated_names)[:5]

    # -- 2. id overlap between splits ---------------------------------------
    ids_by_split = {
        split: set(group["image_id"]) for split, group in manifest.groupby("split", sort=False)
    }
    for a, b in _pairs(ids_by_split):
        shared = ids_by_split[a] & ids_by_split[b]
        report.id_overlap[f"{a}&{b}"] = len(shared)
        if shared:
            report.id_overlap_examples[f"{a}&{b}"] = sorted(shared)[:5]
            report.problems.append(f"{len(shared)} image_ids shared between {a} and {b}")

    # -- 3. patient overlap between splits ----------------------------------
    with_patients = manifest[manifest["patient_id"].notna()]
    report.domains_with_patient_ids = sorted(with_patients["domain"].unique().tolist())
    report.domains_without_patient_ids = sorted(
        set(manifest["domain"].unique()) - set(report.domains_with_patient_ids)
    )

    if len(with_patients):
        # Namespace by domain: two datasets could reuse the same patient string.
        keys_by_split: dict[str, set[str]] = defaultdict(set)
        for split, group in with_patients.groupby("split", sort=False):
            keys_by_split[split] = {
                f"{d}/{p}" for d, p in zip(group["domain"], group["patient_id"])
            }
        for a, b in _pairs(keys_by_split):
            shared = keys_by_split[a] & keys_by_split[b]
            report.patient_overlap[f"{a}&{b}"] = len(shared)
            if shared:
                report.patient_overlap_examples[f"{a}&{b}"] = sorted(shared)[:5]
                report.problems.append(f"{len(shared)} patients shared between {a} and {b}")

    if report.domains_without_patient_ids:
        report.limitations.append(
            "Patient ids are unavailable for "
            f"{', '.join(report.domains_without_patient_ids)}; those domains are split at "
            "image level, so same-patient images could in principle span splits. This "
            "cannot be detected or prevented from the released data and is reported as a "
            "limitation of the datasets, not of the protocol."
        )

    # -- 4. content-hash duplicates -----------------------------------------
    if check_content_hashes:
        report.hashing_performed = True
        paths = manifest["path"].tolist()

        sizes: dict[str, int] = {}
        for path in paths:
            try:
                sizes[path] = os.path.getsize(path)
            except OSError as exc:
                report.problems.append(f"cannot stat {path}: {exc}")

        by_size: dict[int, list[str]] = defaultdict(list)
        for path, size in sizes.items():
            by_size[size].append(path)
        # A unique file size rules out a byte-identical twin, so skip those.
        candidates = [p for group in by_size.values() if len(group) > 1 for p in group]
        log.info(
            "leakage: %d/%d files share a size and will be hashed", len(candidates), len(paths)
        )

        digests: dict[str, str] = {}
        for path in candidates:
            try:
                digests[path] = file_hash(path, algorithm=hash_algorithm)
            except OSError as exc:
                report.problems.append(f"cannot hash {path}: {exc}")
        report.n_files_hashed = len(digests)

        by_digest: dict[str, list[str]] = defaultdict(list)
        for path, digest in digests.items():
            by_digest[digest].append(path)
        duplicate_groups = {d: p for d, p in by_digest.items() if len(p) > 1}
        report.n_duplicate_content_groups = len(duplicate_groups)

        if duplicate_groups:
            lookup = manifest.set_index("path")[["split", "domain", "image_id"]]
            cross_split: Counter[str] = Counter()
            cross_domain = 0
            for digest, group in duplicate_groups.items():
                rows = lookup.loc[group]
                splits = sorted(set(rows["split"]))
                domains = sorted(set(rows["domain"]))
                if len(domains) > 1:
                    cross_domain += 1
                for a, b in _pairs(splits):
                    cross_split[f"{a}&{b}"] += 1
                if len(report.duplicate_content_examples) < 10:
                    report.duplicate_content_examples.append(
                        {
                            "hash": digest[:12],
                            "n_files": len(group),
                            "splits": splits,
                            "domains": domains,
                            "image_ids": rows["image_id"].tolist()[:4],
                        }
                    )
            report.duplicate_content_across_splits = dict(cross_split)
            report.duplicate_content_across_domains = cross_domain

            for pair, count in cross_split.items():
                report.problems.append(
                    f"{count} byte-identical image group(s) span the {pair} splits"
                )
            if cross_domain:
                report.problems.append(
                    f"{cross_domain} byte-identical image group(s) span two domains"
                )

        report.limitations.append(
            "Content hashing detects byte-identical files only. The same photograph "
            "re-encoded or resized would not be flagged; near-duplicate detection would "
            "require perceptual hashing."
        )
    else:
        report.limitations.append("Content hashing was skipped for this run.")

    return report


def format_leakage_report(report: LeakageReport) -> str:
    """Render the leakage audit for the notebook cell."""
    status = "PASS -- no leakage detected" if report.passed else "FAIL"
    lines = [
        "=" * 74,
        "SPLIT-LEVEL LEAKAGE AUDIT",
        "=" * 74,
        f"  status        : {status}",
        f"  images        : {report.n_images:,}",
        f"  split sizes   : {report.split_sizes}",
        "",
        "-- image_id overlap between splits --",
    ]
    lines += [f"    {pair:16s}: {count}" for pair, count in report.id_overlap.items()] or ["    (none)"]

    lines += ["", "-- patient overlap between splits --"]
    if report.patient_overlap:
        lines += [f"    {pair:16s}: {count}" for pair, count in report.patient_overlap.items()]
    else:
        lines.append("    (no domain in this manifest exposes patient ids)")
    lines.append(f"    domains with patient ids   : {report.domains_with_patient_ids or 'none'}")
    lines.append(f"    domains without patient ids: {report.domains_without_patient_ids or 'none'}")

    lines += ["", "-- duplicate content --"]
    if report.hashing_performed:
        lines.append(f"    files hashed              : {report.n_files_hashed:,}")
        lines.append(f"    byte-identical groups     : {report.n_duplicate_content_groups:,}")
        lines.append(f"    groups spanning splits    : {report.duplicate_content_across_splits or '{}'}")
        lines.append(f"    groups spanning domains   : {report.duplicate_content_across_domains}")
        for example in report.duplicate_content_examples[:5]:
            lines.append(f"      {example}")
    else:
        lines.append("    (hashing skipped)")

    lines += ["", "-- manifest hygiene --"]
    lines.append(f"    duplicated rows           : {report.duplicate_rows}")
    lines.append(f"    duplicated file paths     : {report.duplicate_paths}")
    lines.append(
        f"    repeated bare filenames   : {report.repeated_bare_filenames} "
        f"{report.repeated_bare_filename_examples}"
    )
    if report.repeated_bare_filenames:
        lines.append("      (harmless: image_id is namespaced by domain and split)")

    if report.problems:
        lines += ["", "!! PROBLEMS !!"] + [f"    * {p}" for p in report.problems]
    if report.limitations:
        lines += ["", "-- documented limitations --"] + [f"    * {p}" for p in report.limitations]

    lines.append("=" * 74)
    return "\n".join(lines)
