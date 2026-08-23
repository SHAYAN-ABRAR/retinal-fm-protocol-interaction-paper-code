"""Exact-duplicate detection and removal, applied *before* splitting.

Why this exists
---------------
The split-level leakage audit found 224 groups of byte-identical images (456
files) hiding under different filenames across three of the four domains:

    ddr 95 groups | aptos 123 groups | idrid 6 groups

89 of those groups straddled a split boundary, which means a model could be
trained on a file and then evaluated on a byte-identical copy of it.  That is
leakage, and it inflates in-domain results.

Worse, **33 groups contain the same bytes labelled with different grades.** The
ground truth contradicts itself for those images.

Policy
------
Two rules, applied per duplicate group:

``consistent`` group (all members share a grade)
    Keep one representative -- the lexicographically smallest ``image_id``, so
    the choice is deterministic -- and drop the rest.  The copies carry no extra
    information and would distort class frequencies.

``conflicting`` group (members disagree on the grade)
    **Drop the whole group.**  We cannot know which label is correct, and keeping
    an arbitrary one injects known-bad supervision.  The cost is tiny (~70
    images) and the alternative is unjustifiable.

Both counts are reported and written to the run record, never applied silently.

Scope
-----
This finds byte-identical files only.  The same photograph re-encoded or resized
produces different bytes and is not detected; see the limitation noted in
``leakage.py``.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.io import ensure_dir
from ..utils.logging import get_logger
from .inspect import file_hash

log = get_logger("data.deduplicate")

__all__ = [
    "DeduplicationReport",
    "compute_content_hashes",
    "deduplicate_manifest",
    "format_deduplication_report",
]


@dataclass
class DeduplicationReport:
    n_input: int = 0
    n_output: int = 0
    n_files_hashed: int = 0
    n_groups: int = 0
    n_consistent_groups: int = 0
    n_conflicting_groups: int = 0
    n_dropped_redundant: int = 0
    n_dropped_conflicting: int = 0
    groups_per_domain: dict[str, int] = field(default_factory=dict)
    cross_domain_groups: int = 0
    conflicting_examples: list[dict[str, Any]] = field(default_factory=list)
    removed_per_domain: dict[str, int] = field(default_factory=dict)

    @property
    def n_removed(self) -> int:
        return self.n_dropped_redundant + self.n_dropped_conflicting


def compute_content_hashes(
    manifest: "Any",
    *,
    algorithm: str = "md5",
    cache_path: Path | str | None = None,
) -> dict[str, str]:
    """Hash only those files whose size is shared with another file.

    A file whose byte-size is unique in the corpus cannot be byte-identical to
    any other file, so hashing it is wasted I/O. On this corpus the prefilter
    cuts hashing from 51,808 files to about 10,400.

    Results are cached to CSV because re-hashing on every pipeline run is the
    single slowest step in Phase 2.
    """
    import pandas as pd

    paths = manifest["path"].tolist()

    cached: dict[str, str] = {}
    if cache_path is not None and Path(cache_path).exists():
        frame = pd.read_csv(cache_path)
        cached = dict(zip(frame["path"], frame["content_hash"]))
        log.info("loaded %d cached content hashes from %s", len(cached), cache_path)

    by_size: dict[int, list[str]] = defaultdict(list)
    for path in paths:
        try:
            by_size[os.path.getsize(path)].append(path)
        except OSError as exc:
            log.warning("cannot stat %s: %s", path, exc)

    candidates = [p for group in by_size.values() if len(group) > 1 for p in group]
    todo = [p for p in candidates if p not in cached]
    log.info(
        "content hashing: %d/%d files share a size; %d already cached, %d to hash",
        len(candidates), len(paths), len(candidates) - len(todo), len(todo),
    )

    digests = {p: cached[p] for p in candidates if p in cached}
    for path in todo:
        try:
            digests[path] = file_hash(path, algorithm=algorithm)
        except OSError as exc:
            log.warning("cannot hash %s: %s", path, exc)

    if cache_path is not None and todo:
        ensure_dir(Path(cache_path).parent)
        merged = {**cached, **digests}
        pd.DataFrame(
            {"path": list(merged), "content_hash": list(merged.values())}
        ).to_csv(cache_path, index=False)
        log.info("cached %d content hashes -> %s", len(merged), cache_path)

    return digests


def deduplicate_manifest(
    manifest: "Any",
    *,
    algorithm: str = "md5",
    cache_path: Path | str | None = None,
) -> tuple["Any", DeduplicationReport]:
    """Remove exact duplicates according to the documented policy."""
    report = DeduplicationReport(n_input=len(manifest))
    digests = compute_content_hashes(manifest, algorithm=algorithm, cache_path=cache_path)
    report.n_files_hashed = len(digests)

    by_digest: dict[str, list[str]] = defaultdict(list)
    for path, digest in digests.items():
        by_digest[digest].append(path)
    groups = {d: p for d, p in by_digest.items() if len(p) > 1}
    report.n_groups = len(groups)

    if not groups:
        report.n_output = len(manifest)
        return manifest.copy(), report

    lookup = manifest.set_index("path")
    drop_paths: set[str] = set()
    domain_counter: Counter[str] = Counter()
    removed_counter: Counter[str] = Counter()

    for digest, paths in groups.items():
        rows = lookup.loc[paths]
        domains = sorted(set(rows["domain"]))
        for domain in domains:
            domain_counter[domain] += 1
        if len(domains) > 1:
            report.cross_domain_groups += 1

        if rows["grade"].nunique() > 1:
            # Contradictory ground truth -- discard the entire group.
            report.n_conflicting_groups += 1
            report.n_dropped_conflicting += len(paths)
            drop_paths.update(paths)
            for domain in rows["domain"]:
                removed_counter[domain] += 1
            if len(report.conflicting_examples) < 10:
                report.conflicting_examples.append(
                    {
                        "hash": digest[:12],
                        "image_ids": rows["image_id"].tolist(),
                        "grades": [int(g) for g in rows["grade"]],
                        "domains": domains,
                    }
                )
        else:
            # Consistent: keep the lexicographically smallest image_id.
            report.n_consistent_groups += 1
            keep = rows["image_id"].idxmin()
            redundant = [p for p in paths if p != keep]
            report.n_dropped_redundant += len(redundant)
            drop_paths.update(redundant)
            for path in redundant:
                removed_counter[str(lookup.loc[path, "domain"])] += 1

    cleaned = manifest[~manifest["path"].isin(drop_paths)].reset_index(drop=True)
    report.n_output = len(cleaned)
    report.groups_per_domain = dict(domain_counter)
    report.removed_per_domain = dict(removed_counter)

    log.info(
        "deduplication: %d -> %d images (removed %d redundant, %d from conflicting groups)",
        report.n_input, report.n_output,
        report.n_dropped_redundant, report.n_dropped_conflicting,
    )
    return cleaned, report


def format_deduplication_report(report: DeduplicationReport) -> str:
    lines = [
        "=" * 74,
        "EXACT-DUPLICATE REMOVAL (applied before splitting)",
        "=" * 74,
        f"  images in           : {report.n_input:,}",
        f"  images out          : {report.n_output:,}  (removed {report.n_removed:,})",
        f"  files hashed        : {report.n_files_hashed:,}",
        "",
        f"  duplicate groups    : {report.n_groups:,}",
        f"    consistent labels : {report.n_consistent_groups:,} "
        f"-> kept 1 each, dropped {report.n_dropped_redundant:,} redundant copies",
        f"    CONFLICTING labels: {report.n_conflicting_groups:,} "
        f"-> whole group dropped ({report.n_dropped_conflicting:,} images)",
        f"    spanning 2 domains: {report.cross_domain_groups:,}",
        "",
        f"  groups touching each domain : {report.groups_per_domain}",
        f"  images removed per domain   : {report.removed_per_domain}",
    ]
    if report.conflicting_examples:
        lines += ["", "  examples of byte-identical images with contradictory grades:"]
        for example in report.conflicting_examples[:5]:
            lines.append(
                f"    {example['hash']}  grades={example['grades']}  {example['image_ids']}"
            )
    lines.append("=" * 74)
    return "\n".join(lines)
