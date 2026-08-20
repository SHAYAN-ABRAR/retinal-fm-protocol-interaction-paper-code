"""Provenance and integrity audits for the four raw dataset folders.

These are *not* the full leakage audit of Section 5 of the research brief (that
one runs on the constructed train/val/test indices and lives in
``src/data/leakage.py``, Phase 2).  These audits run on the **raw folders**, and
exist because an initial inspection of ``D:\\DB`` found four problems severe
enough that building loaders without checking for them would silently corrupt
every downstream result:

A. ``Dataset 1`` is not EyePACS.  It is a third-party pooled derivative that
   contains EyePACS *and* the complete APTOS dataset -- the very same 3,662
   images that make up Domain 1.  Treating Dataset 1 as "the EyePACS domain"
   would place identical photographs in two supposedly different domains.

B. ``Dataset 1``'s own train/val/test folders leak.  The same base photograph
   appears in more than one split (via augmented ``-GF`` copies), and EyePACS
   patients have their left and right eye in different splits.

C. IDRiD's training and testing folders both start at ``IDRiD_001.jpg``.  File
   names collide across splits, so any index keyed on bare filename silently
   merges two different images.

D. DDR's image folder holds two files that appear in no CSV row.

Every audit re-measures the problem on this machine rather than trusting the
numbers recorded in ``configs/paths.yaml``, so the reviewer can reproduce them.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from ..utils.logging import get_logger
from .inspect import iter_images, read_csv_ids_labels

log = get_logger("data.provenance")

__all__ = [
    "Finding",
    "audit_eyepacs_folder_composition",
    "audit_cross_domain_duplication",
    "audit_provided_split_integrity",
    "audit_filename_collisions",
    "audit_csv_disk_consistency",
    "audit_label_alphabets",
    "run_all_audits",
    "format_findings",
]

SEVERITIES = ("CRITICAL", "WARNING", "INFO")

EYEPACS_ID = re.compile(r"^\d+_(left|right)$")
APTOS_ID = re.compile(r"^[0-9a-f]{12}$")
DERIVED_COPY = re.compile(r"(^GF-|-GF$)")


@dataclass
class Finding:
    """One audit result.  ``severity`` drives whether the pipeline may proceed."""

    audit: str
    severity: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    consequence: str = ""
    remedy: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _base_id(name: str) -> str:
    """Strip the extension and any derived-copy marker from a filename."""
    stem = Path(name).stem
    previous = None
    while previous != stem:
        previous = stem
        stem = DERIVED_COPY.sub("", stem)
    return stem


def _eyepacs_patient(base: str) -> str | None:
    """EyePACS encodes the patient in the filename: ``<patient>_<eye>``."""
    match = EYEPACS_ID.match(base)
    return base.split("_", 1)[0] if match else None


def _class_folder_index(image_root: Path) -> dict[str, list[tuple[str, str]]]:
    """Index a ``<root>/<split>/<class>/*.jpg`` layout.

    Returns ``{base_id: [(split, class_label), ...]}``.
    """
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for image in iter_images(class_dir):
                index[_base_id(image.name)].append((split_dir.name, class_dir.name))
    return index


# ---------------------------------------------------------------------------
# Audit A -- what is actually inside the "EyePACS" folder
# ---------------------------------------------------------------------------

def audit_eyepacs_folder_composition(spec: dict[str, Any]) -> list[Finding]:
    """Report the true composition of the folder mapped to the EyePACS domain."""
    image_root = Path(spec["image_root"])
    findings: list[Finding] = []

    families: Counter[str] = Counter()
    derived = 0
    originals_by_family: dict[str, set[str]] = defaultdict(set)

    for image in iter_images(image_root):
        stem = Path(image.name).stem
        base = _base_id(image.name)
        if base != stem:
            derived += 1
        if EYEPACS_ID.match(base):
            families["eyepacs"] += 1
            originals_by_family["eyepacs"].add(base)
        elif APTOS_ID.match(base):
            families["aptos"] += 1
            originals_by_family["aptos"].add(base)
        else:
            families["unrecognised"] += 1
            originals_by_family["unrecognised"].add(base)

    patients = {p for b in originals_by_family["eyepacs"] if (p := _eyepacs_patient(b))}

    findings.append(
        Finding(
            audit="eyepacs_folder_composition",
            severity="CRITICAL" if originals_by_family["aptos"] else "INFO",
            summary=(
                "The folder mapped to the EyePACS domain is a pooled derivative: it "
                f"contains {len(originals_by_family['eyepacs']):,} unique EyePACS images "
                f"({len(patients):,} patients) AND "
                f"{len(originals_by_family['aptos']):,} unique APTOS images, plus "
                f"{derived:,} augmented duplicate files."
            ),
            evidence={
                "image_root": str(image_root),
                "files_by_family": dict(families),
                "unique_eyepacs_images": len(originals_by_family["eyepacs"]),
                "unique_eyepacs_patients": len(patients),
                "unique_aptos_images": len(originals_by_family["aptos"]),
                "files_that_are_augmented_copies": derived,
            },
            consequence=(
                "Using this folder as-is would (a) put APTOS images in two different "
                "'domains' at once and (b) train on augmented copies of test images."
            ),
            remedy=(
                "Keep only filenames matching ^\\d+_(left|right)$, drop every file "
                "carrying a GF- / -GF marker, and ignore the provided split folders. "
                "Rules are encoded in configs/paths.yaml under domains.eyepacs."
            ),
        )
    )
    return findings


# ---------------------------------------------------------------------------
# Audit B -- the same photograph in two different domains
# ---------------------------------------------------------------------------

def audit_cross_domain_duplication(
    eyepacs_spec: dict[str, Any], aptos_spec: dict[str, Any]
) -> list[Finding]:
    """Check whether the EyePACS folder republishes images from the APTOS domain."""
    aptos_ids: set[str] = set()
    for split in aptos_spec["splits"].values():
        pairs = read_csv_ids_labels(
            split["csv"],
            aptos_spec["csv_columns"]["image"],
            aptos_spec["csv_columns"]["label"],
        )
        aptos_ids.update(image_id for image_id, _ in pairs)

    pooled_aptos: set[str] = set()
    for image in iter_images(Path(eyepacs_spec["image_root"])):
        base = _base_id(image.name)
        if APTOS_ID.match(base):
            pooled_aptos.add(base)

    overlap = pooled_aptos & aptos_ids
    fraction = len(overlap) / len(aptos_ids) if aptos_ids else 0.0

    return [
        Finding(
            audit="cross_domain_duplication",
            severity="CRITICAL" if overlap else "INFO",
            summary=(
                f"{len(overlap):,} of {len(aptos_ids):,} APTOS images "
                f"({fraction:.1%}) are also physically present inside the folder "
                "mapped to the EyePACS domain."
            ),
            evidence={
                "aptos_domain_images": len(aptos_ids),
                "aptos_style_images_in_eyepacs_folder": len(pooled_aptos),
                "overlapping_ids": len(overlap),
                "overlap_fraction_of_aptos": round(fraction, 4),
                "examples": sorted(overlap)[:5],
            },
            consequence=(
                "A leave-one-domain-out experiment holding out APTOS would still train "
                "on every held-out APTOS image via the EyePACS domain. The external "
                "evaluation would be meaningless."
            ),
            remedy=(
                "APTOS images must come only from the APTOS domain. The EyePACS loader "
                "must exclude every 12-hex filename; this is asserted, not assumed."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Audit C -- integrity of vendor-provided splits
# ---------------------------------------------------------------------------

def audit_provided_split_integrity(spec: dict[str, Any]) -> list[Finding]:
    """Measure image-level and patient-level leakage in provided split folders."""
    image_root = Path(spec["image_root"])
    index = _class_folder_index(image_root)

    by_split: dict[str, set[str]] = defaultdict(set)
    label_conflicts: dict[str, list[str]] = {}
    for base, entries in index.items():
        labels = {label for _split, label in entries}
        if len(labels) > 1:
            label_conflicts[base] = sorted(labels)
        for split, _label in entries:
            by_split[split].add(base)

    findings: list[Finding] = []

    image_overlap = {
        f"{a}&{b}": len(by_split[a] & by_split[b]) for a, b in combinations(sorted(by_split), 2)
    }
    patients_by_split = {
        split: {p for b in bases if (p := _eyepacs_patient(b))} for split, bases in by_split.items()
    }
    patient_overlap = {
        f"{a}&{b}": len(patients_by_split[a] & patients_by_split[b])
        for a, b in combinations(sorted(patients_by_split), 2)
    }

    total_image_overlap = sum(image_overlap.values())
    total_patient_overlap = sum(patient_overlap.values())

    findings.append(
        Finding(
            audit="provided_split_integrity",
            severity="CRITICAL" if (total_image_overlap or total_patient_overlap) else "INFO",
            summary=(
                "The provided train/val/test folders leak: "
                f"{total_image_overlap:,} base photographs and "
                f"{total_patient_overlap:,} patients appear in more than one split."
            ),
            evidence={
                "image_root": str(image_root),
                "unique_base_images_per_split": {k: len(v) for k, v in sorted(by_split.items())},
                "base_image_overlap": image_overlap,
                "unique_patients_per_split": {
                    k: len(v) for k, v in sorted(patients_by_split.items())
                },
                "patient_overlap": patient_overlap,
            },
            consequence=(
                "Any in-domain result computed on these splits is optimistically biased "
                "and cannot be reported."
            ),
            remedy=(
                "Discard the provided split folders entirely. Pool the images, then "
                "re-split at patient level (both eyes of a patient stay together). "
                "configs/paths.yaml sets ignore_provided_splits: true for this reason."
            ),
        )
    )

    findings.append(
        Finding(
            audit="label_consistency_across_folders",
            severity="CRITICAL" if label_conflicts else "INFO",
            summary=(
                f"{len(label_conflicts):,} base images carry conflicting grade labels "
                "in different class folders."
            ),
            evidence={
                "n_conflicts": len(label_conflicts),
                "examples": dict(list(label_conflicts.items())[:5]),
            },
            consequence="Conflicting labels would make the folder-derived grade unusable.",
            remedy="None needed if zero; otherwise the folder labels cannot be trusted at all.",
        )
    )
    return findings


# ---------------------------------------------------------------------------
# Audit D -- filename collisions between splits
# ---------------------------------------------------------------------------

def audit_filename_collisions(key: str, spec: dict[str, Any]) -> list[Finding]:
    """Detect identical filenames reused across a domain's split folders."""
    splits = spec.get("splits") or {}
    names: dict[str, set[str]] = {}
    for split, sub in splits.items():
        if "image_dir" not in sub:
            continue
        names[split] = {p.name for p in iter_images(sub["image_dir"])}

    collisions = {
        f"{a}&{b}": sorted(names[a] & names[b]) for a, b in combinations(sorted(names), 2)
    }
    n_collisions = sum(len(v) for v in collisions.values())

    return [
        Finding(
            audit="filename_collisions",
            severity="WARNING" if n_collisions else "INFO",
            summary=(
                f"[{key}] {n_collisions:,} filenames are reused across split folders."
            ),
            evidence={
                "per_split_counts": {k: len(v) for k, v in names.items()},
                "collision_counts": {k: len(v) for k, v in collisions.items()},
                "examples": {k: v[:3] for k, v in collisions.items() if v},
            },
            consequence=(
                "An index keyed on bare filename would merge two distinct images and "
                "assign one of them the wrong grade."
            ),
            remedy=(
                "Namespace every image id as '<domain>/<split>/<filename>'. The unified "
                "index must use that composite key, not the raw filename."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Audit E -- CSV rows vs files on disk
# ---------------------------------------------------------------------------

def audit_csv_disk_consistency(key: str, spec: dict[str, Any]) -> list[Finding]:
    """Cross-check every labelled id against the files actually present."""
    columns = spec["csv_columns"]
    id_includes_ext = bool(spec.get("id_includes_extension", False))
    extension = spec.get("image_ext", "")

    pairs: list[tuple[str, str]] = []
    image_dirs: list[Path] = []

    if "label_csv" in spec:
        pairs += read_csv_ids_labels(spec["label_csv"], columns["image"], columns["label"])
        image_dirs.append(Path(spec["image_dir"]))
    for sub in (spec.get("splits") or {}).values():
        if "csv" in sub and "image_dir" in sub:
            pairs += read_csv_ids_labels(sub["csv"], columns["image"], columns["label"])
            image_dirs.append(Path(sub["image_dir"]))

    if not image_dirs:
        return []

    on_disk: set[str] = set()
    for directory in image_dirs:
        on_disk |= {p.name for p in iter_images(directory)}

    expected = {
        (image_id if id_includes_ext else f"{image_id}{extension}") for image_id, _ in pairs
    }
    missing = sorted(expected - on_disk)
    orphans = sorted(on_disk - expected)

    return [
        Finding(
            audit="csv_disk_consistency",
            severity="WARNING" if (missing or orphans) else "INFO",
            summary=(
                f"[{key}] {len(pairs):,} labelled rows; "
                f"{len(missing):,} have no image file; "
                f"{len(orphans):,} image files have no label row."
            ),
            evidence={
                "n_label_rows": len(pairs),
                "n_files_on_disk": len(on_disk),
                "n_labelled_without_file": len(missing),
                "labelled_without_file_examples": missing[:5],
                "n_files_without_label": len(orphans),
                "files_without_label": orphans[:10],
            },
            consequence=(
                "Unlabelled files must be excluded explicitly; labelled-but-absent rows "
                "would crash the loader mid-epoch."
            ),
            remedy="The unified index keeps only ids that have both a label row and a file.",
        )
    ]


# ---------------------------------------------------------------------------
# Audit F -- label alphabet
# ---------------------------------------------------------------------------

def audit_label_alphabets(paths_config: dict[str, Any]) -> list[Finding]:
    """Verify that every domain really encodes grades as the integers 0-4.

    The five-grade ICDR scale is assumed by the whole study. If any domain used
    a different alphabet (for example DDR's ungradable class 5) it must be
    mapped explicitly, never silently reinterpreted.
    """
    findings: list[Finding] = []
    alphabets: dict[str, dict[str, int]] = {}

    for key, spec in paths_config["domains"].items():
        columns = spec.get("csv_columns")
        if not columns:
            alphabets[key] = {"<labels come from folder names>": -1}
            continue
        counts: Counter[str] = Counter()
        if "label_csv" in spec:
            counts.update(
                label for _, label in read_csv_ids_labels(
                    spec["label_csv"], columns["image"], columns["label"]
                )
            )
        for sub in (spec.get("splits") or {}).values():
            if "csv" in sub:
                counts.update(
                    label for _, label in read_csv_ids_labels(
                        sub["csv"], columns["image"], columns["label"]
                    )
                )
        alphabets[key] = dict(sorted(counts.items()))

    expected = {"0", "1", "2", "3", "4"}
    unexpected = {
        key: sorted(set(counts) - expected)
        for key, counts in alphabets.items()
        if counts and -1 not in counts.values() and set(counts) - expected
    }

    findings.append(
        Finding(
            audit="label_alphabets",
            severity="CRITICAL" if unexpected else "INFO",
            summary=(
                "All CSV-labelled domains use the grade alphabet {0,1,2,3,4}."
                if not unexpected
                else f"Unexpected grade values found: {unexpected}"
            ),
            evidence={"per_domain_label_counts": alphabets},
            consequence=(
                "An unmapped extra class (for example DDR grade 5 = ungradable) would be "
                "trained on as if it were a severity level."
            ),
            remedy=(
                "Any non-0-4 value must get an explicit entry in the label map written "
                "by Phase 2 to outputs/reports/label_mapping.json."
            ),
        )
    )
    return findings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all_audits(paths_config: dict[str, Any]) -> list[Finding]:
    """Run every raw-data audit and return the findings, worst first."""
    domains = paths_config["domains"]
    findings: list[Finding] = []

    log.info("audit 1/6: label alphabets")
    findings += audit_label_alphabets(paths_config)

    log.info("audit 2/6: CSV vs disk consistency")
    for key, spec in domains.items():
        if spec.get("csv_columns"):
            findings += audit_csv_disk_consistency(key, spec)

    log.info("audit 3/6: filename collisions across split folders")
    for key, spec in domains.items():
        if spec.get("splits"):
            findings += audit_filename_collisions(key, spec)

    if "eyepacs" in domains:
        log.info("audit 4/6: EyePACS folder composition (walks ~90k files, ~1 min)")
        findings += audit_eyepacs_folder_composition(domains["eyepacs"])

        if "aptos" in domains:
            log.info("audit 5/6: cross-domain duplication (EyePACS folder vs APTOS domain)")
            findings += audit_cross_domain_duplication(domains["eyepacs"], domains["aptos"])

        log.info("audit 6/6: provided-split integrity (image- and patient-level)")
        findings += audit_provided_split_integrity(domains["eyepacs"])

    order = {name: i for i, name in enumerate(SEVERITIES)}
    findings.sort(key=lambda f: order[f.severity])
    return findings


def format_findings(findings: Iterable[Finding]) -> str:
    """Render findings as a console report."""
    findings = list(findings)
    counts = Counter(f.severity for f in findings)
    lines = [
        "=" * 74,
        "RAW-DATA PROVENANCE AUDIT",
        "=" * 74,
        f"  {counts.get('CRITICAL', 0)} critical | "
        f"{counts.get('WARNING', 0)} warning | {counts.get('INFO', 0)} info",
        "",
    ]
    for finding in findings:
        lines.append(f"[{finding.severity}] {finding.audit}")
        lines.append(f"    {finding.summary}")
        if finding.consequence:
            lines.append(f"    consequence: {finding.consequence}")
        if finding.remedy:
            lines.append(f"    remedy     : {finding.remedy}")
        lines.append("")
    lines.append("=" * 74)
    return "\n".join(lines)
