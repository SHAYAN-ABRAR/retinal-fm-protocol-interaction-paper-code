"""Assumption-free inspection of the four raw dataset folders.

This module *discovers* and *reports*; it never guesses.  Nothing here builds a
training index -- that is Phase 2, and it must be written against the output of
these functions rather than against an assumed layout.

Design rules
------------
* Read the filesystem, not the documentation.  Directory names, CSV column
  names and label encodings are all reported as found.
* Never coerce.  If a CSV column is missing, say so; do not fall back to
  positional indexing.
* Sample, do not exhaust.  Opening 100k JPEGs to read their headers costs
  minutes; a seeded random sample answers the same questions.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..utils.logging import get_logger

log = get_logger("data.inspect")

__all__ = [
    "DirectorySummary",
    "CsvSummary",
    "ImageStats",
    "summarise_directory",
    "summarise_csv",
    "sample_image_properties",
    "filename_family_census",
    "iter_images",
    "file_hash",
    "inspect_domain",
    "inspect_all_domains",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm"}

# Filename families observed across public DR datasets.  Used only for
# *reporting* which source a file appears to come from -- never for labelling.
FILENAME_FAMILIES: dict[str, re.Pattern[str]] = {
    "eyepacs (<n>_left / <n>_right)": re.compile(r"^\d+_(left|right)$"),
    "aptos (12 hex chars)": re.compile(r"^[0-9a-f]{12}$"),
    "idrid (IDRiD_nnn)": re.compile(r"^IDRiD_\d+$", re.IGNORECASE),
    "ddr (17-digit timestamp)": re.compile(r"^\d{17}$"),
    "ddr (nnn-nnnn-nnn)": re.compile(r"^\d{3}-\d{4}-\d{3}$"),
}

# Suffix/prefix markers for derived (augmented / filtered) copies, as used by the
# third-party "dr_unified" derivative in Dataset 1.
#
# NOTE: a resize suffix such as "-600" is deliberately NOT listed here. Stripping
# a trailing numeric group would mangle legitimate DDR filenames of the form
# 007-0004-000, reporting them as derived copies of a non-existent original.
# The only folder that uses "-600"/"-ALL" is augmented_resized_V2, which
# configs/paths.yaml marks as forbidden and this project never reads.
DERIVED_MARKERS = re.compile(r"(^GF-|-GF$)")


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

@dataclass
class DirectorySummary:
    root: str
    exists: bool
    subdirectories: list[str] = field(default_factory=list)
    file_count: int = 0
    extension_counts: dict[str, int] = field(default_factory=dict)
    example_files: list[str] = field(default_factory=list)
    total_bytes: int = 0


def summarise_directory(
    root: Path | str,
    *,
    max_depth: int = 3,
    max_examples: int = 5,
    count_bytes: bool = False,
) -> DirectorySummary:
    """Summarise a directory: sub-tree shape, extension census, examples.

    ``count_bytes`` is off by default because stat-ing 236k files on a spinning
    index takes a while; enable it deliberately.
    """
    root = Path(root)
    summary = DirectorySummary(root=str(root), exists=root.exists())
    if not summary.exists:
        return summary

    extensions: Counter[str] = Counter()
    examples: list[str] = []
    subdirs: list[str] = []
    total_files = 0
    total_bytes = 0

    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth < max_depth:
            for d in sorted(dirnames):
                subdirs.append(str(Path(dirpath, d).relative_to(root)))
        else:
            dirnames[:] = []  # stop descending, but still count files here

        for name in filenames:
            total_files += 1
            extensions[Path(name).suffix.lower()] += 1
            if len(examples) < max_examples:
                examples.append(str(Path(dirpath, name).relative_to(root)))
            if count_bytes:
                try:
                    total_bytes += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass

    summary.subdirectories = subdirs
    summary.file_count = total_files
    summary.extension_counts = dict(extensions.most_common())
    summary.example_files = examples
    summary.total_bytes = total_bytes
    return summary


def iter_images(root: Path | str, *, extensions: Iterable[str] | None = None) -> Iterator[Path]:
    """Yield every image file under ``root`` (recursively), lazily."""
    exts = {e.lower() for e in (extensions or IMAGE_EXTENSIONS)}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in exts:
                yield Path(dirpath, name)


# ---------------------------------------------------------------------------
# CSV / metadata
# ---------------------------------------------------------------------------

@dataclass
class CsvSummary:
    path: str
    exists: bool
    n_rows: int = 0
    columns: list[str] = field(default_factory=list)
    head: list[dict[str, str]] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)
    n_unique_ids: int | None = None
    n_duplicate_ids: int = 0
    duplicate_id_examples: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def summarise_csv(
    path: Path | str,
    *,
    id_column: str | None = None,
    label_column: str | None = None,
    n_head: int = 3,
) -> CsvSummary:
    """Read a label CSV and report its real schema and contents.

    Column names are reported exactly as they appear, including stray trailing
    whitespace (IDRiD's ``"Risk of macular edema "`` genuinely has one).
    """
    path = Path(path)
    summary = CsvSummary(path=str(path), exists=path.exists())
    if not summary.exists:
        return summary

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        summary.columns = list(reader.fieldnames or [])
        rows = list(reader)

    summary.n_rows = len(rows)
    summary.head = rows[:n_head]

    # Drop the all-empty padding columns that trail some hand-exported CSVs.
    padding = [c for c in summary.columns if c == "" or all(not (r.get(c) or "").strip() for r in rows)]
    if padding:
        summary.notes.append(f"empty/padding columns present and ignored: {padding!r}")

    if label_column is not None:
        if label_column not in summary.columns:
            summary.notes.append(
                f"MISSING label column {label_column!r}; available: {summary.columns!r}"
            )
        else:
            counts = Counter((r.get(label_column) or "").strip() for r in rows)
            summary.label_counts = {k: v for k, v in sorted(counts.items())}

    if id_column is not None:
        if id_column not in summary.columns:
            summary.notes.append(
                f"MISSING id column {id_column!r}; available: {summary.columns!r}"
            )
        else:
            ids = [(r.get(id_column) or "").strip() for r in rows]
            counts = Counter(ids)
            summary.n_unique_ids = len(counts)
            dupes = [k for k, v in counts.items() if v > 1]
            summary.n_duplicate_ids = len(dupes)
            summary.duplicate_id_examples = dupes[:5]

    return summary


def read_csv_ids_labels(
    path: Path | str, id_column: str, label_column: str
) -> list[tuple[str, str]]:
    """Return ``(id, label)`` pairs from a CSV, raising if a column is absent."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        for column in (id_column, label_column):
            if column not in fields:
                raise KeyError(
                    f"{path.name}: column {column!r} not found. Present: {fields!r}"
                )
        return [
            ((r.get(id_column) or "").strip(), (r.get(label_column) or "").strip())
            for r in reader
        ]


# ---------------------------------------------------------------------------
# Image properties
# ---------------------------------------------------------------------------

@dataclass
class ImageStats:
    n_sampled: int = 0
    n_failed: int = 0
    failures: list[str] = field(default_factory=list)
    size_counts: dict[str, int] = field(default_factory=dict)
    mode_counts: dict[str, int] = field(default_factory=dict)
    width_min: int | None = None
    width_max: int | None = None
    height_min: int | None = None
    height_max: int | None = None
    aspect_min: float | None = None
    aspect_max: float | None = None
    median_kilobytes: float | None = None


def sample_image_properties(
    paths: Sequence[Path],
    *,
    n_sample: int = 200,
    seed: int = 0,
) -> ImageStats:
    """Open a seeded random sample of images and report header properties.

    Only headers are read (``PIL.Image.open`` is lazy), so this is cheap even
    for 4288x2848 IDRiD scans.
    """
    from PIL import Image

    stats = ImageStats()
    if not paths:
        return stats

    rng = random.Random(seed)
    sample = rng.sample(list(paths), min(n_sample, len(paths)))

    sizes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    kilobytes: list[float] = []

    for path in sample:
        try:
            with Image.open(path) as im:
                width, height = im.size
                sizes[f"{width}x{height}"] += 1
                modes[im.mode] += 1
                widths.append(width)
                heights.append(height)
            kilobytes.append(os.path.getsize(path) / 1024)
        except Exception as exc:  # noqa: BLE001 - corrupt files are the point
            stats.n_failed += 1
            if len(stats.failures) < 10:
                stats.failures.append(f"{path.name}: {type(exc).__name__}: {exc}")

    stats.n_sampled = len(sample)
    stats.size_counts = dict(sizes.most_common(10))
    stats.mode_counts = dict(modes.most_common())
    if widths:
        aspects = [w / h for w, h in zip(widths, heights)]
        stats.width_min, stats.width_max = min(widths), max(widths)
        stats.height_min, stats.height_max = min(heights), max(heights)
        stats.aspect_min, stats.aspect_max = round(min(aspects), 3), round(max(aspects), 3)
        ordered = sorted(kilobytes)
        stats.median_kilobytes = round(ordered[len(ordered) // 2], 1)
    return stats


def file_hash(path: Path | str, *, algorithm: str = "md5", chunk_size: int = 1 << 20) -> str:
    """Content hash of a file, streamed so large images do not load into RAM."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Filename provenance
# ---------------------------------------------------------------------------

def filename_family_census(
    names: Iterable[str], *, strip_derived_markers: bool = True
) -> dict[str, Any]:
    """Classify bare filenames (no extension) into known dataset families.

    This is how we detect that a folder advertised as one dataset actually
    contains images from another.  Reporting only -- it assigns no labels.
    """
    counts: Counter[str] = Counter()
    unmatched: list[str] = []
    derived = 0

    for raw in names:
        stem = Path(raw).stem
        if strip_derived_markers:
            stripped = DERIVED_MARKERS.sub("", stem)
            while stripped != stem:
                stem, stripped = stripped, DERIVED_MARKERS.sub("", stripped)
            if stem != Path(raw).stem:
                derived += 1
        for family, pattern in FILENAME_FAMILIES.items():
            if pattern.match(stem):
                counts[family] += 1
                break
        else:
            counts["UNRECOGNISED"] += 1
            if len(unmatched) < 15:
                unmatched.append(Path(raw).stem)

    return {
        "family_counts": dict(counts.most_common()),
        "n_with_derived_markers": derived,
        "unrecognised_examples": unmatched,
    }


# ---------------------------------------------------------------------------
# Whole-domain inspection
# ---------------------------------------------------------------------------

def _collect_domain_csvs(spec: dict[str, Any]) -> list[tuple[str, Path]]:
    """Return ``(split_name, csv_path)`` for a domain, whatever its layout."""
    out: list[tuple[str, Path]] = []
    if "label_csv" in spec:
        out.append(("all", Path(spec["label_csv"])))
    for split, sub in (spec.get("splits") or {}).items():
        if isinstance(sub, dict) and "csv" in sub:
            out.append((split, Path(sub["csv"])))
    return out


def _collect_domain_image_dirs(spec: dict[str, Any]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if "image_dir" in spec:
        out.append(("all", Path(spec["image_dir"])))
    if "image_root" in spec:
        out.append(("all", Path(spec["image_root"])))
    for split, sub in (spec.get("splits") or {}).items():
        if isinstance(sub, dict) and "image_dir" in sub:
            out.append((split, Path(sub["image_dir"])))
    return out


def inspect_domain(
    key: str,
    spec: dict[str, Any],
    *,
    n_image_sample: int = 200,
    max_files_for_census: int = 300_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Inspect one domain end to end and return a JSON-serialisable report."""
    log.info("inspecting domain %-8s (domain_id=%s) ...", key, spec.get("domain_id"))

    report: dict[str, Any] = {
        "key": key,
        "name": spec.get("name", key),
        "domain_id": spec.get("domain_id"),
        "source_folder": spec.get("source_folder"),
        "structure": {},
        "csvs": {},
        "image_dirs": {},
        "filename_families": {},
        "image_properties": {},
        "declared_expectations": spec.get("verified", {}),
        "mismatches": [],
    }

    root = Path(spec["source_folder"])
    report["structure"] = summarise_directory(root, max_depth=3, max_examples=5)

    columns = spec.get("csv_columns", {})
    id_column, label_column = columns.get("image"), columns.get("label")
    for split, csv_path in _collect_domain_csvs(spec):
        report["csvs"][split] = summarise_csv(
            csv_path, id_column=id_column, label_column=label_column
        )

    all_names: list[str] = []
    for split, image_dir in _collect_domain_image_dirs(spec):
        paths = list(iter_images(image_dir))
        report["image_dirs"][split] = {
            "path": str(image_dir),
            "n_images": len(paths),
        }
        if len(paths) <= max_files_for_census:
            all_names.extend(p.name for p in paths)
        report["image_properties"][split] = sample_image_properties(
            paths, n_sample=n_image_sample, seed=seed
        )

    if all_names:
        report["filename_families"] = filename_family_census(all_names)

    _check_declared_expectations(report, spec)
    return report


def _check_declared_expectations(report: dict[str, Any], spec: dict[str, Any]) -> None:
    """Compare what we just measured against the numbers recorded in paths.yaml.

    ``paths.yaml`` stores counts observed during the original Phase-1 audit.
    Re-checking them here means a changed or partially copied dataset is caught
    immediately instead of silently altering every downstream result.
    """
    expected = spec.get("verified") or {}
    mismatches: list[str] = report["mismatches"]

    if "csv_rows" in expected:
        actual = report["csvs"].get("all", CsvSummary(path="", exists=False)).n_rows
        if actual != expected["csv_rows"]:
            mismatches.append(f"csv_rows: expected {expected['csv_rows']}, found {actual}")

    if "split_sizes" in expected:
        for split, want in expected["split_sizes"].items():
            got = report["image_dirs"].get(split, {}).get("n_images")
            if got != want:
                mismatches.append(f"images in split {split!r}: expected {want}, found {got}")

    if "total_images" in expected:
        total = sum(d.get("n_images", 0) for d in report["image_dirs"].values())
        if total != expected["total_images"]:
            mismatches.append(
                f"total images: expected {expected['total_images']}, found {total}"
            )

    for key_name, split in (("label_counts", "all"),
                            ("train_label_counts", "train"),
                            ("test_label_counts", "test")):
        if key_name not in expected:
            continue
        summary = report["csvs"].get(split)
        if summary is None:
            continue
        got = {str(k): v for k, v in summary.label_counts.items()}
        want = {str(k): v for k, v in expected[key_name].items()}
        if got != want:
            mismatches.append(f"{key_name}: expected {want}, found {got}")


def inspect_all_domains(
    paths_config: dict[str, Any],
    *,
    n_image_sample: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Inspect every domain declared in ``configs/paths.yaml``."""
    reports: dict[str, Any] = {}
    for key, spec in paths_config["domains"].items():
        reports[key] = inspect_domain(key, spec, n_image_sample=n_image_sample, seed=seed)
    return reports


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def format_domain_report(report: dict[str, Any]) -> str:
    """Render one domain report as readable console text."""
    lines: list[str] = []
    header = f"DOMAIN {report['domain_id']} -- {report['name']}  ({report['key']})"
    lines += ["=" * 74, header, "=" * 74]
    lines.append(f"  source folder : {report['source_folder']}")

    structure = report["structure"]
    n_files = structure.file_count if hasattr(structure, "file_count") else structure["file_count"]
    exts = structure.extension_counts if hasattr(structure, "extension_counts") else structure["extension_counts"]
    lines.append(f"  files on disk : {n_files:,}")
    lines.append(f"  extensions    : {exts}")

    if report["csvs"]:
        lines.append("  -- label CSVs --")
        for split, summary in report["csvs"].items():
            rows = summary.n_rows if hasattr(summary, "n_rows") else summary["n_rows"]
            cols = summary.columns if hasattr(summary, "columns") else summary["columns"]
            labels = summary.label_counts if hasattr(summary, "label_counts") else summary["label_counts"]
            notes = summary.notes if hasattr(summary, "notes") else summary["notes"]
            lines.append(f"    [{split}] {rows:,} rows | columns={cols}")
            lines.append(f"        label counts: {labels}")
            for note in notes:
                lines.append(f"        NOTE: {note}")

    if report["image_dirs"]:
        lines.append("  -- image folders --")
        for split, info in report["image_dirs"].items():
            lines.append(f"    [{split}] {info['n_images']:,} images  <- {info['path']}")

    if report["image_properties"]:
        lines.append("  -- sampled image properties --")
        for split, stats in report["image_properties"].items():
            sizes = stats.size_counts if hasattr(stats, "size_counts") else stats["size_counts"]
            n_failed = stats.n_failed if hasattr(stats, "n_failed") else stats["n_failed"]
            w_min = stats.width_min if hasattr(stats, "width_min") else stats["width_min"]
            w_max = stats.width_max if hasattr(stats, "width_max") else stats["width_max"]
            h_min = stats.height_min if hasattr(stats, "height_min") else stats["height_min"]
            h_max = stats.height_max if hasattr(stats, "height_max") else stats["height_max"]
            lines.append(
                f"    [{split}] width {w_min}-{w_max}, height {h_min}-{h_max}, "
                f"unreadable={n_failed}"
            )
            top = list(sizes.items())[:4]
            lines.append(f"        most common sizes: {top}")

    if report["filename_families"]:
        lines.append("  -- filename provenance --")
        for family, count in report["filename_families"]["family_counts"].items():
            lines.append(f"    {family:32s}: {count:,}")
        derived = report["filename_families"]["n_with_derived_markers"]
        if derived:
            lines.append(f"    files carrying derived-copy markers: {derived:,}")

    if report["mismatches"]:
        lines.append("  !! MISMATCH vs the counts recorded in configs/paths.yaml !!")
        for m in report["mismatches"]:
            lines.append(f"     * {m}")
    else:
        lines.append("  OK: matches the counts recorded in configs/paths.yaml")

    return "\n".join(lines)
