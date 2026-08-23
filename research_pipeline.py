"""Interactive research driver for the calibrated-DG diabetic-retinopathy study.

HOW TO USE
----------
This file is a VS Code *Python Interactive* script, not a program to run
top-to-bottom.  Every ``# %%`` marker starts a cell; press "Run Cell" above it
(or Ctrl+Enter) exactly as you would in Colab.

    Cells 1-8    Phase 1 -- environment, config, raw-data inspection, audits.
                 Cheap (~5 min). Run these first, in order.
    Cells 9-13   Phase 2 -- unified manifest, deduplication, splits, leakage
                 audit, image statistics, figures, raw dataloaders. ~10 min the
                 first time; cached afterwards. Cell 13 needs torch.
    Cell 13b     Phase 3 -- build the pre-resized image cache. ~30 min ONCE per
                 resolution. Everything after this reads the cache.
    Cells 14-20  Phase 3 -- model + batch probe, sanity/overfit check, ERM
                 training (~20 min), evaluation, generalization gap, registry,
                 bootstrap CIs, selective prediction.
    Cells 21-25  Phase 4 -- method comparison (ordinal / Deep CORAL / MixStyle),
                 paired bootstrap, per-class recall, representation analysis,
                 multi-seed aggregation. The training itself is launched from a
                 terminal (run_method_comparison.py); these cells read and
                 interpret what the runs produced.
    Cells 26+    Later phases. STUBS: each says what it will do and what it is
                 waiting on. None fabricates results.

Nothing expensive executes on import.  Opening this file costs nothing, and no
cell trains a model unless you deliberately run it.

The heavy lifting lives in ``src/``; this file only orchestrates.
"""

# %%
# ===========================================================================
# CELL 1 -- Imports and environment
# ===========================================================================
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable no matter which directory the interactive window
# started in (VS Code usually starts at the workspace root, one level up).
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _candidate in (_HERE, *_HERE.parents[:2]):
    if (_candidate / "src" / "__init__.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        PROJECT_DIR = _candidate
        break
else:  # pragma: no cover - only hit if the file is moved out of the repo
    raise RuntimeError(
        "Could not locate the project root (no src/__init__.py found). "
        "Open the folder 'dr_domain_generalization' in VS Code and retry."
    )

from src.utils.io import ensure_dir, load_paths_config, project_root, write_json  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.seed import environment_fingerprint, set_global_seed  # noqa: E402

ROOT = project_root()
LOG = configure_logging(ROOT / "outputs" / "logs" / "pipeline.log")

print(f"project root : {ROOT}")
print(f"python       : {sys.version.split()[0]}")
print("Cell 1 OK -- imports resolved.")


# %%
# ===========================================================================
# CELL 2 -- CUDA / GPU diagnostics
# ---------------------------------------------------------------------------
# Prints a full hardware report. It does NOT raise, so you can inspect a broken
# environment. Call ``assert_cuda_ready()`` (bottom of the cell) before any
# training cell -- that one refuses to continue on CPU.
# ===========================================================================
from src.utils.hardware import (  # noqa: E402
    assert_cuda_ready,
    cuda_report,
    format_report,
    system_report,
)

GPU_REPORT = cuda_report(run_smoke_test=True)
SYSTEM_REPORT = system_report()
print(format_report(GPU_REPORT, SYSTEM_REPORT))

# Uncomment once torch is installed; raises rather than silently using the CPU.
# assert_cuda_ready()


# %%
# ===========================================================================
# CELL 3 -- Global configuration and paths
# ---------------------------------------------------------------------------
# All dataset locations live in configs/paths.yaml. Loading verifies that every
# declared path exists and reports ALL missing ones at once.
# ===========================================================================
PATHS = load_paths_config()

DATA_ROOT = Path(PATHS["data_root"])
OUTPUTS = ensure_dir(PATHS["outputs_root"])
for _sub in ("checkpoints", "logs", "tables", "predictions",
             "figures", "embeddings", "reports"):
    ensure_dir(OUTPUTS / _sub)

# Domain numbering is fixed by the research protocol and must never be reordered:
DOMAIN_IDS = {key: spec["domain_id"] for key, spec in PATHS["domains"].items()}
GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

print(f"data root : {DATA_ROOT}")
print(f"outputs   : {OUTPUTS}")
print(f"domains   : {DOMAIN_IDS}")
print(f"grades    : {GRADE_NAMES}")
print("Cell 3 OK -- every path in configs/paths.yaml exists.")


# %%
# ===========================================================================
# CELL 4 -- Seeds and reproducibility record
# ===========================================================================
SEED = 42
DETERMINISTIC = False  # see src/utils/seed.py for the throughput trade-off

set_global_seed(SEED, deterministic=DETERMINISTIC)
FINGERPRINT = environment_fingerprint(SEED, deterministic=DETERMINISTIC)
write_json(OUTPUTS / "reports" / "environment_fingerprint.json", FINGERPRINT)

print(f"seed              : {FINGERPRINT.seed} (deterministic={FINGERPRINT.deterministic})")
print(f"python            : {FINGERPRINT.python_version}")
print(f"platform          : {FINGERPRINT.platform}")
print(f"git commit        : {FINGERPRINT.git_commit or 'not a git repository'}")
print("packages:")
for _name, _version in FINGERPRINT.packages.items():
    print(f"    {_name:16s}: {_version}")
print(f"cuda              : {FINGERPRINT.cuda}")
print("\nsaved -> outputs/reports/environment_fingerprint.json")


# %%
# ===========================================================================
# CELL 5 -- Inspect the raw dataset directory structures
# ---------------------------------------------------------------------------
# Discovers what is actually on disk: folder shape, file counts, extensions.
# Assumes nothing about layout. Takes ~1-2 min because Dataset 1 holds 236k
# files.
# ===========================================================================
from src.data.inspect import summarise_directory  # noqa: E402

STRUCTURES = {}
for _key, _spec in PATHS["domains"].items():
    _summary = summarise_directory(_spec["source_folder"], max_depth=3, max_examples=4)
    STRUCTURES[_key] = _summary
    print(f"--- domain {_spec['domain_id']} : {_spec['name']} ---")
    print(f"    root       : {_summary.root}")
    print(f"    files      : {_summary.file_count:,}")
    print(f"    extensions : {_summary.extension_counts}")
    print(f"    subdirs    : {len(_summary.subdirectories)} (first 8) {_summary.subdirectories[:8]}")
    print(f"    examples   : {_summary.example_files}")
    print()

write_json(OUTPUTS / "reports" / "phase1_directory_structures.json", STRUCTURES)
print("saved -> outputs/reports/phase1_directory_structures.json")


# %%
# ===========================================================================
# CELL 6 -- Load and summarise the label metadata
# ---------------------------------------------------------------------------
# Reads every label CSV, reports its true column names (including IDRiD's
# trailing-space column), row counts, duplicate ids and label distributions.
# Cross-checks each measurement against the counts recorded in paths.yaml and
# flags any mismatch loudly.
# ===========================================================================
from src.data.inspect import format_domain_report, inspect_domain  # noqa: E402

DOMAIN_REPORTS = {}
for _key, _spec in PATHS["domains"].items():
    _report = inspect_domain(_key, _spec, n_image_sample=200, seed=SEED)
    DOMAIN_REPORTS[_key] = _report
    print(format_domain_report(_report))
    print()

write_json(OUTPUTS / "reports" / "phase1_domain_inspection.json", DOMAIN_REPORTS)
print("saved -> outputs/reports/phase1_domain_inspection.json")

_mismatched = {k: v["mismatches"] for k, v in DOMAIN_REPORTS.items() if v["mismatches"]}
if _mismatched:
    print("\n!! Some measurements disagree with configs/paths.yaml:")
    for _k, _v in _mismatched.items():
        print(f"   {_k}: {_v}")
    print("   Investigate before continuing -- do NOT edit paths.yaml to match.")
else:
    print("\nAll domains match the counts recorded in configs/paths.yaml.")


# %%
# ===========================================================================
# CELL 7 -- Verify / map label semantics
# ---------------------------------------------------------------------------
# Confirms every domain encodes the 5-grade ICDR scale identically. Writes the
# explicit mapping so that no class is ever silently reinterpreted.
#
# Findings from the Phase-1 audit that this cell records:
#   * DDR, APTOS and IDRiD all use integer grades 0-4 in their CSVs.
#   * DDR's ungradable class 5 is already absent from this copy (12,522 rows,
#     matching the official gradable count) -- confirmed, not assumed.
#   * EyePACS labels come from PARENT FOLDER NAMES in a third-party derivative,
#     not from an official CSV. That provenance gap is recorded here.
# ===========================================================================
from src.data.provenance import audit_label_alphabets  # noqa: E402

LABEL_FINDINGS = audit_label_alphabets(PATHS)
for _finding in LABEL_FINDINGS:
    print(f"[{_finding.severity}] {_finding.summary}")
    for _domain, _counts in _finding.evidence["per_domain_label_counts"].items():
        print(f"    {_domain:8s}: {_counts}")

LABEL_MAPPING = {
    "scale": "ICDR 5-grade diabetic retinopathy severity",
    "grades": {str(k): v for k, v in GRADE_NAMES.items()},
    "per_domain": {
        "ddr": {
            "source": "DR_grading.csv column 'diagnosis'",
            "native_values": "0,1,2,3,4",
            "mapping": "identity",
            "notes": (
                "Official DDR also defines grade 5 = ungradable. This copy contains "
                "12,522 rows, exactly the official gradable count, so class 5 was "
                "removed upstream. Verified by counting, not assumed."
            ),
        },
        "aptos": {
            "source": "train_1.csv / valid.csv / test.csv column 'diagnosis'",
            "native_values": "0,1,2,3,4",
            "mapping": "identity",
            "notes": "APTOS 2019 uses the ICDR scale directly.",
        },
        "idrid": {
            "source": "IDRiD_Disease Grading_*.csv column 'Retinopathy grade'",
            "native_values": "0,1,2,3,4",
            "mapping": "identity",
            "notes": (
                "The CSV also carries 'Risk of macular edema ' (note trailing space). "
                "That column is NOT the DR grade and is not used as the target."
            ),
        },
        "eyepacs": {
            "source": (
                "parent folder name in the third-party dr_unified_v2 derivative, "
                "VERIFIED against an independent copy of the Kaggle trainLabels.csv "
                "(see Cell 7b and src/data/eyepacs_labels.py)"
            ),
            "native_values": "0,1,2,3,4",
            "mapping": "identity",
            "notes": (
                "The domain is restricted to the 35,108 images whose folder label "
                "agrees with the reference labels (agreement: 100.0000%). The other "
                "50,070 images are EXCLUDED: their folder labels are corrupted, "
                "containing one single grade-3 image where ~1,247 are expected. "
                "Verified class distribution 73.49/6.94/15.06/2.48/2.02 percent "
                "matches published official EyePACS statistics to within 0.02 points. "
                "See docs/DATA_PROVENANCE.md."
            ),
        },
    },
}
write_json(OUTPUTS / "reports" / "label_mapping.json", LABEL_MAPPING)
print("\nsaved -> outputs/reports/label_mapping.json")


# %%
# ===========================================================================
# CELL 7b -- Verify EyePACS labels against an independent reference
# ---------------------------------------------------------------------------
# EyePACS is the one domain whose grades come from folder names in a
# third-party derivative rather than from an official release. This cell
# checks them instead of accepting the limitation.
#
# Downloads a ~500 KB reference trainLabels.csv on first run, corroborates it
# against published EyePACS statistics, then compares it image by image.
#
# Outcome recorded in configs/paths.yaml:
#   * 35,108 images agree 100.0000%  -> these form the EyePACS domain
#   * 50,070 images have corrupted labels (ONE grade-3 image where ~1,247 are
#     expected) -> excluded
#
# Writes outputs/reports/eyepacs_verified_index.csv, which is the ONLY source
# the Phase-2 EyePACS loader is allowed to read.
# ===========================================================================
from src.data.eyepacs_labels import (  # noqa: E402
    download_reference_labels,
    format_verification,
    verify_eyepacs_labels,
    write_verified_index,
)

_eyepacs = PATHS["domains"]["eyepacs"]
REFERENCE_LABELS = download_reference_labels(ROOT / "data_external")
EYEPACS_VERIFICATION = verify_eyepacs_labels(_eyepacs["image_root"], REFERENCE_LABELS)
print(format_verification(EYEPACS_VERIFICATION))

if EYEPACS_VERIFICATION.reference_problems:
    raise RuntimeError(
        "The reference EyePACS label file failed its corroboration checks; "
        "refusing to build a verified index from it.\n  - "
        + "\n  - ".join(EYEPACS_VERIFICATION.reference_problems)
    )

EYEPACS_INDEX = write_verified_index(
    EYEPACS_VERIFICATION, REFERENCE_LABELS, ROOT / _eyepacs["verified_index_csv"]
)
write_json(
    OUTPUTS / "reports" / "eyepacs_label_verification.json",
    {k: v for k, v in EYEPACS_VERIFICATION.__dict__.items() if not k.startswith("_")},
)
print(f"\nsaved -> {_eyepacs['verified_index_csv']}")
print("saved -> outputs/reports/eyepacs_label_verification.json")


# %%
# ===========================================================================
# CELL 8 -- Raw-data provenance and leakage audit
# ---------------------------------------------------------------------------
# Runs every raw-folder audit and re-measures each problem on THIS machine.
# Takes ~2-4 min (it walks all ~90k files under Dataset 1).
#
# This is the raw-folder audit. The split-level leakage audit of the research
# brief (hashes, patient overlap between constructed train/val/test) runs in
# Phase 2 once the unified index exists.
# ===========================================================================
from src.data.provenance import format_findings, run_all_audits  # noqa: E402

AUDIT_FINDINGS = run_all_audits(PATHS)
print()
print(format_findings(AUDIT_FINDINGS))

write_json(
    OUTPUTS / "reports" / "phase1_provenance_audit.json",
    [f.__dict__ for f in AUDIT_FINDINGS],
)
print("saved -> outputs/reports/phase1_provenance_audit.json")

CRITICAL = [f for f in AUDIT_FINDINGS if f.severity == "CRITICAL"]
if CRITICAL:
    print(
        f"\n{len(CRITICAL)} CRITICAL finding(s). These are handled by explicit "
        "exclusion rules in configs/paths.yaml, which Phase 2 enforces with "
        "assertions. Read docs/DATA_PROVENANCE.md before building loaders."
    )


# ===========================================================================
# ===========================================================================
#
#   E N D   O F   P H A S E   2
#
#   Cells 1-13 above are implemented and run on real data.
#   Everything below is a placeholder: each cell prints what it will do and
#   what it is waiting on. No stub fabricates numbers, writes result files or
#   touches the GPU.
#
# ===========================================================================
# ===========================================================================

def _pending(cell: str, phase: str, description: str, blocked_by: str = "") -> None:
    """Announce an unimplemented cell. Never produces or writes results."""
    print(f"[NOT RUN] {cell}  (implemented in {phase})")
    print(f"    {description}")
    if blocked_by:
        print(f"    blocked by: {blocked_by}")


# %%
# ===========================================================================
# CELL 9 -- Build the unified manifest, remove exact duplicates, split
# ---------------------------------------------------------------------------
# One table becomes the source of truth for every later cell: splits, leakage
# audit, figures, dataloaders and error analysis all read from it.
#
# Three steps, in this order for a reason:
#   1. build   -- per-domain loaders, each enforcing its own exclusion rules
#   2. dedupe  -- remove byte-identical images BEFORE splitting, otherwise a
#                 photograph can land in train and a copy of it in test
#   3. split   -- patient-level where patient ids exist, stratified otherwise
#
# Takes ~2-4 min on the first run (it hashes ~10k candidate files); afterwards
# the content hashes are cached and it is fast.
# ===========================================================================
from src.data.deduplicate import (  # noqa: E402
    deduplicate_manifest,
    format_deduplication_report,
)
from src.data.splits import SplitConfig, assign_within_domain_splits  # noqa: E402
from src.data.unified_dataset import (  # noqa: E402
    build_unified_manifest,
    domain_summary,
    save_manifest,
)

RAW_MANIFEST = build_unified_manifest(PATHS, root=ROOT)
save_manifest(RAW_MANIFEST, OUTPUTS / "reports" / "unified_manifest.csv")

MANIFEST, DEDUP_REPORT = deduplicate_manifest(
    RAW_MANIFEST, cache_path=OUTPUTS / "reports" / "content_hashes.csv"
)
print()
print(format_deduplication_report(DEDUP_REPORT))

SPLIT_CONFIG = SplitConfig(seed=SEED, val_fraction=0.15, test_fraction=0.15)
MANIFEST = assign_within_domain_splits(MANIFEST, SPLIT_CONFIG)
save_manifest(MANIFEST, OUTPUTS / "reports" / "unified_manifest_split.csv")

DOMAIN_TABLE = domain_summary(MANIFEST)
print()
print(DOMAIN_TABLE.to_string(index=False))
DOMAIN_TABLE.to_csv(OUTPUTS / "tables" / "table1_dataset_characteristics.csv", index=False)

import pandas as pd  # noqa: E402

print()
print("split sizes per domain:")
print(pd.crosstab(MANIFEST["domain"], MANIFEST["split"]).to_string())
print("\nsaved -> outputs/reports/unified_manifest_split.csv")
print("saved -> outputs/tables/table1_dataset_characteristics.csv")


# %%
# ===========================================================================
# CELL 10 -- Leakage audit on the CONSTRUCTED splits
# ---------------------------------------------------------------------------
# This is the audit that decides whether any result is publishable. It checks
# image-id overlap, patient overlap and byte-identical content across every
# pair of splits.
#
# It must PASS before any training cell is run. If it fails, fix the splits --
# never proceed and caveat it later.
# ===========================================================================
from src.data.leakage import audit_split_leakage, format_leakage_report  # noqa: E402

LEAKAGE = audit_split_leakage(MANIFEST, check_content_hashes=True)
print(format_leakage_report(LEAKAGE))

write_json(
    OUTPUTS / "reports" / "phase2_leakage_audit.json",
    {k: v for k, v in LEAKAGE.__dict__.items()},
)
print("saved -> outputs/reports/phase2_leakage_audit.json")

if not LEAKAGE.passed:
    raise RuntimeError(
        "Leakage detected in the constructed splits. Training on these would "
        "produce optimistically biased results.\n  - "
        + "\n  - ".join(LEAKAGE.problems)
    )


# %%
# ===========================================================================
# CELL 11 -- Per-image statistics (geometry + photometry)
# ---------------------------------------------------------------------------
# Quantifies how different the domains look before any model is involved.
# Runs on a seeded, grade-stratified sample per domain (400 by default) --
# raise N_STATS_SAMPLE to use more, or set it to None for the full corpus
# (slow: it decodes every 4288x2848 IDRiD scan).
# ===========================================================================
from src.data.image_stats import (  # noqa: E402
    compute_image_statistics,
    domain_shift_summary,
    sample_manifest,
)
from src.data.preprocessing import PreprocessConfig  # noqa: E402

IMAGE_SIZE = 224          # development resolution; 384 for the final runs
N_STATS_SAMPLE = 400      # per domain

PREPROCESS = PreprocessConfig(image_size=IMAGE_SIZE)
STATS_SAMPLE = sample_manifest(MANIFEST, n_per_domain=N_STATS_SAMPLE, seed=SEED)
IMAGE_STATS = compute_image_statistics(
    STATS_SAMPLE,
    config=PREPROCESS,
    cache_path=OUTPUTS / "reports" / "image_statistics.csv",
)

SHIFT_TABLE = domain_shift_summary(IMAGE_STATS)
print()
print(SHIFT_TABLE.to_string(index=False))
SHIFT_TABLE.to_csv(OUTPUTS / "tables" / "table1b_domain_appearance.csv", index=False)

_unreadable = int((~IMAGE_STATS["readable"]).sum())
print(f"\nunreadable images in the sample: {_unreadable}")
if _unreadable:
    print(IMAGE_STATS[~IMAGE_STATS["readable"]][["image_id", "error"]].head(10).to_string())
print("saved -> outputs/tables/table1b_domain_appearance.csv")


# %%
# ===========================================================================
# CELL 12 -- Dataset and preprocessing figures
# ---------------------------------------------------------------------------
# Writes 18 publication-resolution (300 DPI) figures to outputs/figures/.
# Cheap; safe to re-run after changing the style or the sample.
# ===========================================================================
from src.visualization.dataset_figures import generate_all  # noqa: E402

FIGURE_PATHS = generate_all(
    MANIFEST,
    IMAGE_STATS,
    OUTPUTS / "figures",
    deduplication=DEDUP_REPORT,
    image_size=IMAGE_SIZE,
    seed=SEED,
    formats=("png",),          # add "pdf" for camera-ready versions
)
print(f"\n{len(FIGURE_PATHS)} figures written to outputs/figures/")
for _path in FIGURE_PATHS:
    print(f"    {_path.name}")


# %%
# ===========================================================================
# CELL 13 -- Dataloaders and a batch sanity check
# ---------------------------------------------------------------------------
# REQUIRES TORCH. Everything above runs without it; this cell is the first that
# does not.
#
# Builds one experiment's loaders and inspects a single batch: shapes, dtypes,
# value ranges, label ranges, domain ids, and throughput. Nothing is trained.
# ===========================================================================
from src.data.splits import build_experiment_split, leave_one_domain_out_plan  # noqa: E402

# Stage C of the development plan: DDR + APTOS -> IDRiD, small enough to iterate on.
EXPERIMENT = build_experiment_split(
    MANIFEST, protocol="lodo", sources=["ddr", "aptos"], target="idrid"
)
print(f"experiment : {EXPERIMENT.name()}")
print(f"sizes      : {EXPERIMENT.sizes()}")
for _note in EXPERIMENT.notes:
    print(f"note       : {_note}")

print("\nfull leave-one-domain-out plan (Stage D):")
for _plan in leave_one_domain_out_plan(["ddr", "aptos", "idrid", "eyepacs"]):
    _split = build_experiment_split(MANIFEST, **_plan)
    print(f"    {_split.name():48s} {_split.sizes()}")

try:
    import torch

    from src.data.augmentations import (
        AugmentationConfig,
        build_eval_transform,
        build_train_transform,
    )
    from src.data.loaders import LoaderConfig, benchmark_loader, build_loaders
except ImportError as _exc:
    print(f"\n[SKIPPED] torch is not installed ({_exc}).")
    print("    Install it before running this cell -- see README section 4.1:")
    print("    pip install torch==2.9.1 torchvision==0.24.1 "
          "--index-url https://download.pytorch.org/whl/cu128")
else:
    import time as _time

    _augment = AugmentationConfig(image_size=IMAGE_SIZE)

    # This cell reads the RAW images, so it passes preprocess=PREPROCESS. It is
    # deliberately slow -- that is the point: it establishes the baseline that
    # Cell 13b's cache improves on. From Cell 15 onwards everything reads the
    # cache and passes preprocess=None.
    _raw_loaders = build_loaders(
        EXPERIMENT,
        loader_config=LoaderConfig(batch_size=16, num_workers=0, seed=SEED),
        train_transform=build_train_transform(_augment),
        eval_transform=build_eval_transform(_augment),
        preprocess=PREPROCESS,
        expected_size=IMAGE_SIZE,
    )

    _start = _time.perf_counter()
    _images, _grades, _domains, _indices = next(iter(_raw_loaders["train"]))
    _elapsed = _time.perf_counter() - _start

    print("\n-- batch sanity check (raw images, before caching) --")
    print(f"    images  : {tuple(_images.shape)} {_images.dtype}")
    print(f"    range   : [{_images.min():.3f}, {_images.max():.3f}] "
          f"(normalised, so negative values are expected)")
    print(f"    grades  : {tuple(_grades.shape)} {_grades.dtype} "
          f"values {sorted(set(_grades.tolist()))}")
    print(f"    domains : {sorted(set(_domains.tolist()))}")
    print(f"    first batch in {_elapsed:.2f}s")

    assert _images.ndim == 4 and _images.shape[1] == 3, "expected NCHW with 3 channels"
    assert _images.shape[2] == _images.shape[3] == IMAGE_SIZE, "unexpected spatial size"
    assert int(_grades.min()) >= 0 and int(_grades.max()) <= 4, "grades outside 0..4"
    assert not torch.isnan(_images).any(), "NaN pixels in the batch"
    print("    all assertions passed")

    _raw_throughput = benchmark_loader(_raw_loaders["train"], n_batches=8, warmup=1)
    print(f"\n    RAW throughput: {_raw_throughput['images_per_second']} images/s")
    print("    Cell 13b builds a pre-resized cache; expect a ~100x speedup, after")
    print("    which training becomes GPU-bound rather than I/O-bound.")
    del _raw_loaders


# %%
# ===========================================================================
# CELL 13b -- Pre-resized image cache  (run once per resolution)
# ---------------------------------------------------------------------------
# Decoding the raw corpus is the bottleneck, not the GPU: 203 ms per APTOS image
# and 376 ms per IDRiD scan versus ~10 ms for DDR. Preprocessing is fully
# deterministic, so it is done once and reused by every epoch of every run.
#
# Measured effect: the loader goes from ~5 images/s (raw APTOS) to 1343 images/s,
# which moves the bottleneck to the GPU (217 images/s for DenseNet121).
#
# ~30 min and ~890 MB at 224px. Resumable -- rerun it if interrupted.
# The cache directory name encodes the preprocessing settings, so changing them
# writes to a new directory instead of silently reusing stale images.
# ===========================================================================
from src.data.cache import build_image_cache, measure_cache_fidelity  # noqa: E402

CACHE_ROOT = Path("D:/DB/_cache")

# What JPEG-95 caching costs, measured rather than asserted.
FIDELITY = measure_cache_fidelity(MANIFEST, PREPROCESS, n_samples=40, quality=95, seed=SEED)
print("cache fidelity vs lossless PNG:")
for _key, _value in FIDELITY.items():
    print(f"    {_key:28s}: {_value}")

CACHED_MANIFEST, CACHE_REPORT = build_image_cache(
    MANIFEST, PREPROCESS, CACHE_ROOT, workers=8, show_progress=True
)
save_manifest(CACHED_MANIFEST, OUTPUTS / "reports" / f"manifest_cached_{IMAGE_SIZE}.csv")
print(
    f"\ncache: {CACHE_REPORT.n_written} written, {CACHE_REPORT.n_reused} reused, "
    f"{CACHE_REPORT.n_failed} failed, {CACHE_REPORT.megabytes} MB, {CACHE_REPORT.seconds}s"
)
print(f"cache dir: {CACHE_REPORT.cache_dir}")


# %%
# ===========================================================================
# CELL 14 -- Instantiate a baseline model and probe feasible batch sizes
# ---------------------------------------------------------------------------
# The probe runs a real forward+backward step at each candidate size and reports
# measured peak VRAM. Guessing on an 8 GB card wastes time and produces
# confusing mid-epoch OOMs.
#
# Whatever is chosen is RECORDED in the experiment registry -- batch size is part
# of the experiment, never silently adjusted.
# ===========================================================================
import torch  # noqa: E402

from src.models.backbones import (  # noqa: E402
    BackboneConfig,
    build_model,
    count_parameters,
    estimate_memory,
    resolve_input_size,
)
from src.utils.hardware import assert_cuda_ready  # noqa: E402

assert_cuda_ready()          # refuses to fall back to CPU
DEVICE = "cuda"

BACKBONE = "densenet121"     # Stage A/C baseline; convnext_tiny and dinov2 next
BACKBONE_CONFIG = BackboneConfig(
    name=BACKBONE,
    pretrained=True,
    num_classes=5,
    head="linear",
    dropout=0.2,
    # DINOv2 is patch-14, so 384 is invalid for it; resolve_input_size snaps and
    # logs the change instead of failing deep inside the model.
    image_size=resolve_input_size(BACKBONE, IMAGE_SIZE),
)

_probe_model = build_model(BACKBONE_CONFIG)
_total, _trainable = count_parameters(_probe_model)
print(f"{BACKBONE}: {_total/1e6:.1f}M parameters ({_trainable/1e6:.1f}M trainable), "
      f"feature dim {_probe_model.feature_dim}, input {BACKBONE_CONFIG.image_size}px")
del _probe_model

print("\n-- batch-size probe (measured peak VRAM, real train step with AMP) --")
BATCH_PROBE = []
for _candidate in (32, 16, 8, 4):
    _result = estimate_memory(BACKBONE_CONFIG, _candidate, amp=True, train=True)
    BATCH_PROBE.append(_result)
    if _result.get("fits"):
        print(f"    batch {_candidate:3d}: OK   peak {_result['peak_gb']:.2f} GB "
              f"(reserved {_result['reserved_gb']:.2f} GB)")
    else:
        print(f"    batch {_candidate:3d}: FAIL {_result.get('error', '')[:70]}")

write_json(OUTPUTS / "reports" / f"batch_size_probe_{BACKBONE}_{IMAGE_SIZE}.json", BATCH_PROBE)
print("\nsaved -> outputs/reports/batch_size_probe_*.json")


# %%
# ===========================================================================
# CELL 15 -- Sanity experiment: can the loop overfit a handful of batches?
# ---------------------------------------------------------------------------
# The standard check before spending GPU-hours. A correct training loop drives
# the loss on ~100 images close to zero within a few dozen steps. If it cannot,
# something is wrong -- labels detached from images, a frozen backbone, a broken
# transform -- and no number of epochs on the full set will fix it.
#
# Verifies, per the brief: loss decreases, the GPU is used, metrics compute,
# no NaNs, and the checkpoint round-trips. Takes about a minute.
# ===========================================================================
import time as _time  # noqa: E402

from src.data.augmentations import (  # noqa: E402
    AugmentationConfig,
    build_eval_transform,
    build_train_transform,
)
from src.data.loaders import LoaderConfig, benchmark_loader, build_loaders  # noqa: E402
from src.data.splits import build_experiment_split  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.losses.classification import build_classification_loss  # noqa: E402
from src.training.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from src.training.trainer import TrainConfig, Trainer, predict  # noqa: E402

# Stage C of the development plan: DDR + APTOS -> IDRiD. Small enough to iterate
# on, and it is a genuine leave-one-domain-out experiment.
EXPERIMENT = build_experiment_split(
    CACHED_MANIFEST, protocol="lodo", sources=["ddr", "aptos"], target="idrid"
)
print(f"experiment: {EXPERIMENT.name()}  {EXPERIMENT.sizes()}")
for _note in EXPERIMENT.notes:
    print(f"    note: {_note}")

AUGMENT = AugmentationConfig(image_size=IMAGE_SIZE)
# num_workers=2: measured throughput is already ~6x the GPU's, and 2 halves the
# RAM cost. See src/data/loaders.py for the numbers.
LOADER_CONFIG = LoaderConfig(batch_size=16, num_workers=2, seed=SEED)

# preprocess=None: the cached manifest points at already-preprocessed images.
# expected_size makes a cached/raw mix-up fail loudly on the first batch.
LOADERS = build_loaders(
    EXPERIMENT,
    loader_config=LOADER_CONFIG,
    train_transform=build_train_transform(AUGMENT),
    eval_transform=build_eval_transform(AUGMENT),
    preprocess=None,
    expected_size=IMAGE_SIZE,
)

print("\n-- loader throughput --")
THROUGHPUT = benchmark_loader(LOADERS["train"], n_batches=20)
print(f"    {THROUGHPUT['images_per_second']} images/s "
      f"({LOADER_CONFIG.num_workers} workers, batch {LOADER_CONFIG.batch_size})")

print("\n-- overfit check: 6 fixed batches, 40 steps --")
_sanity_model = build_model(BACKBONE_CONFIG).to(DEVICE)
_loss_fn, _ = build_classification_loss("none")
_optimizer = torch.optim.AdamW(_sanity_model.parameters(), lr=3e-4, weight_decay=1e-4)
_scaler = torch.amp.GradScaler("cuda", enabled=True)

_fixed = []
for _batch in LOADERS["train"]:
    _fixed.append((_batch[0].to(DEVICE), _batch[1].to(DEVICE)))
    if len(_fixed) == 6:
        break

_sanity_model.train()
_losses = []
_start = _time.perf_counter()
for _step in range(40):
    _images, _targets = _fixed[_step % len(_fixed)]
    with torch.autocast("cuda", dtype=torch.float16):
        _logits = _sanity_model(_images)
    _loss = _loss_fn(_logits.float(), _targets)
    _scaler.scale(_loss).backward()
    _scaler.step(_optimizer)
    _scaler.update()
    _optimizer.zero_grad(set_to_none=True)
    _losses.append(float(_loss.item()))
    if _step % 10 == 0 or _step == 39:
        print(f"    step {_step:3d}: loss {_losses[-1]:.4f}")

_elapsed = _time.perf_counter() - _start
print(f"\n    {_losses[0]:.4f} -> {_losses[-1]:.4f} in {_elapsed:.0f}s "
      f"({40 * LOADER_CONFIG.batch_size / _elapsed:.0f} img/s), "
      f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

assert all(_l == _l for _l in _losses), "NaN loss during the sanity run"
assert _losses[-1] < _losses[0] * 0.5, (
    f"loss did not halve on a fixed batch set ({_losses[0]:.4f} -> {_losses[-1]:.4f}); "
    "the training loop is not learning -- investigate before running anything longer"
)
print("    PASS: loss decreased, no NaNs, GPU used")

print("\n-- checkpoint round-trip --")
_ckpt = OUTPUTS / "checkpoints" / "_sanity" / "roundtrip.pt"
save_checkpoint(_ckpt, model=_sanity_model, optimizer=_optimizer, scaler=_scaler,
                epoch=0, metrics={"qwk": 0.0}, config=BACKBONE_CONFIG.describe())
_reloaded = build_model(BACKBONE_CONFIG).to(DEVICE)
load_checkpoint(_ckpt, model=_reloaded, map_location=DEVICE)
_sanity_model.eval()
_reloaded.eval()
with torch.inference_mode():
    _difference = float((_sanity_model(_fixed[0][0]).float()
                         - _reloaded(_fixed[0][0]).float()).abs().max())
print(f"    max |logit difference| after reload: {_difference:.2e}")
assert _difference < 1e-4, "checkpoint did not restore the model exactly"
print("    PASS: checkpoint saves and restores exactly")

print("\n-- metrics on real predictions --")
_outputs = predict(_sanity_model, LOADERS["val"], device=DEVICE, loss_fn=_loss_fn)
_sanity_metrics = compute_all_metrics(
    _outputs["y_true"], _outputs["y_pred"], _outputs["probabilities"],
    include_per_class=False, include_referable=False,
)
print(f"    val QWK {_sanity_metrics['qwk']:.4f}, macro F1 "
      f"{_sanity_metrics['f1_macro']:.4f}, loss {_outputs['loss']:.4f}")
print("    PASS: metrics compute on real predictions")

del _sanity_model, _reloaded, _fixed, _optimizer
torch.cuda.empty_cache()
print("\nALL SANITY CHECKS PASSED -- safe to run real training.")


# %%
# ===========================================================================
# CELL 16 -- Baseline ERM training  (EXPENSIVE -- run deliberately)
# ---------------------------------------------------------------------------
# Trains the Stage-C leave-one-domain-out baseline: DDR + APTOS -> IDRiD.
# About 60 s/epoch at 224px on the RTX 5060.
#
# Early stopping, checkpoint selection and the scheduler all read SOURCE-domain
# validation metrics. The Trainer is never given the test loader, so the target
# domain cannot influence training even by accident.
#
# Resumable: rerun the cell after a crash and it continues from last.pt.
# ===========================================================================
from src.data.preprocessing import PreprocessConfig  # noqa: E402
from src.utils.registry import make_experiment_id  # noqa: E402

TRAIN_CONFIG = TrainConfig(
    epochs=20,
    batch_size=16,
    accumulation_steps=1,
    learning_rate=3e-4,
    weight_decay=1e-4,
    warmup_epochs=1,
    scheduler="cosine",
    amp=True,
    grad_clip_norm=1.0,
    early_stopping_patience=6,
    monitor="qwk",
    seed=SEED,
)

IMBALANCE_STRATEGY = "none"       # compare against 'weighted_ce' / 'focal' later
TRAIN_CLASS_COUNTS = (
    EXPERIMENT.train["grade"].value_counts().reindex(range(5), fill_value=0).tolist()
)
print(f"train class counts: {TRAIN_CLASS_COUNTS}")

LOSS_FN, LOSS_DESCRIPTION = build_classification_loss(
    IMBALANCE_STRATEGY, class_counts=TRAIN_CLASS_COUNTS, device=DEVICE
)

EXPERIMENT_ID = make_experiment_id(
    protocol=EXPERIMENT.protocol,
    sources=EXPERIMENT.source_domains,
    target=EXPERIMENT.target_domain,
    backbone=BACKBONE,
    method=f"erm-{IMBALANCE_STRATEGY}",
    seed=SEED,
)
print(f"experiment id: {EXPERIMENT_ID}")

MODEL = build_model(BACKBONE_CONFIG)
TRAINER = Trainer(
    MODEL, LOSS_FN, TRAIN_CONFIG,
    device=DEVICE,
    checkpoint_dir=OUTPUTS / "checkpoints",
    experiment_id=EXPERIMENT_ID,
    extra_config={
        **BACKBONE_CONFIG.describe(),
        **LOSS_DESCRIPTION,
        "preprocess": PreprocessConfig(image_size=IMAGE_SIZE).describe(),
        "augmentation": AUGMENT.describe(),
        "loaders": LOADER_CONFIG.describe(),
        "sources": EXPERIMENT.source_domains,
        "target": EXPERIMENT.target_domain,
    },
)
TRAINER.resume()          # no-op when there is nothing to resume

_train_started = _time.perf_counter()
HISTORY = TRAINER.fit(LOADERS["train"], LOADERS["val"])
TRAIN_SECONDS = round(_time.perf_counter() - _train_started, 1)

HISTORY_FRAME = TRAINER.history_frame()
HISTORY_PATH = OUTPUTS / "logs" / f"{EXPERIMENT_ID}_history.csv"
HISTORY_FRAME.to_csv(HISTORY_PATH, index=False)
print()
print(HISTORY_FRAME.to_string(index=False))
print(f"\ntrained in {TRAIN_SECONDS}s -> {HISTORY_PATH}")
print(f"checkpoints    : {TRAINER.checkpoints.summary()}")
print(f"early stopping : {TRAINER.early_stopping.state()}")

from src.visualization.training_figures import generate_training_figures  # noqa: E402

TRAINING_FIGURES = generate_training_figures(
    HISTORY_FRAME, OUTPUTS / "figures", prefix=f"train_{EXPERIMENT_ID}"
)
print(f"{len(TRAINING_FIGURES)} training figures -> outputs/figures/")


# %%
# ===========================================================================
# CELL 17 -- Evaluation: source validation, then the unseen target domain
# ---------------------------------------------------------------------------
# Loads the BEST-VALIDATION-QWK checkpoint (never a checkpoint chosen by target
# performance) and evaluates it.
#
# The order is the protocol:
#   1. score the source validation split
#   2. fit temperature on those validation logits ONLY
#   3. apply that fixed temperature to the unseen target domain
#
# Sample-level predictions are written for every split -- they are the input to
# the bootstrap CIs, selective prediction and the error-analysis galleries.
# ===========================================================================
from src.evaluation.evaluate import evaluate_experiment  # noqa: E402

BEST_CHECKPOINT = TRAINER.checkpoints.best_path
print(f"loading best-validation-QWK checkpoint: {BEST_CHECKPOINT.name}")
EVAL_MODEL = build_model(BACKBONE_CONFIG).to(DEVICE)
load_checkpoint(BEST_CHECKPOINT, model=EVAL_MODEL, map_location=DEVICE)

EVALUATION = evaluate_experiment(
    EVAL_MODEL, LOADERS, EXPERIMENT,
    experiment_id=EXPERIMENT_ID,
    device=DEVICE,
    predictions_dir=OUTPUTS / "predictions",
)

import pandas as pd  # noqa: E402

HEADLINE = pd.DataFrame([r.headline() for r in EVALUATION["results"].values()])
print()
print(HEADLINE.to_string(index=False))
print(f"\ntemperature: {EVALUATION['temperature']}")

HEADLINE.to_csv(OUTPUTS / "tables" / f"{EXPERIMENT_ID}_headline.csv", index=False)
write_json(
    OUTPUTS / "reports" / f"{EXPERIMENT_ID}_evaluation.json",
    {
        "experiment_id": EXPERIMENT_ID,
        "protocol": EVALUATION["protocol"],
        "source_domains": EVALUATION["source_domains"],
        "target_domain": EVALUATION["target_domain"],
        "temperature": EVALUATION["temperature"],
        "results": {k: {"metrics": v.metrics, "calibration": v.calibration,
                        "calibration_scaled": v.calibration_scaled}
                    for k, v in EVALUATION["results"].items()},
    },
)
print("saved -> outputs/tables/*_headline.csv, outputs/reports/*_evaluation.json")
print("saved -> outputs/predictions/*_predictions.csv")


# %%
# ===========================================================================
# CELL 18 -- The domain-generalization gap, and performance figures
# ---------------------------------------------------------------------------
# Quantifies the drop between source validation and the unseen target domain --
# the paper's first claim -- and writes the confusion / ROC / reliability /
# confidence figures for both splits.
# ===========================================================================
from src.evaluation.evaluate import predict_with_logits  # noqa: E402
from src.visualization.performance_figures import (  # noqa: E402
    figure_indomain_vs_external,
    generate_performance_figures,
)
from src.visualization.style import apply_style, save_figure  # noqa: E402

_source = EVALUATION["results"]["source_val"]
_target = EVALUATION["results"]["target_test"]

GAP_ROWS = [{
    "label": f"{'+'.join(EVALUATION['source_domains'])} -> {EVALUATION['target_domain']}",
    **{f"source_{k}": _source.metrics.get(k) for k in ("qwk", "f1_macro", "accuracy")},
    **{f"target_{k}": _target.metrics.get(k) for k in ("qwk", "f1_macro", "accuracy")},
    "source_ece": _source.calibration.get("ece"),
    "target_ece": _target.calibration.get("ece"),
}]
GAP_FRAME = pd.DataFrame(GAP_ROWS)
print(GAP_FRAME.to_string(index=False))
print()
for _metric in ("qwk", "f1_macro", "accuracy"):
    _a, _b = GAP_ROWS[0][f"source_{_metric}"], GAP_ROWS[0][f"target_{_metric}"]
    print(f"    {_metric:10s}: source {_a:.4f} -> target {_b:.4f}   gap {_b - _a:+.4f}")
_a, _b = GAP_ROWS[0]["source_ece"], GAP_ROWS[0]["target_ece"]
print(f"    {'ece':10s}: source {_a:.4f} -> target {_b:.4f}   gap {_b - _a:+.4f} "
      f"({'worse' if _b > _a else 'better'} calibration off-domain)")

GAP_FRAME.to_csv(OUTPUTS / "tables" / f"{EXPERIMENT_ID}_generalization_gap.csv", index=False)

# Re-run inference once to keep the raw logits/probabilities for the figures.
OUTPUTS_BY_SPLIT = {
    "source_val": predict_with_logits(EVAL_MODEL, LOADERS["val"], device=DEVICE),
    "target_test": predict_with_logits(EVAL_MODEL, LOADERS["test"], device=DEVICE),
}
if EVALUATION["temperature"] is not None:
    from src.evaluation.calibration import TemperatureScaler  # noqa: E402

    _scaler_obj = TemperatureScaler(temperature=EVALUATION["temperature"]["temperature"])
    for _key, _out in OUTPUTS_BY_SPLIT.items():
        _out["probabilities_scaled"] = _scaler_obj.apply(_out["logits"])

PERFORMANCE_FIGURES = generate_performance_figures(
    EVALUATION, OUTPUTS_BY_SPLIT, OUTPUTS / "figures", prefix=f"perf_{EXPERIMENT_ID}"
)
apply_style()
PERFORMANCE_FIGURES.extend(save_figure(
    figure_indomain_vs_external(GAP_ROWS, metric="qwk"),
    f"gap_{EXPERIMENT_ID}_qwk", OUTPUTS / "figures",
))
print(f"\n{len(PERFORMANCE_FIGURES)} performance figures -> outputs/figures/")


# %%
# ===========================================================================
# CELL 18b -- Register the finished experiment
# ---------------------------------------------------------------------------
# One append-only row per finished run. Results are never overwritten: a re-run
# of the same configuration adds a new row with a new timestamp.
# ===========================================================================
from src.utils.registry import load_registry, register_experiment  # noqa: E402

REGISTRY_PATH = OUTPUTS / "experiment_registry.csv"
_best = TRAINER.checkpoints.summary()

register_experiment(REGISTRY_PATH, {
    "experiment_id": EXPERIMENT_ID,
    "status": "COMPLETE",
    "protocol": EXPERIMENT.protocol,
    "source_domains": EXPERIMENT.source_domains,
    "target_domain": EXPERIMENT.target_domain,
    "backbone": BACKBONE,
    "head": BACKBONE_CONFIG.head,
    "method": f"erm-{IMBALANCE_STRATEGY}",
    "loss": "cross_entropy",
    "imbalance_strategy": IMBALANCE_STRATEGY,
    "image_size": BACKBONE_CONFIG.image_size,
    "batch_size": TRAIN_CONFIG.batch_size,
    "accumulation_steps": TRAIN_CONFIG.accumulation_steps,
    "effective_batch_size": TRAIN_CONFIG.effective_batch_size,
    "learning_rate": TRAIN_CONFIG.learning_rate,
    "weight_decay": TRAIN_CONFIG.weight_decay,
    "epochs_planned": TRAIN_CONFIG.epochs,
    "epochs_run": len(HISTORY),
    "seed": SEED,
    "deterministic": DETERMINISTIC,
    "n_train": len(EXPERIMENT.train),
    "n_val": len(EXPERIMENT.val),
    "n_test": len(EXPERIMENT.test),
    "best_epoch": _best["best_epoch"],
    "best_val_qwk": _best.get("best_val_qwk"),
    "best_val_loss": _best.get("best_val_loss"),
    "best_checkpoint": str(BEST_CHECKPOINT),
    "test_qwk": _target.metrics.get("qwk"),
    "test_f1_macro": _target.metrics.get("f1_macro"),
    "test_accuracy": _target.metrics.get("accuracy"),
    "test_balanced_accuracy": _target.metrics.get("balanced_accuracy"),
    "test_mae_grade": _target.metrics.get("mae_grade"),
    "test_within_1_grade": _target.metrics.get("within_1_grade"),
    "test_severe_error_rate": _target.metrics.get("severe_error_rate"),
    "test_auroc_macro": _target.metrics.get("auroc_macro"),
    "test_ece": _target.calibration.get("ece"),
    "test_nll": _target.calibration.get("nll"),
    "test_brier": _target.calibration.get("brier"),
    "train_seconds": TRAIN_SECONDS,
    "peak_vram_gb": float(HISTORY_FRAME["peak_vram_gb"].max()) if len(HISTORY_FRAME) else None,
    "params_total_m": round(_total / 1e6, 2),
    "params_trainable_m": round(_trainable / 1e6, 2),
    "predictions_path": _target.predictions_path,
    "history_path": str(HISTORY_PATH),
    "config_json": {**TRAIN_CONFIG.describe(), **BACKBONE_CONFIG.describe()},
    "notes": "Stage C baseline; temperature fitted on source validation only",
})
print(load_registry(REGISTRY_PATH).tail(3).to_string(index=False))


# %%
# ===========================================================================
# CELL 19 -- Bootstrap confidence intervals
# ---------------------------------------------------------------------------
# IDRiD's test split is 507 images. A bare QWK on that is a noisy estimate, and
# the difference between two methods can easily be smaller than the sampling
# error -- so every headline number carries a 95% percentile-bootstrap interval.
#
# Seeded, so a reported interval regenerates exactly from the saved predictions.
# ===========================================================================
from src.evaluation.bootstrap import bootstrap_all_metrics  # noqa: E402

BOOTSTRAP_ROWS = []
for _split, _out in OUTPUTS_BY_SPLIT.items():
    _results = bootstrap_all_metrics(
        _out["y_true"], _out["y_pred"], _out["probabilities"],
        metrics=("qwk", "f1_macro", "auroc_macro", "ece"),
        n_bootstrap=2000, seed=SEED,
    )
    print(f"  --- {_split} (n={len(_out['y_true'])}) ---")
    for _name, _result in _results.items():
        print(f"      {_name:12s} {_result.format()}")
        BOOTSTRAP_ROWS.append({"split": _split, **_result.as_dict()})

BOOTSTRAP_FRAME = pd.DataFrame(BOOTSTRAP_ROWS)
BOOTSTRAP_FRAME.to_csv(OUTPUTS / "tables" / f"{EXPERIMENT_ID}_bootstrap_ci.csv", index=False)
print("\nsaved -> outputs/tables/*_bootstrap_ci.csv")

# The gap is only meaningful if the intervals actually separate.
_source_qwk = next(r for r in BOOTSTRAP_ROWS if r["split"] == "source_val" and r["metric"] == "qwk")
_target_qwk = next(r for r in BOOTSTRAP_ROWS if r["split"] == "target_test" and r["metric"] == "qwk")
_overlap = _target_qwk["ci_upper"] >= _source_qwk["ci_lower"]
print(f"\nQWK intervals overlap: {_overlap} "
      f"(source [{_source_qwk['ci_lower']:.4f}, {_source_qwk['ci_upper']:.4f}], "
      f"target [{_target_qwk['ci_lower']:.4f}, {_target_qwk['ci_upper']:.4f}])")
print("NOTE: these are separate intervals, not a paired test. Use "
      "paired_bootstrap_difference() when comparing two METHODS on the same split.")


# %%
# ===========================================================================
# CELL 20 -- Selective prediction on the unseen target domain
# ---------------------------------------------------------------------------
# Does the model's confidence identify the cases it gets wrong? Measures
# performance as low-confidence predictions are deferred.
#
# Interpretation guard: a good risk-coverage curve shows confidence RANKS
# errors. It does NOT show the system is clinically deployable -- the deferred
# cases still need a clinician, and the threshold would have to be fixed
# prospectively rather than chosen on the test set.
# ===========================================================================
from src.evaluation.selective_prediction import evaluate_selective_prediction  # noqa: E402

_target_outputs = OUTPUTS_BY_SPLIT["target_test"]
SELECTIVE = evaluate_selective_prediction(
    _target_outputs["y_true"],
    _target_outputs["y_pred"],
    _target_outputs["probabilities"].max(axis=1),
)

print(f"AURC {SELECTIVE.aurc:.4f} | error-detection AUROC "
      f"{SELECTIVE.error_detection_auroc:.4f}  (0.5 = confidence is uninformative)")
print()
print(f"  {'cov':>5} {'kept':>5} {'defer':>6} {'acc':>7} {'qwk':>8} {'f1':>7} {'severe':>7}")
for _row in SELECTIVE.at_coverage:
    print(f"  {_row['coverage']:5.2f} {_row['n_retained']:5d} {_row['n_deferred']:6d} "
          f"{_row['accuracy']:7.4f} {_row['qwk']:8.4f} {_row['f1_macro']:7.4f} "
          f"{_row['severe_error_rate']:7.4f}")

pd.DataFrame(SELECTIVE.at_coverage).to_csv(
    OUTPUTS / "tables" / f"{EXPERIMENT_ID}_selective_prediction.csv", index=False
)
write_json(
    OUTPUTS / "reports" / f"{EXPERIMENT_ID}_selective_prediction.json", SELECTIVE.as_dict()
)
print("\nsaved -> outputs/tables/*_selective_prediction.csv")


# %%
# ===========================================================================
# CELL 21 -- Method comparison: ordinal, Deep CORAL, MixStyle  (EXPENSIVE)
# ---------------------------------------------------------------------------
# Trains all six methods on the same Stage-C experiment. Everything except the
# method is held identical -- splits, seed, preprocessing, augmentation,
# optimiser, schedule, epochs and batch size -- so a difference is attributable
# to the method rather than the wiring.
#
# ~100 min for six methods at 20 epochs each on the RTX 5060.
#
# WHY BATCH 32, not the 16 used for the Phase-3 baseline
# ------------------------------------------------------
# Deep CORAL estimates a per-domain feature covariance WITHIN each batch. At the
# natural DDR/APTOS mix of 76/24, a batch of 16 leaves fewer than 4 APTOS samples
# 43% of the time -- a degenerate covariance. At batch 32 that falls to 3%
# (measured; the realised rate in training was 3.3%). The batch size is therefore
# fixed at 32 for EVERY method, including a re-run of ERM.
#
# The two CORALs are different methods sharing an acronym:
#   'ordinal'    -> src/losses/ordinal_coral_loss.py   (Cao et al., label order)
#   'deep_coral' -> src/losses/deep_coral_alignment.py (Sun & Saenko, domains)
# They are separate ablation axes; 'deep_coral_ordinal' combines them.
# ===========================================================================
from src.training.methods import METHODS, MethodConfig, build_method  # noqa: E402

print("available methods:", list(METHODS))
print()
print("Run from a terminal rather than this cell -- it is a long job and the")
print("script handles per-method failure without losing the completed runs:")
print()
print("    python run_method_comparison.py                 # seed 42")
print("    python run_method_comparison.py --seeds 1,2     # two more seeds")
print()
print("Then analyse:")
print("    python analyse_method_comparison.py   # paired bootstrap vs ERM")
print("    python analyse_embeddings.py          # did the DG methods change anything?")
print("    python analyse_seeds.py               # mean +- SD once >1 seed exists")

# Building a method here is cheap and verifies the wiring without training.
_demo = build_method(
    MethodConfig(name="deep_coral_ordinal"), BACKBONE_CONFIG,
    class_counts=TRAIN_CLASS_COUNTS, device=DEVICE,
)
print(f"\nexample build -- deep_coral_ordinal: {_demo.description}")
print(f"    feature loss : {type(_demo.feature_loss).__name__}")
print(f"    batch hook   : {_demo.batch_hook is not None}")
print(f"    predict rule : {_demo.description['prediction_rule']}")
del _demo


# %%
# ===========================================================================
# CELL 22 -- Method-comparison results, with paired bootstrap
# ---------------------------------------------------------------------------
# Reads what the runs actually produced. Prints NOT RUN for anything missing
# rather than inventing a value.
# ===========================================================================
_comparison_path = OUTPUTS / "tables" / "stage_c_method_comparison.csv"
if not _comparison_path.exists():
    _pending(
        "Cell 22: method comparison results", "Phase 4",
        "Reads outputs/tables/stage_c_method_comparison.csv.",
        "run_method_comparison.py has not been run yet.",
    )
else:
    COMPARISON = pd.read_csv(_comparison_path)
    print(COMPARISON.round(4).to_string(index=False))

    _paired_path = OUTPUTS / "tables" / "stage_c_paired_comparison.csv"
    if _paired_path.exists():
        PAIRED = pd.read_csv(_paired_path)
        print("\n-- paired bootstrap vs ERM (same 507 target images) --")
        _shown = PAIRED[PAIRED["metric"].isin(["qwk", "f1_macro", "ece"])]
        for _, _row in _shown.iterrows():
            print(
                f"    {_row['method']:20s} {_row['metric']:9s} "
                f"{_row['difference']:+8.4f}  "
                f"[{_row['ci_lower']:+7.4f}, {_row['ci_upper']:+7.4f}]  "
                f"{'significant' if _row['significant'] else 'not significant'}"
            )
        print("\n    (uncorrected for multiple comparisons)")
    else:
        print("\npaired comparison NOT RUN -- run analyse_method_comparison.py")


# %%
# ===========================================================================
# CELL 23 -- Per-class recall: what the aggregate metrics hide
# ---------------------------------------------------------------------------
# The single most important table in the method comparison. Aggregate QWK and
# ECE can both improve while a model quietly stops predicting the grades that
# drive referral, so per-class recall is reported beside them, never after.
# ===========================================================================
_recall_path = OUTPUTS / "tables" / "stage_c_per_class_recall.csv"
if not _recall_path.exists():
    _pending(
        "Cell 23: per-class recall", "Phase 4",
        "Recall per grade for every method on the unseen target.",
        "analyse_method_comparison.py has not been run yet.",
    )
else:
    RECALL = pd.read_csv(_recall_path)
    print(RECALL.round(3).to_string(index=False))
    print()
    print("Grade 3 is severe non-proliferative DR -- the urgent-referral threshold.")
    _worst = RECALL.loc[RECALL["recall_grade_3"].idxmin()]
    _best = RECALL.loc[RECALL["recall_grade_3"].idxmax()]
    print(f"    best  grade-3 recall: {_best['method']} at {_best['recall_grade_3']:.3f}")
    print(f"    worst grade-3 recall: {_worst['method']} at {_worst['recall_grade_3']:.3f}")
    print("    A method with a better ECE but near-zero grade-3 recall is not better.")


# %%
# ===========================================================================
# CELL 24 -- Did the DG methods change the representation?
# ---------------------------------------------------------------------------
# A domain-generalization method that neither improves the target metrics nor
# reduces domain information in its features has not worked. This separates
# those two claims:
#
#   silhouette / centroid distance -- coarse geometric separation, which is what
#                                     Deep CORAL directly optimises
#   linear probe accuracy          -- whether domain identity is still decodable
#
# The two can disagree, and on this corpus they did.
# ===========================================================================
_separability_path = OUTPUTS / "tables" / "stage_c_domain_separability.csv"
if not _separability_path.exists():
    _pending(
        "Cell 24: representation analysis", "Phase 4",
        "Linear-probe domain separability and silhouette scores per method.",
        "analyse_embeddings.py has not been run yet.",
    )
else:
    SEPARABILITY = pd.read_csv(_separability_path)
    print(SEPARABILITY.round(4).to_string(index=False))
    _chance = float(SEPARABILITY["chance"].iloc[0])
    print(f"\n    chance (majority-domain rate) = {_chance:.3f}")
    print("    lift = 0 means no domain information beyond the class prior;")
    print("    lift = 1 means the domain is perfectly linearly decodable.")


# %%
# ===========================================================================
# CELL 25 -- Multi-seed aggregation  (run after >= 3 seeds exist)
# ---------------------------------------------------------------------------
# The objection a reviewer raises first: is the single-seed ordering just seed
# noise? A difference is treated as real only if it exceeds the pooled
# across-seed SD *and* its paired bootstrap interval excludes zero.
#
# With fewer than two seeds this reports CANNOT ASSESS -- not "within noise",
# which would claim a measurement that was never made.
# ===========================================================================
_seed_path = OUTPUTS / "tables" / "stage_c_seed_summary.csv"
if not _seed_path.exists():
    _pending(
        "Cell 25: multi-seed aggregation", "Phase 4",
        "Mean +- SD per method across seeds, and whether each gap exceeds "
        "across-seed variation.",
        "run_method_comparison.py --seeds 1,2 then analyse_seeds.py.",
    )
else:
    SEED_SUMMARY = pd.read_csv(_seed_path)
    print(SEED_SUMMARY.round(4).to_string(index=False))
    _verdict_path = OUTPUTS / "tables" / "stage_c_seed_verdicts.csv"
    if _verdict_path.exists():
        VERDICTS = pd.read_csv(_verdict_path)
        _qwk = VERDICTS[VERDICTS["metric"] == "test_qwk"]
        print("\n-- target QWK vs ERM, against across-seed variation --")
        for _, _row in _qwk.iterrows():
            _assessed = _row["exceeds_seed_noise"]
            _verdict = (
                f"CANNOT ASSESS ({int(_row['n_seeds'])} seed)"
                if pd.isna(_assessed)
                else ("EXCEEDS seed noise" if _assessed else "within seed noise")
            )
            print(
                f"    {_row['method']:20s} delta {_row['delta']:+7.4f}  "
                f"pooled SD {_row['pooled_seed_sd']:.4f}  {_verdict} ({_row['direction']})"
            )


# %%
# ===========================================================================
# CELL 26 -- Selective prediction
# ===========================================================================
_pending(
    "Cell 26: selective prediction across ALL experiments",
    "Phase 4",
    "Apply the Cell-20 machinery (src/evaluation/selective_prediction.py -- "
    "already implemented and used for the Stage-C baseline) to every LODO run, "
    "and plot the risk-coverage curves together.",
    "Cell 24 -- the other LODO runs do not exist yet.",
)


# %%
# ===========================================================================
# CELL 27 -- Feature embeddings
# ===========================================================================
_pending(
    "Cell 27: embeddings",
    "Phase 4",
    "PCA / UMAP / optional t-SNE coloured by domain and by grade; "
    "domain-centroid distances.",
    "Cell 24.",
)


# %%
# ===========================================================================
# CELL 28 -- Ablations
# ===========================================================================
_pending(
    "Cell 28: ablation study",
    "Phase 5",
    "The ordinal x DG x calibration grid. The full system is not assumed to win.",
    "Cells 20-23.",
)


# %%
# ===========================================================================
# CELL 29 -- Statistical comparison
# ===========================================================================
_pending(
    "Cell 29: statistics",
    "Phase 5",
    "Paired bootstrap difference intervals and McNemar tests on shared test "
    "samples. No test applied where its assumptions do not hold.",
    "Cell 28.",
)


# %%
# ===========================================================================
# CELL 30 -- Generate all figures
# ===========================================================================
_pending("Cell 30: figures", "Phase 5", "Rebuild the full figure library at 300 DPI.", "Cells 17-29.")


# %%
# ===========================================================================
# CELL 31 -- Generate all tables
# ===========================================================================
_pending("Cell 31: tables", "Phase 5", "Tables 1-9 as CSV and LaTeX.", "Cells 17-29.")


# %%
# ===========================================================================
# CELL 32 -- Experiment summary report
# ===========================================================================
_pending(
    "Cell 32: summary report",
    "Phase 5",
    "Markdown synthesis of what actually happened, including negative results "
    "and limitations.",
    "Cells 17-31.",
)
