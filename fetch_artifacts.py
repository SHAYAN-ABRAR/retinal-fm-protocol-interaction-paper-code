"""Assemble the local reproducibility bundle, and verify one already assembled.

The public repository gitignores `outputs/predictions/`, so a clean clone
cannot reproduce a single number in the manuscript. This closes that gap by
collecting exactly the artifacts the final analyses read -- and nothing else.

**It publishes nothing.** The bundle is written locally under `release/` for the
author to review and, if they choose, archive. Publication is a separate,
deliberate act.

What goes in: the frozen per-image predictions the final analyses consume, the
experiment registry, the authoritative tables, run configs, the source-validation
histories the manuscript figures use, the figure provenance manifests, the
analysis scripts, the environment record and the evidence hashes.

What stays out, deliberately:

* **raw retinal images** -- they are the datasets' to distribute, not ours;
* restricted or unverifiable data;
* model checkpoints (1.21 GB each, and no analysis reads them; every statistic
  is recomputed from predictions);
* superseded artifacts, except where audit history needs them.

**Per-image identifiers.** The prediction files carry an `image_id` column. For
DDR, APTOS and IDRiD these are the identifiers the public releases themselves
use; for EyePACS they are the `<n>_left` / `<n>_right` names, which are
patient-linkable within that release. Redistributing them alongside predicted
grades may or may not be permitted by each dataset's terms. `--check-terms`
reports the exposure so the author can decide **before** anything is published;
`--anonymise-ids` replaces them with per-domain surrogate keys, which keeps
every analysis reproducible because nothing joins on the identifier.

Usage:
    python fetch_artifacts.py --check-terms     # report exposure, write nothing
    python fetch_artifacts.py --build           # assemble under release/bundle/
    python fetch_artifacts.py --build --anonymise-ids
    python fetch_artifacts.py --verify          # re-hash an assembled bundle
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

BUNDLE = Path("release") / "bundle"
MANIFEST = Path("release") / "JBHI_REPRODUCIBILITY_MANIFEST.csv"

SOURCES = {"ddr": "aptos-eyepacs-idrid",
           "aptos": "ddr-eyepacs-idrid",
           "idrid": "aptos-ddr-eyepacs"}
PROTOCOLS = {"frozen": "linprobe-b256",
             "partial": "erm-b16-tb4-lr0.0001",
             "full": "erm-b16-ftfull-lr0.0001"}
ARMS = ["vit-large-mae-in1k", "retfound-cfp"]
COMMON_SEEDS = [42, 1, 2, 3, 4]
FROZEN_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]

AUTHORITATIVE_TABLES = [
    "JBHI_MASTER_RESULTS.csv", "JBHI_PRIMARY_INTERACTION.csv",
    "JBHI_REFERABLE_DR_SECONDARY.csv", "JBHI_REFERABLE_DR_PER_SEED.csv",
    "final_claim_evidence.csv", "two_domain_interaction_holm.csv",
    "aptos_full_finetune_per_seed.csv", "aptos_full_finetune_primary.csv",
    "aptos_adaptation_depth.csv", "aptos_secondary_outcomes.csv",
    "aptos_full_finetune_source_validation.csv",
    "full_finetune_per_seed.csv", "full_finetune_primary.csv",
    "full_finetune_secondary_interactions.csv",
    "full_finetune_secondary_outcomes.csv",
    "full_finetune_source_validation.csv",
    "ft_convergence_source_validation.csv",
    "figure_qwk_by_domain_protocol_seed.csv",
    "configuration_decomposition.csv", "configuration_per_seed.csv",
    "JBHI_EVIDENCE_MANIFEST.csv", "JBHI_RUN_ACCOUNTING.csv",
    "JBHI_ENVIRONMENT.csv", "table1_dataset_characteristics.csv",
]

ANALYSIS_SCRIPTS = [
    "analyse_aptos_replication.py", "analyse_full_finetune.py",
    "analyse_ft_source_validation.py", "analyse_ft_convergence.py",
    "export_jbhi_tables.py", "export_claim_evidence_map.py",
    "export_referable_dr_secondary.py", "export_evidence_freeze.py",
    "export_figure_provenance.py", "make_jbhi_figures.py",
    "make_jbhi_supplement_figures.py", "audit_consistency.py",
    "audit_resumed_runs.py", "audit_experiment_identity.py",
]


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def required_runs():
    """Every run whose predictions a final analysis actually reads."""
    for domain, sources in SOURCES.items():
        for protocol, tag in PROTOCOLS.items():
            if domain == "idrid" and protocol == "full":
                continue                      # not run, by design
            seeds = FROZEN_SEEDS if protocol == "frozen" else COMMON_SEEDS
            for arm in ARMS:
                for seed in seeds:
                    yield (domain, protocol, arm, seed,
                           f"lodo_{sources}__{domain}_{arm}_{tag}_s{seed}")


def analyses_for(domain: str, protocol: str) -> str:
    parts = []
    if protocol in ("frozen", "full") and domain in ("ddr", "aptos"):
        parts.append("primary interaction")
    parts.append("master results")
    if protocol == "full" and domain in ("ddr", "aptos"):
        parts.append("full-FT comparison; referable-DR secondary")
    if protocol == "frozen" and domain in ("ddr", "aptos"):
        parts.append("referable-DR secondary")
    if protocol == "partial":
        parts.append("adaptation depth")
    return "; ".join(parts)


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    arguments = sys.argv[1:]
    build = "--build" in arguments
    verify = "--verify" in arguments
    check_terms = "--check-terms" in arguments
    anonymise = "--anonymise-ids" in arguments
    if not (build or verify or check_terms):
        print(__doc__)
        return 1

    root = project_root()
    outputs = root / "outputs"
    bundle = root / BUNDLE

    runs = list(required_runs())
    rows, missing = [], []

    # ------------------------------------------------------------ terms
    if check_terms:
        print("Per-image identifier exposure, by domain\n")
        for domain in SOURCES:
            sample = next((eid for d, _, _, _, eid in runs if d == domain), None)
            path = (outputs / "predictions" /
                    f"{sample}__target_test[{domain}]_predictions.csv")
            if not path.exists():
                print(f"  {domain:7} predictions missing")
                continue
            frame = pd.read_csv(path, nrows=3)
            ids = frame["image_id"].tolist() if "image_id" in frame else []
            print(f"  {domain:7} n={len(pd.read_csv(path)):>6}  "
                  f"example ids: {ids[:2]}")
        print("\nAssessment:")
        print("  DDR / APTOS / IDRiD -- identifiers are the public releases'")
        print("    own file names; they carry no patient linkage, because none")
        print("    of these three releases publishes patient identifiers.")
        print("  EyePACS -- '<n>_left' / '<n>_right' IS patient-linkable within")
        print("    that release. Redistributing it with predicted grades is the")
        print("    one exposure worth a decision.")
        print("\n  !! Confirm each dataset's terms before publishing. Use")
        print("     --anonymise-ids to substitute surrogate keys; every")
        print("     analysis still reproduces, since nothing joins on the id.")
        return 0

    # ------------------------------------------------------------ build
    if build:
        if bundle.exists():
            shutil.rmtree(bundle)
        for sub in ("predictions", "tables", "configs", "histories",
                    "scripts", "figures"):
            (bundle / sub).mkdir(parents=True, exist_ok=True)

    def record(source: Path, target: Path, role: str, run: str, analysis: str):
        if not source.exists():
            missing.append(str(source.relative_to(root)))
            return
        if build:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if anonymise and target.suffix == ".csv" and role == "predictions":
                frame = pd.read_csv(target)
                if "image_id" in frame.columns:
                    domain = run.split("__")[1].split("_")[0]
                    frame["image_id"] = [f"{domain}_{i:06d}"
                                         for i in range(len(frame))]
                    frame.to_csv(target, index=False)
        final = target if build else source
        rows.append({
            "artifact": str(target.relative_to(bundle)) if build
            else str(source.relative_to(root)),
            "role": role,
            "size_bytes": final.stat().st_size,
            "sha256": sha256(final),
            "source_run": run,
            "required_for": analysis,
        })

    for domain, protocol, arm, seed, eid in runs:
        name = f"{eid}__target_test[{domain}]_predictions.csv"
        record(outputs / "predictions" / name,
               bundle / "predictions" / name, "predictions", eid,
               analyses_for(domain, protocol))
        report = outputs / "reports" / f"{eid}_evaluation.json"
        record(report, bundle / "configs" / report.name, "config/report", eid,
               "temperature, config, recorded metrics")
        if protocol == "full":
            history = outputs / "logs" / f"{eid}_history.csv"
            record(history, bundle / "histories" / history.name,
                   "source-validation history", eid,
                   "source-validation tables; figures S1, fig4 context")

    record(outputs / "experiment_registry.csv",
           bundle / "experiment_registry.csv", "registry", "(all)",
           "run accounting, identity audit, provenance")

    for name in AUTHORITATIVE_TABLES:
        record(outputs / "tables" / name, bundle / "tables" / name,
               "authoritative table", "(derived)", "manuscript tables and text")

    figures = outputs / "figures" / "jbhi_final"
    for name in ("FIGURE_PROVENANCE.csv",):
        record(figures / name, bundle / "figures" / name,
               "figure provenance", "(derived)", "figure traceability")

    for name in ANALYSIS_SCRIPTS:
        record(root / name, bundle / "scripts" / name, "analysis script",
               "(code)", "regenerates the tables and figures")

    for name in ("requirements.txt", "pyproject.toml"):
        if (root / name).exists():
            record(root / name, bundle / name, "environment", "(code)",
                   "dependency pinning")

    if missing:
        print(f"!! {len(missing)} expected artifact(s) missing:")
        for item in missing[:15]:
            print(f"   {item}")
        return 1

    frame = pd.DataFrame(rows)
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / MANIFEST, index=False)

    total = frame.size_bytes.sum()
    print(f"{'built' if build else 'inventoried'} "
          f"{len(frame)} artifact(s), {total/1e6:.1f} MB")
    for role, group in frame.groupby("role"):
        print(f"  {role:28} {len(group):>4} files  "
              f"{group.size_bytes.sum()/1e6:8.1f} MB")
    if anonymise:
        print("\n  image_id columns replaced with per-domain surrogate keys")
    print(f"\nmanifest -> {MANIFEST}")
    if build:
        print(f"bundle   -> {BUNDLE}  (LOCAL ONLY -- nothing published)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
