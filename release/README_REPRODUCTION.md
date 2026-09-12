# Reproducing the results

**281 artifacts, 127.7 MB.** Everything needed to recompute every number in the
manuscript from frozen per-image predictions — without retraining anything and
without access to the raw retinal images.

> **NOT PUBLISHED.** This bundle was assembled locally for the author to review.
> Publishing it is a separate, deliberate act requiring author approval and a
> check of each dataset's redistribution terms (see *Licensing*, below).

Rebuild or re-verify with:

```
python fetch_artifacts.py --check-terms    # report identifier exposure only
python fetch_artifacts.py --build          # assemble release/bundle/
python fetch_artifacts.py --verify         # re-hash without rebuilding
```

## Why a bundle is needed

The public repository gitignores `outputs/predictions/`. A clean clone can read
the code but cannot reproduce a single reported number, because **every
statistic in this study is recomputed from saved predictions, not from model
checkpoints.** That design is what makes this bundle small enough to archive:
125.8 MB of predictions replaces ~60 GB of weights.

## Contents

| role | files | size | what it is |
|---|---|---|---|
| predictions | 110 | 125.8 MB | per-image target-test predictions and class probabilities — the primary evidence |
| config/report | 110 | 1.0 MB | per-run evaluation JSON: temperature, config, recorded metrics |
| authoritative table | 24 | 0.1 MB | every table the manuscript draws on |
| source-validation history | 20 | 0.1 MB | per-epoch source-validation curves for the 20 full-FT runs |
| analysis script | 14 | 0.2 MB | regenerates all tables and figures |
| registry | 1 | 0.4 MB | `experiment_registry.csv` — run accounting and provenance |
| figure provenance | 1 | — | per-panel source, columns, filters, seed set, hashes |
| environment | 1 | — | dependency pinning |

Each row of `JBHI_REPRODUCIBILITY_MANIFEST.csv` carries: `artifact`, `role`,
`size_bytes`, `sha256`, `source_run`, `required_for`.

### Deliberately excluded

- **raw retinal images** — the datasets' to distribute, not ours
- **model checkpoints** — 1.21 GB each; no analysis reads them
- restricted or unverifiable data
- superseded artifacts, except where audit history needs them

## The 110 runs

| held out | frozen probe | partial FT | full FT |
|---|---|---|---|
| DDR | 20 (10 seeds × 2 arms) | 10 (5 × 2) | 10 (5 × 2) |
| APTOS | 20 | 10 | 10 |
| IDRiD | 20 | 10 | **0 — not run** |

Seeds: 42, 1, 2, 3, 4 for both adaptation protocols; those plus 5–9 for the
frozen probe. **IDRiD has no full fine-tuning arm and no value is imputed for
it anywhere.**

## Reproducing

With the bundle beside a clone of the repository, restore the artifacts into
`outputs/` and run:

```
python export_jbhi_tables.py            # master + primary interaction tables
python analyse_aptos_replication.py     # the APTOS replication analysis
python export_claim_evidence_map.py     # claim-evidence map
python export_referable_dr_secondary.py # post-hoc referable-DR secondary
python make_jbhi_figures.py             # main figures 1-5
python make_jbhi_supplement_figures.py  # supplementary S1-S6
```

Expected: the regenerated tables and figures are **byte-identical** to the
committed ones. This has been verified — all five authoritative tables and all
23 figure files reproduce exactly.

### Two checks worth running

```
python export_evidence_freeze.py    # re-hash every artifact
python audit_consistency.py         # 3090 cross-checks
```

## Reproducibility caveat that is not a bug

Training runs are **not bit-reproducible**, by recorded design:
`run_lodo.py` sets `deterministic: False`, so cuDNN autotuning is on and
deterministic algorithms are off. Measured directly — two from-scratch
executions of the identical command at the identical seed differ from epoch 0.

**This bundle sidesteps that entirely.** The predictions are frozen, so every
*analysis* is exactly reproducible even though the *training* is not. Retraining
from scratch would reproduce the distribution over seeds, not the individual
numbers.

## Licensing — read before publishing

The prediction files carry an `image_id` column.

- **DDR, APTOS, IDRiD** — identifiers are the public releases' own file names.
  None of these three releases publishes patient identifiers, so the ids carry
  no patient linkage.
- **EyePACS** — its `<n>_left` / `<n>_right` names *are* patient-linkable within
  that release. **However, EyePACS is never a held-out target in this study, so
  no EyePACS per-image identifier appears in this bundle at all.** Verified: 0
  EyePACS target-test prediction files.

`fetch_artifacts.py --anonymise-ids` replaces every `image_id` with a per-domain
surrogate key if the author prefers. No analysis joins on the identifier, so
everything still reproduces.

**Still required before any public release:**

1. confirm DDR, APTOS, IDRiD and EyePACS terms permit redistributing derived
   per-image predictions;
2. author approval;
3. a decision on surrogate identifiers.

## Intended publication route

1. **GitHub source release** — code, documentation, authoritative tables
2. **Versioned Zenodo archive with a DOI** — this bundle, if the author approves

Neither has been done. Nothing has been published.

## Environment

Python 3.14.3 · PyTorch 2.9.1+cu128 · torchvision 0.24.1+cu128 · timm 1.0.28 ·
CUDA 12.8 · NVIDIA RTX 5060 Laptop (8 GB) · Windows 11.

Full record in `tables/JBHI_ENVIRONMENT.csv`, including the RETFound checkpoint
SHA256. Analyses need only CPU; no GPU is required to reproduce any table or
figure.
