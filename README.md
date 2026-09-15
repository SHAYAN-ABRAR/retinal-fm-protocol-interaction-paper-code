# Does linear probing predict fine-tuned cross-domain performance of a retinal foundation model?

A matched-initialisation comparison of **RETFound** against the exact
**ImageNet-MAE** checkpoint it was continued from, evaluated under three
adaptation protocols on held-out diabetic-retinopathy domains.

> ## Status: scientific programme CLOSED · evidence FROZEN
>
> **110/110 authoritative runs COMPLETE · 134.3 GPU-hours · 438/438 tests ·
> `audit_consistency` 3,090 checks.**
>
> The DDR + APTOS full fine-tuning replication is the final experimental
> evidence. **No further experiments are planned or recommended.**

---

## The result

The relative cross-domain QWK difference between the two initialisations
**depends strongly on the downstream adaptation protocol**.

Define, within a protocol, `Δ = QWK(ImageNet-MAE) − QWK(RETFound)`. The
protocol-by-initialisation interaction is `I_full = Δ_full − Δ_frozen`, paired
within seed:

| held out | *n* test | **I_full** | crossed 95% CI | *p* | **Holm *p*** | sign |
|---|---|---|---|---|---|---|
| DDR | 12,424 | **−0.1141** | [−0.1548, −0.0728] | 0.0066 | **0.0132** | 5/5 |
| APTOS | 3,504 | **−0.0992** | [−0.1595, −0.0397] | 0.0370 | **0.0370** | 5/5 |

Statistically supported on both held-out domains, in the same direction, with
**all ten seed-level interactions negative**. The large ImageNet-MAE advantage
measured under frozen linear probing is attenuated by roughly **0.10 QWK**
under matched full fine-tuning.

The mechanism in plain view, on APTOS:

| protocol | ImageNet-MAE | RETFound | Δ |
|---|---|---|---|
| frozen linear probe | 0.5872 | 0.4796 | **+0.1075** |
| partial FT (last 4 of 24 blocks) | 0.8334 | 0.8261 | +0.0073 |
| full fine-tuning | 0.8123 | 0.8040 | +0.0083 |

Both arms gain enormously from adaptation (RETFound 0.4796 → 0.8040), so this
is not a ceiling effect.

## What this does **not** say

- **No difference is demonstrated between the two initialisations under full
  fine-tuning — and that is not equivalence.** A null at five seeds bounds the
  effect only as tightly as its interval.
- **RETFound is not shown to be superior after fine-tuning.**
- The frozen-probe direction is **not uniform**: DDR and APTOS favour
  ImageNet-MAE, **IDRiD favours RETFound** (−0.0429, ten seeds).
- **IDRiD has no full fine-tuning arm.** That experiment was not run and no
  value is imputed for it anywhere; the interaction is a two-domain result.
- **This is not a priority claim.** RETFound's own paper already compares
  against an SSL-ImageNet baseline. The contribution is the *within-lineage
  protocol interaction*, and the wording is "to our knowledge", never "first".
- Nothing generalises to other retinal foundation models, architectures, or
  adaptation budgets.

Full allowed/forbidden wording per claim:
[`docs/FINAL_CLAIM_EVIDENCE_MAP.md`](docs/FINAL_CLAIM_EVIDENCE_MAP.md).

## Design

Two checkpoints of **one lineage**, identical architecture (ViT-L/16, 224 px):

```
ImageNet-MAE (timm vit_large_patch16_224.mae)
        │  continued MAE pretraining on retinal colour fundus photographs
        ▼
RETFound-CFP (RETFound_mae_natureCFP.pth)
```

Each initialisation enters each of three protocols — **frozen linear probe**,
**partial FT** (last 4 of 24 blocks), **full FT** (all 303,306,757 encoder
parameters) — under one fixed recipe, evaluated **leave-one-domain-out** across
four public DR datasets.

**Target labels are never used** for model selection, early stopping,
hyperparameter choice or temperature scaling. Temperature is fitted on source
validation only.

| held out | frozen probe | partial FT | full FT |
|---|---|---|---|
| DDR | 20 (10 seeds × 2 arms) | 10 (5 × 2) | 10 (5 × 2) |
| APTOS | 20 | 10 | 10 |
| IDRiD | 20 | 10 | **0 — not run** |

Seeds: 42, 1, 2, 3, 4 for both adaptation protocols; those plus 5–9 for the
frozen probe. **Interactions use the common five**, because the contrast is
paired within seed.

## Statistical framework

| instrument | role |
|---|---|
| crossed seed × case bootstrap (2,000 replicates) | **uncertainty interval** — not a test |
| paired seed-level *t*-test | **formal inference** |
| Holm | **specified multiplicity correction**, across the two interaction tests only |
| exact sign-flip permutation | **sensitivity** (floor 2/2⁵ = 0.0625 at *n*=5) |
| seed SD, sign count | **descriptive diagnostics** |

An effect whose multiplicity-adjusted *p* < 0.05 **and** whose crossed interval
excludes zero is *statistically supported under the study's inferential
framework*. **Where the two disagree, the formal test governs** — this occurs
for the DDR full-FT comparison, whose interval excludes zero while *p* = 0.1035,
and where consequently nothing is claimed.

## Study chronology

Stated as it happened, because it changes how the two-domain family should be
read:

1. The **DDR** full-FT experiment was specified, frozen, run — and **observed**.
2. **Only then** was APTOS specified as a confirmatory replication. Its protocol
   was committed before the first APTOS run, its recipe copied unchanged, and
   its analysis script written while seven of the ten runs did not yet exist.
3. Holm adjustment across the two domains is applied for conservative final
   reporting.

**The two-domain family was not pre-specified before DDR was seen.**

## Repository layout

```
src/                    library: data, models, losses, training, evaluation
tests/                  438 tests
configs/                experiment configuration
analyse_*.py            analyses that read frozen predictions
export_*.py             authoritative tables, claim map, provenance
make_jbhi_figures.py    main figures 1–5
make_jbhi_supplement_figures.py   supplementary S1–S6
audit_*.py              consistency, resume, identity audits
fetch_artifacts.py      assembles the reproducibility bundle
outputs/tables/         authoritative result tables
outputs/figures/jbhi_final/   11 figures, PNG (400 dpi) + vector PDF
docs/                   documentation — start at docs/README.md
paper/                  manuscript scaffold, verified bibliography, checklists
release/                reproducibility bundle (payload gitignored)
```

## Documentation

**Start at [`docs/README.md`](docs/README.md)** — it separates authoritative
final sources from superseded historical records, which matters here because
the project ran through thirteen phases and the older documents contradict the
frozen state.

| document | what it settles |
|---|---|
| [`docs/JBHI_EVIDENCE_FREEZE.md`](docs/JBHI_EVIDENCE_FREEZE.md) | hashes, run accounting, datasets, checkpoints, hypotheses, seed sets, chronology |
| [`docs/JBHI_DRAFTING_MANIFEST.md`](docs/JBHI_DRAFTING_MANIFEST.md) | which artifact each manuscript section draws on, with claim guardrails |
| [`docs/FINAL_CLAIM_EVIDENCE_MAP.md`](docs/FINAL_CLAIM_EVIDENCE_MAP.md) | every claim with allowed and forbidden wording |
| [`docs/JBHI_DATASET_PROVENANCE.md`](docs/JBHI_DATASET_PROVENANCE.md) | dataset table, contamination disclosure, integrity controls |
| [`docs/LITERATURE_NOVELTY_AUDIT.md`](docs/LITERATURE_NOVELTY_AUDIT.md) | corrected novelty position and the full search record |
| [`docs/JBHI_FIGURE_REVIEW.md`](docs/JBHI_FIGURE_REVIEW.md) | per-figure purpose, placement, reviewer-misreading notes |
| [`paper/CLAIM_2024_CHECKLIST.md`](paper/CLAIM_2024_CHECKLIST.md) | CLAIM 2024 compliance audit, 44 items |

## Data and integrity controls

Four public DR datasets. **DDR**, **APTOS 2019** and **IDRiD** serve as held-out
targets; **EyePACS** is a source domain only.

Two controls a reader needs to know about, both enforced in code:

1. **The EyePACS-derivative folder physically contains all 3,662 APTOS images.**
   The loader keeps only `^\d+_(left|right)$` filenames and rejects every 12-hex
   APTOS id (`src/data/eyepacs.py:99,144`). Without this, holding out APTOS
   would have trained on 100% of its own test set. This is the single most
   consequential integrity control in the study.
2. **EyePACS appears in RETFound's published pretraining corpus** and is a
   *source* domain here. This is disclosed, is **not target leakage** — no
   held-out target image was seen in pretraining or training — and applies
   identically to both protocol arms, so it cannot generate the interaction.

EyePACS labels were verified against an independent copy of the official labels
at 100.0000% agreement; the 50,070 unverifiable images were excluded. Full
detail: [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## Reproduction

`outputs/predictions/` is gitignored, so a clean clone can read the code but
cannot recompute the numbers. Every statistic in this study is derived from
**saved predictions, not checkpoints**, so a 127.7 MB bundle replaces ~60 GB of
weights.

```bash
python fetch_artifacts.py --check-terms   # report identifier exposure only
python fetch_artifacts.py --build         # assemble release/bundle/ locally
```

With the artifacts restored under `outputs/`:

```bash
python export_jbhi_tables.py             # master + primary interaction tables
python analyse_aptos_replication.py      # the APTOS replication analysis
python export_claim_evidence_map.py      # claim-evidence map
python export_referable_dr_secondary.py  # post-hoc referable-DR secondary
python make_jbhi_figures.py              # main figures 1-5
python make_jbhi_supplement_figures.py   # supplementary S1-S6
```

All five authoritative tables and all 23 figure files regenerate
**byte-identically**. Analyses need only CPU.

See [`release/README_REPRODUCTION.md`](release/README_REPRODUCTION.md). **The
bundle has not been published**; that awaits author approval and a check of each
dataset's redistribution terms.

### Training runs are not bit-reproducible, by recorded design

`run_lodo.py` sets `deterministic: False`, so cuDNN autotuning is on. Measured
directly: two from-scratch executions of the identical command at the identical
seed differ from epoch 0. Reproducing this study reproduces the *distribution
over seeds*, not individual numbers. **The frozen predictions sidestep this
entirely** — every analysis is exactly reproducible even though training is not.

## Environment

Python 3.14.3 · PyTorch 2.9.1+cu128 · torchvision 0.24.1+cu128 · timm 1.0.28 ·
CUDA 12.8 · NVIDIA RTX 5060 Laptop (8 GB) · Windows 11.

The RTX 5060 is Blackwell (**sm_120**); kernels ship only in CUDA 12.8+ builds.
An older CUDA wheel imports cleanly and reports `cuda.is_available() == True`,
then dies on the first kernel launch.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`src/utils/hardware.py` refuses to fall back to CPU — this study is not feasible
on CPU, so silence would be worse than a crash.

Full environment record including the RETFound checkpoint SHA256:
`outputs/tables/JBHI_ENVIRONMENT.csv`.

## Verification

```bash
python -m pytest -q                 # 438 tests
python audit_consistency.py         # 3,090 cross-checks
python audit_resumed_runs.py        # learning-rate schedule integrity
python export_evidence_freeze.py    # re-hash every artifact
python paper/check_numbers.py       # manuscript numbers trace to generators
python paper/check_report_numbers.py
python paper/check_latex.py
```

## Honesty policy

The practices this repository actually follows, stated so they can be checked:

- **Unrun work is marked `NOT RUN`, never estimated.** IDRiD full fine-tuning
  was not run, and no value is imputed for it anywhere.
- **Nulls are reported as nulls.** The full-FT comparison failed to demonstrate
  a difference on both domains, and is reported that way rather than as
  equivalence or as a near-miss.
- **Superseded results are retained**, not deleted. Abandoned executions are
  archived under names recording why they were abandoned.
- **Corrections are made in place and left visible.** Where a claim was
  narrowed or an audit tool was found wrong, the record says so — see the
  novelty correction in
  [`docs/LITERATURE_NOVELTY_AUDIT.md`](docs/LITERATURE_NOVELTY_AUDIT.md) §0 and
  the citation corrections in
  [`docs/DATASET_CITATION_NOTES.md`](docs/DATASET_CITATION_NOTES.md).
- **Every manuscript number traces to a generator CSV**, enforced by
  `check_numbers.py` and `check_report_numbers.py`.
- **Target blindness was maintained and is auditable** — the APTOS analysis code
  predates seven of its ten runs in the git history.

## Historical record

The mid-project README is preserved at
[`docs/README_HISTORICAL.md`](docs/README_HISTORICAL.md), and the thirteen phase
reports carry banners marking them non-authoritative. They are kept because they
document what was believed at each stage; where they disagree with the frozen
state, **the frozen state is correct**.

## Citation

The manuscript is in preparation. Citation metadata, licence, and an archival
DOI are author decisions still outstanding — see
[`paper/OPEN_ITEMS.md`](paper/OPEN_ITEMS.md).
