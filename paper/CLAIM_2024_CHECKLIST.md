# CLAIM 2024 checklist audit

Against **Tejani AS et al., *Checklist for Artificial Intelligence in Medical
Imaging (CLAIM): 2024 Update*, Radiology: Artificial Intelligence
6(4):e240300, 2024. DOI 10.1148/ryai.240300, PMID 38809149.**
44 items. Item numbers and headings taken from the open-access record
(PMC11304031).

> **Compliance is a property of the manuscript, not of this repository.**
> Every row below marked "action" is *not yet satisfied*, even where the
> repository already holds the information. An item counts as met only once the
> final manuscript reports it.

**Journal-format note.** CLAIM suggests a structured abstract. JBHI uses its own
abstract format. Where the journal's instructions supersede checklist
presentation, the **content requirement is satisfied in the JBHI-compatible
format** rather than forcing a structured abstract; this is flagged at item 2.

## Legend

- **Applicable** — yes / no / partial, with the reason when not "yes"
- **Section** — where in `paper/jbhi_manuscript.tex` it will be addressed
- **Evidence** — the repository artifact that supplies the content
- **Action** — what still has to be written

---

## Title and Abstract

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 1 | Identify as an AI study; specify technology category | yes | Title | — | State that this is an evaluation of self-supervised vision-transformer representations under differing adaptation protocols. Title must not claim priority |
| 2 | Abstract: summary of design, methods, results, conclusions | yes | Abstract | `JBHI_PRIMARY_INTERACTION.csv`; `FINAL_CLAIM_EVIDENCE_MAP.md` | **Content satisfied in JBHI's abstract format, not a structured abstract** (see note above). Must carry both interaction estimates with Holm-adjusted *p*, and the explicit non-equivalence statement |

## Introduction

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 3 | Scientific/clinical background and intended use | yes | I | `LITERATURE_NOVELTY_AUDIT.md` §4 | Frame as an evaluation-methodology question. **Intended use: none — no clinical deployment is proposed** |
| 4 | Study aims, objectives, hypotheses | yes | I, III-E | `JBHI_EVIDENCE_FREEZE.md` §6 | State the protocol-by-initialisation interaction as the pre-specified primary hypothesis, two-sided, on QWK |

## Methods — Study Design

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 5 | Prospective or retrospective | yes | III-A | `JBHI_DATASET_PROVENANCE.md` | **Retrospective**, using four previously published public datasets |
| 6 | Study goal and design description | yes | IV | `JBHI_EVIDENCE_FREEZE.md` §10 | Describe the LODO design and the DDR→APTOS chronology exactly as recorded |

## Methods — Data

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 7 | Data sources | yes | III-A | `JBHI_DATASET_PROVENANCE.md` | Name all four releases with citations |
| 8 | Inclusion and exclusion criteria | yes | III-A | `DATA_PROVENANCE.md` §3D | Report the 35,108-of-85,178 EyePACS label verification and the 50,070 exclusions |
| 9 | Data preprocessing | yes | III-A/C | `src/data/preprocessing.py` | Retina crop, square pad, resize to 224 px; augmentation in the training transform only |
| 10 | Selection of data subsets | yes | III-A | `JBHI_RUN_ACCOUNTING.csv` | Report LODO pooling and the train/val/test sizes per held-out domain |
| 11 | De-identification | partial | III-A | dataset documentation | All four releases are already de-identified by their publishers. **State that and do not claim to have performed de-identification** |
| 12 | Missing data handling | yes | III-A | `DATA_PROVENANCE.md` | State that ungradable/grade-5 DDR images are already absent and no imputation is used |
| 13 | Image acquisition protocol | partial | III-A | `s2_domain_profile` | Not reported by the public releases at image level. Report the observable acquisition heterogeneity instead and say the acquisition protocols are not published |

## Methods — Reference Standard

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 14 | Definition of reference standard | yes | III-A/D | `src/evaluation/metrics.py` | ICDR 0–4 grading as released; referable DR defined as grade ≥ 2 for the secondary analysis |
| 15 | Rationale for reference standard | yes | III-D | — | ICDR is the released label schema; no re-grading was performed |
| 16 | Source of annotations | partial | III-A | `DATA_PROVENANCE.md` | Labels come with the releases. **EyePACS labels were verified against an independent copy of the official labels at 100.0000% agreement** — report this |
| 17 | Annotation of test set | partial | III-A | as above | Same provenance as training labels; no additional annotation by us |
| 18 | Inter/intrarater variability | **no** | VII | — | **Not available.** No release publishes rater-level data and no re-grading was performed. State as a limitation |

## Methods — Data Partitions

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 19 | Partition assignment | yes | III-A, IV-D | `JBHI_RUN_ACCOUNTING.csv` | LODO: one domain held out entirely; remaining pooled and split into train/source-validation |
| 20 | Level of partition disjointness | yes | III-A | `DATA_PROVENANCE.md` §3B | **Patient level for EyePACS** (both eyes kept together); **image level elsewhere, because DDR, APTOS and IDRiD publish no usable patient identifiers.** Report both facts |

## Methods — Testing Data

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 21 | Intended sample size | partial | IV | `JBHI_EVIDENCE_FREEZE.md` §3 | No formal power calculation was performed. Report the fixed test-set sizes and the five-seed design, and state the sign-flip floor (0.0625) as the power constraint |

## Methods — Model

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 22 | Detailed model description | yes | III-B/C | `JBHI_EVIDENCE_FREEZE.md` §5 | ViT-L/16 at 224 px; 303,306,757 trainable parameters under full FT |
| 23 | Software libraries and frameworks | yes | III-C | `JBHI_ENVIRONMENT.csv` | PyTorch 2.9.1+cu128, timm 1.0.28, CUDA 12.8, Python 3.14.3 |
| 24 | Model parameter initialization | yes | III-B | `JBHI_EVIDENCE_FREEZE.md` §5 | **The heart of the study.** `vit_large_patch16_224.mae` vs `RETFound_mae_natureCFP.pth` (SHA256 pinned); 294 tensors at 100% loaded, asserted at runtime |

## Methods — Training

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 25 | Training approach details | yes | III-C | `APTOS_FULL_FINETUNE_REPLICATION_PROTOCOL.md` | AdamW, lr 1e-4, wd 1e-4, 1-epoch warmup, cosine, 20 epochs, batch 16, gradient checkpointing, AMP |
| 26 | Final model selection method | yes | III-C, IV-D | `*_source_validation.csv` | **Best source-validation QWK; early stopping patience 6 on source validation. Target labels never used** |
| 27 | Ensembling technique | **no** | — | — | No ensembling. State explicitly that each reported model is a single run |

## Methods — Evaluation

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 28 | Performance metrics | yes | III-D | `JBHI_MASTER_RESULTS.csv` | QWK primary; eight secondary metrics named; referable-DR secondary |
| 29 | Statistical significance measures | yes | III-E | `JBHI_EVIDENCE_FREEZE.md` §7 | Crossed bootstrap = interval; paired seed-level *t*-test = test; Holm across the two interaction tests; sign-flip = sensitivity. **Retire "two-bar rule"** |
| 30 | Robustness/sensitivity analysis | yes | III-E, V-E | `JBHI_PRIMARY_INTERACTION.csv` | Exact sign-flip permutation; across-seed SD and sign counts; adaptation-depth gradient |
| 31 | Explainability/interpretability | **no** | VII | — | **Not performed and not planned.** Saliency/attention maps do not bear on a protocol-interaction claim. State this rather than leaving it silent |
| 32 | Evaluation on internal data | partial | Supplement | in-domain runs | In-domain runs exist but are not the manuscript's subject; point to the supplement |
| 33 | Testing on external data | yes | IV-D, V | `JBHI_MASTER_RESULTS.csv` | **Every headline number is external**: leave-one-domain-out, target never seen in training |
| 34 | Clinical trial registration | **no** | — | — | Not a clinical trial; no registration applicable. **No prospective registration of this computational protocol exists**, though the APTOS replication protocol was committed to version control before its first run — state that honestly, and do not call it trial registration |

## Results — Data

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 35 | Numbers included/excluded | yes | V-A, TABLE I | `JBHI_DATASET_PROVENANCE.md` | Per-domain counts and the EyePACS exclusions |
| 36 | Demographic and clinical characteristics | **no** | VII | §7 of this audit | **Not available.** No release publishes age, sex, race or ethnicity. Report the mandated limitation sentence verbatim (below) |

## Results — Model Performance

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 37 | Performance estimates with uncertainty | yes | V-B/C/D | `JBHI_MASTER_RESULTS.csv`, `JBHI_PRIMARY_INTERACTION.csv` | Crossed 95% intervals on every estimate; seed SDs |
| 38 | Diagnostic performance estimates | partial | V-E, referable table | `JBHI_REFERABLE_DR_SECONDARY.csv` | Sensitivity/specificity/PPV/F1/AUROC at the fixed grade ≥ 2 threshold. **Post-hoc secondary; QWK remains primary** |
| 39 | Failure analysis | partial | V-E, S6 | `s6_error_pattern` | Row-normalised confusion and severe-error rates. Note the shared failure mode: grade 1 is almost never predicted by either arm |

## Discussion

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 40 | Study limitations | yes | VII | `FINAL_CLAIM_EVIDENCE_MAP.md` L1–L4 | Must include: two domains only; post-hoc pairing; EyePACS exposure; no demographics; one architecture; five seeds; sign-flip floor; non-determinism |
| 41 | Implications for practice | yes | VI | `LITERATURE_NOVELTY_AUDIT.md` §4 | **Implication is for evaluation practice, not clinical practice.** No deployment recommendation |

## Other Information

| # | Requirement | Applicable | Section | Evidence | Action |
|---|---|---|---|---|---|
| 42 | Study protocol reference | partial | IV-E | `APTOS_FULL_FINETUNE_REPLICATION_PROTOCOL.md`, `FULL_FINETUNE_PROTOCOL.md` | Protocols were version-controlled before their runs. Cite the repository, and **do not describe this as prospective registration** |
| 43 | Software/model/data availability | yes | Back matter | `release/README_REPRODUCTION.md` | `[CODE REPOSITORY / DOI]` and `[DATA AVAILABILITY LANGUAGE]` placeholders; reproducibility bundle prepared but **not published** |
| 44 | Funding sources and funder role | yes | Back matter | — | `[FUNDING]` placeholder — author action |

---

## Summary

| status | count |
|---|---|
| applicable, content available, **manuscript must still report it** | 31 |
| partial — available but limited by the public releases | 8 |
| not applicable, and to be stated as such | 5 (items 18, 27, 31, 34, 36) |

**No item is marked satisfied.** Every one requires the manuscript to report it.

## Mandated limitation text (items 36 and 40)

Verified true on 2026-09-12 by inspecting the label files of all four releases:
the only columns present are image identifiers, DR grade, macular-oedema risk
and lesion coordinates. **No age, sex, race or ethnicity field exists in any
release.**

> "Demographic subgroup fairness could not be evaluated because harmonized
> patient demographic attributes were not available across the public releases."

Do not attempt to infer demographics from images, and do not report any
subgroup analysis.
