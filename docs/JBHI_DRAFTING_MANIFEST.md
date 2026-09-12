# JBHI drafting manifest

**The authoritative source list for every manuscript section, and the claim
constraints that apply while writing it.**

Read this, `FINAL_CLAIM_EVIDENCE_MAP.md` and `JBHI_EVIDENCE_FREEZE.md` before
writing a sentence. Draft into `paper/jbhi_manuscript.tex`.

> **`paper/main.tex` is NOT a source.** It is historical scaffolding with stale
> framing, retained for the record and deliberately not patched. Do not draft
> from it, and do not reconcile the new manuscript to its structure.
>
> **The phase reports (`docs/PHASE*.md`) are NOT authoritative.** They record
> what was believed at each stage, including superseded terminology and claims
> that were later narrowed. Where a final artifact exists, the final artifact
> wins. They may be cited only as a record of process, never for a number.

## 1. Where each section's content comes from

| section | authoritative artifacts |
|---|---|
| **Abstract** | `outputs/tables/JBHI_PRIMARY_INTERACTION.csv`; `docs/FINAL_CLAIM_EVIDENCE_MAP.md` |
| **I. Introduction** | `docs/LITERATURE_NOVELTY_AUDIT.md` §0, §4 |
| **II-A Retinal foundation models** | `docs/LITERATURE_NOVELTY_AUDIT.md` §2 (rows A–C) |
| **II-B Evaluation protocols** | `docs/LITERATURE_NOVELTY_AUDIT.md` §2 (rows B–D), §3 |
| **II-C Cross-dataset DR generalization** | `docs/LITERATURE_NOVELTY_AUDIT.md` §2 (rows D–E) |
| **III-A Datasets and provenance** | `docs/JBHI_DATASET_PROVENANCE.md` → **TABLE I** |
| **III-B Matched pretraining lineage** | `docs/JBHI_EVIDENCE_FREEZE.md` §5 → **Fig. 1** |
| **III-C Adaptation protocols** | `docs/JBHI_EVIDENCE_FREEZE.md` §5–6 |
| **III-D Metrics** | `docs/JBHI_EVIDENCE_FREEZE.md` §6 |
| **III-E Statistical analysis** | `docs/JBHI_EVIDENCE_FREEZE.md` §7 |
| **IV-A/B/C Designs** | `docs/JBHI_EVIDENCE_FREEZE.md` §3, §8; `outputs/tables/JBHI_RUN_ACCOUNTING.csv` |
| **IV-D External protocol** | `docs/JBHI_DATASET_PROVENANCE.md` |
| **IV-E APTOS replication** | `docs/JBHI_EVIDENCE_FREEZE.md` §10; `docs/APTOS_FULL_FINETUNE_REPLICATION_PROTOCOL.md` |
| **V-A Frozen evaluation** | `outputs/tables/JBHI_MASTER_RESULTS.csv` (`seed_set = all-10`) |
| **V-B Adaptation depth** | `outputs/tables/JBHI_MASTER_RESULTS.csv` (`seed_set = common-5`) → **Fig. 2**, **TABLE II** |
| **V-C Primary interaction** | `outputs/tables/JBHI_PRIMARY_INTERACTION.csv` → **Fig. 3**, **TABLE III** |
| **V-D Full-FT comparison** | `outputs/tables/JBHI_MASTER_RESULTS.csv` (`protocol = full`) → **Fig. 5** |
| **V-E Secondary reliability** | `outputs/tables/aptos_secondary_outcomes.csv`, `outputs/tables/full_finetune_secondary_outcomes.csv` → S4, S5, S6 |
| **VI. Discussion** | `docs/FINAL_CLAIM_EVIDENCE_MAP.md`; `docs/LITERATURE_NOVELTY_AUDIT.md` §4 |
| **VII. Limitations** | `docs/FINAL_CLAIM_EVIDENCE_MAP.md` L1–L3 |
| **VIII. Conclusion** | nothing new; restate V-C and V-D only |
| **Figure captions** | `docs/JBHI_FIGURE_REVIEW.md` (per-figure emphasis and misreading notes) |
| **Main vs supplement** | `docs/JBHI_CONTENT_PLACEMENT.md` |
| **Bibliography** | `paper/refs_jbhi_verified.bib`; caveats in `docs/DATASET_CITATION_NOTES.md` |
| **Reproducibility / hashes** | `docs/JBHI_EVIDENCE_FREEZE.md` §1–2; `outputs/tables/JBHI_EVIDENCE_MANIFEST.csv` |

### Deep-dive reports (context, not primary numbers)

- `docs/FULL_FINETUNE_FINAL_REPORT.md` — the DDR full-FT result in full
- `docs/APTOS_FULL_FINETUNE_FINAL_REPORT.md` — the APTOS replication in full
- `docs/APTOS_EXECUTION_LOG.md` — execution history, interruptions, determinism

Prefer `JBHI_MASTER_RESULTS.csv` and `JBHI_PRIMARY_INTERACTION.csv` for any
number that appears in the manuscript; the reports agree with them and exist to
explain them.

## 2. Claim guardrails

### The allowed central claim

> **"The relative cross-domain QWK difference between the ImageNet-MAE
> initialization and RETFound depends strongly on downstream adaptation
> protocol."**

### Also allowed

> "The large ImageNet-MAE advantage observed under frozen linear probing was
> attenuated by approximately 0.10 QWK under matched full fine-tuning on both
> DDR and APTOS."

> "The protocol-by-initialization interaction was statistically supported on
> both held-out domains in the final two-domain analysis."

> "No difference between the two initialisations was demonstrated under matched
> full fine-tuning on either domain."

> "To our knowledge, the direct within-lineage protocol-by-initialisation
> interaction has not previously been estimated."

### NOT allowed — none of these is supported

| forbidden | why |
|---|---|
| "RETFound is superior after fine-tuning." | the full-FT comparison is a null on both domains |
| "The models are equivalent after full fine-tuning." | a null at five seeds is not equivalence |
| "Linear probing reverses the ranking." | on APTOS the mean ordering never changes sign |
| "Retinal pretraining is useless." | not tested; out of scope |
| "This applies to every retinal foundation model." | one lineage pair, one architecture |
| "We are the first to compare RETFound and ImageNet-MAE." | RETFound's own paper does this |

Further per-claim allowed/forbidden wording, including the DDR
interval-versus-test disagreement, is in `FINAL_CLAIM_EVIDENCE_MAP.md`.

## 3. Chronology guardrail

State the sequence exactly as it happened:

1. The **DDR** full fine-tuning experiment was specified, frozen, run — and its
   interaction **observed**.
2. **APTOS was then** specified as a confirmatory replication. Its protocol was
   committed before the first APTOS full-FT run, its recipe was copied from DDR
   unchanged, and its analysis script was written before seven of the ten runs
   existed. No APTOS target outcome was inspected until all ten completed.
3. The **Holm adjustment across DDR and APTOS** is applied for final two-domain
   reporting.

> **Do not imply the two-domain family was pre-specified before DDR was seen.**
> It was not. DDR is the originating result; APTOS is its replication; the Holm
> adjustment is conservative final reporting, not evidence of a pre-planned
> two-domain design.

## 4. Terminology

The phrase **"two-bar rule" is retired** and must not appear.

| instrument | how to describe it |
|---|---|
| crossed seed × case bootstrap | **uncertainty interval** — never a test |
| paired seed-level *t*-test | **formal inferential test** |
| Holm | **multiplicity adjustment**, across the two interaction tests only |
| exact sign-flip permutation | **distribution-free sensitivity analysis** (floor 0.0625 at *n*=5) |
| seed SD, sign agreement | **descriptive stability diagnostics** |

An effect with a multiplicity-adjusted seed-level *p* < 0.05 **and** a crossed
interval excluding zero is **"statistically supported under the study's
inferential framework"**. Where the two disagree, **the formal test governs**
and nothing is claimed — this is the DDR full-FT case.

## 5. Numbers that must not be mixed

- **Never difference a ten-seed frozen estimate against a five-seed adaptation
  estimate.** Interactions are paired within seed and use the common five. The
  `seed_set` column in `JBHI_MASTER_RESULTS.csv` marks every row.
- **Never pool DDR and APTOS into one *p*-value.** Different test-set sizes,
  different shift structures.
- **Never impute an IDRiD full-FT value.** It was not run.

## 6. Front matter — do not guess

`[AUTHOR LIST]` · `[AUTHOR ORDER]` · `[AFFILIATIONS]` ·
`[CORRESPONDING AUTHOR]` · `[ORCID]` · `[FUNDING]` ·
`[ETHICS DETERMINATION]` · `[CONFLICTS OF INTEREST]` ·
`[CREDIT CONTRIBUTIONS]` · `[CODE REPOSITORY / DOI]` ·
`[DATA AVAILABILITY LANGUAGE]`

Every one requires a human. Leave the placeholder rather than inventing a
plausible value.

## 7. Before submission

- [ ] obtain the current IEEEtran template — **not in this repository**
- [ ] resolve the CLAIM-version question (`DATASET_CITATION_NOTES.md` §Uncertain)
- [ ] add page numbers for the three non-Crossref proceedings entries if the
      style requires them
- [ ] confirm Kaggle competition terms for EyePACS and APTOS
- [ ] re-check whether the Lancet Digital Health article has been paginated
- [ ] verify every manuscript number against `check_numbers.py`
