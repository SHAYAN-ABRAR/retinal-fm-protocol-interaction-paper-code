# STARD-AI applicability assessment

Against **Sounderajah V, Guni A, Liu X, Collins GS, et al., *The STARD-AI
reporting guideline for diagnostic accuracy studies using artificial
intelligence*, Nature Medicine 31(10):3283–3289, 2025.
DOI 10.1038/s41591-025-03953-8.** Verified via Crossref (51 authors).

## Conclusion first

> **This study is not a conventional diagnostic accuracy study, and STARD-AI
> should not be presented as its primary reporting framework.**

STARD-AI governs studies whose object is *the diagnostic accuracy of an index
test against a reference standard in a defined clinical population*. The object
here is different: it is a **methodological comparison of evaluation protocols**
applied to two checkpoints of one pretraining lineage. The estimand is an
interaction between adaptation protocol and initialisation, not the accuracy of
a test intended for patient care.

Claiming full STARD-AI compliance would misrepresent the study design. Claiming
STARD-AI is irrelevant would waste a genuinely useful transparency check on the
one part that *is* accuracy-shaped — the post-hoc referable-DR analysis.

**Recommended position for the manuscript:**

> "This study evaluates adaptation protocols rather than the diagnostic accuracy
> of a deployable index test, so STARD-AI was used as a secondary transparency
> cross-check rather than as the primary reporting framework. The primary
> reporting checklist is CLAIM 2024."

## Why not TRIPOD+AI as the primary framework either

TRIPOD+AI governs the development and validation of **clinical prediction models
for individual patients**. This study develops no model intended for individual
prediction, proposes no clinical use, and reports no risk estimate for a person.
Several TRIPOD+AI items — target population, predictors, outcome timing,
clinical utility, model presentation for use — have no referent here.

TRIPOD+AI remains **cited and applicability-checked**, and its transparency
items about data, missingness and reporting of uncertainty are honoured. It is
not claimed as the governing checklist.

**Primary framework: CLAIM 2024.** It is the checklist written for medical
imaging AI research reporting, which is what this is.

## Item-level applicability

STARD-AI extends STARD 2015. Assessed by STARD's standard section structure.

### Relevant — use as a cross-check

These bear on the **post-hoc referable-DR secondary analysis**
(`JBHI_REFERABLE_DR_SECONDARY.csv`), which is the only accuracy-shaped
component.

| STARD area | why relevant here | where addressed |
|---|---|---|
| Index test description | the classifier and its decision rule must be reproducible | III-C |
| **Definition of test positivity, and whether the threshold was pre-specified** | **the most important item for us.** Referable DR is the fixed ICDR definition, grade ≥ 2, **not tuned on either held-out domain and not chosen after seeing target performance** | III-D, V-E |
| Reference standard and its rationale | ICDR grades as released; no re-grading | III-A, III-D |
| Rationale for the reference standard | published label schema | III-D |
| Participants: eligibility, sampling, recruitment | four public releases, retrospective, full released sets | III-A |
| Flow of participants / numbers analysed | per-domain test sizes, EyePACS exclusions | V-A, TABLE I |
| Estimates of accuracy with uncertainty | sensitivity, specificity, PPV, F1, AUROC, each mean ± SD over five seeds | V-E |
| Handling of indeterminate results | none: every image receives a grade; DDR grade 5 already absent | III-A |
| Adverse events / harms | none — no patient was exposed to anything | not applicable, state once |
| Funding and role of funder | back matter | `[FUNDING]` |
| Registration | none exists for this computational study | see below |
| Data availability | reproducibility bundle | back matter |

### Not applicable — and why

| STARD area | why it does not apply |
|---|---|
| Intended clinical use of the index test | **no clinical use is proposed.** This is an evaluation-methodology study |
| Clinical setting and site(s) of recruitment | no recruitment; previously published public datasets |
| Time interval between index test and reference standard | both derive from the same stored image; no temporal separation exists |
| Blinding of readers to the reference standard | no human reader; the classifier has no access to labels at inference |
| Sample-size calculation for accuracy | none performed; test-set sizes are fixed by the releases. The design constraint is the **five-seed sign-flip floor of 0.0625**, which is reported |
| Prospective registration | **none.** The protocols were committed to version control before their runs, which is auditable but is **not** trial registration and must not be described as such |
| Subgroup / fairness analyses | **impossible here**: no release publishes age, sex, race or ethnicity. See the mandated limitation |
| Comparison against an alternative index test in clinical use | no comparator test; both arms are research checkpoints |

## What STARD-AI usefully adds

Two items it enforces that the manuscript should state explicitly, because they
are exactly the places an accuracy claim usually goes wrong:

1. **Threshold pre-specification.** The referable-DR threshold is the fixed
   ICDR definition. It was not selected, tuned or moved after inspecting target
   performance. The manuscript must say this in the same sentence as the
   sensitivity/specificity numbers, because that is the claim a reviewer will
   test hardest.
2. **Spectrum and prevalence.** Referable prevalence differs between held-out
   domains (DDR 0.453, APTOS 0.391), and PPV is prevalence-dependent. The
   manuscript must not compare PPV across domains as though it were a property
   of the model.

## Summary for the manuscript

- **Primary reporting checklist: CLAIM 2024** — `paper/CLAIM_2024_CHECKLIST.md`
- **STARD-AI: secondary transparency cross-check**, applied to the post-hoc
  referable-DR analysis only
- **TRIPOD+AI: cited and applicability-checked**, not claimed as governing
- This is **not** presented as a diagnostic accuracy study, and **not** as a
  conventional individualised clinical prediction-model development paper
