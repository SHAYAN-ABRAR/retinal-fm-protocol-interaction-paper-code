# Main paper versus supplement — recommended placement

The risk this document exists to prevent: **the domain-generalisation and
resolution work is voluminous and will crowd out the central retinal
foundation-model evaluation** if placed by volume rather than by relevance. The
DG benchmark, the resolution/batch decomposition and the calibration suite
represent a large share of the project's total compute and page count, and
almost none of the manuscript's contribution.

The paper is about **one question**: does the adaptation protocol change the
estimated value of RETFound's retinal continuation pretraining relative to its
ImageNet-MAE starting point? Everything that does not serve that question is
supplement.

## Main paper

| section | content | supporting artifact |
|---|---|---|
| Introduction | frozen probing as the cheap way to rank representations; the untested assumption that it predicts fine-tuned behaviour | `LITERATURE_NOVELTY_AUDIT.md` §4 |
| Related work | RETFound (Nature 2023) as closest prior art; RETFound-Green; pre-training-data-effects; the frozen calibration benchmark; the label-efficiency study | `LITERATURE_NOVELTY_AUDIT.md` §2 |
| Data | four DR datasets, LODO design, the splits, and **the APTOS-inside-EyePACS guard** | `JBHI_DATASET_PROVENANCE.md` |
| Methods — lineage | the matched-initialisation design: one architecture, one checkpoint lineage, one continuation step | Figure 1 |
| Methods — protocols | frozen linear probe / partial FT (last 4 of 24) / full FT (303,306,757 params) under one fixed recipe | Figure 1 |
| Methods — statistics | crossed bootstrap = interval; paired seed-level *t*-test = test; Holm across two domains; sign-flip = sensitivity; seed SD and sign count = diagnostics | `JBHI_EVIDENCE_FREEZE.md` §7 |
| Results — primary | the two-domain interaction | **Table: `JBHI_PRIMARY_INTERACTION.tex`**, Figure 3 |
| Results — depth | frozen → partial → full attenuation, paired per seed | **Table: `JBHI_MASTER_RESULTS.tex`**, Figure 2 |
| Results — chronology | DDR first and frozen; APTOS specified afterwards as confirmatory replication with the recipe frozen before any target outcome | `JBHI_EVIDENCE_FREEZE.md` §10 |
| Discussion | what a frozen probe does and does not tell you; concise clinical / domain-shift framing | `FINAL_CLAIM_EVIDENCE_MAP.md` |
| Limitations | L1 two domains only; L2 post-hoc pairing; L3 EyePACS exposure; one architecture; five seeds; sign-flip floor; non-determinism | `FINAL_CLAIM_EVIDENCE_MAP.md` L1–L3 |

**Optional main-paper figure if space allows:** Figure 4 (source-validation
optimisation behaviour), as descriptive mechanistic context only. It is the
first thing to cut.

## Supplement / secondary

| content | why it is not main-paper |
|---|---|
| broad DG-method benchmarking (ERM, MixStyle, CORAL, GroupDRO, IRM…) | a different question — which DG algorithm wins — and its headline finding is a null |
| resolution and batch-size decomposition | a configuration study; informs the recipe, not the contribution |
| extensive calibration tables, temperature-scaling detail | secondary outcomes; ECE appears in the main text only as an uncorrected exploratory metric that is explicitly not claimed |
| selective prediction / risk-coverage | not part of the primary question |
| full per-class metrics, confusion matrices | grade-1 counts are too small on IDRiD to support per-class claims |
| seed-instability case studies | methodological colour; the seed SD and sign counts in the main tables carry what is needed |
| provenance audit detail (leak counts, label verification, duplicate analysis) | summarised in the main Data section; the audit belongs in the supplement |
| checkpoint retention, storage engineering, run accounting | reproducibility appendix |
| extended training curves | Figure 4 carries the summary if it survives |
| in-domain and single-source experiments | different evaluation setting |

## Ordering rule

If a reviewer reads only the main paper, they should be able to state:

1. what the two arms are and why they are a matched pair;
2. what the three protocols are;
3. that the interaction is negative and statistically supported on two
   independent held-out domains;
4. that no difference is demonstrated under full fine-tuning, and that this is
   **not** an equivalence claim;
5. that the frozen-probe direction is not uniform across domains (IDRiD
   reverses it);
6. the three named limitations.

Anything that does not help a reader reach those six points belongs in the
supplement.
