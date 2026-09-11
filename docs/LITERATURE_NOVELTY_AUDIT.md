# Literature and novelty audit — final

**Revised 2026-09-11, after the evidence freeze, incorporating a correction
that narrows the contribution.**

## 0. The correction, stated first

An earlier version of this audit framed the contribution as the **first
comparison of RETFound against the ImageNet-MAE checkpoint from which it
originated**.

**That claim is wrong and must not appear anywhere in the manuscript.**

The RETFound paper (Zhou et al., *Nature* 2023) already compares RETFound
against an **SSL-ImageNet** baseline. That paper states that RETFound *uses the
weights of SSL-ImageNet as a baseline before extending to retinal images* —
i.e. the lineage relationship is explicit in the original work. Its comparison
models share the architecture, its downstream models use matched fine-tuning,
and it includes cross-dataset DR evaluation in which RETFound is reported as
significantly better than SSL-ImageNet on external sets.

So the pairing itself is not new. Neither is comparing the two arms under
fine-tuning, nor evaluating them across DR datasets.

### What we may claim

> A **within-lineage evaluation of how downstream adaptation protocol changes
> the estimated value of RETFound's additional retinal-domain MAE continuation
> pretraining relative to its SSL-ImageNet starting point**, with a **direct
> protocol-by-initialisation interaction test** replicated across independent
> held-out DR domains.

The novel object is the **interaction**, estimated directly and with an explicit
inferential framework, not the pair and not either protocol alone.

### What we must not claim

- "First comparison of RETFound against its own ImageNet-MAE initialisation."
- "First lineage-matched evaluation of RETFound."
- "First to show ImageNet-MAE can match or beat RETFound."
- Any uniqueness claim not supported by §2 and the search in §3.

## 1. Search performed

Executed 2026-09-11 against the live literature for the conjunction:

> RETFound versus its SSL-ImageNet/MAE starting checkpoint × frozen linear probe
> versus full fine-tuning × external DR grading × direct interaction analysis.

**No publication performing that specific conjunction was found.** Adjacent work
is enumerated in §2, and every element of the conjunction individually has prior
art. The manuscript should therefore claim the *interaction analysis* as the
contribution and cite §2 generously, rather than claim priority over any
component.

This is a negative search result, not proof of absence. The manuscript wording
should be "to our knowledge" rather than "first".

## 2. The comparison matrix

| | pairing | protocols compared | external DR? | interaction estimated? | relation to us |
|---|---|---|---|---|---|
| **A. RETFound** — Zhou et al., *Nature* 2023, `10.1038/s41586-023-06555-x` | RETFound vs **SSL-ImageNet** (its own starting point), SL-ImageNet; same ViT-L architecture | fine-tuning only, matched across arms | yes — cross-dataset DR incl. APTOS-2019, IDRiD, MESSIDOR-2 | **no** | **Closest prior art.** Establishes the lineage pair and matched fine-tuning. Does **not** contrast frozen probing against matched full fine-tuning for that pair, and estimates no protocol interaction. |
| **B. RETFound-Green** — *Nat. Commun.* 2025, `10.1038/s41467-025-62123-z` ("half the data, 400× less compute") | new efficient retinal FM vs RETFound and others | linear probing and full fine-tuning both appear | yes | no | Model identity and adaptation protocol **vary together**; it is not one checkpoint pair evaluated under both protocols. |
| **C. Pre-training data effects** — *Nat. Commun.* 17:3309 (2026), `10.1038/s41467-026-70077-z`, PMID 41764179 | parallel retinal FMs pretrained on two different 904,170-image cohorts (Moorfields; Shanghai) | linear probing and full encoder fine-tuning, five seeds, multiple external sets | yes | no | Compares FMs built from **different pretraining cohorts**, not the before/after of a continuation step on one lineage. Methodologically the nearest neighbour on protocol and seed design. |
| **D. Frozen-transfer calibration benchmark** — *Front. Med.* 2026, `10.3389/fmed.2026.1815982`, PMID 42078464 | MedSigLIP vs RETFound vs EfficientNet-B0 (ImageNet-supervised) | **frozen only** for the principal benchmark | yes — APTOS internal, MESSIDOR-2 external | no | **Reports RETFound below an ImageNet-supervised baseline under frozen external transfer** (external AUC 0.697 vs 0.745) and already argues MAE-derived frozen representations may need nonlinear adaptation. Different architectures and pretraining families; no within-lineage frozen-vs-full interaction. |
| **E. Label efficiency** — *Lancet Digit. Health* 2026, PMID 42665469, online 28 Aug 2026 (preprint arXiv:2501.12016) | RETFound vs ResNet50, ViT-Base, SwinV2 (ImageNet-pretrained) | full fine-tuning at varying label fractions | yes, ocular + systemic | no | **For ocular disease with larger labelled sets, ImageNet-pretrained models perform comparably to RETFound after full fine-tuning**; RETFound's advantage concentrates in systemic tasks at small label counts. Directly relevant context for our full-FT null. Different architectures; not the lineage pair; no interaction test. |

**D is the paper closest to our frozen-arm result** and **E is the paper closest
to our full-FT result**. Both must appear in Related Work, and the manuscript
should present our contribution as *connecting* the two observations via a
direct interaction on a single lineage pair, rather than as discovering either.

## 3. Where each element already has prior art

| element | prior art | so we cannot claim |
|---|---|---|
| RETFound vs its SSL-ImageNet start | A | novelty of the pairing |
| linear probing vs full fine-tuning of retinal FMs | B, C | novelty of the protocol contrast |
| external/LODO DR grading | A, C, D, E | novelty of the evaluation setting |
| RETFound underperforming an ImageNet baseline | D (frozen), E (full FT, ocular) | novelty of the direction of the finding |
| "frozen probes mislead about fine-tuned transfer" as a general idea | broad transfer-learning literature, incl. medical-imaging benchmarks reporting linear probing beating fine-tuning on external data | novelty of the hypothesis |
| **direct protocol × initialisation interaction, estimated with CI and multiplicity-adjusted test, replicated on a second held-out domain** | **none found** | — this is the contribution |

## 4. Honest framing for the manuscript

Recommended positioning, in order:

1. **Motivation.** Frozen linear probing is widely used to rank foundation-model
   representations cheaply (B, C, D). Whether that ranking predicts what happens
   after full adaptation is an assumption, rarely tested directly.
2. **Gap.** Prior work either varies model identity and protocol together (B, C),
   evaluates frozen representations only (D), or fine-tunes only (A, E). The
   protocol effect is therefore confounded with model identity, or not estimated.
3. **Approach.** Hold model identity to a single lineage step — RETFound versus
   the exact SSL-ImageNet MAE checkpoint it continued from — and vary only the
   adaptation protocol, estimating the interaction directly.
4. **Result.** The interaction is statistically supported on two independent
   held-out DR domains, in the same direction, with all ten seed-level
   interactions negative.
5. **Consequence.** A frozen-probe comparison of these two checkpoints does not
   transfer to their matched fine-tuned comparison. Reported representation
   rankings that rest on frozen probes should be read with that in mind.

## 5. Required citations

At minimum A–E above, plus the checkpoint sources:

- Zhou et al., *Nature* 2023 — RETFound (A)
- RETFound-Green, *Nat. Commun.* 2025 (B)
- Pre-training data effects, *Nat. Commun.* 2026 (C)
- Frozen-transfer calibration benchmark, *Front. Med.* 2026 (D)
- Label efficiency, *Lancet Digit. Health* 2026 (E)
- He et al., MAE (the `vit_large_patch16_224.mae` lineage)
- Dataset sources: DDR, APTOS 2019, IDRiD, EyePACS

**BibTeX status:** entries for A–E must be added to `paper/refs.bib` by whoever
drafts the manuscript, from the DOIs/PMIDs recorded above. They are recorded
here rather than fabricated into `refs.bib` with invented page numbers, volume
numbers or author lists. **Verify every field against the publisher record
before submission.**

## 6. Reviewer objections to pre-empt

| objection | response available from our evidence |
|---|---|
| "RETFound already compared against SSL-ImageNet." | Correct, and cited as the closest prior art. Our object is the protocol interaction, which that paper does not estimate. |
| "Your fine-tuning isn't fine-tuning." | Full fine-tuning of all 303,306,757 encoder parameters, asserted at runtime, in addition to a partial-FT arm. |
| "Five seeds is too few." | Acknowledged and quantified: the sign-flip floor at five seeds is 0.0625 and is reported. The formal test is the paired seed-level *t*-test; ten of ten seed-level interactions are negative. |
| "You tuned on the target." | Selection, early stopping and temperature are all on source validation. Target blindness during APTOS is auditable in the git history. |
| "EyePACS was in RETFound's pretraining." | Disclosed. EyePACS is a *source* domain, never a held-out target; it applies identically to frozen and full arms and so cannot generate the interaction. |
| "One architecture, one budget." | Stated as a scope limit, not defended. |
