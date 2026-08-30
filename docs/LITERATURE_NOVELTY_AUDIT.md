# Literature and novelty audit

**Date:** 2026-08-30 · **Status:** first pass, sources verified where accessible
**Purpose:** establish what may and may not be claimed before any novelty wording
enters the manuscript.

**Verification status.** The February 2026 Nature Communications paper has now
been checked against the full article; the fields previously marked *unverified*
are confirmed below and the marker removed. RETFound-Green has been read for its
adaptation-protocol comparison. Remaining gaps are listed in §4.

---

## 1. The paper that constrains us most

**Understanding pre-training data effects in retinal foundation models using two
large fundus cohorts** — *Nature Communications*, February 2026.
<https://www.nature.com/articles/s41467-026-70077-z>

| aspect | what it does | verified? |
|---|---|---|
| Models | FM-MEH and FM-SDPP — two foundation models pretrained with identical pipelines on two retinal cohorts (Moorfields, 904,170 images; Shanghai DPP, 904,170) | yes |
| Adaptation | **Both** full encoder fine-tuning (all model parameters tuned) **and** linear probing (all parameters frozen, one linear classifier trained) | yes |
| Tasks | DR detection, diabetic macular oedema, ischaemic stroke | yes |
| Metrics | AUROC, AUPRC | yes |
| Evaluation | held-out MEH and SDPP data plus public datasets | yes |
| Seeds | **five random seeds**, mean and SD reported across them | yes |
| Statistics | **two-sided Welch $t$-test with Holm–Bonferroni correction** | yes |
| Downstream public datasets | include **APTOS2019 and IDRiD** — the same cohorts we hold out | yes |
| Lineage-matched control | **No** — compares two *retinal* models to each other | yes |
| Contribution | pre-training data demographics shape generalisability and fairness (age gaps; sex and ethnicity minimal) | yes |

### What this forbids

The following claims are **not available** and must not appear:

- "first to compare linear probing and fine-tuning" — they do both, explicitly,
  with five seeds, Welch $t$-tests and Holm correction
- "first to show adaptation strategy changes conclusions" — their linear-probe
  results already differ from their fine-tuned results by subgroup
- "first to evaluate on APTOS or IDRiD under domain shift" — both are among
  their downstream public datasets
- any unqualified "first", "novel", "unprecedented", or "no prior work" about
  protocol-dependent evaluation of retinal foundation models

**RETFound-Green also compares protocols**, setting a linear probe of
RETFound-Green against a fully fine-tuned RETFound-MEH and reporting broadly
comparable performance. That is a protocol comparison, so it further forecloses
any general novelty claim about comparing probing with fine-tuning. It differs
from our design in the same way the 2026 paper does: it varies the *model* and
the *protocol* together, so it cannot isolate what either contributes. Ours
holds the model pair fixed by construction — the same weights before and after
one pretraining stage — and varies only the protocol.

### What it leaves open

They compare **retinal cohort A against retinal cohort B**. Neither model is
compared against the general-purpose checkpoint it was built from, so their
design cannot answer *what the retinal pretraining added relative to its own
starting point*. That is the gap our experiment occupies.

---

## 2. Comparison matrix

| Paper | Datasets | Backbone / FMs | Pretraining comparison | Adaptation protocols | External evaluation | Seeds | Statistics | Domain-shift design | Main contribution | Overlap with us | What remains ours |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **RETFound** (Nature 2023) | MEH-MIDAS + EyePACS pretrain; multiple downstream | ViT-L/16 MAE | vs ImageNet-supervised and SSL baselines | fine-tuning (primary) | yes | not seed-focused | AUROC CIs | internal/external splits | a retinal FM improves downstream ocular and systemic prediction | our reference model | it does not compare against its own MAE init |
| **Pre-training data effects** (Nat Commun 2026) | MEH, SDPP, public | two parallel retinal FMs | retinal cohort A vs B | **full FT + linear probe** | yes | 5 | Welch $t$-test + Holm | demographic subgroups | pretraining demographics drive fairness | **protocol comparison — closest work** | lineage-matched control; ordinal DR grading; LODO shift |
| **RETFound-Green** (Nat Commun 2025) | retinal | efficient retinal FM (half data, ~400× less compute) | vs RETFound-MEH | **linear probe (Green) vs full fine-tuning (RETFound-MEH)** | yes | — | — | — | a cheaper FM reaches broadly comparable performance | **compares protocols across models** | it varies model *and* protocol together; ours holds both fixed around one pretraining stage |
| **CauDR** (Comput Biol Med 2024) | 4 public DR sets | CNN | — | supervised | yes | — | — | causal DG benchmark | causal DG for DR | DR + multi-dataset DG | no FM; no protocol comparison |
| **Phase-augmentation DG for DR** (2026) | public DR sets | CNN | — | supervised | yes | — | — | augmentation-based DG | DG method for DR grading | DR + DG | no FM; no protocol comparison |
| **DomainBed** (Gulrajani & Lopez-Paz 2021) | DG benchmarks | ResNet | — | supervised | yes | multiple | careful model selection | canonical DG | DG methods do not beat ERM | our DG null replicates it | medical modality; FM protocol question |
| **This work** | DDR, APTOS, IDRiD, EyePACS | RETFound vs **its own ImageNet-MAE init**; DenseNet121; ConvNeXt | **lineage-matched** | linear probe + partial FT (+ full FT pending) | leave-one-dataset-out | **10 frozen / 5 partial FT** | crossed seed×case bootstrap (intervals) + paired $t$-test with Holm (inference) | 4 datasets as domains, LODO | protocol can change the ranking of a lineage-matched pair | — | see §3 |

---

## 3. The narrow claim we can defend

> An exact lineage-matched comparison of a retinal foundation model against the
> precise general-purpose checkpoint from which it was constructed, under
> identical downstream protocols, on unseen-domain ordinal DR grading — showing
> whether the ranking of that pair depends on the adaptation protocol.

Four components, each of which we have checked is not jointly present above:

1. **Lineage-matched.** RETFound's own checkpoint records
   `resume='./mae_pretrain_vit_large_full.pth'`, and that exact checkpoint is
   public as `vit_large_patch16_224.mae`. Architecture, parameter count,
   objective and initialisation are therefore identical by construction, and
   the intervention is the additional retinal-domain MAE pretraining stage
   itself. No paper in §2 does this.
2. **Cross-dataset ordinal grading**, 5-class ICDR with QWK and severe-error
   rate, rather than binary detection with AUROC.
3. **Leave-one-dataset-out** shift across four public cohorts, with the
   pretraining-contaminated dataset excluded as a target by construction.
4. **Seed-aware inference**: ten seeds frozen, five partial FT, crossed
   seed × case bootstrap with Holm correction.

**Wording that is permitted:** "to our knowledge, no prior work compares a
retinal foundation model against the exact general-purpose checkpoint from
which it was initialised". This is a *narrow* to-our-knowledge claim about a
specific experimental design, not a claim about protocol comparison in general.

**Wording that is forbidden:** anything implying we are first to compare
probing with fine-tuning, or first to observe protocol-dependent conclusions.

---

## 4. Outstanding before submission

- [ ] Obtain the Nat Commun 2026 full text; confirm seed count, multiplicity
      method, and whether any lineage-matched control appears in supplementary
- [ ] Read RETFound-Green in full — it compares against RETFound and may
      contain an initialisation ablation
- [ ] Search for PEFT/LoRA-vs-probing comparisons in retinal imaging published
      after 2026-01
- [ ] Confirm CauDR and the phase-augmentation paper do not include an FM arm
- [ ] Re-run this audit immediately before submission; the field is moving fast
      enough that a three-month-old audit is not evidence
