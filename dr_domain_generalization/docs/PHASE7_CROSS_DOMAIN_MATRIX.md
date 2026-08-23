# Phase 7 — The full cross-domain matrix, and what it does to the multi-source story

**Run:** ERM, DenseNet121, batch 32, 20 epochs, **3 seeds (42, 1, 2)**
**Date:** 2026-08-21 · 3h04m wall clock (59 min × 3)
**Protocol:** train on one domain, evaluate on every other. Four models, twelve
cross-domain evaluations. Combined with the four in-domain runs this completes
every (train-domain, test-domain) cell.

> **Updated with 3 seeds.** §0 below carries the final numbers and supersedes
> the single-seed tables in §1–§2, which are kept for the record. The §5 caveat
> has been resolved.
>
> **⚠ CORRECTION 2026-08-22.** §4 and §6 attributed EyePACS's low in-domain score
> to label noise. A 512 px re-run of the identical in-domain protocol reaches
> **0.8004** against 0.7090 at 224 px, on the identical test images (Phase 6
> §0a). Resolution, not label quality, set that number. §4 and §6 are corrected
> in place; the superseded wording is quoted so the error stays on the record.
> **Everything in this document is a 224 px measurement** — the 512 px matrix is
> running, and until it lands no number here may be differenced against one.

---

## 0. Three-seed results (supersedes §1–§2)

36 cross-domain evaluations: 4 sources × 3 targets × 3 seeds.

### 0.1 The matrix, mean ± SD

| train \ test | DDR | APTOS | IDRiD | EyePACS |
|---|---|---|---|---|
| **DDR** | *in-domain* | 0.774 ± 0.008 | 0.588 ± 0.011 | 0.367 ± 0.010 |
| **APTOS** | 0.566 ± 0.005 | *in-domain* | 0.703 ± 0.019 | 0.431 ± 0.007 |
| **IDRiD** | 0.517 ± 0.022 | 0.510 ± **0.088** | *in-domain* | 0.239 ± 0.017 |
| **EyePACS** | 0.731 ± 0.004 | 0.855 ± 0.004 | 0.750 ± 0.039 | *in-domain* |

### 0.2 Multi-source vs the best single source

| Target | 3-source | best single | source | Δ | pooled SD | verdict |
|---|---|---|---|---|---|---|
| DDR | 0.738 ± 0.006 | 0.731 ± 0.004 | EyePACS | +0.007 | 0.005 | 3-source, marginal (1.4×) |
| APTOS | 0.859 ± 0.005 | 0.855 ± 0.004 | EyePACS | +0.004 | 0.005 | within seed noise |
| IDRiD | 0.741 ± 0.024 | 0.750 ± 0.039 | EyePACS | −0.009 | 0.033 | within seed noise |
| EyePACS | 0.415 ± 0.008 | 0.431 ± 0.007 | APTOS | **−0.016** | 0.007 | **single better (2.3×)** |

**Multi-source training does not consistently help, and on the hardest target it
hurts.** Two targets are indistinguishable. DDR's +0.007 clears its pooled SD by
only 1.4× — and EyePACS is 89% of that pool, so the entire measured benefit of
"three domains" there is what 3,144 extra images from DDR and IDRiD buy. On
EyePACS, adding DDR and IDRiD to APTOS makes the model **worse** by 2.3× the
pooled SD, and the paired bootstrap on 35,108 images agrees at p < 0.0001,
surviving Holm–Bonferroni across the whole 13-comparison family.

### 0.3 What is now settled

- **Source identity and scale dominate; source count does not.** The EyePACS row
  spans 0.73–0.86; the IDRiD row spans 0.24–0.52. That range dwarfs every
  multi-source effect in §0.2.
- **The severe-error advantage of APTOS-only on EyePACS holds**: 0.172 vs 0.241
  for three sources, with disjoint ranges across seeds.
- **The IDRiD row cannot be quoted cell-by-cell.** IDRiD→APTOS has an SD of
  **0.088** across three seeds (0.597 / 0.511 / 0.421) — sixteen times
  DDR→APTOS's 0.008 on the same test set. 335 training images do not determine
  a model.

### 0.4 ⚠ Correction to the single-seed reading

Seed 42 alone showed EyePACS→IDRiD at 0.756 against 3-source 0.741, which read
as single-source winning by +0.015. Across three seeds the single-source mean is
**0.750 ± 0.039** against 0.741 ± 0.024 — a difference of 0.009 against a pooled
SD of 0.033. **Not established**, and the apparent advantage was noise on the
project's most volatile target.

---

## 1. The matrix (seed 42 — superseded by §0)

Target QWK. Rows = trained on, columns = tested on. Diagonal is in-domain
(Phase 6); off-diagonal is single-source external.

| train \ test | DDR | APTOS | IDRiD | EyePACS | n_train |
|---|---|---|---|---|---|
| **DDR** | *0.876* | 0.782 | 0.600 | 0.372 | 8,697 |
| **APTOS** | 0.568 | *0.909* | 0.723 | 0.433 | 2,809 |
| **IDRiD** | 0.516 | 0.597 | *0.588* | 0.236 | 335 |
| **EyePACS** | **0.731** | **0.856** | **0.756** | *0.709* | 24,574 |

Two structures are immediate. **EyePACS is the hardest column** — no source
exceeds 0.44 against a 224 px in-domain reference of 0.709 (0.8004 at 512 px;
see the correction above). **IDRiD is the weakest row**,
which is its 335-image budget rather than anything about the domain.

**Transfer is directional.** DDR→IDRiD 0.600 against IDRiD→DDR 0.516;
DDR→APTOS 0.782 against APTOS→DDR 0.568. The useful direction runs from the
larger, more heterogeneous source, not symmetrically.

## 2. Finding: multi-source training buys essentially nothing

| Target | 3-source pool | 3-source QWK | best single source | Δ |
|---|---|---|---|---|
| DDR | 27,718 | 0.738 ± 0.006 | EyePACS → 0.731 | +0.007 |
| APTOS | 33,606 | 0.859 ± 0.005 | EyePACS → 0.856 | +0.003 |
| IDRiD | 36,080 | 0.741 ± 0.024 | EyePACS → **0.756** | −0.015 |
| EyePACS | 11,841 | 0.415 ± 0.008 | APTOS → **0.433** | −0.018 |

**One source matches or beats three on all four targets.** The largest advantage
from combining domains is +0.007, which is inside the seed SD everywhere.

## 3. Why: the pools were mostly EyePACS

| Target | 3-source pool | EyePACS share |
|---|---|---|
| DDR | 27,718 | **89%** |
| APTOS | 33,606 | **73%** |
| IDRiD | 36,080 | **68%** |
| EyePACS | 11,841 | 0% |

The "multi-source" models were largely EyePACS models. Adding the other domains
contributes 11–32% more images and moves QWK by less than run-to-run noise.

**Consequence for interpretation.** Phase 5's LODO results are not a measurement
of source diversity. They are close to a measurement of *training on EyePACS*,
plus a small perturbation. Any claim of the form "combining clinical datasets
improves generalization" is unsupported by this data — what improved things was
using the single largest source.

**Source identity and scale matter; source diversity does not.** That runs
against the default assumption in the DG literature and is worth stating plainly.

## 4. Finding: the lowest-scoring dataset is the best source

EyePACS scores **0.709 in-domain at 224 px** — the lowest of the four, and the
dataset the Phase-1 audit found 50,070 corrupted labels in. It is nonetheless
the best source for *every* other domain.

Trained on EyePACS and tested on DDR it scores **0.731, above its own
validation score of 0.712** (both at 224 px, so the comparison is internally
consistent). The same model beats three-source training on every target.

> **Corrected.** This section was previously headed *"the worst-labelled dataset
> is the best source"* and explained the effect as: *"because DDR's labels are
> cleaner than the ones it was trained on … label noise in a source depresses
> measured performance on that source without proportionally degrading what the
> model learns."*
>
> **The mechanism is not established.** Phase 6 §0a shows EyePACS's own score
> rises to 0.8004 at 512 px, so most of what looked like a label-noise floor was
> a resolution limit. A competing explanation now fits equally well: EyePACS
> images are simply *harder at 224 px* — more small lesions, more variable
> capture quality — so the model scores worse on them than on DDR's cleaner
> captures, with label quality doing none of the work.
>
> Distinguishing the two needs DDR at 512 px as well. If DDR gains far less than
> EyePACS's +0.091, the gap was resolution; if both gain alike, the residual
> difference is something else. **That run is queued and the question is open.**

What survives is the practically useful part, which does not depend on the
mechanism: **the dataset with the lowest in-domain score was the most valuable
source for every other domain.** For corpus assembly that still inverts the
instinct to discard the dataset that scores badly on itself — a source's own
score is a poor guide to its worth as training data, whatever the reason it
scores badly.

## 5. What one seed could not settle — RESOLVED in §0

Three of the four Δ values in §2 (+0.007, +0.003, −0.015) are smaller than the
LODO across-seed SD for that target. Single-source variance is **not measured at
all** in this phase. Therefore:

- **Supported:** multi-source training shows no advantage over the best single
  source on any target.
- **Not supported:** that single-source is *better*. The two negative Δ values
  (−0.015 IDRiD, −0.018 EyePACS) are not established.
- **Not supported:** any ranking among sources separated by less than ~0.03.

**Resolved.** Seeds 1-2 landed; see §0. Two of the three uncertain deltas turned out to be noise (APTOS +0.004, IDRiD -0.009); the EyePACS one (-0.016) survived and is now supported by both bars.

## 6. Bounding the EyePACS size confound

Phases 5 and 6 could not separate domain shift from training-set size for
EyePACS. Three points now exist:

| Training set | n_train | EyePACS QWK |
|---|---|---|
| DDR only (cross-domain) | 8,697 | 0.372 |
| DDR+APTOS+IDRiD (cross-domain) | 11,841 | 0.415 |
| EyePACS's own (in-domain) | 24,574 | 0.709 |

A 36% increase in cross-domain data bought **+0.043**. Extrapolating
log-linearly, reaching the in-domain budget of 24,574 predicts roughly **+0.10**,
landing near 0.52 — against the 0.709 achieved in-domain **at 224 px**.

On that basis, of the 0.294 gap roughly **one third is training-set size and two
thirds is domain shift**. This is a bound from two points under a log-linear
assumption at one seed, not a measurement.

> **Corrected — the reference point moved.** This decomposition uses the
> in-domain score as its upper anchor, and Phase 6 §0a raised that anchor from
> 0.709 to **0.8004** at 512 px. Against the 512 px anchor the gap is 0.385
> rather than 0.294, so the *same* size extrapolation of +0.10 covers a smaller
> share: roughly **one quarter size, three quarters shift**.
>
> Neither split should be quoted yet. The 0.385 version mixes a 512 px anchor
> with 224 px cross-domain points, which charges the resolution change to domain
> shift — exactly the error `run_in_domain.py` now refuses to commit. **The
> honest statement today is that the size/shift split is bounded somewhere
> between one quarter and one third size**, and the 512 px LODO matrix will
> settle it at a single resolution.

## 7. Calibration across the matrix

The weakest models are the best calibrated. IDRiD→EyePACS reaches ECE **0.068**
after temperature — the best post-scaling calibration of any cross-domain cell —
at QWK **0.236**, barely above ordinal chance.

**Good calibration is not evidence of a usable model.** It means only that
stated uncertainty matches accuracy; here the model accurately reports that it
does not know. Calibration must be read alongside discrimination, never instead
of it.

A second observation: single-source models are roughly twice as miscalibrated as
three-source models, and a source-fitted temperature that *worked* on APTOS in
the 3-source setting (ECE → 0.040) fails in the single-source setting (→ 0.111).
The number of training domains affects whether post-hoc recalibration transfers
at all, not only how accurate the model is.

## 8. Corrections issued during this phase

Two statements made while results were arriving were wrong and are withdrawn:

1. *"Multi-source wins every time, by 0.04–0.14."* Based on the DDR row alone.
   Falsified by the APTOS and EyePACS rows.
2. *"Source diversity helps or hurts depending on the pair."* Closer, but wrong
   about the mechanism. The gaps were measuring *which* source, not *how many*.
3. *"The worst-labelled dataset is the best source"* and the label-noise
   mechanism in §4 and §6. Falsified as a mechanism by the 512 px in-domain
   result (Phase 6 §0a); the observation survives, the explanation does not.

The correct statement is §3.

## 9. Artifacts

| Path | Contents |
|---|---|
| `outputs/tables/single_source_results.csv` | 12 cross-domain rows |
| `outputs/tables/selective_prediction_erm.csv` | abstention analysis, 4 targets × 3 seeds |
| `outputs/tables/risk_coverage_curves_erm.csv` | full risk–coverage curves |
| `outputs/logs/single_source_erm_s42.log` | training log |

Reproduce: `python run_single_source.py` (~1 h, no target ever seen in training —
asserted for all 12 pairs).
