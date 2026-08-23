# Phase 5 — The full leave-one-domain-out matrix

**Run:** ERM, DenseNet121, batch 32, 20 epochs, **3 seeds (42, 1, 2)**
**Date:** 2026-08-20 · 2h24m wall clock (2288 + 2317 + 2946 + 829 s)
**Protocol:** train on three domains, test on the fourth. The held-out domain
contributes test data only — never training, validation, early stopping, model
selection, or temperature fitting (asserted in `src/data/splits.py`).

> **Superseded in part.** Sections 1–6 below report seed 42 alone and are kept
> for the record. **Section 0 supersedes them** with 3-seed means ± SD.
> Two numbers changed enough to matter; see §0.3.

---

## 0. Three-seed results (supersedes §1)

12 runs, 2026-08-21, 5h07m wall clock. All four targets × seeds 42, 1, 2.

### 0.1 Per-target, mean ± SD

| Target | n_test | target QWK | target ECE | after T | severe |
|---|---|---|---|---|---|
| DDR | 12,424 | 0.7383 ± 0.0055 | 0.1296 ± 0.0083 | 0.0472 ± 0.0068 | 0.1598 ± 0.0061 |
| APTOS | 3,504 | 0.8590 ± 0.0054 | 0.1239 ± 0.0137 | 0.0397 ± 0.0076 | 0.0628 ± 0.0059 |
| IDRiD | 507 | 0.7413 ± **0.0244** | 0.2294 ± 0.0151 | 0.1213 ± 0.0176 | 0.1177 ± 0.0060 |
| EyePACS | 35,108 | 0.4147 ± 0.0081 | 0.2846 ± 0.0074 | 0.1729 ± 0.0088 | 0.2407 ± 0.0046 |

Per-seed target QWK, so a single divergent run stays visible:

| Target | seed 42 | seed 1 | seed 2 | range |
|---|---|---|---|---|
| DDR | 0.7334 | 0.7373 | 0.7443 | 0.0109 |
| APTOS | 0.8591 | 0.8644 | 0.8537 | 0.0107 |
| IDRiD | 0.7543 | 0.7132 | 0.7566 | **0.0434** |
| EyePACS | 0.4077 | 0.4128 | 0.4237 | 0.0159 |

### 0.2 Validation performance does not predict generalization

On IDRiD the three runs are **indistinguishable on source validation** and
differ substantially on the target:

| | source val QWK | target QWK |
|---|---|---|
| seed 1 | 0.8034 | 0.7132 |
| seed 2 | 0.8030 | 0.7566 |
| seed 42 | 0.8017 | 0.7543 |
| **SD** | **0.0009** | **0.0244** |

**A ratio of 28×.** Three checkpoints that cannot be told apart at selection
time — validation QWK agreeing to the third decimal — produce unseen-domain
scores spanning 0.043. Under this protocol, model selection may only use source
validation, so selection is close to blind with respect to the quantity that
matters. This is the strongest available argument for the two-bar criterion, and
it explains why single-seed DG comparisons are fragile.

### 0.3 ⚠ Corrections to the single-seed numbers

| Target | source-gap ΔQWK (seed 42) | 3-seed mean ± SD |
|---|---|---|
| DDR | −0.0186 | **−0.0105 ± 0.0090** |
| APTOS | +0.0625 | +0.0657 ± 0.0028 |
| IDRiD | −0.0475 | **−0.0614 ± 0.0250** |
| EyePACS | −0.4832 | −0.4709 ± 0.0113 |

**DDR's source gap was overstated by 77%.** Quote −0.011, not −0.019. This
strengthens the thesis rather than weakening it: QWK falls even less than
reported while ECE still rises 0.076 → 0.130.

**IDRiD's −0.061 ± 0.025 is not established.** The SD is 41% of the effect.

### 0.4 The prior-shift artifact is reproducible

APTOS, all three seeds, same predictions:

| seed | ΔQWK | ΔF1 |
|---|---|---|
| 1 | +0.0676 | −0.1062 |
| 2 | +0.0671 | −0.1040 |
| 42 | +0.0625 | −0.0894 |
| **mean ± SD** | **+0.0657 ± 0.0028** | **−0.0999 ± 0.0092** |

QWK rises 23× its own noise while macro-F1 falls 11× its noise, in opposite
directions, every time. Reported alone, QWK would support the false claim that
the model generalises to unseen APTOS better than to held-in data.

### 0.5 Temperature scaling: what replicates

| Target | source ECE | target ECE after T | corrected? |
|---|---|---|---|
| DDR | 0.076 | 0.047 ± 0.007 | yes, all 3 seeds |
| APTOS | 0.060 | 0.040 ± 0.008 | yes, all 3 seeds |
| IDRiD | 0.073 | 0.121 ± 0.018 | **no, all 3 seeds** |
| EyePACS | 0.082 | 0.173 ± 0.009 | **no, all 3 seeds** |

No seed reverses any of these four verdicts.

---

## 1. Headline (seed 42 — superseded by §0)

| Target | n_train | n_test | src QWK | tgt QWK | ΔQWK | src ECE | tgt ECE | after T | severe |
|---|---|---|---|---|---|---|---|---|---|
| DDR | 27,718 | 12,424 | 0.7520 | 0.7334 | −0.019 | 0.077 | 0.138 | **0.055** | 0.167 |
| APTOS | 33,606 | 3,504 | 0.7966 | 0.8591 | +0.063 | 0.046 | 0.108 | **0.041** | 0.062 |
| IDRiD | 36,080 | 507 | 0.8017 | 0.7543 | −0.048 | 0.074 | 0.216 | 0.105 | 0.116 |
| EyePACS | 11,841 | 35,108 | 0.8909 | **0.4077** | **−0.483** | 0.082 | 0.289 | 0.180 | 0.245 |

Bootstrap 95% CIs (2,000 resamples) in `outputs/tables/lodo_erm_s42_bootstrap_ci.csv`:

| Target | QWK | macro F1 | ECE | MAE |
|---|---|---|---|---|
| DDR | 0.7334 [0.7225, 0.7453] | 0.5083 [0.4960, 0.5210] | 0.1383 [0.1310, 0.1460] | 0.468 |
| APTOS | 0.8591 [0.8475, 0.8707] | 0.5047 [0.4874, 0.5218] | 0.1082 [0.0949, 0.1227] | 0.343 |
| IDRiD | 0.7543 [0.7054, 0.7976] | 0.4899 [0.4446, 0.5340] | 0.2159 [0.1785, 0.2572] | 0.544 |
| EyePACS | 0.4077 [0.3967, 0.4177] | 0.3758 [0.3653, 0.3863] | 0.2887 [0.2837, 0.2937] | 0.675 |

**Do not quote a mean over these four.** It is −0.122, and the range is −0.483
to +0.063. No target is near the mean; it describes none of them.

---

## 2. Finding: calibration degrades even where accuracy does not

DDR is the clean case. QWK falls by 0.019 — a model deployed on unseen DDR
performs almost as well as on its own validation data. But ECE **doubles**,
0.077 → 0.138. A study reporting only discrimination would conclude DDR
transfer is essentially solved, and would ship a model whose confidence is
twice as wrong as it was in development.

This is the project's thesis and it holds on every target: **ECE degrades on
all four, including the one where QWK barely moves.**

## 3. Finding: temperature scaling transfers — until it doesn't

The temperature is fitted **only on source validation**; it never sees the
target. On DDR and APTOS it pulls target ECE *below* the source-domain ECE
(0.055 vs 0.077; 0.041 vs 0.046). There, cross-domain miscalibration is
essentially one global confidence offset, and a scalar removes it.

It breaks down as the shift grows:

| Target | src ECE | tgt ECE after T | verdict |
|---|---|---|---|
| DDR | 0.077 | 0.055 | fully corrected |
| APTOS | 0.046 | 0.041 | fully corrected |
| IDRiD | 0.074 | 0.105 | **partial** — 42% above source |
| EyePACS | 0.082 | 0.180 | **fails** — 120% above source |

So "recalibrate on source data before deployment" is sound advice for mild
shift and false advice for severe shift — and it fails precisely where the
guarantee matters most. A practitioner cannot tell which regime they are in
without target labels, which is the situation the method is meant to cover.

## 4. Finding: QWK and macro-F1 disagree under prior shift

APTOS scores **higher** than its own source validation (+0.063 QWK), which
would read as generalising better to an unseen domain than to held-in data.
Macro-F1 on the same predictions **falls** by 0.089.

The cause is the label distribution:

| | g0 | g1 | g2 | g3 | g4 |
|---|---|---|---|---|---|
| APTOS (target) | 51.3% | 9.6% | 26.3% | 5.1% | 7.7% |
| its source val | 66.7% | 6.6% | 20.7% | 2.6% | 3.5% |

QWK normalises by *expected* disagreement, which grows when the true-label
marginal is wider. The same per-image error quality therefore scores higher on
APTOS. The +0.063 is largely a **prior-shift artifact**, not better transfer.

**Methodological consequence: QWK alone would have supported a false
generalisation claim.** Report macro-F1 alongside it, and print class support.

## 5. Finding: the middle grades collapse

Per-class recall on each unseen target (%):

| Grade | DDR | APTOS | IDRiD | EyePACS |
|---|---|---|---|---|
| 0 — none | 95.2 | 91.8 | 61.1 | 63.0 |
| **1 — mild** | **7.6** | **1.8** | 47.8 | **13.2** |
| 2 — moderate | 51.0 | 69.1 | 81.2 | 68.2 |
| **3 — severe** | 47.5 | 53.1 | **12.1** | **14.8** |
| 4 — proliferative | 74.6 | 53.9 | 63.9 | 35.5 |

Grade 1 recall is **1.8% on APTOS** — the model essentially never identifies
mild NPDR there. Grade 3 recall is 12–15% on IDRiD and EyePACS. The endpoints
survive; the ordinal interior does not. This mirrors the Stage-C ablation,
where the ordinal head's ECE gain came with collapse of exactly grades 1 and 3.

Clinically this is the worst possible error pattern: grade 1 and 3 are the
decision boundaries for referral timing.

## 6. EyePACS is the worst target — and the result is confounded

QWK 0.408, severe-error rate 24.5%: **one image in four is misgraded by two or
more levels.** This is not a deployable model by any reading.

Three explanations are entangled and this run cannot separate them:

1. **Domain shift** — the honest hypothesis, and the one the paper is about.
2. **Training-set size** — holding out EyePACS leaves only **11,841** source
   images, a third of what the other targets get. Every other target trains on
   27k–36k.
3. **Label noise** — EyePACS grading has high inter-reader variability, and
   this project already excluded 50,070 of its labels as corrupted (see
   `docs/DATA_PROVENANCE.md`). The 35,108 retained were verified against the
   official `trainLabels.csv` at 100% agreement, so the *test* labels are
   sound, but the underlying grading protocol is still the noisiest of the four.

**Do not attribute the EyePACS collapse to domain shift alone.** The
single-source-external experiment (`NOT_RUN`) would separate cause 2 by varying
the training pool size at fixed target.

---

## 7. What this does not establish

- **Anything at n=1 within ±0.032.** The DDR (−0.019) and IDRiD (−0.048)
  deltas are inside or near ERM's measured across-seed SD.
- **That adding EyePACS as a third source helps IDRiD.** Stage C (DDR+APTOS →
  IDRiD) gave 0.7235 ± 0.032; this run (DDR+APTOS+EyePACS → IDRiD) gave 0.7543.
  The +0.031 difference is exactly the size of the noise floor.
- **The cost of cross-domain deployment.** Every Δ here compares source
  validation against the target — different images from different domains. The
  clinically meaningful comparison is against a model *trained on* the target
  domain, and no such model exists yet (`in_domain` is `NOT_RUN`). When it does,
  the LODO side must be restricted to the target's held-out test split, or the
  comparison is unfair.

## 8. Artifacts

| Path | Contents |
|---|---|
| `outputs/tables/lodo_results.csv` | one row per target |
| `outputs/tables/lodo_erm_s42_headline.csv` | headline table above |
| `outputs/tables/lodo_erm_s42_bootstrap_ci.csv` | 2,000-resample CIs |
| `outputs/tables/lodo_erm_s42_class_support.csv` | true vs predicted support |
| `outputs/figures/lodo_summary_qwk.png` | source vs target, 4 targets |
| `outputs/figures/domain_matrix_{qwk,ece}_erm_s42.png` | cross-domain matrices |
| `outputs/logs/lodo_erm_s42.log` | full training log |

Figures are built from exactly the four runs in the table — filtered by method
*and* seed, so no cell is a mean over experiments.
