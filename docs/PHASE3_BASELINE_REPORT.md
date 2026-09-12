# Phase 3 — Baseline result: DDR + APTOS → IDRiD

> **HISTORICAL PROJECT RECORD — NOT AUTHORITATIVE FOR THE FINAL MANUSCRIPT.**
> See [`JBHI_EVIDENCE_FREEZE.md`](JBHI_EVIDENCE_FREEZE.md) and
> [`JBHI_DRAFTING_MANIFEST.md`](JBHI_DRAFTING_MANIFEST.md) for the frozen final
> state. Statements below were true at the time they were written and may have
> been superseded — including older test and audit counts, the retired
> "two-bar" terminology, and novelty framing that has since been narrowed.


**Date:** 2026-08-19
**Experiment ID:** `lodo_aptos-ddr__idrid_densenet121_erm-none_s42`
**Reproduce with:** `research_pipeline.py`, Cells 13b–18b
**Artefacts:** `outputs/experiment_registry.csv`, `outputs/tables/`, `outputs/predictions/`, `outputs/figures/`

> **Scope.** This is **one run, one seed, one backbone, one target domain.** It is
> a working baseline that shows the pipeline produces meaningful numbers — not a
> paper result. No ablation, no multi-seed variance, no full leave-one-domain-out
> sweep, no DG method has been run yet.

---

## 1. Setup

| | |
|---|---|
| Protocol | Leave-one-domain-out (Stage C) |
| Source domains | DDR + APTOS — 11,506 train / 2,206 val |
| **Unseen target** | **IDRiD — 507 images, never seen in training, validation, early stopping or calibration** |
| Backbone | DenseNet121 (ImageNet), 7.0M parameters |
| Loss | Cross-entropy, no imbalance handling (`erm-none`) |
| Optimiser | AdamW, lr 3e-4, wd 1e-4, cosine + 1 warmup epoch |
| Batch / precision | 16, AMP fp16, grad-clip 1.0 |
| Epochs | 20 (early-stopping patience 6; **did not trigger**) |
| Model selection | Best **source-validation** QWK → epoch 18 (val QWK 0.8787) |
| Wall clock | 1,111 s (~55 s/epoch), peak 1.12 GB VRAM |

---

## 2. The headline result — accuracy degrades, calibration degrades more

95% bootstrap CIs, 2,000 resamples, seed 42.

| Metric | Source validation (n=2,206) | **Unseen IDRiD (n=507)** | Gap |
|---|---|---|---|
| **QWK** | 0.8787 [0.8618, 0.8948] | **0.6823 [0.6177, 0.7383]** | **−0.196** |
| Macro F1 | 0.6683 [0.6337, 0.7012] | 0.4323 [0.3870, 0.4752] | −0.236 |
| Accuracy | 0.8454 | 0.4911 | −0.354 |
| Balanced accuracy | 0.6458 | 0.5221 | −0.124 |
| AUROC (macro) | 0.9443 [0.9359, 0.9522] | 0.8672 [0.8473, 0.8862] | −0.077 |
| MAE (grades) | 0.227 | 0.647 | +0.420 |
| Within ±1 grade | 0.933 | 0.892 | −0.041 |
| Severe error (≥2) | 0.067 | 0.108 | +0.041 |
| **ECE** | 0.0797 [0.0686, 0.0948] | **0.3456 [0.3049, 0.3875]** | **+0.266 (4.3×)** |
| NLL | 0.543 | 2.140 | +1.597 |
| Brier | 0.247 | 0.815 | +0.568 |

**Three observations, stated at the strength the evidence supports:**

1. **Discrimination degrades substantially.** QWK falls 0.196 (22% relative) and
   the confidence intervals do not overlap. Ranking ability survives better than
   thresholded decisions: AUROC drops only 0.077 while accuracy drops 0.354 —
   the model still orders severity reasonably, but its decision boundaries do
   not transfer.

2. **Calibration degrades disproportionately.** ECE rises 4.3× and NLL almost
   quadruples. The CIs are far apart ([0.069, 0.095] vs [0.305, 0.388]), so this
   is not sampling noise. The reliability diagram
   (`perf_..._target_test_reliability.png`) shows *every* confidence bin sitting
   below the diagonal — systematic over-confidence, not scattered error.

3. **This is exactly the motivating claim of the project**, and it now has a
   measurement behind it rather than an assumption. Whether ordinal learning,
   Deep CORAL, MixStyle or a combined method reduce the gap is still open — none
   has been run.

---

## 3. Temperature scaling helps, but cannot repair domain-induced miscalibration

Temperature was fitted on **source validation logits only** (T = 1.790), then
applied unchanged to IDRiD. The target domain influenced nothing.

| | Source val | Unseen IDRiD |
|---|---|---|
| ECE before | 0.0797 | 0.3456 |
| ECE after | **0.0203** | **0.2261** |
| Relative reduction | **−75%** | **−35%** |
| NLL after | 0.446 (from 0.543) | 1.444 (from 2.140) |

In-domain, temperature scaling works very well — a 75% ECE reduction. On the
unseen domain it recovers only 35%, and the residual ECE (0.226) is still **11×**
the calibrated in-domain value (0.020).

This is a useful negative result: **a single scalar fitted in-domain cannot
absorb the miscalibration that domain shift introduces.** It is consistent with
the shift changing the *shape* of the confidence–accuracy relationship, not just
its scale. It also motivates the rest of the study — if temperature scaling were
sufficient, the DG methods would have little to add.

*(Because temperature scaling divides all logits by a positive scalar, it cannot
change the argmax. QWK, F1 and accuracy are identical before and after by
construction — verified in `tests/test_phase3.py`.)*

---

## 4. Selective prediction — confidence ranks errors, but weakly off-domain

Rejecting the least-confident predictions on IDRiD:

| Coverage | n kept | Accuracy | QWK | Macro F1 | Severe-error rate |
|---|---|---|---|---|---|
| 100% | 507 | 0.4911 | 0.6823 | 0.4323 | 0.1085 |
| 95% | 482 | 0.5062 | 0.7127 | 0.4519 | 0.0975 |
| 90% | 456 | 0.5110 | 0.7184 | 0.4521 | 0.0965 |
| 80% | 406 | 0.5320 | 0.7446 | 0.4695 | 0.0788 |
| 70% | 355 | 0.5493 | 0.7618 | 0.4786 | 0.0704 |
| 50% | 254 | 0.6181 | 0.8077 | 0.5163 | 0.0472 |

- **AURC 0.3865**, **error-detection AUROC 0.6540**.
- Every metric improves monotonically as low-confidence cases are deferred, so
  confidence does carry information about correctness.
- But 0.654 is only modestly above chance (0.5). On a *held-out* domain the
  model's confidence is a weak error detector — which is the same finding as §2
  seen from another angle: an over-confident model separates its errors poorly.
- Deferring half the cases raises QWK from 0.682 to 0.808 — real, but it means
  half the workload returns to a clinician.

**This does not make the system clinically deployable**, and the figures are
labelled to say so. The deferred cases still need a human; the threshold would
have to be fixed prospectively rather than picked on the test set; and nothing
here addresses a confidently-wrong proliferative case.

---

## 5. Where the errors are

Per-class recall on IDRiD (see `perf_..._target_test_per_class.png` and the
confusion matrices):

- The collapse is concentrated in the **minority and mid-severity grades**.
  Balanced accuracy (0.522) falls much less than raw accuracy (0.491) because
  IDRiD's grade distribution is far flatter than the sources' — 32.9% no-DR
  versus 49–51% in DDR/APTOS, and 18.0% severe versus 1.9%.
- **Label shift and covariate shift are confounded here.** IDRiD is not merely a
  different camera; it has a different case mix. Part of the measured gap is
  prior shift, which a DG method targeting covariate shift would not fix. Any
  claim about *why* the gap exists needs that separated, and it is not yet.

Sample-level predictions for every image (`outputs/predictions/*.csv`) include
the five class probabilities, confidence, correctness and absolute grade error,
which is what the error-analysis galleries in Phase 4 will use.

---

## 6. Engineering results worth recording

| Finding | Value |
|---|---|
| Image cache | 51,543 images, 892 MB, 0 failures, 30 min |
| Cache fidelity (JPEG-95 vs PNG) | MAE 1.02/255, PSNR 44.4 dB, 3.3× smaller |
| Loader throughput (cached) | 576 / 1343 / 2234 img/s at 0 / 2 / 4 workers |
| GPU step throughput | DenseNet121 217 img/s, ConvNeXt-Tiny 317 img/s |
| **Bottleneck after caching** | **GPU, not I/O** — by ~2.7× even at 0 workers |
| Peak VRAM (batch 16, 224px, AMP) | **1.12 GB** of 8 GB |

Two consequences: raising `num_workers` past 2 buys nothing, and there is ample
VRAM headroom for 384px or larger batches — the 8 GB card is **not** the binding
constraint at 224px.

---

## 7. Bugs the tests caught

Worth recording because each would have corrupted results silently:

1. **`FocalLoss` was broken** — it passed `label_smoothing` to `F.nll_loss`,
   which does not accept it. Any focal-loss run would have crashed. Now uses
   `F.cross_entropy`.
2. **Experiment registry wrote empty timestamps** — `setdefault` on a key that
   already existed with value `None`. Every row would have been undated.
3. **Manifest schema rejected its own `split` column**, so a split manifest could
   not be reloaded.
4. **`estimate_retina_bbox` threshold was too permissive** (0.10 → 0.20), leaving
   IDRiD essentially uncropped.

The suite now has **128 tests**, including null cases: a model compared with
itself must not be significant, and random confidence must score at chance.

---

## 8. What is NOT done

Marked explicitly, per the honesty policy:

- `NOT RUN` — ConvNeXt-Tiny and DINOv2 backbones (implemented, never trained)
- `NOT RUN` — ordinal CORAL objective (implemented and unit-tested, never trained)
- `NOT RUN` — Deep CORAL and MixStyle (Deep CORAL implemented and unit-tested)
- `NOT RUN` — the other three leave-one-domain-out experiments
- `NOT RUN` — in-domain and single-source-external protocols
- `NOT RUN` — multi-seed runs; **everything above is a single seed**, so no
  claim about run-to-run variance is possible
- `NOT RUN` — ablation grid, combined method, embedding figures, error galleries

**No number anywhere in this repository is fabricated.** Every value in this
report traces to `outputs/experiment_registry.csv` and the CSVs beside it.
