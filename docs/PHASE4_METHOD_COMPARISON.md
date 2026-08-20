# Phase 4 — Method comparison: DDR + APTOS → unseen IDRiD

**Date:** 2026-08-20
**Protocol:** leave-one-domain-out, DenseNet121, 224px, batch 32, 20 epochs, **3 seeds (42, 1, 2)**
**Reproduce with:** `run_method_comparison.py --seeds 42,1,2`, then `analyse_seeds.py`, `analyse_method_comparison.py`, `analyse_embeddings.py`
**Artefacts:** `outputs/tables/stage_c_*.csv`, `outputs/experiment_registry.csv`, `outputs/figures/stage_c_*.png`

> **Headline: no method beat the ERM baseline on target discrimination.** Every
> one of the five alternatives is worse on target QWK, and the gap survives
> three seeds. The ordinal objective produced a large, robust improvement in
> calibration — but by collapsing two clinically important grades.
>
> **Now replicated over 3 seeds.** Section 0 supersedes the single-seed numbers
> in sections 2–3 and corrects one claim that did not survive.

---

## 0. Multi-seed replication (3 seeds) — supersedes §2–3

**Updated 2026-08-20.** All six methods re-run at seeds 1 and 2 (seed 42 already
existed), same split, same everything else. `outputs/tables/stage_c_seed_*.csv`.

A difference is called real here only if it clears **both** bars:

1. the gap exceeds the pooled **across-seed SD** (does it survive a different
   random initialisation?), and
2. its **paired bootstrap** interval on the 507 test images excludes zero (does
   it survive resampling the test set?).

These are different sources of uncertainty. Neither substitutes for the other,
and one claim below passes the second bar while failing the first.

### Mean ± SD over 3 seeds, unseen IDRiD (n=507)

| Method | Target QWK | Target macro F1 | Target ECE | Severe-error rate |
|---|---|---|---|---|
| **ERM** | **0.7235 ± 0.0320** | **0.4383 ± 0.0280** | 0.2973 ± 0.0304 | **0.1065 ± 0.0287** |
| Ordinal | 0.6545 ± 0.0111 | 0.3901 ± 0.0065 | **0.0610 ± 0.0068** | 0.2156 ± 0.0023 |
| Deep CORAL | 0.6743 ± 0.0158 | 0.4253 ± 0.0206 | 0.3264 ± 0.0347 | 0.1374 ± 0.0131 |
| MixStyle | 0.6617 ± 0.0197 | 0.4143 ± 0.0278 | 0.3122 ± 0.0229 | 0.1486 ± 0.0148 |
| MixStyle + Ordinal | 0.6596 ± 0.0170 | 0.3980 ± 0.0246 | 0.0632 ± 0.0091 | 0.2110 ± 0.0123 |
| Deep CORAL + Ordinal | 0.6778 ± 0.0068 | 0.4027 ± 0.0126 | 0.0745 ± 0.0182 | 0.2032 ± 0.0000 |

### Verdicts against ERM

| Method | ΔQWK | vs seed SD | paired CI | **verdict** |
|---|---|---|---|---|
| Ordinal | −0.069 | > 0.024 ✓ | [−0.095, −0.043] ✓ | **worse, real** |
| Deep CORAL | −0.049 | > 0.025 ✓ | [−0.074, −0.023] ✓ | **worse, real** |
| MixStyle | −0.062 | > 0.027 ✓ | [−0.092, −0.034] ✓ | **worse, real** |
| MixStyle + Ordinal | −0.064 | > 0.026 ✓ | [−0.090, −0.037] ✓ | **worse, real** |
| Deep CORAL + Ordinal | −0.046 | > 0.023 ✓ | [−0.072, −0.018] ✓ | **worse, real** |

**ERM's advantage on target QWK survives.** All five gaps clear both bars.

| Method | ΔECE | vs seed SD | paired CI | **verdict** |
|---|---|---|---|---|
| Ordinal | −0.236 | > 0.022 ✓ | [−0.273, −0.198] ✓ | **better, real** (10× the noise) |
| MixStyle + Ordinal | −0.234 | > 0.022 ✓ | [−0.271, −0.196] ✓ | **better, real** |
| Deep CORAL + Ordinal | −0.223 | > 0.025 ✓ | [−0.263, −0.187] ✓ | **better, real** |
| Deep CORAL | +0.029 | **< 0.033 ✗** | [+0.009, +0.049] ✓ | **NOT ESTABLISHED** |
| MixStyle | +0.015 | **< 0.027 ✗** | [−0.008, +0.036] ✗ | **NOT ESTABLISHED** |

### ⚠ Correction to the single-seed report

**§2 claimed "Deep CORAL significantly worsens calibration (+0.071, CI [+0.028,
+0.110])". That claim does not survive.** Across three seeds the mean effect is
+0.029 against a pooled across-seed SD of 0.033 — *within* seed noise. It still
passes the paired bootstrap (which resamples images, not seeds), which is
exactly why both bars are required. The honest statement is that Deep CORAL's
effect on calibration is **not resolvable** at this sample size.

The MixStyle F1 and Deep CORAL F1 deltas likewise fail the seed bar and are
withdrawn as findings.

Everything else in §2–3 holds, and two claims strengthen considerably:

- The ordinal ECE improvement is ~10× the across-seed noise — the most robust
  effect in the study.
- The **severe-error rate** result is unambiguous: ERM 0.107 ± 0.029 against
  0.216 ± 0.002 for the ordinal variants. Every method makes *more* clinically
  dangerous errors than ERM, and the gap clears both bars for all five.

### ERM is the best *and* the least stable

ERM has the largest across-seed SD of any method on QWK (0.0320, range
0.689–0.753); the ordinal variants sit at 0.007–0.017. So ERM wins on the mean
while being the most sensitive to initialisation. Two consequences:

- A single ERM run could land at 0.689 and appear to tie the alternatives. The
  single-seed Phase-3 baseline scoring 0.682 at batch 16 was not an outlier so
  much as a draw from a wide distribution.
- **Three seeds is the minimum here, not a formality.** With one seed the
  ordering of the middle four methods is not recoverable.

---

## 1. Why batch 32, when the Phase-3 baseline used 16

Deep CORAL estimates a per-domain feature covariance **within each batch**. At
the natural DDR/APTOS mix of 76/24, a batch of 16 leaves fewer than 4 APTOS
samples **43% of the time** — a degenerate covariance. At batch 32 that falls to
**3%** (measured, not estimated; the realised rate in training was 3.3%).

Batch size is therefore fixed at 32 for **every** method, including a re-run of
ERM, so the comparison is not confounded. All six runs share identical splits,
seed, preprocessing, augmentation, optimiser, schedule and epoch count.

*Side observation worth carrying forward:* ERM at batch 16 scored target QWK
0.682; at batch 32 it scored **0.753**. A single hyperparameter moved the target
metric by 0.07 — as much as the gap between methods below. On a 507-image target
split, these numbers are not stable to configuration.

---

## 2. Results (single seed -- superseded by section 0)

All metrics on the unseen IDRiD split (n=507). 95% bootstrap CIs, 2,000
resamples, seed 42.

| Method | Val QWK | **Target QWK** | Target macro F1 | **Target ECE** | ECE after T-scaling | Severe-error rate |
|---|---|---|---|---|---|---|
| **ERM** | 0.880 | **0.753** [0.705, 0.794] | **0.468** [0.424, 0.509] | 0.265 [0.224, 0.306] | 0.185 | **0.079** |
| Ordinal | 0.879 | 0.665 [0.610, 0.715] | 0.394 [0.356, 0.435] | **0.059** [0.043, 0.104] | 0.113 | 0.217 |
| Deep CORAL | 0.879 | 0.665 [0.603, 0.722] | 0.420 [0.376, 0.465] | 0.335 [0.294, 0.375] | 0.206 | 0.152 |
| MixStyle | 0.879 | 0.643 [0.581, 0.700] | 0.419 [0.376, 0.463] | 0.302 [0.261, 0.344] | 0.204 | 0.164 |
| MixStyle + Ordinal | 0.873 | 0.662 [0.608, 0.714] | 0.378 [0.349, 0.407] | 0.063 [0.037, 0.107] | 0.135 | 0.215 |
| Deep CORAL + Ordinal | 0.871 | 0.671 [0.614, 0.723] | 0.394 [0.364, 0.422] | 0.071 [0.044, 0.117] | **0.097** | 0.203 |

**Source-validation QWK is essentially identical across all six methods
(0.871–0.880).** Every difference above is an out-of-domain difference.

### Paired bootstrap, each method minus ERM (same 507 images)

| Method | ΔQWK [95% CI] | Δmacro F1 [95% CI] | ΔECE [95% CI] |
|---|---|---|---|
| Ordinal | **−0.088** [−0.137, −0.042] ✓ | **−0.074** [−0.120, −0.024] ✓ | **−0.205** [−0.247, −0.133] ✓ |
| Deep CORAL | **−0.088** [−0.137, −0.044] ✓ | **−0.048** [−0.091, −0.004] ✓ | **+0.071** [+0.028, +0.110] ✓ |
| MixStyle | **−0.110** [−0.163, −0.062] ✓ | **−0.049** [−0.090, −0.005] ✓ | +0.037 [−0.002, +0.073] |
| MixStyle + Ordinal | **−0.091** [−0.140, −0.045] ✓ | **−0.090** [−0.134, −0.045] ✓ | **−0.202** [−0.255, −0.129] ✓ |
| Deep CORAL + Ordinal | **−0.082** [−0.130, −0.039] ✓ | **−0.074** [−0.116, −0.029] ✓ | **−0.194** [−0.246, −0.122] ✓ |

✓ = interval excludes zero. Uncorrected for multiple comparisons.

**Every alternative is significantly worse than ERM on both QWK and macro F1.**
Deep CORAL significantly *worsens* calibration; MixStyle worsens it
non-significantly. The three ordinal variants improve ECE by ~0.20, significantly.

McNemar on exact accuracy is mostly non-significant (p = 0.05–0.83). That
contrast is informative: the methods differ less in *how often* they are right
than in *how far wrong* they are when wrong, which is exactly what QWK measures
and plain accuracy does not.

---

## 3. The ordinal calibration gain is not what it looks like

Per-grade recall on IDRiD:

| Method | Grade 0 (167) | Grade 1 (23) | Grade 2 (165) | **Grade 3 (91)** | Grade 4 (61) |
|---|---|---|---|---|---|
| **ERM** | 0.389 | **0.652** | 0.770 | **0.242** | 0.574 |
| Ordinal | 0.515 | 0.043 | 0.903 | 0.011 | 0.705 |
| Deep CORAL | 0.503 | 0.435 | 0.788 | 0.066 | 0.508 |
| MixStyle | 0.401 | 0.609 | 0.806 | 0.066 | 0.557 |
| MixStyle + Ordinal | 0.539 | **0.000** | 0.903 | 0.011 | 0.656 |
| Deep CORAL + Ordinal | 0.593 | **0.000** | 0.885 | 0.011 | 0.672 |

The ordinal variants achieve their low ECE largely by **abandoning grades 1 and
3**: recall 0.000–0.043 on grade 1 and 0.011 on grade 3, against ERM's 0.652 and
0.242. They concentrate predictions on grades 0, 2 and 4, where they are indeed
better calibrated — because they are answering an easier question.

Grade 3 is *severe non-proliferative DR*: the point at which urgent referral is
indicated. A model that finds 1 of 91 such cases is not usable, however good its
ECE. **This is the clearest argument in the whole study for never reporting
aggregate calibration without per-class recall beside it.**

ERM is the only method with non-trivial recall across all five grades — and even
its grade-3 recall (0.242) is poor.

---

## 4. Did the DG methods do anything at all?

Domain separability of the learned features, measured on the source-validation
split (which contains both DDR and APTOS). A logistic probe predicts the
**domain** from pooled features; chance is 0.845 (the majority-domain rate).

| Method | Probe accuracy | Lift over chance | Silhouette by domain | Centroid distance / spread |
|---|---|---|---|---|
| ERM | 0.981 | 0.877 | 0.085 | 0.448 |
| Ordinal | 0.972 | 0.818 | 0.072 | 0.405 |
| **Deep CORAL** | 0.981 | 0.877 | **0.025** | 0.405 |
| MixStyle | 0.976 | 0.845 | 0.061 | 0.388 |
| MixStyle + Ordinal | 0.970 | 0.806 | 0.076 | 0.357 |
| Deep CORAL + Ordinal | 0.977 | 0.853 | 0.032 | **0.318** |

Two things are true at once, and the distinction matters:

- **Deep CORAL did what it optimises.** It cut silhouette-by-domain 3.4× (0.085 →
  0.025) and pulled the domain centroids together. Second-order alignment
  worked.
- **It changed nothing a linear probe cares about.** Probe accuracy is 0.981 —
  *identical to ERM* — and the lift is unchanged at 0.877. Domain identity
  remains almost perfectly linearly decodable.

That is precisely the limitation stated in `deep_coral_alignment.py`: matching
covariance is necessary but not sufficient for invariance. Domain information
survives in directions the covariance objective does not constrain. It is
therefore coherent — not surprising — that the target metrics did not improve.

**No method reduced the probe lift below 0.81.** On this corpus, none of them
produced a representation that meaningfully hides the source dataset.

---

## 5. Temperature scaling can make target calibration *worse*

Fitted on source validation only, then applied unchanged to IDRiD:

| Method | Source val ECE | Target ECE before | Target ECE after | Effect |
|---|---|---|---|---|
| ERM | 0.051 | 0.265 | 0.185 | improved |
| Deep CORAL | 0.085 | 0.335 | 0.206 | improved |
| MixStyle | 0.070 | 0.302 | 0.204 | improved |
| **Ordinal** | 0.177 | **0.059** | **0.113** | **worse** |
| MixStyle + Ordinal | 0.168 | 0.063 | 0.135 | **worse** |
| Deep CORAL + Ordinal | 0.180 | 0.071 | 0.097 | **worse** |

The ordinal models were *under*-confident on source validation, so the fitted
temperature was below 1 (T ≈ 0.63–0.73) and sharpened their predictions. On the
target they were already well calibrated, so sharpening broke them.

**A temperature fitted in-domain can actively damage out-of-domain calibration
when the direction of miscalibration differs between domains.** This is a
concrete argument against treating post-hoc calibration as a free safety net,
and it is invisible if only the source domain is examined.

---

## 6. Remaining limitations

- **One seed.** Every number is a single run. The ERM batch-16 vs batch-32
  comparison (QWK 0.682 vs 0.753) shows this setup moves by ~0.07 under a change
  that is not even a seed. Several of the method gaps above are ~0.08–0.11.
  **Three seeds per method are needed before any of these conclusions is
  reportable.**
- **One target domain.** IDRiD is the smallest domain (507 images) and the most
  distributionally distinct (18.0% severe vs 1.9% in DDR). A method that fails
  here might succeed on the others.
- **Label shift is confounded with covariate shift.** IDRiD differs in case mix
  as well as appearance, and neither Deep CORAL nor MixStyle targets prior shift.
  Part of the gap they were asked to close was never theirs to close.
- **No hyperparameter search.** Deep CORAL's λ was fixed at 1.0 and MixStyle's
  p at 0.5 — the papers' defaults, not values tuned here. A tuned λ (on source
  validation only) could change the picture. As it stands the honest claim is
  "these methods at their default settings did not help", not "these methods do
  not help".
- **Uncorrected comparisons.** Fifteen paired tests were run; no multiplicity
  correction was applied.

---

## 7. A bug that reached a run, and what it changed

The first sweep crashed on all three ordinal methods:
`probabilities must have shape (2206, 5), got (2206, 4)`. The evaluation path
was softmaxing the CORAL head's K−1 *cumulative* logits. Training had completed
and checkpoints were saved, so the three runs were **re-scored from checkpoint**
rather than retrained.

Fixing it surfaced a genuine methodological point. CORAL admits two decision
rules, and they are not equivalent:

| | Agreement with the other rule | Changed by temperature scaling |
|---|---|---|
| argmax of the differenced distribution | — | **23% of samples** |
| CORAL native threshold count | 71% | **0.00%** |

The native rule is *exactly* invariant to temperature (sigmoid(x/T) > 0.5 ⟺
x > 0 for T > 0). Using argmax would let a calibration step silently change
accuracy and QWK — destroying the property that makes temperature scaling a
clean, isolated intervention in this study. The native rule is therefore used
for predictions, and confidence is reported as the probability of the
*predicted* class rather than the maximum probability, so the two stay
consistent. Regression tests cover both.

---

## 8. What this means for the paper

The result so far is a **negative one, and a specific one**:

1. Standard DG methods (Deep CORAL, MixStyle) at default settings did not improve
   cross-domain DR grading here, and the embedding analysis shows why: they
   reduced coarse geometric domain separation without reducing linearly decodable
   domain information.
2. Ordinal learning bought a large calibration improvement at an unacceptable
   clinical price — collapse of the two grades that drive referral decisions.
3. Post-hoc temperature scaling helped in-domain and for the softmax models, but
   *harmed* the ordinal models out-of-domain.

Together these argue the paper's contribution is likely to be **the evaluation
protocol and the failure analysis**, not a method that wins. That is a
defensible contribution — the brief anticipated it ("negative results are
allowed") — but it needs the multi-seed runs and the remaining three target
domains before it can be written.

**Nothing in this document is fabricated.** Every value traces to a CSV in
`outputs/tables/` or a row in `outputs/experiment_registry.csv`.
