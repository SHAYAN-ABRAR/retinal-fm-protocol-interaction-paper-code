# Phase 6 — In-domain ceilings and the cost of cross-domain deployment

> **HISTORICAL PROJECT RECORD — NOT AUTHORITATIVE FOR THE FINAL MANUSCRIPT.**
> See [`JBHI_EVIDENCE_FREEZE.md`](JBHI_EVIDENCE_FREEZE.md) and
> [`JBHI_DRAFTING_MANIFEST.md`](JBHI_DRAFTING_MANIFEST.md) for the frozen final
> state. Statements below were true at the time they were written and may have
> been superseded — including older test and audit counts, the retired
> "two-bar" terminology, and novelty framing that has since been narrowed.


**Run:** ERM, DenseNet121, batch 32, 20 epochs, **seed 42 only**
**Date:** 2026-08-20 · 49 min wall clock (620 + 235 + 48 + 2026 s)
**Protocol:** train, validate and test entirely within one domain.

This supplies the term Phase 5 was missing. Every gap there compared a LODO
model's target score against its own *source validation* — different images from
different domains. The number that matters clinically is different: **how much
worse is a model that never saw this domain than one trained on it?**

> **⚠ CORRECTION 2026-08-22 — the EyePACS ceiling claim in §1 was wrong.**
> §1 originally stated that EyePACS's 0.709 ceiling "is set by grading noise in
> the labels, not by the architecture or the budget." **A 512 px re-run of the
> identical protocol reaches 0.8004 on the identical test images.** Resolution
> was the binding constraint, not label noise. §0a below carries the measurement;
> §1 and §4 have been corrected in place and the superseded wording is quoted
> there so the error stays on the record.

---

## 0a. The resolution correction

Same domain, same splits, same architecture, same seed, same 5,268 test images.
Only the input resolution differs (224 px batch 32 → 512 px batch 16).

| Metric | 224 px | 512 px | Δ | 95% CI | p |
|---|---|---|---|---|---|
| QWK | 0.7090 | **0.8004** | **+0.0914** | [+0.0694, +0.1142] | < 0.0002 |
| Severe-error rate (\|err\|≥2) | 0.0919 | **0.0575** | **−0.0344** | [−0.0418, −0.0272] | < 0.0002 |
| Macro F1 | 0.5423 | 0.6130 | +0.0707 | [+0.0446, +0.0977] | < 0.0002 |
| MAE of grade | 0.2984 | 0.2124 | −0.0860 | [−0.1025, −0.0701] | < 0.0002 |
| Within ±1 grade | 0.9081 | 0.9425 | +0.0344 | [+0.0272, +0.0418] | < 0.0002 |
| ECE | 0.0733 | 0.0606 | −0.0127 | — | — |
| ECE after T | 0.0364 | 0.0252 | −0.0112 | — | — |

Paired bootstrap, 5,000 resamples over the shared image ids. Every interval
excludes zero and no resample crossed it, so p is bounded by 1/5000.

**What this establishes.** The 0.709 figure was a property of the input pipeline,
not of the labels. DR grading turns on microaneurysms of roughly 50–100 µm; in a
~2000 px fundus photograph downsampled to 224 px those lesions approach a single
pixel. Grade 1 is *defined* by their presence, which is why macro F1 gains most.

**What this does not establish.**

- **That label noise is absent.** Published work on EyePACS reaches ≈0.85; 0.8004
  is still short of it. Resolution explains the gap that was attributed to
  labels — it does not prove the remaining gap is not labels.
- **That 0.8004 is the ceiling.** No resolution above 512 px was tried.
- **Bar 1 of the two-bar criterion.** In-domain runs exist at seed 42 only, so
  there is no across-seed SD for this comparison. The nearest proxy is the LODO
  across-seed SD on EyePACS, 0.0081, which +0.0914 exceeds by ~11×. That is a
  proxy, not a measurement, and is labelled as such wherever it is cited.

**Consequence for the rest of this project.** Every 224 px number in this
document and in Phases 5, 7 and 8 is a 224 px number. Resolution is a confound
of that size on at least one domain, so no 224 px result may be differenced
against a 512 px result — `run_in_domain.py` refuses to do it, and
`in_domain_results.csv` is keyed on `image_size` so the two cannot overwrite
each other. The 512 px LODO matrix is running to make the comparison like-for-like.

## 1. In-domain ceilings

| Domain | n_train | n_test | val QWK | test QWK | macro F1 | ECE | after T | severe |
|---|---|---|---|---|---|---|---|---|
| DDR | 8,697 | 1,862 | 0.8901 | 0.8757 | 0.665 | 0.087 | 0.011 | 0.071 |
| APTOS | 2,809 | 354 | 0.8966 | 0.9091 | 0.666 | 0.072 | 0.051 | 0.040 |
| IDRiD | 335 | 102 | 0.8567 | **0.5884** | 0.437 | 0.172 | 0.171 | 0.225 |
| EyePACS | 24,574 | 5,268 | 0.7090 | 0.7090 | 0.542 | 0.073 | 0.036 | 0.092 |

Two of these are not "ceilings" in any useful sense:

**IDRiD** has 335 training images. Its validation QWK (0.857) and test QWK
(0.588) differ by 0.27 across 70 and 102 images — neither split is large enough
to estimate anything stably. Temperature scaling did essentially nothing
(0.1720 → 0.1712) because a temperature fitted on 70 images has nothing to fit.
This row is reported as a measured limit of the domain, not as a tuned result.

**EyePACS** plateaus at 0.709 with 24,574 of its *own* images, while DDR and
APTOS reach 0.88–0.91 on theirs. Training loss kept falling over the last three
epochs (0.357 → 0.324) while validation QWK moved 0.003.

> **Corrected.** This paragraph previously concluded: *"The ceiling here is set
> by grading noise in the labels, not by the architecture or the budget."*
> **That was wrong.** The same protocol at 512 px reaches 0.8004 on the same
> test images (§0a). The plateau was a resolution limit. The training-loss
> observation above is still accurate — the model had stopped extracting signal
> — but the reason is that at 224 px the signal was no longer in the input.

## 2. The cost of cross-domain deployment

Both models scored on **identical test images** — the LODO predictions are
restricted to the in-domain test split before differencing, otherwise the LODO
model would be credited for images the in-domain model trained on.

> **Updated 2026-08-23: three seeds on BOTH sides, paired.** In-domain runs at
> seeds 1 and 2 were added, so the reference is no longer a single point
> estimate. Seeds are now paired — the LODO model at seed *s* is differenced
> against the in-domain model at seed *s* — which means the SD below is the
> run-to-run variation **of the difference itself**, not of either model alone.
> The earlier version, which held the seed-42 in-domain score fixed, is in §2b.

| Domain | n_test | in-domain (3 seeds) | LODO (3 seeds) | Δ | Δ SD | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| DDR | 1,862 | 0.8702 ± 0.0086 | 0.7327 ± 0.0038 | **−0.1376** | 0.0119 | [−0.1760, −0.0954] | **REAL (both bars)** |
| APTOS | 354 | 0.9085 ± 0.0097 | 0.8625 ± 0.0103 | −0.0460 | 0.0127 | [−0.1002, +0.0035] | CI spans zero |
| IDRiD | 102 | 0.5846 ± 0.0191 | 0.6803 ± 0.0504 | +0.0957 | 0.0695 | [−0.1630, +0.3661] | CI spans zero |
| EyePACS | 5,268 | 0.7063 ± 0.0139 | 0.4153 ± 0.0082 | **−0.2910** | 0.0202 | [−0.3305, −0.2375] | **REAL (both bars)** |

**Two of four are established, and both are large.** DDR's −0.1376 is 11.6× the
SD of its own difference; EyePACS's −0.2910 is 14.4×.

**The point estimates barely moved; the uncertainty roughly tripled.** Against
the old single-seed reference these were −0.1430 and −0.2937 — a shift of under
0.006 — but the Δ SDs went 0.0038 → 0.0119 and 0.0082 → 0.0202. That increase is
the honest accounting: the reference used to be treated as exact, and it is not.
The claims clear the wider bar comfortably, which is the point of having run the
extra seeds rather than asserting they would not matter.

**APTOS is the second documented case of the two bars disagreeing.** Its Δ of
−0.046 *exceeds* the Δ SD of 0.0127 — bar 1 passes — but the paired interval on
354 images includes zero, so bar 2 fails. The first such case was Deep CORAL's
calibration effect in Phase 4, which cleared the bootstrap and failed the seed
bar. The two bars fail independently in both directions, which is the argument
for requiring both.

**IDRiD is unresolvable, twice over.** The SD of its difference on the matched
102-image split is 0.0695 — 73% of the effect — and the paired interval spans
0.53 of QWK. No claim about IDRiD's direction is supportable from this data.

### 2b. Single-seed version (seed 42, superseded)

| Domain | n_test | in-domain | LODO | Δ | 95% CI | verdict |
|---|---|---|---|---|---|---|
| DDR | 1,862 | 0.8757 | 0.7287 | −0.1469 | [−0.1760, −0.1164] | REAL |
| APTOS | 354 | 0.9091 | 0.8742 | −0.0349 | [−0.0733, +0.0035] | not resolved |
| IDRiD | 102 | 0.5884 | 0.6662 | +0.0778 | [−0.0905, +0.2598] | not resolved |
| EyePACS | 5,268 | 0.7090 | 0.4072 | −0.3018 | [−0.3300, −0.2751] | REAL |

**Two of four are resolved, and both are large.** On DDR, never having seen the
domain costs 0.147 QWK. On EyePACS it costs 0.302.

**Two are not resolved, and the reason is test-set size.** APTOS contributes 354
test images and IDRiD 102. At that scale the paired interval spans zero even
where the point estimate is sizeable.

### 2c. The same cost in severe errors

QWK is the field's metric; it is not the one a clinician cares about. A
**severe error** is a grade misread by two or more steps — the error that sends
a referable patient home. Same protocol as §2: matched images, LODO mean over
three seeds, paired bootstrap on seed 42, both bars required.

| Domain | n | In-domain (3 seeds) | Cross-domain (3 seeds) | Δ | Δ SD | 95% CI | Relative | Verdict |
|---|---|---|---|---|---|---|---|---|
| DDR | 1,862 | 0.0788 ± 0.0105 | 0.1631 ± 0.0054 | **+0.0843** | 0.0150 | [+0.0806, +0.1155] | **+107%** | **REAL (both bars)** |
| APTOS | 354 | 0.0395 ± 0.0056 | 0.0669 ± 0.0091 | +0.0273 | 0.0099 | [−0.0085, +0.0424] | — | CI spans zero |
| IDRiD | 102 | 0.2092 ± 0.0283 | 0.1536 ± 0.0344 | −0.0556 | 0.0463 | [−0.1373, +0.0588] | — | CI spans zero |
| EyePACS | 5,268 | 0.0938 ± 0.0028 | 0.2396 ± 0.0075 | **+0.1457** | 0.0091 | [+0.1433, +0.1689] | **+155%** | **REAL (both bars)** |

**Where the deployment cost is established, the severe-error rate more than
doubles.** (Paired across three seeds on both sides, 2026-08-23. Against the
earlier single-seed reference these read +128% and +161%; the in-domain mean is
now over three seeds rather than seed 42's particularly good run, which is why
they came down.) The two established domains are the same two as in §2, which is what
should happen — the metrics disagree about magnitude, not about which effects
are real.

**This is the number that resists the obvious fix.** Temperature scaling is
monotonic: it rescales confidences without reordering them, so it cannot move an
argmax and cannot change a single severe error. A perfectly calibrated model
here still misgrades 24% of EyePACS cases by two or more steps. Calibration
makes the failure *legible*; it does not make it smaller.

**APTOS and IDRiD are unresolved for the same reason as in §2** — 354 and 102
test images. IDRiD's point estimate is negative, but on 102 images with a
seed SD of 0.034 that is not a finding.

Figure: `outputs/figures/fig_severe_error_deployment.png`.
Reproduce: `python analyse_severe_error.py` (no GPU, reads saved predictions).

## 3. Correction: cross-domain training does *not* demonstrably beat in-domain on IDRiD

An earlier reading of the raw per-run numbers compared IDRiD's in-domain test
QWK (0.5884, on 102 images) against the LODO run's target QWK (0.7543, on all
507 images) and concluded that the cross-domain model beat the in-domain one.

**That comparison was invalid** — different test sets, and the 507-image set
includes the 335 images the in-domain model trained on.

On the matched 102 images the LODO model scores 0.6662 against 0.5884, a
difference of **+0.0778 with a 95% CI of [−0.0905, +0.2598]**. The direction
still favours cross-domain training, but the interval spans zero. With 102 test
images this cannot be resolved.

The defensible statement is: *IDRiD's own 335 training images do not produce a
model that clearly outperforms one trained on other domains.* That is a weaker
claim than "cross-domain training wins", and it is the one the data supports.

## 4. What Phase 5's numbers looked like, and what they mean

Phase 5 reported DDR's gap as −0.019 against source validation, which reads as
near-perfect transfer. The like-for-like figure is **−0.147**, nearly eight times
larger. Both are correct; they answer different questions. The source-validation
gap is flattering because the mixed-domain validation set is itself harder than
DDR's own test split.

| Domain | vs source validation (Phase 5) | vs in-domain model (Phase 6) |
|---|---|---|
| DDR | −0.019 | **−0.147** |
| APTOS | +0.063 | −0.035 |
| IDRiD | −0.048 | +0.078 |
| EyePACS | −0.483 | **−0.302** |

EyePACS moves the other way: −0.483 against source validation but −0.302
against an in-domain model, because the 224 px in-domain reference is itself
only 0.709.

> **Corrected.** This paragraph previously read: *"Roughly 40% of what looked
> like domain-shift damage is the domain's own label noise, which no
> domain-generalization method can remove."* **The attribution was wrong.** That
> ~40% is the gap between the source-validation view and the in-domain
> reference, and §0a shows the reference was depressed by resolution rather than
> by labels. The arithmetic stands; the causal claim does not. What fraction is
> label noise is **unmeasured**, and the 512 px LODO run is what will let the
> comparison be made at one resolution throughout.

## 5. Calibration in-domain vs cross-domain

| Domain | in-domain ECE → after T | cross-domain ECE → after T |
|---|---|---|
| DDR | 0.087 → **0.011** | 0.138 → 0.055 |
| APTOS | 0.072 → 0.051 | 0.108 → 0.041 |
| IDRiD | 0.172 → 0.171 | 0.216 → 0.105 |
| EyePACS | 0.073 → **0.036** | 0.289 → 0.180 |

In-domain models calibrate almost perfectly with a temperature (DDR 0.011,
EyePACS 0.036). Cross-domain models do not, and the shortfall grows with the
shift. IDRiD is the exception in the other direction: its *in-domain*
temperature is useless because it was fitted on 70 images, while the LODO
temperature — fitted on thousands of source-validation images — works better.

## 6. What this does not establish

- **Anything at n=1 within ±0.032.** APTOS's −0.035 is at the noise floor.
- **The IDRiD direction**, per section 3.
- **That EyePACS's remaining −0.302 is all domain shift.** Its LODO model
  trained on only 11,841 images. Separating training-set size from domain shift
  needs the single-source-external runs (done in Phase 7 §6).
- **That the 224 px in-domain scores are ceilings at all.** §0a shows EyePACS's
  was not. DDR, APTOS and IDRiD have not been re-run at 512 px yet, so whether
  their 224 px scores are ceilings or resolution limits is **unmeasured**. The
  three re-runs are queued.

## 7. Artifacts

| Path | Contents |
|---|---|
| `outputs/tables/in_domain_results.csv` | one row per (domain, resolution) |
| `outputs/tables/resolution_effect_eyepacs_s42.csv` | the §0a paired bootstrap |
| `outputs/tables/severe_error_comparison.csv` | the §2c severe-error comparison |
| `outputs/figures/fig_severe_error_deployment.png` | §2c, both panels |
| `outputs/figures/fig_risk_coverage.png` | abstention curves, four unseen domains |
| `outputs/tables/in_domain_vs_lodo_erm_s42.csv` | the matched-image comparison |
| `outputs/logs/in_domain_erm_s42.log` | full training log |

Regenerate the comparison without retraining:
`python run_in_domain.py --compare-only`
