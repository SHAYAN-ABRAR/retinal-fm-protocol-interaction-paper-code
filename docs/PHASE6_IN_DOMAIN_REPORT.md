# Phase 6 — In-domain ceilings and the cost of cross-domain deployment

**Run:** ERM, DenseNet121, batch 32, 20 epochs, **seed 42 only**
**Date:** 2026-08-20 · 49 min wall clock (620 + 235 + 48 + 2026 s)
**Protocol:** train, validate and test entirely within one domain.

This supplies the term Phase 5 was missing. Every gap there compared a LODO
model's target score against its own *source validation* — different images from
different domains. The number that matters clinically is different: **how much
worse is a model that never saw this domain than one trained on it?**

---

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
epochs (0.357 → 0.324) while validation QWK moved 0.003. The ceiling here is set
by grading noise in the labels, not by the architecture or the budget.

## 2. The cost of cross-domain deployment

Both models scored on **identical test images** — the LODO predictions are
restricted to the in-domain test split before differencing, otherwise the LODO
model would be credited for images the in-domain model trained on.

> **Updated 2026-08-21 with 3 seeds.** The table below now reports the LODO mean
> over seeds 42, 1, 2, with the across-seed SD, and applies the two-bar
> criterion. The single-seed version is in §2b.

| Domain | n_test | in-domain | LODO mean | seed SD | Δ | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| DDR | 1,862 | 0.8757 | 0.7327 | 0.0038 | **−0.1430** | [−0.1760, −0.1113] | **REAL (both bars)** |
| APTOS | 354 | 0.9091 | 0.8625 | 0.0103 | −0.0466 | [−0.0932, +0.0035] | CI spans zero |
| IDRiD | 102 | 0.5884 | 0.6803 | 0.0504 | +0.0920 | [−0.1420, +0.3342] | CI spans zero |
| EyePACS | 5,268 | 0.7090 | 0.4153 | 0.0082 | **−0.2937** | [−0.3300, −0.2577] | **REAL (both bars)** |

**Two of four are established, and both are large.** DDR's −0.143 is 38× its
across-seed SD; EyePACS's −0.294 is 36×.

**APTOS is the second documented case of the two bars disagreeing.** Its Δ of
−0.047 *exceeds* the seed SD of 0.0103 — bar 1 passes — but the paired interval
on 354 images includes zero, so bar 2 fails. The first such case was Deep
CORAL's calibration effect in Phase 4, which cleared the bootstrap and failed
the seed bar. The two bars fail independently in both directions, which is the
argument for requiring both.

**IDRiD is unresolvable, twice over.** Its across-seed SD on the matched
102-image split is 0.0504 — 55% of the effect — and the paired interval spans
0.48 of QWK. No claim about IDRiD's direction is supportable from this data.

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
against an in-domain model, because the in-domain ceiling itself is only 0.709.
Roughly 40% of what looked like domain-shift damage is the domain's own label
noise, which no domain-generalization method can remove.

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
  needs the single-source-external runs (`NOT_RUN`).

## 7. Artifacts

| Path | Contents |
|---|---|
| `outputs/tables/in_domain_results.csv` | one row per domain |
| `outputs/tables/in_domain_vs_lodo_erm_s42.csv` | the matched-image comparison |
| `outputs/logs/in_domain_erm_s42.log` | full training log |

Regenerate the comparison without retraining:
`python run_in_domain.py --compare-only`
