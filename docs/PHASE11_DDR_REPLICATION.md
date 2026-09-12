# Phase 11 — The negative result replicates on a second target

> **HISTORICAL PROJECT RECORD — NOT AUTHORITATIVE FOR THE FINAL MANUSCRIPT.**
> See [`JBHI_EVIDENCE_FREEZE.md`](JBHI_EVIDENCE_FREEZE.md) and
> [`JBHI_DRAFTING_MANIFEST.md`](JBHI_DRAFTING_MANIFEST.md) for the frozen final
> state. Statements below were true at the time they were written and may have
> been superseded — including older test and audit counts, the retired
> "two-bar" terminology, and novelty framing that has since been narrowed.


**Run:** DenseNet121, 224 px, batch 32, 20 epochs, LODO target **DDR**
**Methods:** Deep CORAL · MixStyle · GroupDRO · IRMv1, against an ERM control
**Sampler:** natural shuffling
**Seeds:** 42 / 1 / 2 · **Test set:** 12,424 DDR images, identical for every row
**Date:** 2026-08-26 → 2026-08-27 · ~9 h GPU
**Generator:** `analyse_methods.py --target ddr` → `outputs/tables/method_comparison_lodo_ddr.csv`

Phase 10 established that no domain-generalization method beats ERM on the
EyePACS target. One target is a case study. This phase asks whether the same
holds where the shift is different in kind: EyePACS is the largest domain and
the hardest target; DDR is a mid-sized target whose source pool is dominated by
EyePACS.

---

## 0. Answer

**No method beats ERM on DDR either.** The claim now holds on two targets.

| method | QWK | Δ vs ERM | Δ SD | Δ/SD | verdict |
|---|---|---|---|---|---|
| **ERM** | **0.7383 ± 0.0055** | — | — | — | — |
| MixStyle | 0.7444 | +0.0061 | 0.0139 | 0.44× | within seed noise |
| Deep CORAL | 0.7338 | −0.0046 | 0.0073 | 0.62× | within seed noise |
| GroupDRO | 0.7333 | −0.0051 | 0.0072 | 0.71× | within seed noise |
| IRMv1 | — | — | — | — | **diverged (§2)** |

---

## 1. MixStyle, and why one seed is not a result

MixStyle finished seed 42 at **0.7528** against ERM's 0.7334 — **+0.0194**, about
3.5× ERM's across-seed SD, and its paired bootstrap CI was [+0.0102, +0.0286]
with p < 0.001. On that seed alone it is the first method in this project to
beat ERM anywhere.

Then seed 1 came in at +0.0070 and seed 2 at **−0.0083**.

| seed | MixStyle | ERM | Δ |
|---|---|---|---|
| 42 | 0.7528 | 0.7334 | **+0.0194** |
| 1 | 0.7443 | 0.7373 | +0.0070 |
| 2 | 0.7360 | 0.7443 | **−0.0083** |
| | | mean | +0.0061, SD 0.0139 → **0.44×** |

The seeds disagree in sign, so bar 1 fails by a factor of two. This is the same
shape MixStyle showed on EyePACS, where it was +0.0214 on seed 42 and −0.0056
averaged over three.

**This is the single clearest justification for the two-bar criterion in the
whole project.** A paper reporting seed 42 with its bootstrap interval would
have claimed a real effect, with a correct p-value, from a correctly-implemented
method — and been wrong. The bootstrap resamples *images* and answers "is this
model better than that model on this test set", which it genuinely was. It
cannot see that a different seed produces a different model. Only the
across-seed SD sees that.

---

## 2. IRMv1 diverged, twice, for two different reasons

### First attempt: the published default

All three seeds crashed with `ValueError: Input contains NaN`.

| seed | trajectory (train loss) | outcome |
|---|---|---|
| 42 | 1.37 → 1135 | NaN after epoch 1 |
| 1 | 5.19 → 210 | NaN after epoch 1 |
| 2 | 0.76 → 359 → 2009 → 4622 → 6420 → 13752 | NaN after epoch 5 |

The cause is arithmetic, and it is a genuine finding about IRMv1's defaults:

| split | train images | batches/epoch | λ 1→100 switch at |
|---|---|---|---|
| EyePACS | 11,841 | 370 | epoch **1.35** |
| DDR | 27,718 | 866 | epoch **0.58** |

`anneal_iters` is specified in **steps**, so the point at which the penalty
switches on depends on the size of the source pool. On EyePACS the network gets
one complete epoch before the penalty dominates; on DDR the switch lands
mid-first-epoch, before the network is a usable predictor, and the penalty
term runs away.

**IRMv1's published anneal default does not transfer across dataset sizes.**

### Second attempt: the anneal matched to the split

Re-run with `--irm-anneal-iters 1170` — 1.35 × 866, reproducing exactly the
fraction of training EyePACS got before its switch.

**Two of three seeds still diverged to NaN.** The third completed at QWK
**0.230**, with its best epoch at 0 — the same signature as EyePACS, where every
IRM run selected the last checkpoint before the penalty engaged.

This second run is what makes the negative result reportable. The divergence
alone could be answered with "you used a default that obviously did not fit your
data", and that answer would have been correct. It is no longer available.

Both configurations are in the registry: `irm-b32` with `status=DIVERGED`, and
`irm-b32-a1170` as a separate experiment id. Nothing was overwritten and no
metrics were invented for runs that produced none.

---

## 3. The methods were starved worse here than on EyePACS

| method | degenerate batches, EyePACS | degenerate batches, **DDR** |
|---|---|---|
| Deep CORAL | alignment inactive in 4% | **59%** |
| GroupDRO | a domain was a single image in 62% | **83%** |

DDR's source pool is APTOS + IDRiD + EyePACS, and EyePACS supplies 24,574 of
27,718 training images — 89%. A batch of 32 is therefore ~28 EyePACS, ~3 APTOS
and ~0.4 IDRiD, so most batches contain effectively one domain and the
multi-domain machinery has nothing to work with.

**This limits what the DDR arm alone can claim**, and it is why the
domain-balanced sampler was run on EyePACS: there, forcing every domain into
every batch drove these rates to 0% and did not change the ranking. The DDR arm
inherits that answer rather than re-establishing it — a balanced arm here would
cost ~15 h to re-answer a question already answered.

---

## 4. Calibration and severe errors

| method | ECE (scaled) | Δ | severe | Δ | Δ SD |
|---|---|---|---|---|---|
| ERM | 0.0472 | — | 0.1598 | — | — |
| Deep CORAL | 0.0431 | −0.0041 | 0.1647 | +0.0049 | 0.0052 |
| MixStyle | 0.0461 | −0.0011 | 0.1601 | +0.0003 | 0.0109 |
| GroupDRO | 0.0449 | −0.0023 | 0.1620 | +0.0022 | 0.0058 |

Every difference is within seed noise, and **no method reduces severe errors** —
each is marginally worse than ERM. Consistent with Phase 10.

DDR is far better calibrated than EyePACS post-scaling (0.047 against 0.173),
which is a statement about the targets, not the methods.

---

## 5. What this does NOT establish

- **Not that DG methods never work.** Two targets, one backbone, one resolution,
  published defaults throughout. This is a fixed-configuration result.
- **Not a fair test of the methods' design intent on DDR**, because the natural
  sampler starves them (§3). The EyePACS balanced arm covers that.
- **Not that IRM is a bad method** — that it is unstable here under two anneal
  settings, one of them chosen to be favourable.
- **Not tuned per-domain**, because tuning on the target would violate the
  external-test protocol. That is the correct choice and also a real limit.

---

## 6. What this means for the paper

Phase 10's claim was "no method beats ERM on the hardest target". It is now
**"no method beats ERM on either target tested, under either sampler where both
were run, across four methods and three method families"**, with the added
observation that a method can look like a winner on one seed with a valid
bootstrap interval behind it.

The MixStyle near-miss is worth a paragraph in its own right. It is a concrete,
in-house demonstration that seed variance in this problem is large enough to
manufacture a publishable-looking effect, which is the strongest available
argument for the paper's methodology section.

---

## 7. Artifacts

| what | where |
|---|---|
| Generator | `analyse_methods.py --target ddr` |
| Result table | `outputs/tables/method_comparison_lodo_ddr.csv` |
| Registry rows | `outputs/experiment_registry.csv` (incl. `status=DIVERGED`) |
| Training logs | `outputs/logs/lodo_dg_ddr_*.log` |
| IRM anneal flag | `run_lodo.py --irm-anneal-iters` |
