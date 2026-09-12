# Phase 10 — Four DG methods, two samplers, and why none of them wins

> **HISTORICAL PROJECT RECORD — NOT AUTHORITATIVE FOR THE FINAL MANUSCRIPT.**
> See [`JBHI_EVIDENCE_FREEZE.md`](JBHI_EVIDENCE_FREEZE.md) and
> [`JBHI_DRAFTING_MANIFEST.md`](JBHI_DRAFTING_MANIFEST.md) for the frozen final
> state. Statements below were true at the time they were written and may have
> been superseded — including older test and audit counts, the retired
> "two-bar" terminology, and novelty framing that has since been narrowed.


**Run:** DenseNet121, 224 px, batch 32, 20 epochs, LODO target **EyePACS**
**Methods:** Deep CORAL · MixStyle · GroupDRO · IRMv1, each against an **ERM control on the same sampler**
**Samplers:** natural shuffling · domain-balanced batches
**Seeds:** 42 / 1 / 2 on every cell — 30 runs
**Test set:** 35,108 EyePACS images, identical for every row
**Date:** 2026-08-25 → 2026-08-26 · ~7.9 h GPU (10.7 h wall; one seed absorbed a machine suspend)
**Generator:** `analyse_methods.py` → `outputs/tables/method_comparison_lodo_eyepacs.csv`

Phase 4 made this project's central negative claim from two feature-space
methods on a two-source configuration with a 507-image test set. That was too
thin to carry a paper. This phase rebuilds the claim on four methods spanning
three method families, on the largest domain gap in the study, with 69× the test
images and a paired seed design.

---

## 0. Answer

**No method beats ERM. Not under either sampler, not on either bar.**

| sampler | ERM QWK | best method | Δ | over-SD | verdict |
|---|---|---|---|---|---|
| natural | 0.4147 ± 0.0081 | Deep CORAL 0.4225 | +0.0077 | 0.77× | within seed noise |
| domain-balanced | 0.3963 ± 0.0643 | Deep CORAL 0.4073 | +0.0110 | 0.24× | within seed noise |

The two-bar criterion requires a difference to exceed the across-seed SD **and**
have a paired bootstrap CI excluding zero. Every method fails bar 1. The single
largest effect in the table belongs to IRM, and it points the wrong way.

**The objection this phase exists to close** is "you did not give the methods a
fair run" — on natural batches a batch of 32 from this pool is ~24 DDR, ~8 APTOS
and **zero IDRiD**, which starves every one of these methods of the multi-domain
batch they are defined on. The domain-balanced arm removes that objection
outright, and the diagnostics prove it:

| method | degenerate batches, natural | degenerate batches, balanced |
|---|---|---|
| Deep CORAL | alignment inactive in **4%** | **0%** |
| GroupDRO | a domain was a single image in **62%** | **0%** |
| IRMv1 | penalty skipped in **61%** | **0%** |

**The machinery went from firing on a minority of steps to firing on every
step, and the ranking did not change.** That is a much stronger negative result
than Phase 4's, and it is the version that belongs in the paper.

---

## 1. IRM did not underperform — it diverged

This must be stated plainly, because reporting IRM's −0.077 QWK as though it
were a fair method comparison would be misleading, and a reviewer would be right
to object.

**All six IRM runs — both samplers, all three seeds — stopped at epoch 7 with
`best_epoch = 0`.** The evaluated IRM model is the checkpoint from the first
epoch. The trajectory (balanced, seed 42; the other five are the same shape):

| epoch | train loss | val loss | val QWK | |
|---|---|---|---|---|
| 0 | 0.98 | 0.79 | **0.679** | λ = 1 for the whole epoch — best checkpoint |
| 1 | 1.92 | 10.52 | 0.476 | λ → 100 at step 500, mid-epoch |
| 2 | 23.5 | 44.1 | 0.504 | |
| 3 | 129.5 | 29.7 | 0.519 | |
| 4 | 337.8 | 18.4 | 0.543 | |
| 5 | 760.8 | 63.9 | 0.599 | |
| 6 | 1579.6 | 22.8 | 0.478 | early stopping fires |

The arithmetic is exact: 11,841 training images at batch 32 is 370 steps per
epoch, and `anneal_iters = 500` puts the λ = 1 → 100 switch at epoch 1.35.
**Epoch 0 is the only epoch trained entirely before the penalty switches on, and
it is the epoch every run selects.** After the switch the training loss grows by
three orders of magnitude — and that is the loss *after* division by λ, so the
undivided objective reached the order of 10^5. The penalty runs away and the NLL
term stops mattering.

This is the DomainBed-faithful configuration (λ = 100, 500 anneal steps), not a
setting chosen to make IRM look bad. But the honest description of the result is
**"IRMv1 was unstable on this data under its standard schedule"**, not "IRM is
worse than ERM at domain generalization." The paper will say the former. A λ
sweep would be needed to say anything stronger, and that is named as future work
rather than quietly implied.

### The calibration trap this creates

IRM has by far the best calibration in the table — post-scaling ECE **0.059**
against ERM's 0.173, a 66% reduction, the largest calibration effect anywhere in
this project.

**It is an artefact and must not be reported as a finding.** That number is the
calibration of a model that trained for one epoch and stopped. An under-trained
network is under-confident, and ECE rewards under-confidence exactly as much as
it rewards being right. The QWK column shows what it cost: 0.338 against 0.415.

This is a useful illustration for the paper's calibration section — **ECE alone
can be improved by damaging the model** — and it is worth one sentence there. It
is not evidence that invariance penalties improve calibration.

---

## 2. The full table

Δ is method − ERM **under the same sampler**. Positive favours the method. The
interval and p-value belong to `Δ_seed42`; `Δ_qwk` is the 3-seed mean and the SD
is across paired seeds.

### Natural sampler — ERM 0.4147

| method | QWK | Δ | Δ SD | Δ/SD | Δ seed 42 | CI | p | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Deep CORAL | 0.4225 | +0.0077 | 0.0100 | 0.77× | +0.0084 | [+0.0014, +0.0154] | 0.020 | 0.040 | within seed noise |
| MixStyle | 0.4091 | −0.0056 | 0.0239 | 0.23× | +0.0214 | [+0.0144, +0.0285] | <0.001 | <0.001 | within seed noise |
| GroupDRO | 0.4142 | −0.0005 | 0.0017 | 0.30× | −0.0024 | [−0.0095, +0.0048] | 0.514 | 0.514 | within seed noise |
| IRMv1 | 0.3379 | −0.0768 | 0.0162 | 4.73× | −0.0588 | [−0.0709, −0.0462] | <0.001 | <0.001 | **worse** (diverged — §1) |

### Domain-balanced sampler — ERM 0.3963

| method | QWK | Δ | Δ SD | Δ/SD | Δ seed 42 | CI | p | verdict |
|---|---|---|---|---|---|---|---|---|
| Deep CORAL | 0.4073 | +0.0110 | 0.0469 | 0.24× | −0.0282 | [−0.0357, −0.0205] | <0.001 | within seed noise |
| MixStyle | 0.3933 | −0.0030 | 0.0410 | 0.07× | −0.0339 | [−0.0421, −0.0259] | <0.001 | within seed noise |
| GroupDRO | 0.4069 | +0.0106 | 0.0518 | 0.20× | −0.0274 | [−0.0352, −0.0194] | <0.001 | within seed noise |
| IRMv1 | 0.3548 | −0.0415 | 0.0482 | 0.86× | −0.0780 | [−0.0883, −0.0673] | <0.001 | within seed noise |

**Three rows have a mean and an interval with opposite signs.** Deep CORAL is
+0.0110 averaged but −0.0282 on seed 42; GroupDRO is +0.0106 against −0.0274;
MixStyle likewise. The generator prints a note whenever this happens rather than
letting a reader pair the two columns by eye. The bootstrap is computed on the
anchor seed, so a sign disagreement means the seeds disagree — which is the
finding, not a defect. It is the reason bar 1 exists.

### Multiplicity

Eight comparisons are made against ERM and Holm–Bonferroni is reported beside
the raw p-value, **but the uncorrected p is primary here.** Correction makes
differences harder to detect, and the claim being defended is that there are
none — so correcting would make the claim *easier* to assert. A method that
fails to beat ERM uncorrected has already failed the more demanding test.

---

## 3. The sampler is a bigger effect than any method

| ERM configuration | QWK | across-seed SD |
|---|---|---|
| natural shuffling | **0.4147** | 0.0081 |
| domain-balanced batches | 0.3963 | **0.0643** |

Domain-balanced batching costs 0.018 QWK and multiplies the seed SD by **eight**.
The cause is structural: the sampler oversamples IDRiD's 335 images roughly
twelve times per epoch, which changes the domain prior and drags the class prior
with it, and how much damage that does depends on which twelve copies a seed
draws.

This is why the ERM control on both samplers was non-negotiable. Without it,
Deep CORAL's 0.4073 under balanced batches would sit below its own 0.4225 under
natural batches and read as the method failing, when in fact it is the sampler.
Every Δ in §2 is computed within a sampler for this reason.

It also means a 0.0643 SD swamps bar 1 by construction, so the balanced arm has
low power to detect a small true effect. That is stated as a limitation rather
than being used to claim the balanced arm proves more than it does.

---

## 4. Calibration and severe errors

Post-temperature-scaling ECE, and severe-error rate (|error| ≥ 2 grades), both
on the 35,108-image target:

| sampler | method | ECE (scaled) | Δ ECE | severe | Δ severe |
|---|---|---|---|---|---|
| natural | ERM | 0.1729 | — | 0.2407 | — |
| natural | Deep CORAL | 0.1746 | +0.0016 | 0.2278 | −0.0129 |
| natural | MixStyle | 0.1639 | −0.0091 | 0.2424 | +0.0017 |
| natural | GroupDRO | 0.1736 | +0.0007 | 0.2387 | −0.0020 |
| natural | IRMv1 | 0.0587 | −0.1142 | 0.2562 | +0.0155 |
| balanced | ERM | 0.1634 | — | 0.2539 | — |
| balanced | Deep CORAL | 0.1507 | −0.0127 | 0.2340 | −0.0199 |
| balanced | MixStyle | 0.1478 | −0.0157 | 0.2475 | −0.0064 |
| balanced | GroupDRO | 0.1523 | −0.0111 | 0.2413 | −0.0126 |
| balanced | IRMv1 | 0.0525 | −0.1110 | 0.2532 | −0.0007 |

Excluding IRM (§1), every calibration difference is ≤ 0.016 ECE and every
severe-error difference is ≤ 0.020, against seed SDs of 0.012–0.053 — **all of it
within seed noise**. Deep CORAL's −0.0199 severe under balanced batches has a
seed SD of 0.0505 and fails bar 1 by a factor of two and a half.

**No DG method in this comparison reduces severe errors.** Doubling the input
resolution reduces them by 22–48% (Phase 9). That contrast is the paper's point.

---

## 5. What this does NOT establish

- **Not that DG methods never work.** One target, one backbone, one resolution,
  one hyperparameter setting per method. DomainBed's finding is that DG methods
  do not beat ERM *under a fair sweep*; this is a fixed-configuration result and
  is stated as such.
- **Not that IRM is a bad method.** §1: it diverged under its standard schedule
  on this data. A λ sweep is the missing experiment.
- **Not a claim about the balanced arm's small effects.** SD 0.0643 gives it low
  power; it can rule out large effects, not small ones.
- **Not generalizable to the other three targets.** DDR, APTOS and IDRiD were not
  run with the full method set. ~7 h GPU for DDR is the obvious next experiment,
  and is listed as such rather than assumed.
- **The methods were not tuned per-domain**, because tuning on the target would
  violate the external-test protocol. Every method uses its published default.
  This is the correct choice and it is also a real limit on how strong the
  negative claim can be.

---

## 6. What this means for the paper

Phase 4's negative claim is now defensible. It rests on:

1. **Four methods, three families** — feature alignment, style augmentation,
   group reweighting, invariance. Not two variants of one idea.
2. **35,108 test images**, not 507.
3. **Paired seeds** with an across-seed SD bar, so a lucky seed cannot carry a
   claim.
4. **A sampler that lets every method run as designed** — 0% degenerate batches —
   with an ERM control on that same sampler.
5. **Diagnostics proving the machinery engaged**, so "your implementation was
   broken" is answerable with numbers.

Combined with Phase 9, the paper's thesis sharpens to something worth saying:
**the interventions the domain-generalization literature studies do not move
cross-domain DR grading, and the intervention it does not study — input
resolution — moves it more than all of them combined.** Phase 10 is the negative
half of that sentence and Phase 9 is the positive half.

---

## 7. Artifacts

| what | where |
|---|---|
| Generator | `analyse_methods.py` |
| Result table | `outputs/tables/method_comparison_lodo_eyepacs.csv` |
| 30 registry rows | `outputs/experiment_registry.csv` |
| Per-run evaluation reports | `outputs/reports/lodo_aptos-ddr-idrid__eyepacs_*_evaluation.json` |
| Training histories | `outputs/logs/lodo_aptos-ddr-idrid__eyepacs_*_history.csv` |
| Method implementations | `src/losses/deep_coral_alignment.py`, `src/losses/group_dro.py`, `src/losses/irm.py`, `src/training/methods.py` |
| Balanced sampler | `src/data/loaders.py`, `src/losses/deep_coral_alignment.py` |
| Tests | `tests/test_domain_objectives.py`, `tests/test_domain_balanced_batches.py`, `tests/test_method_comparison.py` |

Reproduce the analysis with no GPU:

```
python analyse_methods.py
```
