# Phase 9 — Input resolution: the largest effect in the project

**Run:** ERM, DenseNet121, **512 px, batch 16**, 20 epochs
**Compared against:** ERM, DenseNet121, **224 px, batch 32**, identical otherwise
**Seeds:** 42 on every cell; **42 / 1 / 2** on LODO APTOS, IDRiD and EyePACS
**Date:** 2026-08-22 → 2026-08-25 · 29.9 h GPU over 14 runs
**Generator:** `analyse_resolution.py` → `outputs/tables/resolution_comparison.csv`

Phases 4 and 5 established that no domain-generalization method in the
comparison beats plain ERM. Phase 8 established that a 4×-larger backbone buys
discrimination and not calibration. This phase asks the question those two leave
open: **if the method does not matter and the architecture barely does, what
does?**

The answer is the one nobody writes a paper about. It is the input resolution.

---

## 0. Answer

**Doubling the input resolution improves cross-domain grading on all four
targets, by more than any method, any backbone, and four times the training
data.**

| What was changed | Best QWK gain on EyePACS (LODO) | Cost |
|---|---|---|
| **224 px → 512 px** | **+0.0967** | 1.0 h GPU per run |
| DenseNet121 → ConvNeXt-Tiny | +0.0554 | 0.2 h GPU per run |
| 4× the training data (extrapolated to the in-domain budget) | +0.0177 | — |
| Every DG method in Phase 4 | none beat ERM | 0.3 h GPU per run |

Severe errors — misgrades of two steps or more — fall on all four LODO targets
and by **22–48%** on the three where the fall clears both bars.

**The uncomfortable part:** this is the largest, cheapest and least novel
intervention in the study, and it is the one a methods-focused paper would never
have run. It is reported here in full rather than buried, because a domain-
generalization result that a resolution change beats is a result the field
should know is beatable that way.

---

## 1. What is actually being compared

**Not resolution alone.** 512 px at batch 32 does not fit in 8 GB of VRAM, so
the 512 px arm runs at batch 16. The comparison is between two configurations:

```
224 px, batch 32     vs     512 px, batch 16
```

Resolution is the dominant term and the intended one, but it is not the only
thing that changed, and no run on this hardware isolates it. Every row of
`resolution_comparison.csv` carries a `configuration` column saying so, and
`tests/test_resolution_comparison.py` fails if a row names only the resolution.

Measured VRAM at 512 px / batch 16: **5.33–5.48 GB peak**, against 2.42 GB at
224 px / batch 32. Batch 32 at 512 px projects to roughly 11 GB, which is why it
was not run.

**What this means for the paper.** The honest claim is *"a 512 px, batch-16
configuration outperforms a 224 px, batch-32 one"*. Attributing it entirely to
pixels would be claiming an experiment that was never run. The attribution is
nonetheless reasonable — batch size 32 → 16 at a fixed learning rate is a small
perturbation next to a 4× increase in input pixels — but reasonable is not
measured, and the distinction is stated wherever the result is.

## 2. Leave-one-domain-out — the result that matters

| Target | n | seeds | 224 px | 512 px | Δ QWK | × seed SD | 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|
| DDR | 12,424 | 1 | 0.7334 | **0.7795** | **+0.0461** | 8.4× ¹ | [+0.0366, +0.0561] | **established** |
| APTOS | 3,504 | 3 | 0.8590 | 0.8745 | +0.0155 | **1.6×** | [+0.0085, +0.0317] | established, but marginal |
| IDRiD | 507 | 3 | 0.7413 | **0.8293** | **+0.0880** | 3.4× | [+0.0348, +0.1074] | **established** |
| EyePACS | 35,108 | 3 | 0.4147 | **0.5115** | **+0.0967** | **4.8×** | [+0.1112, +0.1280] | **established** |

¹ borrowed bar — see §5. APTOS clears bar 1 by 1.6× against a one-SD threshold
and should be read as suggestive; see §5.

**EyePACS is the headline.** The hardest target in the study, on which no
method, no source combination and no recalibration has ever helped, gains
**+0.0967 QWK** at 4.8× its own paired seed SD. That single change closes a
third of the gap between cross-domain and in-domain performance.

### Severe errors

The clinically load-bearing metric: a two-step misgrade is what sends a
referable patient home.

| Target | 224 px | 512 px | Δ | relative | 95% CI | verdict |
|---|---|---|---|---|---|---|
| DDR | 0.1666 | 0.1295 | −0.0371 | **−22%** | [−0.0428, −0.0311] | **established** |
| APTOS | 0.0628 | 0.0548 | −0.0080 | −13% | [−0.0194, −0.0023] | within seed noise |
| IDRiD | 0.1177 | 0.0611 | −0.0565 | **−48%** | [−0.0750, −0.0256] | **established** |
| EyePACS | 0.2407 | 0.1822 | −0.0585 | **−24%** | [−0.0809, −0.0722] | **established** |

Three of four clear both bars. On EyePACS this removes roughly one severe error
in four, on a test set of 35,108 images; on IDRiD nearly half, on 507.

APTOS falls in the right direction on every seed but by less than the seed-to-
seed variation of that fall. At one seed it read as established (−18%); three
seeds withdrew it. It is the same target whose QWK row is marginal, and for the
same reason: APTOS is the easiest domain in the study and has the least room to
improve.

## 3. In-domain — where it does *not* help

The same comparison on each domain's own held-out split:

| Domain | n | 224 px | 512 px | Δ QWK | × seed SD | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| DDR | 1,862 | 0.8757 | 0.8777 | +0.0020 | 0.2× ¹ | [−0.0156, +0.0197] | within seed noise |
| APTOS | 354 | 0.9091 | 0.9162 | +0.0071 | 0.7× ¹ | [−0.0249, +0.0411] | within seed noise |
| IDRiD | 102 | 0.5884 | **0.7197** | **+0.1313** | 6.9× ¹ | [+0.0137, +0.2618] | **established** |
| EyePACS | 5,268 | 0.7090 | **0.8004** | **+0.0914** | 6.6× ¹ | [+0.0691, +0.1136] | **established** |

**This is the most interesting table in the phase.** On DDR and APTOS — the two
domains where in-domain performance was already high — extra resolution buys
**nothing**. On IDRiD and EyePACS it buys a great deal.

Two different reasons, and they should not be conflated:

- **IDRiD** has 102 test images and a 224 px in-domain QWK of 0.5884, *below* its
  own LODO score. It is data-starved, not resolution-starved, and the interval
  ([+0.0137, +0.2618]) is correspondingly enormous. Treat the direction as real
  and the magnitude as unresolved.
- **EyePACS** is the genuine finding: 24,574 training images, and 224 px was
  still leaving 0.09 QWK on the table. Its images are the lowest-quality in the
  study, and small lesions — microaneurysms, the grade-1 signal — are the first
  thing lost to downsampling.

**So resolution buys generalization more reliably than it buys accuracy.** Under
LODO all four targets improve; in-domain only two do. That asymmetry is the
paper-relevant part: the gain is largest exactly where the model is being asked
to work on images it has never seen.

## 4. Calibration — descriptive only

Post-temperature ECE, both arms, temperature fitted on source validation:

| | DDR | APTOS | IDRiD | EyePACS |
|---|---|---|---|---|
| LODO, 224 px | 0.0551 | 0.0406 | 0.1047 | 0.1729 |
| LODO, 512 px | 0.0480 | 0.0626 | 0.0870 | 0.1523 |
| in-domain, 224 px | 0.0107 | 0.0514 | 0.1712 | 0.0364 |
| in-domain, 512 px | 0.0219 | 0.0491 | 0.1240 | 0.0252 |

**None of these differences has been tested, and none should be quoted as an
effect.** They have no across-seed SD on the 512 px side except EyePACS LODO,
and no paired bootstrap has been run on them.

This warning is not boilerplate. Phase 8 reported exactly this kind of
sign-counted calibration comparison as a headline — "worse on three of four" —
and three seeds dissolved it entirely. The same trap is available here: the
table above is "better on three of four under LODO", which is precisely the
shape of claim that did not survive last time.

What can be said: **512 px does not repair the calibration failure.** EyePACS
under LODO stays at ECE ≈ 0.15 after temperature scaling, against ≈ 0.03 in
domain. The thesis is unaffected by the resolution change, which is the point
worth making.

## 5. What this does NOT establish

- **DDR under LODO, and all four in-domain rows, still use a borrowed bar.**
  Those cells have one 512 px seed, so "× seed SD" divides by the **224 px**
  across-seed SD rather than a measured one.

  The proxy is consistently optimistic. Measured four times:

  | Comparison | borrowed SD | true paired SD | inflation |
  |---|---|---|---|
  | ConvNeXt vs DenseNet, DDR | 0.0055 | 0.0165 | 3.0× |
  | ConvNeXt vs DenseNet, APTOS | 0.0054 | 0.0109 | 2.0× |
  | 512 px vs 224 px, EyePACS LODO | 0.0081 | 0.0201 | 2.5× |
  | 512 px vs 224 px, APTOS LODO | 0.0054 | 0.0094 | 1.7× |

  APTOS and IDRiD were run with seeds 1 and 2 on 2026-08-24 (13 h) precisely
  because they were the rows a real SD could move. **Both survived**, but not
  equally: IDRiD strengthened from 2.8× to **3.4×**, while APTOS fell from 3.8×
  to **1.64×** and is now the weakest established row in the phase.

  DDR's remaining borrowed ratio of 8.4× would still clear at any inflation
  observed so far. The in-domain rows are 6.9× (IDRiD) and 6.6× (EyePACS)
  established, and 0.2× / 0.7× null, so neither direction is close enough to
  the boundary for the proxy to be deciding them.

- **Bar 1 is a one-SD threshold, and APTOS is close to it.** The project's
  criterion throughout is `|Δ| > SD(Δ)` — the effect must exceed *one* across-
  seed standard deviation, not two. APTOS clears it at 1.64× on three seeds,
  where the SD is itself estimated from three numbers and carries roughly 40%
  relative uncertainty. **Read APTOS as suggestive rather than solid**, and do
  not quote it beside DDR, IDRiD and EyePACS as though the four were equally
  supported. Two more seeds would settle it; nothing else will.

- **That the cause is pixels rather than batch size.** See §1. No run separates
  them on 8 GB.

- **That 512 px is optimal.** Two points do not locate a maximum. 768 px and
  1024 px were not run and will not fit at any useful batch size on this GPU.

- **Anything about resolution × backbone.** Every run here is DenseNet121. The
  two largest interventions in the project have not been combined, and given
  ConvNeXt's seed variance (§7 of Phase 8) that combination would need at least
  two seeds to mean anything.

- **That the deployment cost shrinks proportionally.** The 512 px deployment
  cost is measured at seed 42 on both sides only. It is smaller than at 224 px
  on DDR and EyePACS and newly resolves on APTOS, but it is single-seed and
  therefore subject to the same correction as everything else in this list.

## 6. What this means for the paper

1. **Report it.** A domain-generalization study whose largest effect is an input
   resolution change should say so. Omitting it would leave the paper's method
   comparison looking more consequential than the evidence supports.
2. **It strengthens the negative result rather than undermining it.** Phase 4
   found no DG method beating ERM. That reads as a weak experiment until you can
   show the pipeline *is* capable of moving the metric — by 0.0967 on the
   hardest target — when something that matters is changed. The null result and
   this one are the same argument.
3. **It sharpens the clinical framing.** Severe errors down 18–42% for a
   configuration change requiring no new data, no new labels and no new method
   is a directly actionable finding for anyone deploying a screening model.
4. **It is not novel and must not be dressed as such.** That higher resolution
   helps fine-grained retinal grading is known and expected. The contribution
   here is the *comparison*: resolution against methods, backbones and data
   volume, on identical splits under one protocol, with intervals on all of it.

## 7. Artifacts

| Path | Contents |
|---|---|
| `analyse_resolution.py` | generator; both protocols, paired seeds, two bars |
| `outputs/tables/resolution_comparison.csv` | §2, §3, §4 |
| `outputs/tables/lodo_results_r512.csv` | 512 px LODO rows |
| `outputs/tables/in_domain_results.csv` | both resolutions, keyed on `image_size` |
| `outputs/tables/resolution_effect_eyepacs_s42.csv` | superseded single-seed table, kept as a record |
| `outputs/tables/deployment_cost_r512_s42.csv` | 512 px deployment cost, seed 42 |
| `tests/test_resolution_comparison.py` | five regression tests |

Reproduce:

```
python run_lodo.py      --image-size 512 --batch-size 16 --seeds 42
python run_in_domain.py --image-size 512 --batch-size 16 --seeds 42
python analyse_resolution.py
```

`analyse_resolution.py` re-derives the seed-42 EyePACS in-domain figure on every
run and exits non-zero if it disagrees with the superseded table, so the two
cannot drift apart unnoticed.

## 8. A note on how this was nearly missed

512 px was run to check whether the in-domain ceilings were resolution-limited,
not as an experiment in its own right. It produced the largest effect in the
project and then sat undocumented for two days while smaller findings were
written up, because it had not been *planned* as a finding.

The 512 px in-domain runs for DDR, APTOS and IDRiD were added only after
noticing that a 512 px LODO model differenced against a 224 px in-domain
reference would charge the resolution change to domain shift — which
`run_in_domain.py` now refuses to do. Had that gone unnoticed, the deployment
cost, the paper's central number, would have absorbed this entire effect.
