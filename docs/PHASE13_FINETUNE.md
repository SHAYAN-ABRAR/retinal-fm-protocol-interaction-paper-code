# Phase 13 — Fine-tuning erases the difference Phase 12 found

**Protocol:** partial fine-tuning, last 4 of 24 blocks (50.4 M of 303.3 M trainable, **identical on both models**)
**Backbones:** RETFound CFP · ImageNet-MAE ViT-L/16
**Targets:** DDR (5 seeds) · APTOS (5 seeds) · IDRiD (3 seeds)
**Settings:** batch 16, lr 1e-4, 20 epochs, early stopping patience 6
**Date:** 2026-08-27 → 2026-08-29 · ~40 h GPU over 26 runs
**Generator:** `analyse_finetune.py` → `outputs/tables/finetune_comparison.csv`

Phase 12 found RETFound's frozen features transfer worse than the ImageNet MAE
checkpoint it was built from. There is exactly one fatal objection to that:
**RETFound is meant to be fine-tuned** — its own paper reports fine-tuning, not
linear probing — so a frozen-feature gap might say nothing about how anyone
would actually use it. This phase answers that objection, and the answer is not
the one Phase 12 predicted.

---

## 0. Answer

**The difference does not survive fine-tuning. No target favours the ImageNet
initialisation, and IDRiD favours RETFound.**

| target | RETFound | ImageNet-MAE | Δ | Δ/SD | seeds agreeing | verdict |
|---|---|---|---|---|---|---|
| DDR (n=5) | 0.6966 | 0.7183 | +0.0217 | 0.82× | 3/5 | within seed noise |
| APTOS (n=5) | 0.8261 | 0.8334 | +0.0073 | 0.42× | 4/5 | within seed noise |
| IDRiD (n=3) | **0.7844** | 0.7285 | −0.0559 | **1.25×** | 3/3 | **RETFound better** |

Set against Phase 12's frozen result on the same models, targets and splits:

| target | frozen Δ (Δ/SD) | fine-tuned Δ (Δ/SD) |
|---|---|---|
| DDR | **+0.0702 (2.42×)** | +0.0217 (0.82×) |
| APTOS | **+0.1075 (1.95×)** | +0.0073 (0.42×) |
| IDRiD | −0.0426 (1.24×, CI spans 0) | **−0.0559 (1.25×)** |

Fine-tuning shrinks the DDR gap by 69% and the APTOS gap by 93%, and flips
IDRiD from a null into a result favouring RETFound.

---

## 1. How the three-seed version was wrong

This phase was reported internally as holding on two targets before seeds 3 and
4 existed. It did not survive them, and the record of how it failed is more
useful than the final number.

| target | 3 seeds | 5 seeds |
|---|---|---|
| DDR | +0.0311, **1.03×** — cleared bar 1 | +0.0217, **0.82×** — fails |
| APTOS | +0.0156, **1.58×** — cleared both bars | +0.0073, **0.42×** — fails |

The per-seed deltas show why:

| seed | DDR Δ | APTOS Δ |
|---|---|---|
| 42 | +0.0463 | +0.0152 |
| 1 | −0.0036 | +0.0059 |
| 2 | +0.0506 | +0.0256 |
| **3** | **−0.0048** | **−0.0207** |
| **4** | +0.0199 | +0.0103 |

On both targets the first three seeds happened to be the favourable ones. Two
more moved DDR from a marginal pass to a clear fail and APTOS from 1.58× to
0.42×.

**This is the third time in this project that a three-seed result has evaporated
at five** — MixStyle in Phase 11 was the first, the DDR fine-tune the second,
APTOS the third.

The bootstrap does not warn you. On the anchor seed it is emphatic in the
opposite direction from the verdict:

| target | bootstrap CI (seed 42) | p | across-seed verdict |
|---|---|---|---|
| DDR | [+0.0363, +0.0563] | <0.001 | **within seed noise** |
| APTOS | [+0.0016, +0.0293] | 0.030 | **within seed noise** |
| IDRiD | [−0.1501, −0.0605] | <0.001 | RETFound better |

Both nulls carry an interval excluding zero. That is not a contradiction: the
bootstrap resamples *images within one seed* and correctly reports that this
model beat that model on this test set. It cannot see that a different seed
produces a different model. Only the across-seed SD sees that, and three seeds
is not enough to estimate it. This phase is the clearest evidence in the project
for why both bars exist.

---

## 2. IDRiD, the target that was run because it might disagree

IDRiD was deliberately added after DDR and APTOS, because the fine-tuned arm
otherwise covered exactly the two targets where the frozen comparison had gone
one way, and omitted the one where it had not. Reporting two wins with the
ambiguous case absent is not defensible regardless of the reason.

It came out favouring RETFound: −0.0559, 1.25× the across-seed SD, all three
seeds agreeing, bootstrap CI [−0.1501, −0.0605] excluding zero. Severe errors
agree — 0.0861 against the control's 0.1368.

Two caveats, stated because they cut against the result being over-read:

- **507 test images.** The smallest target in the project by an order of
  magnitude. Bootstrap intervals are correspondingly wide.
- **Three seeds, not five.** By this phase's own argument (§1), that is not
  enough to trust an SD. The honest reading is "IDRiD does not support the
  Phase 12 direction and may point the other way", not "RETFound wins on IDRiD".

---

## 3. Source fit, again

| target | source QWK | | target QWK | | source → target drop | |
|---|---|---|---|---|---|---|
| | RETFound | MAE | RETFound | MAE | RETFound | MAE |
| DDR | 0.7076 | 0.7029 | 0.6966 | 0.7183 | +0.0109 | −0.0154 |
| APTOS | 0.7614 | 0.7506 | 0.8261 | 0.8334 | −0.0647 | −0.0827 |
| IDRiD | 0.7671 | 0.7586 | 0.7844 | 0.7285 | −0.0172 | +0.0301 |

RETFound fits the source domains marginally *better* on all three targets once
fine-tuned, which it did not do as a frozen extractor. Whatever the frozen
representation lacked for transfer, four unfrozen blocks and 20 epochs are
enough to recover it.

Severe errors (|error| ≥ 2 grades) tell the same story — no separation on the
two nulls, and IDRiD favouring RETFound:

| target | RETFound | ImageNet-MAE |
|---|---|---|
| DDR | 0.2066 | 0.1906 |
| APTOS | 0.0906 | 0.0874 |
| IDRiD | **0.0861** | 0.1368 |

Compare Phase 12's frozen severe-error gaps on the same targets — 0.3201 against
0.2881 on DDR, and 0.4219 against 0.3037 on APTOS. Fine-tuning roughly halves
the severe-error rate for both models and closes the gap between them.

---

## 4. What this does NOT establish

- **Not that RETFound and its initialisation are equivalent.** Two targets are
  nulls, not demonstrated equalities, and a null at n=5 with these SDs cannot
  exclude an effect of ~0.02 QWK.
- **Not full fine-tuning.** The last 4 of 24 blocks were unfrozen. Full
  fine-tuning was measured to fit in 5.67 GB with gradient checkpointing at
  batch 16, so it is affordable here and remains the obvious follow-up.
- **Not tuned per model.** Both get lr 1e-4 and the same schedule, which is what
  makes the comparison matched, and also means neither is at its own optimum.
- **IDRiD is three seeds** (§2).

---

## 5. What this means for the paper

The claim Phase 12 appeared to support — *domain-specific pretraining hurts
cross-domain transfer* — is not supported. What the two phases jointly support
is narrower and, for a methods-conscious venue, more useful:

**A linear probe and a fine-tune rank these two models differently, on the same
data, with everything else held constant.** The frozen protocol shows a large
gap on two of three targets; the fine-tuned protocol shows none, and reverses on
the third. Linear probing is the standard cheap benchmark for a foundation
model, and here it does not predict the behaviour of the model as deployed.

The paper's honest statement is therefore about **evaluation protocol**, not
about RETFound: how you measure a medical foundation model determines the
conclusion you reach, and the cheap measurement disagrees with the expensive
one. That is a claim this project can defend at five seeds, with a matched
architecture, on three targets, at two protocol levels.

---

## 6. Artifacts

| what | where |
|---|---|
| Generator | `analyse_finetune.py` |
| Result table | `outputs/tables/finetune_comparison.csv` |
| Per-run rows | `outputs/tables/lodo_results.csv` (`trainable_blocks == 4`) |
| Registry | `outputs/experiment_registry.csv` |
| Training logs | `outputs/logs/ft_{target}_{backbone}.log` |
| Flags | `run_lodo.py --trainable-blocks --learning-rate --backbone` |

Reproduce the analysis with no GPU:

```
python analyse_finetune.py
```
