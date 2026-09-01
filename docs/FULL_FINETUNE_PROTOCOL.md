# Full fine-tuning — protocol, pre-registered

**Fixed 2026-08-31, before any seed-42 target metric was seen.**

## What this experiment is

**Matched fixed-budget full fine-tuning.** Both initialisations receive the
*same* downstream optimisation, so the only thing that differs between the two
arms is the state the encoder starts from.

## What it is not

It is **not** a reproduction of the canonical RETFound fine-tuning recipe, and
it is **not** separately hyperparameter-optimal for either initialisation.
Published RETFound and recent retinal foundation-model work use different and
generally longer fine-tuning schedules — more epochs, layer-wise learning-rate
decay, different augmentation, larger effective batch. A recipe tuned per model
would answer "which model can be made to perform best", which is a different and
also legitimate question. This one answers "does the retinal pretraining stage
help, holding adaptation fixed", and the fixed budget is what makes it a control
rather than a comparison of tuning effort.

This is a limitation and a sensitivity concern, stated rather than hidden: a
budget favourable to one initialisation and not the other could produce this
result. It belongs in the paper's limitations section, and it bounds the claim
to the protocol tested.

## Protocol

| | |
|---|---|
| trainable | full encoder + head |
| trainable parameters | **303,306,757** (asserted at runtime, not assumed) |
| resolution | 224 px |
| physical batch | 16 |
| accumulation steps | 1 |
| **effective batch** | **16** |
| gradient checkpointing | on |
| optimiser | AdamW |
| learning rate | 1e-4 |
| weight decay | 1e-4 |
| warmup | 1 epoch |
| schedule | cosine |
| **maximum epochs** | **20** |
| checkpoint selection | source-validation QWK |
| early stopping | source-validation QWK, patience 6 |
| target domain | **DDR — never used for selection, stopping or calibration** |

## Seeds — pre-committed

**42, 1, 2, 3, 4** for **both** ImageNet-MAE and RETFound-CFP.

Ten runs. This set is fixed now and does not change after target results are
seen. Specifically:

- no stopping early because the target-domain difference is or is not significant;
- no adding seeds to push a borderline result over a threshold;
- no dropping a seed that disagrees with the others.

The seeds match the partial fine-tuning arm, so the protocol interaction
(frozen / partial / full) is testable on a common seed set.

## Recipe adequacy

Whether 20 epochs is an adequate optimisation budget is judged on
**source-validation evidence only** — train loss, source-validation loss, QWK
and macro-F1, the learning-rate curve, the best epoch, early-stopping state,
gradient norms, NaN/Inf and peak VRAM. If either arm's best source-validation
performance lands at the end of the budget while still improving materially,
the 20-epoch cap is flagged as potentially truncating.

The DDR target plays no part in that judgement. Using it would tune the protocol
on the held-out domain, which is the one thing this study exists to avoid.

## Cost

| | measured |
|---|---|
| step time | 0.50 s |
| steps per epoch | 1,732 |
| per epoch | 865 s |
| per 20-epoch run | ~4.8 h + ~0.7 h evaluation |
| **ten runs** | **~55 h** |

## Storage

With the slimmed checkpoint policy, a full-FT run keeps `best_qwk.pt` (~1.2 GB,
model weights only) and, until pruned, `last.pt` (~3.6 GB). `best_loss.pt` is
disabled for full fine-tuning — nothing in this repository reads it.
`prune_checkpoints.py` deletes `last.pt` only after the run is COMPLETE,
evaluated, predicted, historied, audited and its `best_qwk.pt` verified to load.
