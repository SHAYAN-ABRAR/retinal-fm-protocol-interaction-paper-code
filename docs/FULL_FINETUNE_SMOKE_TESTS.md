# Full fine-tuning — engineering validation

**Date:** 2026-08-31 · **Hardware:** RTX 5060 Laptop, 8 GB VRAM

> **These are not results.** Two epochs of a 20-epoch cosine schedule, one seed,
> no comparison. They exist to prove the full-FT path runs, fits in memory, and
> checkpoints correctly before ~55 h of GPU time is committed. Nothing here may
> be cited as scientific evidence, and the QWK values below in particular must
> not be read as a model comparison.

## Commands

```bash
python run_lodo.py --method erm --backbone vit_large_mae_in1k --targets ddr \
    --seeds 42 --full-finetune --gradient-checkpointing \
    --batch-size 16 --learning-rate 1e-4 --epochs 2

RETFOUND_CHECKPOINT="...\RETFound_mae_natureCFP.pth" \
python run_lodo.py --method erm --backbone retfound_cfp --targets ddr \
    --seeds 42 --full-finetune --gradient-checkpointing \
    --batch-size 16 --learning-rate 1e-4 --epochs 2
```

## Telemetry

| | ImageNet-MAE | RETFound CFP |
|---|---|---|
| trainable parameters | **303,306,757 / 303,306,757 (100.0%)** | **303,306,757 / 303,306,757 (100.0%)** |
| weight load | timm `vit_large_patch16_224.mae` | **294 tensors, 100.0%** of parameters |
| physical batch | 16 | 16 |
| accumulation | 1 | 1 |
| **effective batch** | **16** | **16** |
| gradient checkpointing | on | on |
| peak VRAM (trainer) | **5.68 GB** | **5.68 GB** |
| peak VRAM (nvidia-smi, incl. context) | 6342 MiB of 8151 | 6342 MiB of 8151 |
| epoch 0 / epoch 1 wall | 865 s / 860 s | 858 s / 851 s |
| **seconds per step** | **0.50** (1732 steps/epoch) | 0.50 |
| NaN / Inf | none | none |
| checkpoint written | ✅ epoch 0 and 1 | ✅ epoch 0 and 1 |
| checkpoint reloaded | ✅ `best_qwk.pt` | ✅ `best_qwk.pt` |
| resumable state | ✅ model + optimiser moments + epoch + scaler + rng | ✅ same |
| source-val QWK trajectory | 0.2499 → 0.5796 | 0.5669 → 0.6810 |
| evaluation pass (18,101 images) | 41 min | 41 min |

Batch 16 fits, so the documented fallback ladder — `4×4`, `2×8`, `1×16`, all
holding effective batch at 16 — was not needed. It is implemented and tested
(`tests/test_gradient_accumulation.py`) against the day it is.

## Resume

`best_qwk.pt` reloading during evaluation proves save and load. It does **not**
prove resumability: a checkpoint missing the optimiser state reloads fine and
silently restarts Adam's moments from zero. Both `last.pt` files were therefore
opened and inspected directly:

```
top-level keys  ['config', 'epoch', 'global_step', 'metrics', 'model',
                 'optimizer', 'rng', 'scaler', 'scheduler']
model tensors   296 holding 303,306,757 parameters
optimiser state 296 tensors with moments
epoch recorded  1
```

## Two operational findings

**Disk is the binding constraint, not VRAM.** One full-FT run writes **7.91 GB**
of checkpoints (`last.pt` 3.4 + `best_qwk.pt` 3.4 + `best_loss.pt` 1.2), because
a ViT-L checkpoint carries 303 M parameters plus two AdamW moment tensors. Ten
runs need **79 GB**. After these two smoke tests, **13 GB** remained free. The
five-seed experiment cannot complete as configured and would most likely die
mid-checkpoint-write. Mitigations, none applied without a decision:

1. delete each run's checkpoints once it registers COMPLETE — predictions and
   reports are what every analysis reads; caps peak at ~16 GB;
2. stop writing `best_loss.pt`, which no analysis reads — saves 12 GB over ten runs;
3. reclaim from the existing 73.5 GB of checkpoints.

**The two arms must run sequentially.** The RETFound arm crashed on its first
launch with `CUBLAS_STATUS_INTERNAL_ERROR`, 18 seconds after the ImageNet arm
*registered*. Registering is not exiting: that process was still alive writing
its tables and figures, still holding 7.9 GB of VRAM. Two ViT-L full fine-tunes
do not coexist in 8 GB. Wait for process exit, not for the registry row.

## Load shedding

Two outages during these runs (23:04–23:30 the previous night, 21:36–21:39).
Both put the machine into Modern Standby, which **freezes** the process rather
than killing it; no work was lost either time. A hard power-off would cost at
most one epoch, since `last.pt` is written every epoch. The recorded
`train_seconds` for any run spanning a standby includes the frozen time and must
not be quoted as compute cost.
