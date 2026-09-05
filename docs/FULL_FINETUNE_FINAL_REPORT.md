# Full fine-tuning on DDR — final five-seed report

**Frozen 2026-09-05.** Ten runs, seeds 42/1/2/3/4 × {ImageNet-MAE, RETFound-CFP},
all COMPLETE. Protocol pre-registered in `docs/FULL_FINETUNE_PROTOCOL.md` before
any target metric was read; nothing in it was changed after.

Generators: `analyse_full_finetune.py`, `analyse_ft_convergence.py`.
Tables: `outputs/tables/full_finetune_*.csv`. Every number below is read from
those files, none retyped.

## How uncertainty and inference are separated

The **crossed seed × case bootstrap** gives the point estimate and a 95%
uncertainty interval. It gives **no p-value**: its distribution is built around
the empirical estimate rather than under H0, so its tail mass is not a
calibrated test. **Seed-level inference** — a one-sample *t*-test on the
per-seed paired effects, with the exact sign-flip permutation as a sensitivity
check — supplies the formal test.

A claim is treated as established only when **both** agree: the seed-level test
reaches significance *and* the crossed interval excludes zero. They answer
different questions (would another seed agree / would another sample of
patients agree), and a generalisation claim needs both.

At five seeds the exact sign-flip test cannot return a two-sided *p* below
**0.0625**. Every sign-flip value below is at or near that floor and none can
be read as evidence of absence.

## 1. The five-seed table (QWK on held-out DDR)

| seed | ImageNet-MAE | RETFound | Δ (ImageNet − RETFound) |
|---|---|---|---|
| 42 | 0.6494 | 0.6706 | -0.0212 |
| 1 | 0.5428 | 0.6607 | -0.1180 |
| 2 | 0.6119 | 0.6657 | -0.0538 |
| 3 | 0.6536 | 0.6473 | +0.0062 |
| 4 | 0.6468 | 0.6796 | -0.0329 |

## 2. PRIMARY — model comparison under full fine-tuning

| | |
|---|---|
| mean paired Δ | **−0.0439** |
| SD of paired Δ | 0.0467 |
| crossed 95% CI | [−0.0861, −0.0103] |
| paired *t*-test *p* | **0.1035** |
| sign-flip *p* | 0.1250 (floor 0.0625) |
| sign agreement | 4/5 |

**No difference is demonstrated under full fine-tuning.** The crossed interval
excludes zero but the seed-level test does not reach significance, and under
the two-bar rule that is not established. The disagreement is informative: the
effect is reasonably consistent across *patients* but not across *seeds*, and
seed variance is the binding uncertainty here.

**This is not evidence of equivalence.** A null at five seeds bounds the effect
loosely; the interval says how loosely. Only one comparison was pre-specified as
primary, so no Holm family is invented for it.

## 3. PRIMARY — protocol interaction, full vs frozen

I_full(s) = D_full(s) − D_frozen(s), where D is ImageNet − RETFound within a
protocol. The bootstrap resamples cases **once per replicate and applies that
one sample to all four conditions**, preserving the pairing.

| seed | D_frozen | D_full | I_full |
|---|---|---|---|
| 42 | +0.0477 | -0.0212 | -0.0688 |
| 1 | +0.0517 | -0.1180 | -0.1696 |
| 2 | +0.0663 | -0.0538 | -0.1201 |
| 3 | +0.0652 | +0.0062 | -0.0590 |
| 4 | +0.1200 | -0.0329 | -0.1529 |

| | |
|---|---|
| mean I_full | **−0.1141** |
| SD | 0.0493 |
| crossed 95% CI | **[−0.1548, −0.0728]** |
| paired *t*-test *p* | **0.0066** |
| sign-flip *p* | 0.0625 (at the floor) |
| sign agreement | **5/5** |

**This is the strongest result in the study.** Both bars agree and every seed
moves the same way. The relative advantage of the two initialisations *does*
change with adaptation depth: the general-purpose initialisation is ahead under
a frozen probe on all five seeds, and that advantage is gone — point estimates
mostly reversed — under full fine-tuning.

Note what this does and does not license. It establishes that the **contrast
between protocols** is real. It does **not** establish that RETFound beats
ImageNet-MAE under full fine-tuning: §2 is a null. The defensible claim is that
the protocol you choose changes the answer, which is the paper's thesis, tested
directly rather than inferred from one comparison being significant and another
not.

## 4. SECONDARY interactions

Holm-corrected within this two-member family. They do not replace §3.

| contrast | mean | crossed 95% CI | *p* | Holm | sign |
|---|---|---|---|---|---|
| D_partial − D_frozen | −0.0485 | [−0.0834, −0.0153] | 0.0543 | 0.0843 | 5/5 |
| D_full − D_partial | −0.0656 | [−0.1031, −0.0229] | 0.0422 | 0.0843 | 4/5 |

Neither survives correction. The pattern is monotone — the ImageNet-MAE
advantage attenuates progressively as more of the network is unfrozen — but
each individual step is underpowered at five seeds. Only the full span
(frozen → full) is established.

## 5. SECONDARY outcomes, full fine-tuning

Uncorrected and exploratory. **QWK remains primary**; none of these replaces it.

| metric | ImageNet-MAE | RETFound | Δ | SD | *p* | sign |
|---|---|---|---|---|---|---|
| **QWK (primary)** | 0.6209 | 0.6648 | −0.0439 | 0.0467 | 0.1035 | 4/5 |
| macro F1 | 0.4457 | 0.4404 | +0.0052 | 0.0311 | 0.7254 | 4/5 |
| balanced accuracy | 0.4608 | 0.5004 | −0.0396 | 0.0263 | 0.0282 | 5/5 |
| severe-error rate | 0.2372 | 0.2168 | +0.0204 | 0.0183 | 0.0673 | 5/5 |
| MAE grade | 0.6057 | 0.5981 | +0.0075 | 0.0596 | 0.7911 | 2/5 |
| within-1-grade | 0.7628 | 0.7832 | −0.0204 | 0.0183 | 0.0673 | 5/5 |
| macro AUROC | 0.7956 | 0.7993 | −0.0037 | 0.0226 | 0.7360 | 3/5 |
| ECE (temperature-scaled) | 0.0925 | 0.1103 | −0.0177 | 0.0204 | 0.1239 | 4/5 |
| NLL (temperature-scaled) | 1.0062 | 1.0642 | −0.0580 | 0.0748 | 0.1583 | 4/5 |

Balanced accuracy reaches *p* = 0.0282 uncorrected with 5/5 sign agreement.
It is **not** promoted to primary because QWK is inconvenient — it is reported
as one exploratory outcome among nine, where roughly one result at *p* < 0.05
is expected by chance alone.

## 6. Source-validation behaviour — the seed-42 observation replicates

| model | seed | epochs | early stop | best epoch | best QWK | final train loss | val loss min→final | overfit ratio |
|---|---|---|---|---|---|---|---|---|
| ImageNet-MAE | 42 | 20 | no | 16 | 0.6687 | 0.4228 | 0.6599 → 0.7500 | 1.14 |
| ImageNet-MAE | 1 | 20 | no | 18 | 0.5881 | 0.6170 | 0.7017 → 0.7049 | 1.00 |
| ImageNet-MAE | 2 | 20 | no | 16 | 0.6637 | 0.4351 | 0.6691 → 0.7498 | 1.12 |
| ImageNet-MAE | 3 | 20 | no | 19 | 0.6725 | 0.4397 | 0.6546 → 0.7196 | 1.10 |
| ImageNet-MAE | 4 | 20 | no | 19 | 0.6643 | 0.4473 | 0.6609 → 0.7445 | 1.13 |
| RETFound | 42 | 17 | **yes** | 10 | 0.6604 | 0.0570 | 0.6769 → 1.8484 | **2.73** |
| RETFound | 1 | 13 | **yes** | 6 | 0.6783 | 0.2053 | 0.6641 → 1.1998 | **1.81** |
| RETFound | 2 | 16 | **yes** | 9 | 0.6783 | 0.0765 | 0.6696 → 1.7078 | **2.55** |
| RETFound | 3 | 17 | **yes** | 10 | 0.6720 | 0.0645 | 0.6638 → 1.6999 | **2.56** |
| RETFound | 4 | 15 | **yes** | 8 | 0.6700 | 0.1025 | 0.6732 → 1.5468 | **2.30** |

| | ImageNet-MAE | RETFound |
|---|---|---|
| best epoch | 17.6 ± 1.5 | **8.6 ± 1.7** |
| epochs run | 20.0 | 15.6 |
| early stopped | **0/5** | **5/5** |
| overfit ratio | 1.10 ± 0.05 | **2.39 ± 0.36** |

**The seed-42 observation was not seed-specific — it replicates on all five.**
RETFound converges roughly twice as early (best epoch 8.6 vs 17.6), early-stops
in every run while ImageNet-MAE never does, and overfits the source pool far
harder (final validation loss 2.4× its minimum, against 1.1×; final training
loss as low as 0.057).

This is descriptive evidence about **optimisation behaviour, not general
superiority**. It does not say RETFound is better or worse — it says the two
initialisations reach their best source-validation point at very different
times under one fixed budget. Since selection is on best source-validation QWK,
the reported model is unaffected. But **one fixed budget is not equally
well-matched to both initialisations**, and that belongs in the limitations: a
budget tuned per model would be a different (also legitimate) experiment, and
this one deliberately holds adaptation fixed instead.

**RETFound is clearly not budget-truncated:** all five runs peak earlier and
early-stop. **ImageNet-MAE peaks before the final epoch in three seeds but at
the final epoch in two seeds** (best epochs 16, 18, 16, 19, 19 under a
20-epoch budget indexed 0–19); residual optimization-budget sensitivity
therefore cannot be excluded.

This does not weaken the design. It is a **matched fixed-budget** experiment,
not a claim that either model was individually hyperparameter-optimal, and the
budget was fixed before any target result was seen. No DDR target result was
used to justify changing it.

## 7. Cost and integrity

| | |
|---|---|
| total GPU runtime, 10 runs | **42.3 h** (ImageNet 23.7 h, RETFound 18.6 h) |
| peak VRAM, every run | 5.68 GB of 8.15 GB |
| trainable parameters | 303,306,757 / 303,306,757 (100.0%), asserted at runtime |
| effective batch | 16 (physical 16 × accumulation 1) |
| NaN / Inf | none |
| disk after pruning | 17 GB free; 1.21 GB `best_qwk.pt` retained per run |
| test suite | 411 passed |
| audit_consistency | PASS, 2977 checks |
| experiment identity | 0 conflicts involving any full-FT run |
| check_numbers / LaTeX gate | clean |

One CUDA abort (`CUDAEvent::createEvent`) hit seed 3 RETFound a minute into
training. No checkpoint or registry row existed, the driver logged no reset, and
the retry ran clean — recorded as `ft_retfound_ddr_s3_cudaabort.log`.

Seed 2 RETFound was restarted from scratch after a scheduler-restore bug was
found (below); its first attempt is not part of the frozen set.

## 8. Two resume bugs found during this queue

Both were found because a deliberate pause exercised the resume path, and both
would have gone unnoticed otherwise.

**RNG state was never restored.** `torch.set_rng_state` needs a CPU uint8
tensor; `load_checkpoint` uses `map_location="cuda"`, so the saved state
arrived on the device and was rejected. All four streams sat under one
try/except, so it logged one vague warning and continued — resuming while no
longer reproducible from its seed.

**The scheduler was never restored.** `resume()` could not restore it: the
scheduler needs `steps_per_epoch` and is built in `fit()`, which runs *after*
`resume()`. The learning-rate schedule therefore restarted from step zero at
every resume. A run interrupted at epoch 2 of 20 re-ran warmup and then followed
a cosine two epochs behind. It completed, registered COMPLETE and looked normal;
only the per-epoch lr in the log revealed it.

`audit_resumed_runs.py` reconstructs each run's intended cosine and compares it
epoch by epoch. **No run in this frozen full-FT set is affected.** Two *earlier*
runs have a reported model trained under a corrupted schedule and are flagged,
untouched, for a re-run decision:

- `lodo_aptos-ddr-eyepacs__idrid_vit-large-mae-in1k_erm-b16-tb4-lr0.0001_s3`
- `lodo_aptos-eyepacs-idrid__ddr_densenet121_mixstyle-b32_s1`

## 9. Should APTOS full FT be added?

**Recommendation: yes, but it is not required for the paper to stand.**

For: the primary interaction (§3) currently rests on one held-out domain.
Replicating it on APTOS would move the central claim from "on DDR" to "on two
domains", which is the single cheapest strengthening available — and APTOS
already has frozen and partial arms at the same five seeds, so the interaction
is computable the moment the full-FT arm exists.

Against: ~42 h GPU and ~10 GB disk, and it does not fix the study's real
limitation, which is five seeds rather than one domain. A reviewer objecting to
the seed budget is not answered by another domain.

The strongest version of the paper has both domains. The honest version with
one domain is publishable provided §3 is stated as a single-domain result.

**Not launched.** No experiment has been started automatically.
