# Phase 12 — Retinal foundation pretraining and the frozen representation

**Protocol:** frozen features, linear probe, leave-one-domain-out
**Backbones:** RETFound CFP · ImageNet-MAE ViT-L/16 · DenseNet121 (ImageNet)
**Targets:** DDR · APTOS · IDRiD — **EyePACS excluded, see §1**
**Seeds:** 42 / 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 — **ten** · 90 probes · 224 px
**Date:** 2026-08-26 → 2026-08-29 · ~1.5 h GPU
**Generators:** `run_retfound_probe.py --backbone …` · `analyse_linear_probe.py --control …`

> **Read this with [Phase 13](PHASE13_FINETUNE.md).** This phase measures the
> frozen *representation*. Phase 13 fine-tunes both models and finds the
> difference **does not survive** — which is the more important result of the
> pair, and changes what this phase is allowed to claim.

Every negative result in this project invites one question: *is this a
small-model artefact?* DenseNet121 is 7 M parameters, ConvNeXt-Tiny 28 M.
RETFound (Zhou et al., Nature 2023) is a 304 M ViT-L/16 pretrained with a masked
autoencoder on ~1.6 M retinal images, and is the reference foundation model for
this modality.

---

## 0. Answer

**As a frozen feature extractor, RETFound transfers worse across domains than the
general-purpose checkpoint it was built from — on two of three targets, at ten
seeds.**

| target | RETFound | ImageNet-MAE ViT-L | Δ | Δ/SD | verdict |
|---|---|---|---|---|---|
| DDR | 0.5130 | **0.5756** | +0.0626 | **2.46×** | ImageNet features better |
| APTOS | 0.4829 | **0.5963** | +0.1134 | **2.07×** | ImageNet features better |
| IDRiD | **0.6792** | 0.6363 | −0.0429 | 1.25× | within seed noise (CI spans zero) |

This survived three seeds, then five, then ten, and **strengthened** on APTOS at
every step (1.37× → 1.95× → 2.07×). That matters, because the fine-tuned arm
did not survive the same test — see Phase 13, and §5 here.

---

## 1. Why this comparison is the right one

RETFound's checkpoint records its own initialisation in its `args`:

```
resume = './mae_pretrain_vit_large_full.pth'
mask_ratio = 0.85,  epochs = 801,  model = mae_vit_large_patch16
```

**RETFound is `vit_large_patch16_224.mae` plus 801 further MAE epochs on retinal
images.** That checkpoint is in timm, so the control is not an approximation —
it is literally the model RETFound started from.

| | RETFound | ImageNet-MAE ViT-L |
|---|---|---|
| architecture | ViT-L/16 | ViT-L/16 ✓ |
| parameters | 303.3 M | 303.3 M ✓ |
| objective | MAE | MAE ✓ |
| initialisation | this checkpoint | — |
| **pretraining corpus** | **+1.6 M retinal** | ImageNet |

Everything downstream is identical: same 1024-d features, same per-dimension
standardisation fitted on source-training only, same linear head, same 200-epoch
schedule, same splits, same temperature scaling. **The pretraining corpus is the
only variable left.**

### EyePACS cannot be a target here

RETFound's CFP model was pretrained on MEH-MIDAS **and EyePACS**. Using EyePACS
as a held-out target would report leakage as generalization.
`assert_target_not_pretrained` raises rather than warns, because a warning in a
log still produces a results row indistinguishable from an honest one. The same
guard is now in `run_lodo.py`, which originally lacked it.

EyePACS remains a **source** for the other three targets, which is correct —
sources are seen by construction. RETFound has therefore seen source images the
control has not, which if anything should favour RETFound. It does not (§2).

DDR, APTOS and IDRiD are *not named* in RETFound's pretraining description, but
that description does not enumerate its public datasets. They are recorded as
`"unknown"`, not `"clean"`, and this belongs in the limitations.

---

## 2. The mechanism: same fit, worse transfer

| target | source QWK | | target QWK | | **source → target drop** | |
|---|---|---|---|---|---|---|
| | RETFound | MAE | RETFound | MAE | RETFound | MAE |
| DDR | 0.6001 | 0.5953 | 0.5130 | 0.5756 | **0.0872** | 0.0198 |
| APTOS | 0.6343 | 0.6495 | 0.4829 | 0.5963 | **0.1514** | 0.0532 |
| IDRiD | 0.6472 | 0.6591 | 0.6792 | 0.6363 | −0.0320 | 0.0228 |

**Source fit is near-identical** — as it must be for one architecture on one
task. RETFound does not fit the training domains better and then fail to
transfer. It fits them the *same* and transfers *worse*: its drop from source to
target is 4.4× larger on DDR and 2.8× larger on APTOS.

Severe errors (|error| ≥ 2 grades) follow on the two targets where the QWK gap
clears both bars:

| target | RETFound | ImageNet-MAE |
|---|---|---|
| DDR | 0.3183 | 0.2869 |
| APTOS | **0.4286** | 0.2997 |
| IDRiD | 0.1970 | 0.1980 |

On APTOS, RETFound severely misgrades 43% of images against the control's 30%.

### The DenseNet control, and the confound it left open

A second control was run first: DenseNet121, ImageNet, frozen, same pipeline.

| target | RETFound | DenseNet121 | Δ | Δ/SD | verdict |
|---|---|---|---|---|---|
| DDR | 0.5130 | 0.5504 | +0.0375 | 2.79× | ImageNet features better |
| APTOS | 0.4829 | 0.5217 | +0.0388 | 0.92× | within seed noise |
| IDRiD | 0.6792 | 0.7018 | +0.0226 | 0.94× | within seed noise |

ImageNet features score higher on 3 of 3, but only DDR clears both bars — and
this comparison confounds four variables at once (architecture, scale,
objective, corpus). **It could have been "CNN features transfer better than ViT
features".** The matched-architecture control rules that out, and is the reason
the claim in §0 can be made at all.

---

## 3. What would have been published without the controls

Two near-misses, both of which would have produced a striking, wrong result.

**The probe measured its own normalisation.** The first implementation fed raw
features through `nn.LayerNorm`, which normalises each sample across its 1024
dimensions and so discards the relative scale between dimensions a linear
classifier depends on. Measured on DDR:

| input to the linear head | source QWK | target QWK |
|---|---|---|
| LayerNorm (as first run) | 0.172 | **0.064** |
| per-dimension standardised | 0.503 | 0.437 |
| standardised + converged | 0.588 | **0.523** |
| exact L-BFGS optimum | 0.596 | 0.521 |

At 0.064 QWK the headline would have been *"a retinal foundation model is
useless for cross-domain DR grading"* — publishable, striking and false. The
head never predicted grades 1 or 3 at all.

The fix: standardisation fitted on the source-training split only, and 200
epochs at lr 5e-3, chosen by convergence **against an exact L-BFGS reference on
source validation**. No target-test number took part in that choice.

**Zero-variance dimensions.** With `std + 1e-6` as the divisor, a dimension that
never varies in training but fires on the target is amplified a millionfold.
DenseNet121 has 4–5 such dimensions per split; RETFound has none — checked, not
assumed, so its numbers were unaffected.

---

## 4. What this does NOT establish

- **This is a frozen-feature result and nothing more.** Phase 13 fine-tunes both
  models and the difference disappears. Any sentence of the form "RETFound is
  worse for cross-domain DR" is unsupported; the supported sentence is
  "RETFound's *frozen representation* is worse."
- **IDRiD does not agree**, and under fine-tuning it reverses into a result
  favouring RETFound at five seeds (Phase 13). It is also the smallest target,
  at 507 images.
- **Not a claim about foundation models in general** — one model, one modality,
  one downstream task.
- **Overlap is not ruled out** for DDR, APTOS and IDRiD (§1).

---

## 5. What this means for the paper

Taken alone this phase would say "domain-specific pretraining hurts transfer".
Taken with Phase 13 it says something more useful and more defensible:

**A linear probe ranks these two models in a way that fine-tuning does not
reproduce.** The frozen gap is large (+0.0626, +0.1134) and clears both bars on two targets
at ten seeds; after fine-tuning no target favours the ImageNet initialisation
and IDRiD favours RETFound with all five seeds agreeing. Probing is the standard shortcut for benchmarking a
foundation model, and here it gives the wrong answer about how the model will
actually be used.

That is the contribution: not "RETFound is bad", but **"how you evaluate a
medical foundation model determines the answer you get, and the cheap protocol
disagrees with the deployed one."**

---

## 6. Artifacts

| what | where |
|---|---|
| Extraction + probe | `run_retfound_probe.py --backbone {retfound_cfp,densenet121,vit_large_mae_in1k}` |
| Comparison | `analyse_linear_probe.py --control {densenet121,vit_large_mae_in1k}` |
| Per-run results | `outputs/tables/linear_probe_results.csv` |
| Comparisons | `outputs/tables/linear_probe_comparison_*.csv` |
| Loader + guards | `src/models/retfound.py` |
| Backbone registry | `src/models/backbones.py` |
| Tests | `tests/test_retfound.py` |
