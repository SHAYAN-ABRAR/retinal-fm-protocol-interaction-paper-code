# Phase 12 — Retinal foundation pretraining made cross-domain transfer worse

**Protocol:** frozen features, linear probe, leave-one-domain-out
**Backbones:** RETFound CFP · ImageNet-MAE ViT-L/16 · DenseNet121 (ImageNet)
**Targets:** DDR · APTOS · IDRiD — **EyePACS excluded, see §1**
**Seeds:** 42 / 1 / 2 · 27 probes · 224 px
**Date:** 2026-08-26 → 2026-08-27 · ~1 h GPU total
**Generators:** `run_retfound_probe.py --backbone …` · `analyse_linear_probe.py --control …`

Every negative result in this project invites one question: *is this an artefact
of small models?* DenseNet121 is 7 M parameters and ConvNeXt-Tiny 28 M. RETFound
(Zhou et al., Nature 2023) is a 304 M ViT-L/16 pretrained with a masked
autoencoder on ~1.6 M retinal images, and is the reference foundation model for
this modality.

The answer turned out to be more interesting than the question.

---

## 0. Answer

**With architecture, scale, objective and lineage held constant, continuing MAE
pretraining on 1.6 M retinal images made cross-domain transfer *worse* on two of
three targets.**

| target | RETFound (retinal) | ImageNet-MAE ViT-L | Δ | Δ/SD | verdict |
|---|---|---|---|---|---|
| DDR | 0.5143 | **0.5695** | +0.0552 | **5.63×** | ImageNet better |
| APTOS | 0.4801 | **0.5844** | +0.1043 | **1.37×** | ImageNet better |
| IDRiD | **0.6827** | 0.6404 | −0.0423 | 1.53× | CI spans zero |

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
log still produces a results row indistinguishable from an honest one.

EyePACS remains a **source** for the other three targets, which is correct —
sources are seen by construction. This does mean RETFound has seen some source
images the control has not, which if anything should favour RETFound. It does
not (§2).

DDR, APTOS and IDRiD are *not named* in RETFound's pretraining description, but
that description does not enumerate its public datasets. They are recorded as
`"unknown"`, not `"clean"`, and this belongs in the paper's limitations.

---

## 2. The mechanism: identical fit, worse transfer

| target | source QWK | | target QWK | | **source → target drop** | |
|---|---|---|---|---|---|---|
| | RETFound | MAE | RETFound | MAE | RETFound | MAE |
| DDR | 0.5986 | 0.5953 | 0.5143 | 0.5695 | **0.0844** | 0.0259 |
| APTOS | 0.6343 | 0.6510 | 0.4801 | 0.5844 | **0.1542** | 0.0666 |
| IDRiD | 0.6490 | 0.6591 | 0.6827 | 0.6404 | −0.0337 | 0.0187 |

**Source fit is near-identical** — as it must be for one architecture on one
task. RETFound does not fit the training domains better and then fail to
transfer. It fits them the *same* and transfers *worse*: its drop from source to
target is 3.3× larger on DDR and 2.3× larger on APTOS.

Severe errors (|error| ≥ 2 grades) follow:

| target | RETFound | ImageNet-MAE |
|---|---|---|
| DDR | 0.3195 | 0.2936 |
| APTOS | **0.4431** | 0.3107 |
| IDRiD | 0.1920 | 0.1913 |

On APTOS, RETFound produces severe misgrades on 44% of images against the
control's 31%.

### The DenseNet control, and the confound it left open

A second control was run first: DenseNet121, ImageNet, frozen, same pipeline.

| target | RETFound | DenseNet121 | Δ | Δ/SD |
|---|---|---|---|---|
| DDR | 0.5143 | 0.5607 | +0.0465 | 4.11× |
| APTOS | 0.4801 | 0.5171 | +0.0370 | 0.53× |
| IDRiD | 0.6827 | 0.7033 | +0.0206 | 0.79× |

ImageNet features scored higher on 3 of 3, but only DDR cleared both bars — and
the comparison confounds four variables at once (architecture, scale, objective,
corpus). **It could have been "CNN features transfer better than ViT features".**
The matched-architecture control rules that out, and is the reason the claim in
§0 can be made at all.

---

## 3. What would have been published without the controls

Two near-misses are worth recording, because both would have produced a
striking, wrong result.

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

- **This is a frozen-feature result.** RETFound is designed to be fine-tuned and
  its own paper reports fine-tuning. The claim is about the *representation*,
  not about RETFound's ceiling. **This is the single largest open question and
  the obvious next experiment** — VRAM was measured at 5.67 GB for full ViT-L
  fine-tuning with gradient checkpointing at batch 16, so it is affordable here.
- **IDRiD does not agree.** RETFound is +0.0423 there, and though the CI spans
  zero, the direction is opposite. IDRiD is also the smallest target at 507
  images. Two of three is not three of three, and is reported as such.
- **APTOS clears bar 1 by only 1.37×** (SD 0.0759). It is the largest effect and
  the least stable one. More seeds would settle it cheaply.
- **Not a claim about foundation models in general** — one model, one modality,
  one downstream task, linear probes.
- **Overlap is not ruled out** for DDR, APTOS and IDRiD (§1).

---

## 5. What this means for the paper

This changes the paper's centre of gravity. The DG-method results (Phases 4, 10,
11) confirm what DomainBed already found; the resolution result (Phase 9) is
useful but unglamorous. **This is the first finding in the project that is both
surprising and clinically consequential**: the field's default move — take the
domain-specific foundation model — measurably *hurt* cross-domain transfer here,
against the exact general-purpose checkpoint it was built from.

It also reframes everything else. "No method beats ERM" and "resolution beats
every method" become instances of a single, sharper claim: **interventions that
improve in-domain fit do not improve cross-domain transfer, and some of them cost
it.**

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
