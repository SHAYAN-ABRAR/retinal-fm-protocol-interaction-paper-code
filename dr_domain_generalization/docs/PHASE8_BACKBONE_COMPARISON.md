# Phase 8 — Does a stronger backbone fix the calibration problem?

**Run:** ERM, **ConvNeXt-Tiny**, batch 32, 224 px, 20 epochs, **seed 42 only**
**Compared against:** ERM, DenseNet121, identical protocol, seeds 42 / 1 / 2
**Date:** 2026-08-22 · 1 h 58 m wall clock for the four ConvNeXt runs
**Protocol:** the same four LODO experiments as Phase 5, changing only the
backbone. Same manifest, same splits, same resolution, same batch size, same
learning rate and weight decay, same seed, same 20-epoch budget.

The question this phase exists to answer: **the project's thesis is that
calibration degrades on unseen domains even where accuracy does not. Is that a
statement about DenseNet121, or about the problem?** If a stronger, more modern
backbone closed the calibration gap, the thesis would be an artefact of a weak
model and the paper would not survive review.

---

## 0. Answer

**No. A stronger backbone buys discrimination and does not buy calibration.**

ConvNeXt-Tiny is better on target QWK on every domain and decisively better on
two. Its calibration is **worse on three of four** after temperature scaling,
and on APTOS it more than doubles post-temperature ECE while delivering a QWK
gain that is not even established.

That is the thesis, reproduced on a second architecture with four times the
parameters.

---

## 1. Discrimination

Both models evaluated on identical test images, seed 42. The "× seed SD" column
divides the difference by **DenseNet121's** across-seed SD for that target,
because ConvNeXt has only one seed — see §4.

| Target | n | DenseNet121 | ConvNeXt-Tiny | Δ | × DN seed SD | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| DDR | 12,424 | 0.7334 | **0.7580** | **+0.0246** | 4.5× | [+0.0153, +0.0343] | **established** |
| APTOS | 3,504 | 0.8591 | 0.8657 | +0.0066 | 1.2× | [−0.0039, +0.0173] | CI spans zero |
| IDRiD | 507 | 0.7543 | 0.7797 | +0.0254 | 1.0× | [−0.0185, +0.0703] | CI spans zero |
| EyePACS | 35,108 | 0.4077 | **0.4662** | **+0.0585** | 7.2× | [+0.0508, +0.0662] | **established** |

**The gain is largest where the problem is hardest.** EyePACS — the target no
method, no source combination and no amount of recalibration has helped — gains
+0.0585, more than twice DDR's gain and 7.2× the seed noise. APTOS, the easiest
target, gains nothing established.

**Severe errors fall on all four**, though only the two established rows should
be quoted: DDR 0.1666 → 0.1500, EyePACS 0.2454 → 0.2122 (−14% relative).

## 2. Calibration — the point of the phase

| Target | ECE, DN → CN | after T, DN → CN | T fitted (DN / CN) |
|---|---|---|---|
| DDR | 0.1383 → 0.1310 | 0.0551 → **0.0571** | 1.63 / 1.43 |
| APTOS | 0.1082 → **0.1936** | 0.0406 → **0.0843** | 1.38 / 1.90 |
| IDRiD | 0.2159 → 0.2057 | 0.1047 → **0.1180** | 1.63 / 1.48 |
| EyePACS | 0.2887 → 0.3010 | 0.1801 → 0.1655 | 1.85 / 2.39 |

**After temperature scaling — the only fair comparison, since both get the same
post-hoc correction — ConvNeXt is worse on DDR, APTOS and IDRiD, and better only
on EyePACS.**

**APTOS is the sharp case.** ConvNeXt improves QWK by +0.0066, an amount inside
the noise, while its post-temperature ECE goes 0.0406 → 0.0843. It is *more than
twice as miscalibrated* for no established accuracy gain. Its raw ECE nearly
doubles (0.1082 → 0.1936) and its fitted temperature rises from 1.38 to 1.90 —
the model is substantially more over-confident, and the source-fitted
temperature cannot correct it on the target.

**Discrimination and calibration are not the same axis, and improving the
backbone moves only one of them.** A reader who selects an architecture on
target QWK will, on this evidence, select one that is worse calibrated.

## 3. What this means for the paper

The thesis survives its most obvious attack. Three specific statements are now
supported that were not before:

1. **The calibration failure is not a DenseNet121 artefact.** It reproduces on a
   different architecture family with 4× the parameters (27.8 M vs 7.0 M).
2. **Accuracy gains do not carry calibration gains.** ConvNeXt is better on QWK
   on all four targets and better calibrated after temperature on one.
3. **The domain ranking is architecture-invariant.** Both backbones order the
   targets identically — APTOS easiest, then IDRiD and DDR, then EyePACS by a
   wide margin — and both fail to recalibrate IDRiD and EyePACS. The problem is
   a property of the domain shift, not of the model.

ConvNeXt was also **cheaper**: 7,077 s total against DenseNet's 9,147 s, despite
4× the parameters, because it converged earlier (best epoch 12, early-stopped at
19; DenseNet used all 20 with best at 17).

## 4. What this does NOT establish

- **Anything at ConvNeXt seed 42 alone.** ConvNeXt has **one seed**. Bar 1 of
  the two-bar criterion is unavailable for it, and the "× seed SD" column in §1
  uses **DenseNet's** SD as a proxy. That is a proxy, not a measurement. The two
  rows called established clear it by 4.5× and 7.2×, which is large enough to be
  reasonably safe; the two that do not clear it are correctly reported as
  unestablished, and their CIs span zero independently of the proxy.
- **That ConvNeXt is worse-calibrated *in general*.** Three of four is not four
  of four, and none of the calibration differences has been through a paired
  bootstrap — ECE is a binned statistic and the project's paired-difference
  machinery is set up for it, but it has not been run here. **The calibration
  column of §2 is descriptive.** Treat "worse on three of four after T" as an
  observation, not a tested effect.
- **That architecture is responsible rather than capacity.** ConvNeXt-Tiny has
  4× DenseNet121's parameters. This compares two named models under one budget,
  not two architecture families at matched capacity.
- **Anything at 512 px.** Both runs are 224 px. Phase 6 §0a shows resolution is
  worth +0.09 QWK on EyePACS in-domain — larger than every effect in §1. The
  backbone comparison has not been repeated at 512 px.

## 5. The recommendation

**Keep DenseNet121 as the paper's primary backbone** and report ConvNeXt-Tiny as
a robustness check, for three reasons:

1. All three seeds, all four LODO targets, the in-domain ceilings, the
   cross-domain matrix and the entire method comparison are DenseNet121. Making
   ConvNeXt primary would mean re-running the project.
2. The paper's claim is about calibration under domain shift. DenseNet121 is the
   backbone on which that claim is measured with three seeds and full
   statistical treatment.
3. ConvNeXt's role here is stronger *as a check* than as a headline: it answers
   "would a better model fix this?" with "no", which is exactly what a reviewer
   will ask.

**Do not report the two as if they were one experiment.** They were merged into
a single results table once by accident — see §7.

## 6. Artifacts

| Path | Contents |
|---|---|
| `outputs/tables/lodo_results.csv` | both backbones, keyed on `backbone` |
| `outputs/tables/backbone_comparison_s42.csv` | §1 and §2, with paired CIs |
| `outputs/predictions/lodo_*_convnext-tiny_erm-b32_s42__*.csv` | per-image predictions |

Reproduce: `python run_lodo.py --backbone convnext_tiny --seeds 42` (~2 h GPU).

## 7. ⚠ A contamination that reached the exported LaTeX

`lodo_results.csv` was keyed on `(target, method, seed)` with **no backbone**.
The ConvNeXt seed-42 run therefore **replaced** the DenseNet121 seed-42 rows.
The file still held twelve well-formed rows afterwards, so nothing looked wrong.

Consequences, and their extent:

- **The phase documents were never affected.** `analyse_lodo_seeds.py` loads
  predictions by experiment id, which encodes the backbone, so Phases 5–7 were
  computed from the correct files throughout. Every documented value was
  re-derived from predictions on 2026-08-22 and matches to 4 decimal places.
- **`table_lodo.tex` was wrong.** Exported from the contaminated CSV, it
  reported DDR as `0.7465 ± 0.0105` against the true `0.7383 ± 0.0055` — an
  across-*architecture* SD presented as an across-seed SD, nearly double, on a
  bar the two-bar criterion depends on. It has been regenerated and now matches
  `PHASE5_LODO_REPORT.md` exactly.

Fixed: `backbone` and `image_size` are part of the dedup key via
`src/utils/registry.merge_results_table`; `analyse_lodo.py`,
`analyse_lodo_seeds.py`, `analyse_selective.py` and `export_paper_tables.py` all
filter on backbone explicitly; ten regression tests in
`tests/test_results_merge.py`.

**The experiment registry is the authoritative record** — its key is the
experiment id, which encodes backbone, resolution, method, and seed. Summary
tables are derived. Where they disagree, trust the registry.
