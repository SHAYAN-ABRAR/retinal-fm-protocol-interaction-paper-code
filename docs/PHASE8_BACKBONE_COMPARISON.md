# Phase 8 — Does a stronger backbone fix the calibration problem?

**Run:** ERM, **ConvNeXt-Tiny**, batch 32, 224 px, 20 epochs, **seeds 42 / 1 / 2**
**Compared against:** ERM, DenseNet121, identical protocol, seeds 42 / 1 / 2
**Date:** 2026-08-22 (seed 42) · 2026-08-24 (seeds 1, 2) · 5 h 48 m GPU total
**Protocol:** the same four LODO experiments as Phase 5, changing only the
backbone. Same manifest, same splits, same resolution, same batch size, same
learning rate and weight decay, same seeds, same 20-epoch budget.
**Generator:** `analyse_backbone.py` → `outputs/tables/backbone_comparison.csv`

The question this phase exists to answer: **the project's thesis is that
calibration degrades on unseen domains even where accuracy does not. Is that a
statement about DenseNet121, or about the problem?** If a stronger, more modern
backbone closed the calibration gap, the thesis would be an artefact of a weak
model and the paper would not survive review.

> **⚠ This phase was rewritten on 2026-08-24.** Its first version ran ConvNeXt at
> one seed and reported the calibration comparison by sign, with the significance
> bar borrowed from DenseNet. Seeds 1 and 2 overturned that comparison. What
> changed and why is recorded in §7 — the superseded claims are kept there rather
> than deleted.

---

## 0. Answer

**No — and the reason is more interesting than the one first reported.**

ConvNeXt-Tiny grades better than DenseNet121 on two of four targets. After
temperature scaling, **there is no established calibration difference between
the two architectures on any target.** Both land at the same place: well
calibrated on DDR and APTOS, badly calibrated on IDRiD and EyePACS, at post-
temperature ECE around 0.12 and 0.17 whichever backbone produced them.

So a stronger backbone buys discrimination and leaves calibration exactly where
it was. On EyePACS it gains **+0.0554 QWK** — established under both bars —
while its post-temperature ECE moves **−0.0029 ± 0.0268**, which is nothing.

**That is the dissociation, in its cleaner form.** Not "a better model is worse
calibrated", which was a single-seed artefact, but *a better model is not better
calibrated*. The residual miscalibration under domain shift is a property of the
shift, not of the network.

---

## 1. Discrimination

Three paired seeds. ConvNeXt at seed *s* differenced against DenseNet at seed
*s*, on identical test images. The bar is the SD of those per-seed differences.

| Target | n | DenseNet121 | ConvNeXt-Tiny | Δ | × paired SD | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| DDR | 12,424 | 0.7383 | **0.7636** | **+0.0253** | 2.1× | [+0.0154, +0.0340] | **established** |
| APTOS | 3,504 | 0.8590 | 0.8748 | +0.0157 | 1.9× | [−0.0040, +0.0175] | CI spans zero |
| IDRiD | 507 | 0.7413 | 0.7946 | +0.0533 | 2.2× | [−0.0176, +0.0715] | CI spans zero |
| EyePACS | 35,108 | 0.4147 | **0.4701** | **+0.0554** | 4.1× | [+0.0507, +0.0665] | **established** |

**The gain is largest where the problem is hardest.** EyePACS — the target no
method, no source combination and no amount of recalibration has helped — gains
+0.0554 at 4.1× its own seed noise. APTOS, the easiest target, gains nothing
established.

**Severe errors fall on all four**, and on three by more than their seed SD:

| Target | DenseNet121 | ConvNeXt-Tiny | Δ | × paired SD |
|---|---|---|---|---|
| DDR | 0.1598 | 0.1418 | −0.0180 | 2.0× |
| APTOS | 0.0628 | 0.0548 | −0.0080 | 1.3× |
| IDRiD | 0.1177 | 0.0914 | −0.0263 | 2.1× |
| EyePACS | 0.2407 | 0.2002 | **−0.0405** | **5.1×** |

EyePACS is the row that matters clinically: one in six severe errors removed by
changing the backbone alone.

## 2. Calibration — the point of the phase

Three paired seeds. Means across seeds; the SD is of the per-seed difference.

| Target | ECE, DN → CN | after T, DN → CN | Δ after T | × paired SD | verdict |
|---|---|---|---|---|---|
| DDR | 0.1296 → 0.1143 | 0.0472 → 0.0438 | −0.0034 | 0.3× | within seed noise |
| APTOS | 0.1239 → 0.1582 | 0.0397 → 0.0660 | +0.0263 | 1.0× | within seed noise |
| IDRiD | 0.2294 → 0.2400 | 0.1213 → 0.1231 | +0.0018 | 0.0× | within seed noise |
| EyePACS | 0.2846 → 0.3134 | 0.1729 → 0.1700 | −0.0029 | 0.1× | within seed noise |

**After temperature scaling — the only fair comparison, since both get the same
post-hoc correction — no target shows an established difference between the two
backbones.** Every difference is smaller than the seed-to-seed variation of that
same difference.

**What survives is the level, not the contrast.** Both architectures end at
post-temperature ECE ≈ 0.12 on IDRiD and ≈ 0.17 on EyePACS, against ≈ 0.04–0.05
on DDR and APTOS. A source-fitted temperature corrects the near domains and
fails on the far ones, and it fails by the same amount for a 7.0 M-parameter
DenseNet and a 27.8 M-parameter ConvNeXt.

**Before scaling, ConvNeXt is the more over-confident model on three targets**
(APTOS 0.1239 → 0.1582 is the largest), and its fitted temperatures run higher
where that happens (APTOS 1.38 → 1.90, EyePACS 1.85 → 2.39). Temperature scaling
absorbs that difference. The distinction matters: raw ECE differences here are
about how each network's logits are scaled, and that is exactly what a
single-parameter post-hoc correction is for.

## 3. What this means for the paper

The thesis survives its most obvious attack, and its supporting statement is now
stronger than the one first written:

1. **The calibration failure is not a DenseNet121 artefact.** It reproduces at
   the same magnitude on a different architecture family with 4× the parameters
   (27.8 M vs 7.0 M). Not "worse on the other model" — *the same on the other
   model*, which is the harder claim to explain away.
2. **Accuracy gains do not carry calibration gains.** ConvNeXt is better on QWK
   on all four targets, established on two, and better calibrated after
   temperature on none.
3. **The domain ranking is architecture-invariant.** Both backbones order the
   targets identically — APTOS easiest, then IDRiD and DDR, then EyePACS by a
   wide margin — and both fail to recalibrate IDRiD and EyePACS. The problem is
   a property of the domain shift, not of the model.

ConvNeXt was also **cheaper per run**: 4,606 s mean against DenseNet's 5,151 s
despite 4× the parameters, because it converges earlier.

## 4. What this does NOT establish

- **That the two backbones are equally calibrated.** "Within seed noise" is a
  failure to establish a difference, not evidence of equivalence. With three
  seeds the SD of the difference is itself poorly estimated — on IDRiD it is
  0.0450, wider than most of the effects being tested. Five seeds would tighten
  it; three cannot settle APTOS (+0.0263 against SD 0.0270) either way.
- **That ConvNeXt's QWK advantage is small.** Two of four targets are
  established at 2.1× and 4.1×. The other two have CIs spanning zero, and for
  IDRiD that is a test-set problem (n = 507) rather than a seed problem — more
  seeds will not narrow a bootstrap over 507 images.
- **That architecture is responsible rather than capacity.** ConvNeXt-Tiny has
  4× DenseNet121's parameters. This compares two named models under one budget,
  not two architecture families at matched capacity.
- **Anything at 512 px.** Both arms are 224 px. Phase 9 shows the resolution
  change is worth more than the backbone change on every target. The backbone
  comparison has not been repeated at 512 px, and given ConvNeXt's seed variance
  it would need at least two seeds there to mean anything.

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
   "would a better model fix this?" with "it fixes the accuracy and not the
   calibration", which is exactly what a reviewer will ask.

**Do not report the two as if they were one experiment.** They were merged into
a single results table once by accident — see §8.

## 6. Artifacts

| Path | Contents |
|---|---|
| `analyse_backbone.py` | generator for §1 and §2; paired seeds, two bars |
| `outputs/tables/backbone_comparison.csv` | current table, three seeds |
| `outputs/tables/backbone_comparison_s42.csv` | superseded single-seed table, kept as a record |
| `outputs/tables/lodo_results.csv` | both backbones, keyed on `backbone` |
| `outputs/predictions/lodo_*_convnext-tiny_erm-b32_s*__*.csv` | per-image predictions |
| `tests/test_backbone_comparison.py` | five regression tests |

Reproduce: `python run_lodo.py --backbone convnext_tiny --seeds 42,1,2` (~5.8 h
GPU), then `python analyse_backbone.py`.

`analyse_backbone.py` re-derives the seed-42 intervals on every run and exits
non-zero if they disagree with `backbone_comparison_s42.csv`, so the superseded
table cannot drift away from the current one unnoticed.

## 7. ⚠ What seeds 1 and 2 overturned

The first version of this phase ran ConvNeXt at seed 42 only. Two consequences,
both now corrected:

**The significance bar was borrowed.** With one seed there was no ConvNeXt
across-seed SD, so §1 divided by **DenseNet's**. That was labelled as a proxy in
§4 of the original, but the ratios in the table were not, and they read as
measured. ConvNeXt's own variance turned out to be substantially larger:

| Target | DenseNet SD | ConvNeXt SD | ratio |
|---|---|---|---|
| DDR | 0.0055 | 0.0165 | 3.0× |
| APTOS | 0.0054 | 0.0109 | 2.0× |
| IDRiD | 0.0244 | 0.0286 | 1.2× |
| EyePACS | 0.0081 | 0.0086 | 1.1× |

DDR's headline therefore fell from **4.5× to 2.1×** and EyePACS's from **7.2× to
4.1×**. Both remain established; neither is as comfortable as it looked.

**The calibration finding did not survive.** The original §0 read *"Its
calibration is worse on three of four after temperature scaling, and on APTOS it
more than doubles post-temperature ECE"*, and §2 called APTOS *"more than twice
as miscalibrated for no established accuracy gain"* — 0.0406 → 0.0843 at seed 42.

Across three seeds that difference is +0.0263 against a paired SD of 0.0270: it
does not clear bar 1. The seed-42 bootstrap interval still excludes zero, because
it only ever described seed 42. All four targets are now within seed noise.

The original §4 did say the calibration column was *"descriptive… an observation,
not a tested effect"*. That caveat was correct and it was not enough: §0 and §2
stated the observation far more strongly than the caveat permitted, and a reader
would have carried away the strong version. **An observation stated as a headline
is a claim, whatever the caveats section says.**

This is the two-bar criterion working exactly as designed, on the phase that most
needed it.

## 8. ⚠ A contamination that reached the exported LaTeX

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

A related recording bug surfaced when seeds 1 and 2 landed: the table held seed
42 as `convnext-tiny` and the new rows as `convnext_tiny`, one architecture under
two labels. Since `backbone` is part of the dedup key, re-running seed 42 would
have appended a duplicate rather than replaced it — the same failure again, from
the opposite direction. Normalised to the registry's spelling by
`backfill_result_tables.py`.

**The experiment registry is the authoritative record** — its key is the
experiment id, which encodes backbone, resolution, method, and seed. Summary
tables are derived. Where they disagree, trust the registry.
