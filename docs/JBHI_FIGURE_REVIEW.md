# JBHI figure review

**Eleven candidate figures, all rendered and visually inspected 2026-09-12.**
Five main, six supplementary. Every plotted quantity traces to an authoritative
CSV or is recomputed from saved predictions; provenance per panel is in
`outputs/figures/jbhi_final/FIGURE_PROVENANCE.csv`.

Generators: `make_jbhi_figures.py` (main), `make_jbhi_supplement_figures.py`
(supplement), `export_figure_provenance.py` (manifest and completeness gate).
Contact sheet: `outputs/figures/jbhi_final/CONTACT_SHEET.png`.

**Use the PDFs for submission.** They are vector and resolution-independent.
The PNGs are 400 dpi previews.

---

## Figure 1 — study design · **MAIN**

**Purpose.** Establish that the comparison is a *lineage intervention*, not two
unrelated architectures, and that both initialisations pass through the same
three protocols and the same evaluation.

**Claim supported.** The design premise behind P1; no numeric claim.

**Rebuilt from scratch.** The earlier version drew six diagonal arrows from two
source boxes to three protocol boxes; they collided with each other and with
the "Partial FT" / "Full FT" / "303.3M" labels. Both arms now feed a single
horizontal bus and each protocol drops from it — the same statement with no
crossing lines.

**Reviewer misunderstanding to pre-empt.** That this is RETFound versus "an
ImageNet model". The row-1 caption line must say the arms share architecture
*and* checkpoint lineage.

**Caption emphasis.** Spell out the wording the figure only gestures at:
"RETFound continues MAE pretraining from the ImageNet-MAE initialisation on
retinal CFP data." Also state that the 303.3M figure is the asserted trainable
parameter count under full FT.

---

## Figure 2 — adaptation depth · **MAIN**

**Purpose.** Show the attenuation across all three depths with the underlying
per-seed data, not just means.

**Claim supported.** P1 (visually), plus the descriptive per-protocol deltas.

**Design notes.** Five seeds are individually plotted and joined; seeds are
offset horizontally so trajectories stay separable where they bunch near zero.
Mean bars carry a value label placed *beside* the bar — above or below it, the
label landed on a data point in four of the six groups.

**Reviewer misunderstanding to pre-empt.** Reading the near-zero APTOS partial
and full means as equivalence. **The caption must say no difference was
demonstrated there and that this is not an equivalence claim.**

**Caption emphasis.** "$n=5$ common seeds; the interaction is paired within
seed." Do not describe this panel as a ranking reversal — on APTOS the mean
ordering never changes sign.

---

## Figure 3 — primary interaction forest · **MAIN**

**Purpose.** The primary result, per domain, never pooled.

**Claim supported.** P1 directly.

**Shows.** Point estimate, crossed seed × case 95% interval, Holm-adjusted *p*,
sign agreement, and *n* test cases for each domain.

**Reviewer misunderstanding to pre-empt.** That a negative value means RETFound
won. It does not: it means the *difference* shrank. The in-figure title already
says so; the caption must repeat it.

**Caption emphasis.** "Negative values mean the ImageNet-MAE minus RETFound QWK
difference is reduced when moving from frozen probing to full fine-tuning."
Plus: intervals quantify uncertainty and are not tests; no pooled two-domain
*p*-value is reported.

**Redundancy.** Overlaps Figure 4 in message but not in content — Figure 3 is
the inferential summary, Figure 4 is the raw pairing. Keep both; if forced to
cut one, keep Figure 3.

---

## Figure 4 — per-seed frozen→full slopes · **MAIN** (new)

**Purpose.** Make the pairing visible and show that all ten seed-level
interactions are negative, without a bar chart and without hiding any seed.

**Claim supported.** P1, and the "5/5 negative on each domain" stability
diagnostic.

**Design notes.** Every seed is drawn and labelled; labels that would collide
are pushed apart along y with a hairline leader back to the point. Panel titles
carry the mean interaction and the sign count.

**Reviewer misunderstanding to pre-empt.** That the downward slopes prove
RETFound becomes better. They show the *gap* closing; on APTOS every full-FT
point still sits at or above zero.

**Caption emphasis.** "All ten seed-level interactions are negative." State it
plainly and stop there — no adjectives.

---

## Figure 5 — full-FT paired QWK · **MAIN** (new)

**Purpose.** This figure exists specifically to **prevent** the misreading that
the interaction proves RETFound wins after fine-tuning. It shows absolute
performance of both arms, per seed, with the paired difference and its interval.

**Claim supported.** S1-DDR and S1-APTOS — both nulls.

**Design notes.** Statistics live in the panel titles; placed under the axis
they overprinted the tick labels. A shared legend sits outside the axes.

**Reviewer misunderstanding to pre-empt — both directions.**
DDR's mean delta is negative (−0.0439) and its crossed interval excludes zero,
which *looks* like a RETFound win; the formal seed-level test gives *p* =
0.1035 and under this study's framework nothing is claimed. APTOS's delta is
near zero, which *looks* like equivalence; it is not.

**Caption emphasis.** Use close to the supplied wording: "Under full
fine-tuning, no model difference is demonstrated by the pre-specified
seed-level formal test; absence of evidence is not interpreted as equivalence."
For DDR, add that the interval and the test disagree and that the test governs.

---

## S1 — source-validation optimisation behaviour · **SUPPLEMENT**

Was main-paper Figure 4; **moved to supplement** as instructed.

**Purpose.** Descriptive context: best epoch, loss inflation, early-stopping
counts (RETFound 10/10, ImageNet-MAE 0/10 across DDR + APTOS).

**Claim supported.** D3 — descriptive only.

**Language constraint.** The words *mechanism*, *explains* and *causes* must not
appear in the caption or the text referring to it. Optimisation behaviour does
not establish why the interaction occurs.

**Caption emphasis.** "Source validation only; the held-out target is not
consulted. Reported as context, not as a mechanism."

---

## S2 — domain profile · **SUPPLEMENT**

**Purpose.** Show that the four datasets differ observably in class composition
and acquisition characteristics.

**Design decision — licensing.** **No retinal image thumbnails.** Redistribution
terms for DDR, APTOS, IDRiD and EyePACS were not verified, so this is
quantitative only, as instructed when licensing is uncertain.

**Panels.** Grade composition (all images); image size and mean brightness from
the 400-image-per-domain sampled audit.

**Caption must state.** These summary statistics do **not** fully characterise
domain shift; they demonstrate observable heterogeneity, nothing more. Panels B
and C are a 400-image sample per domain, not the full corpus.

---

## S3 — configuration decomposition · **SUPPLEMENT**

**Purpose.** Separate the batch-size contrast from the isolated resolution
contrast and the combined change, per held-out domain.

**Claim supported.** The Q1 configuration finding, honestly bounded.

**Design notes.** Filled marker = survives Holm; open = does not. Per-seed
points shown (three seeds).

**Reviewer misunderstanding to pre-empt.** The old confounded "resolution
explains everything" claim. The figure shows the EyePACS resolution contrast
surviving Holm (*p* = 0.030) while the IDRiD resolution contrast does not
(*p* = 0.088), with IDRiD's combined contrast surviving (*p* = 0.028).

**Caption emphasis.** DenseNet-121, three seeds, and explicitly: the isolated
resolution effect is established on EyePACS and underpowered on IDRiD.

---

## S4 — calibration · **SUPPLEMENT**

**Purpose.** Reliability of both arms, frozen and full, on both primary domains.

**Claim supported.** None new. Secondary evidence only.

**Design notes.** Temperature scaled on source validation, matching how ECE is
reported everywhere else. Curves are the mean over five seeds. **Bins holding
fewer than 50 pooled images are not drawn** — two panels had a vertical spike
to 0.0 from a near-empty low-confidence bin. Annotated ECE still uses all data.

**Caption must state.** Calibration is secondary; the two uncorrected
sub-0.05 metrics in the APTOS full-FT secondary set are **not** claimed.

---

## S5 — risk–coverage · **SUPPLEMENT**

**Purpose.** Retrospective selective prediction: error rate among retained
cases as coverage falls.

**Claim supported.** None new.

**Caption must state.** "Retrospective selective-prediction analysis." It is
**not** evidence of deployment readiness, and no operating point is recommended.

---

## S6 — clinical error pattern · **SUPPLEMENT**

**Purpose.** Row-normalised confusion under full FT for both arms on both
primary domains, with severe-error rate per panel.

**Claim supported.** Descriptive error structure; severe-error rates match
`*_secondary_outcomes.csv` exactly (DDR 0.2372 / 0.2168, APTOS 0.0973 / 0.1103).

**Worth noting in the text.** Grade 1 is almost never predicted by either arm on
either domain — a shared failure mode, not a difference between them.

**Caption must state.** Rows sum to 1; pooled over five seeds; severe error is
|true − predicted| ≥ 2. Avoid per-class claims on IDRiD (not shown here) where
grade-1 counts are tiny.

---

## Recommended placement

| figure | placement | if space is tight |
|---|---|---|
| 1 study design | main | keep |
| 2 adaptation depth | main | keep |
| 3 interaction forest | main | keep — the primary result |
| 4 seed slopes | main | **first main-paper candidate to move to supplement** |
| 5 full-FT paired | main | keep — it prevents the central misreading |
| S1–S6 | supplement | S3 and S5 are the first to drop entirely |

## Redundancy assessment

- **Figures 3 and 4** carry the same message at different levels (inferential
  summary vs raw pairing). Both earn their place, but Figure 4 is the one to
  demote if the main paper must lose a figure.
- **Figures 2 and 4** overlap on the frozen and full columns; Figure 2 adds
  the partial-FT midpoint, Figure 4 adds per-seed identity. Not redundant, but
  a reviewer may say so — consider merging only if forced.
- **Nothing is recommended for outright deletion.** S1–S6 are each the only
  visual for their evidence class.

## Quality control performed

| check | result |
|---|---|
| PNG at ≥300 dpi | all 11 at 399 dpi |
| vector PDF | all 11; each smaller than its PNG, confirming vector not wrapped raster |
| visual inspection of rendered PNG | all 11 inspected; five layout defects found and fixed |
| text overlap / clipped labels | none remaining |
| colour-blind safety | Okabe-Ito palette; arms additionally separated by marker shape and line style |
| greyscale legibility | arm colours differ in luminance; S6 is greyscale by construction |
| IEEE width | main figures authored at 7.16 in (two-column); Figure 3 also legible at one-column |
| no hand-typed scientific values | verified — severe-error rates and all deltas cross-checked against authoritative CSVs |
| reproducibility | re-running the generators reproduces the same figures from the same data |
| completeness gate | `tests/test_manuscript_figures.py`, 24 tests |

### Defects found and fixed during inspection

1. **Figure 1** — six crossing arrows overprinting the protocol labels; rebuilt
   with a bus layout.
2. **Figure 1** — a dead white band below the evaluation panel; axis limits
   trimmed to the drawn content.
3. **Figure 2** — mean-value labels landing on data points; moved beside the
   mean bars.
4. **Figure 4** — seed labels colliding where frozen values bunch; collision-
   aware placement with leader lines.
5. **Figure 5** — statistics annotation overprinting tick labels and the axis
   label, and a suptitle overprinting the panel titles; moved into the titles
   with constrained layout and an outside legend.
6. **S2** — grade legend overlapping the rotated "EyePACS" tick label; moved to
   a figure-level legend.
7. **S4** — reliability spikes from near-empty bins; bins under 50 pooled
   images suppressed.
