# Project Handbook — where everything lives, and how to turn it into a paper

**Last updated:** 2026-08-21
**Project:** *Beyond In-Domain Accuracy: Calibrated Domain Generalization for
Reliable Diabetic Retinopathy Grading Across Clinical Datasets*
**Status:** Phases 1–6 complete on seed 42. LODO seeds 1–2 running.
**Tests:** 227 passing.

This document has two halves. **Part A** is a map: every directory, every file
type, what produced it and what it is for. **Part B** is a writing guide: what
the paper should claim, which artifact supplies each number, what a reviewer
will attack, and what is still missing.

Read Part A when you need to find something. Read Part B when you sit down to
write.

---

# PART A — WHERE EVERYTHING IS

## A1. Top-level layout

```
D:\Research Code\dr_domain_generalization\
├── research_pipeline.py        35 interactive `# %%` cells — the guided tour
├── run_lodo.py                 the four leave-one-domain-out experiments
├── run_in_domain.py            in-domain ceilings + deployment-cost comparison
├── run_method_comparison.py    Stage-C six-method ablation
├── analyse_lodo.py             single-seed LODO analysis
├── analyse_lodo_seeds.py       multi-seed LODO analysis (two-bar criterion)
├── run_single_source.py        one source, every other domain (the 4x4 matrix)
├── analyse_seeds.py            Stage-C multi-seed analysis
├── analyse_selective.py        risk-coverage / abstention analysis
├── analyse_severe_error.py     severe-error deployment cost, matched images
├── analyse_corrected.py        Holm-Bonferroni across the whole comparison family
├── audit_consistency.py        ⭐ verify every table still agrees with the registry
├── export_paper_tables.py      the paper's LaTeX tables, generated never retyped
├── regenerate_figures.py       rebuild figures from saved predictions
├── requirements.txt            dependencies + the sm_120/cu128 warning
├── README.md                   project overview
├── configs/                    4 YAML files
├── docs/                       8 phase reports + this handbook (md + html)
├── src/                        58 modules, the reusable library
├── tests/                      11 test files, 271 tests
└── outputs/                    everything the experiments produced
```

**The separation that matters:** `src/` holds reusable, tested library code.
The `run_*.py` and `analyse_*.py` scripts at top level are thin experiment
drivers — they compose `src/` and own no logic worth testing in isolation.
`research_pipeline.py` is the Colab-style narrative version for exploring
interactively. Nothing important lives only in the pipeline file.

## A2. `src/` — the library, module by module

### `src/data/` — provenance, labels, splits (18 modules)

| Module | Responsibility |
|---|---|
| `inspect.py` | Discovers the actual directory layout of each raw dataset. Nothing assumes a structure. |
| `provenance.py` | Identifies which physical folder is which dataset, from evidence (label histograms, filename patterns), not from folder names. |
| `ddr.py`, `aptos.py`, `idrid.py`, `eyepacs.py` | One loader per domain. Each knows that domain's label file format and filename conventions. |
| `eyepacs_labels.py` | Verifies EyePACS folder labels against the official `trainLabels.csv`. Holds `REFERENCE_SHA256` and asserts 35,108 rows before trusting it. |
| `schema.py` | The unified manifest schema every domain maps into. |
| `unified_dataset.py` | Builds and loads the cross-domain manifest. |
| `deduplicate.py` | Size-prefiltered content hashing. Consistent duplicate group → keep lexicographically smallest id; **conflicting group (different grades) → drop the whole group.** |
| `leakage.py` | The split-level audit: id overlap, patient overlap, byte-identical content, manifest hygiene. |
| `splits.py` | Patient-level splits and the three protocols (`in_domain`, `single_source`, `lodo`). Contains `_assert_target_isolation`. |
| `preprocessing.py` | Retina cropping (threshold 0.20, chosen by sweep), resize, normalisation. |
| `augmentations.py` | Train/eval transform construction. |
| `cache.py` | The 224 px image cache that moved training from I/O-bound to GPU-bound. |
| `loaders.py` | DataLoader construction, worker counts, samplers. |
| `image_stats.py` | Per-domain appearance statistics (brightness, contrast, size) for the domain-shift evidence. |

### `src/models/`

| Module | Responsibility |
|---|---|
| `backbones.py` | timm wrappers for DenseNet121 / ConvNeXt-Tiny / DINOv2 ViT-S/14; `resolve_input_size` handles the patch-14 snap (384 → 378). |
| `mixstyle.py` | MixStyle (Zhou et al., ICLR 2021). No-op in `eval()`; pairs across domains; insertion is asserted, not assumed. |

### `src/losses/` — **read this section before writing the methods section**

| Module | Responsibility |
|---|---|
| `ordinal_coral_loss.py` | **CORAL ordinal regression** (Cao et al. 2020). K−1 rank-consistent binary threshold tasks. |
| `deep_coral_alignment.py` | **Deep CORAL domain alignment** (Sun & Saenko 2016). Second-order feature covariance matching. |
| `classification.py` | Cross-entropy, focal loss, class-balanced weighting. |

These two CORALs are unrelated methods that share a name. Both files carry a
`!! NAME COLLISION WARNING !!` header pointing at the other. **The paper must
disambiguate them on first use** — see B6.

### `src/training/`

`trainer.py` (AMP, cosine schedule with warmup, gradient clipping, resume),
`checkpointing.py`, `early_stopping.py`, and `methods.py` — which assembles the
six methods and returns a `BuiltMethod` bundling model, loss, feature loss,
batch hook, probability converter, and prediction rule.

### `src/evaluation/`

| Module | Responsibility |
|---|---|
| `metrics.py` | QWK, macro F1, balanced accuracy, MCC, MAE-of-grade, within-±1, severe-error rate, per-class, referable-DR. |
| `calibration.py` | ECE (equal-width), adaptive ECE (equal-mass), NLL, Brier, reliability curves, `TemperatureScaler`, `fit_temperature`. |
| `bootstrap.py` | Percentile CIs, **paired** bootstrap difference, McNemar. |
| `selective_prediction.py` | Risk–coverage curves, AURC, error-detection AUROC. |
| `embeddings.py` | Feature extraction, linear-probe domain separability, silhouette. |
| `evaluate.py` | Orchestrates evaluation of one experiment; fits temperature **on source validation only**. |

### `src/visualization/` and `src/reporting/`

Nine figure modules (`dataset_`, `domain_`, `performance_`, `calibration_`,
`error_`, `feature_`, `training_`, `pipeline_diagram`, `style`) and two
reporting modules (`tables.py` with LaTeX export, `summary.py`).

### `src/utils/`

`config.py` (YAML → dataclasses, **rejects unknown keys**), `hardware.py`
(`assert_cuda_ready` — fails loudly rather than training on CPU), `io.py`,
`logging.py`, `registry.py` (append-only experiment registry), `seed.py`.

## A3. `configs/`

| File | Contents |
|---|---|
| `paths.yaml` | Dataset locations, verified counts, and the **hard exclusion rules**: `forbidden_paths`, `restrict_to_verified_ids`, `ignore_provided_splits`, filename regexes. |
| `baseline.yaml` | The ERM reference configuration. |
| `domain_generalization.yaml` | Extends `baseline.yaml` with DG method settings. |
| `experiments.yaml` | The status ledger — **26 COMPLETE, 4 NOT_RUN** — plus a `protocol_rules` block naming the code that enforces each rule. |

`experiments.yaml` is the single source of truth for what has and has not been
run. A test asserts every status is one of `COMPLETE` / `NOT_RUN` / `PARTIAL`,
so a blank can never be read as "done".

## A4. `outputs/` — every artifact the experiments produced

| Directory | Count | What is in it |
|---|---|---|
| `outputs/tables/` | 24 | CSV and LaTeX result tables |
| `outputs/figures/` | 140 | All figures (+ 26 archived in `superseded/`) |
| `outputs/predictions/` | 54 | **Per-image predictions for every evaluated split** |
| `outputs/reports/` | 46 | Evaluation JSONs, the manifest, audit reports |
| `outputs/logs/` | 31 | Training logs and per-epoch history CSVs |
| `outputs/checkpoints/` | 29 | Model weights (~5.1 GB) |
| `outputs/embeddings/` | — | Cached feature arrays |
| `outputs/experiment_registry.csv` | 27 rows | One row per run: config, metrics, paths, timings |

**`outputs/predictions/` is the most valuable directory in the project.** Every
file has one row per image with `image_id`, `true_grade`, `predicted_grade`,
five class probabilities, `confidence`, `correct`, `absolute_grade_error`. Any
metric, CI, figure, or subgroup analysis can be recomputed from these without
touching a GPU. `regenerate_figures.py` and both `analyse_*` scripts work
entirely from them.

Naming convention:

```
{experiment_id}__{split}_predictions.csv
lodo_aptos-eyepacs-idrid__ddr_densenet121_erm-b32_s42__target_test[ddr]_predictions.csv
└─ protocol ─┘└ sources ┘ └target┘└─backbone──┘└method┘└seed┘  └── split ──┘
```

The `experiment_id` is built by `src/utils/registry.py::make_experiment_id`.
Never construct one by hand — tests assert that the runners and analysers agree
on it, because a drift makes finished runs silently report as `NOT RUN`.

## A5. `docs/` — the findings record

| Document | Covers |
|---|---|
| `DATA_PROVENANCE.md` | Which physical folder is which dataset, and the four data defects found by audit. |
| `PHASE2_DATA_REPORT.md` | Manifest construction, deduplication, splits, leakage audit. |
| `PHASE3_BASELINE_REPORT.md` | First baseline (batch 16) — **superseded**, kept for the record. |
| `PHASE4_METHOD_COMPARISON.md` | Six methods × 3 seeds on DDR+APTOS → IDRiD. Section 0 supersedes the single-seed numbers and carries a correction. |
| `PHASE5_LODO_REPORT.md` | The four LODO experiments, seed 42. |
| `PHASE6_IN_DOMAIN_REPORT.md` | In-domain ceilings and the cost of cross-domain deployment. |
| `PHASE7_CROSS_DOMAIN_MATRIX.md` | The full 4x4 matrix. Multi-source training buys nothing; the lowest-scoring dataset is the best source. Carries a 2026-08-22 correction withdrawing the label-noise mechanism. |
| `PHASE8_BACKBONE_COMPARISON.md` | ConvNeXt-Tiny vs DenseNet121 on all four LODO targets. **A stronger backbone buys discrimination, not calibration** — the thesis survives its most obvious attack. Also records a contamination that reached the exported LaTeX. |
| `PROJECT_HANDBOOK.md` | This file. |
| `PROJECT_HANDBOOK.html` | Same content as a navigable page (sticky contents, semantic colour on the claim tables). Open it directly from this folder in any browser — no server, no build step, no external host. |

Each phase document is written to be readable on its own and states explicitly
what it does *not* establish.

## A6. `tests/` — 259 tests

| File | Guards |
|---|---|
| `test_phase1.py` | Paths, CUDA detection, dataset inspection. |
| `test_phase2.py` | Manifest, labels, deduplication, splits, leakage. |
| `test_phase3.py` | Training loop, checkpointing, metrics. |
| `test_phase3_statistics.py` | Bootstrap, McNemar, CI correctness. |
| `test_phase4.py` | MixStyle, method assembly, ordinal head, CORAL prediction rule. |
| `test_phase5_lodo.py` | LODO protocol, target isolation, partial-matrix refusal. |
| `test_lodo_seeds.py` | Multi-seed aggregation, matched-image restriction. |
| `test_calibration_figures.py` | Temperature reconstruction identity, argmax invariance. |
| `test_config_and_reporting.py` | Config validation, ledger statuses, table formatting. |

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`

## A7. Environment

Python 3.14.3, Windows 11, RTX 5060 Laptop (Blackwell **sm_120**, 8 GB).

**The one thing that will break for anyone else:** sm_120 requires a CUDA 12.8+
PyTorch build. A default `pip install torch` gives a wheel that fails with *"no
kernel image is available for execution on the device"*.

```
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verified stack: torch 2.9.1+cu128, timm 1.0.28, albumentations 2.0.8,
numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, matplotlib 3.11.1.

Peak VRAM observed across all runs: **2.8 GB of 8 GB.**

## A8. ⚠ Known staleness

`README.md` still says *"Status: Phase 4"* and *"158 tests pass"*. Both are out
of date (Phases 5–7 are done; **255 tests**). It should be refreshed before the
repository is shared or submitted as an artifact.

**Resolution labelling.** Every result in `docs/` other than
`PHASE6_IN_DOMAIN_REPORT.md` §0a is a **224 px** measurement. The 512 px LODO
matrix is in progress; until it lands, no 224 px number may be differenced
against a 512 px one. `run_in_domain.py` refuses to, and both
`in_domain_results.csv` and `single_source_results.csv` are keyed on
`image_size` so the two cannot overwrite each other.

**A results table was silently overwritten once (2026-08-22).** The 512 px
EyePACS run replaced the 224 px row in `in_domain_results.csv`, because the
dedup key was `(domain, method, seed)` and carried no resolution — the file
still held four well-formed rows afterwards, so nothing looked wrong. It was
recovered exactly from the surviving predictions and the experiment registry
(whose key *is* the experiment id, which encodes resolution). The merge is now
`src/utils/registry.merge_results_table`, with six regression tests in
`tests/test_results_merge.py`. **The registry is the authoritative record; the
summary tables are derived.** If they ever disagree, trust the registry.

---

# PART B — WRITING THE PAPER

## B1. What this project actually found

Four results, in descending order of how well they are established.

**1. Calibration degrades on unseen domains even where accuracy does not.**
On held-out DDR, QWK falls only 0.019 against source validation while ECE
doubles (0.077 → 0.138). Every one of the four targets loses calibration.
*This is the paper's thesis and it is the best-supported claim.*

**2. Source-fitted temperature scaling transfers under mild shift and fails
under severe shift.** Fitted only on source validation, it pulls target ECE
*below* the source ECE on DDR (0.055 vs 0.077) and APTOS (0.041 vs 0.046),
leaves IDRiD 42% above (0.105 vs 0.074), and fails on EyePACS (0.180 vs 0.082).
A practitioner cannot tell which regime they are in without target labels.

**3. Cross-domain deployment costs 0.138–0.291 QWK.** Measured against
in-domain models on identical test images, **three seeds on both sides, paired
seed-to-seed**: DDR −0.1376 ± 0.0119 [−0.1760, −0.0954], EyePACS
−0.2910 ± 0.0202 [−0.3305, −0.2375]. APTOS and IDRiD are not resolvable at their
test-set sizes (354 and 102 images).

**4. No domain-generalization method beat ERM.** Across 3 seeds on
DDR+APTOS → IDRiD, all five alternatives (ordinal CORAL, Deep CORAL, MixStyle,
and two combinations) are worse on target QWK than plain ERM, clearing both the
seed-SD and paired-bootstrap bars. The ordinal head does improve ECE by ~10×
the seed noise — but at the cost of collapsing grades 1 and 3.

**5. The same deployment cost, stated clinically: severe errors more than
double.** On matched images, the rate of misgrading by two or more steps rises
0.0788 → 0.1631 on DDR (**+107%**, CI [+0.0806, +0.1155]) and 0.0938 → 0.2396
on EyePACS (**+155%**, CI [+0.1433, +0.1689]). Both clear both bars; APTOS and
IDRiD are unresolvable at 354 and 102 test images. **This is the strongest
framing for a clinical venue**, and it is the number recalibration cannot
touch — temperature scaling is monotonic and cannot move an argmax.
See `PHASE6_IN_DOMAIN_REPORT.md` §2c.

**6. Input resolution was a binding constraint, and 224 px understated the
in-domain reference.** At 512 px the in-domain EyePACS model reaches 0.8004
against 0.7090 at 224 px on identical test images (+0.0914, CI [+0.0694,
+0.1142]), and its severe-error rate falls 0.0919 → 0.0575. This **falsified**
the earlier claim that 0.709 was a label-noise ceiling. Every 224 px result in
this project is now labelled as such, and the 512 px LODO matrix is running so
the comparison can be made at one resolution. See `PHASE6_IN_DOMAIN_REPORT.md`
§0a.

**7. A stronger backbone does not fix calibration.** ConvNeXt-Tiny (27.8 M
params) beats DenseNet121 (7.0 M) on target QWK on all four LODO targets —
decisively on DDR (+0.0246, 4.5× seed SD) and EyePACS (+0.0585, 7.2×) — yet is
**worse calibrated after temperature scaling on three of four**. On APTOS it
more than doubles post-temperature ECE (0.0406 → 0.0843) for a QWK gain that is
not established. *This is the answer to "would a better model fix this?" and a
reviewer will ask it.* One seed only; see `PHASE8_BACKBONE_COMPARISON.md` §4.

## B2. Recommended paper structure

### Title and framing

Lead with the **negative-plus-diagnostic** result, not a proposed method. This
paper's contribution is a rigorous measurement and a falsification, and framing
it as "we propose X" would invite reviewers to ask why X does not win.

### 1. Introduction

The gap: DR grading models report strong in-domain accuracy; deployment means an
unseen clinic. Existing DG work optimises accuracy metrics and rarely reports
calibration. Contributions:

1. A leakage-audited four-dataset benchmark with a documented protocol.
2. Measurement of accuracy **and** calibration degradation across all four
   leave-one-domain-out configurations.
3. Evidence that source-fitted recalibration transfers only under mild shift.
4. A negative result: four established DG methods do not beat ERM, verified
   across seeds with a two-bar significance criterion.

### 2. Related work

DR grading; domain generalization (Deep CORAL, MixStyle, and the DomainBed
finding that ERM is a strong baseline — **this paper independently reproduces
that in a medical setting**, which is worth stating); ordinal regression (CORAL);
calibration (temperature scaling, ECE, and the literature on calibration under
distribution shift).

### 3. Data — *this section is a strength, write it at length*

The audit findings are unusual and reviewers will value them. Source:
`docs/DATA_PROVENANCE.md`.

| Finding | Number | Consequence |
|---|---|---|
| One vendor folder contained **100% of APTOS** | 3,662 / 3,662 ids | A LODO run holding out APTOS would have trained on its entire test set |
| Corrupted EyePACS labels | 50,070 excluded; **35,108 verified at 100.0000% agreement** against official `trainLabels.csv` | Grade-3 count was 1 observed vs ~1,247 expected |
| Vendor splits leak patients | 6,748 / 6,679 / 817 patient overlaps across train/val/test | Provided splits discarded entirely; patient-level splits rebuilt |
| Byte-identical duplicates | 224 groups, 33 with **contradictory grades**; 265 of 51,808 images removed | Conflicting groups dropped whole, not resolved by majority |

Final corpus: **51,543 images, 17,561 patients, 4 domains** — DDR 12,424,
APTOS 3,504, IDRiD 507, EyePACS 35,108.

State the domain numbering once and use it throughout: **0 = DDR, 1 = APTOS,
2 = IDRiD, 3 = EyePACS.**

### 4. Method

Keep this short — the components are established. Cover the backbone
(DenseNet121, 7.0 M parameters), the four method axes, and the two CORALs
(see B6). Emphasise that calibration is a **post-hoc axis**: it needs no
separate training run, so every trained model yields both an uncalibrated and a
calibrated row.

### 5. Protocol — *write this defensively; it is your credibility*

- **Target isolation.** The held-out domain is used for final evaluation only —
  never training, validation, early stopping, model selection, temperature
  fitting, hyperparameter search, or augmentation selection. Enforced in code at
  `src/data/splits.py::_assert_target_isolation`,
  `src/evaluation/evaluate.py::evaluate_experiment`, and
  `src/training/checkpointing.py`.
- **Patient-level splits**, both eyes in the same split.
- **Temperature fitted on source validation only.**
- **Two-bar significance.** An effect is real only if it exceeds the across-seed
  SD *and* its paired bootstrap CI excludes zero. Justify this: the bootstrap
  resamples images, the SD resamples initialisations, and this project has a
  documented case that cleared one and failed the other.

### 6. Results

Order: LODO headline → calibration → temperature transfer → deployment cost →
method comparison → per-class collapse.

### 7. Discussion

The clinical framing: **severe-error rate** (|error| ≥ 2 grades) is the number
that matters, and it ranges 0.062–0.245 across targets. Temperature scaling is
monotonic — it changes *no* prediction — so good calibration and clinical safety
are separate axes. Say this explicitly; it is a point many calibration papers
elide.

### 8. Limitations — write these yourself before a reviewer does

See B5.

## B3. Figures and tables

### Main paper (6 figures, 3 tables)

| # | Figure | File |
|---|---|---|
| 1 | Protocol diagram | `outputs/figures/diagram_protocols.png` |
| 2 | Domain shift evidence | `14_domain_shift_boxplots.png`, `04_domain_class_heatmap.png` |
| 3 | LODO headline | `lodo_summary_qwk.png` |
| 4 | **Reliability, 4 targets, before/after T** | `calibration_reliability_grid_erm_s42.png` |
| 5 | **QWK vs ECE with temperature arrows** | `calibration_qwk_vs_ece_erm_s42.png` |
| 6 | Method comparison across seeds | `stage_c_seed_comparison.png` |
| 7 | **Severe-error deployment cost + CIs** | `fig_severe_error_deployment.png` |
| 8 | **Risk–coverage, four unseen domains** | `fig_risk_coverage.png` |

**Figures 4 and 5 are the paper's identity.** Figure 4 shows what a source-fitted
temperature can and cannot repair, across all four domains at once. Figure 5 puts
the thesis in one panel: three domains cluster at QWK 0.73–0.86 while their ECE
spans 0.108–0.216.

**Figures 7 and 8 are what a clinical reviewer reads first.** Figure 7 states the
cost in the currency of patient harm and marks the two domains where it is not
established, so the two-bar criterion reaches the figure rather than living only
in the text. Figure 8 forecloses the reviewer's obvious rebuttal — *just let the
model abstain* — by showing that abstaining on 30% of cases buys a 17–33% error
reduction and works **worst** on EyePACS, where it is needed most.

| # | Table | Source |
|---|---|---|
| 1 | Dataset characteristics | `table1_dataset_characteristics.tex` |
| 2 | LODO results with CIs | `lodo_erm_s42_headline.csv`, `lodo_erm_s42_bootstrap_ci.csv` |
| 3 | Cost of deployment | `in_domain_vs_lodo_erm_s42.csv`, `table_deployment_cost.tex` |
| 4 | **Severe-error cost** | `severe_error_comparison.csv`, `table_severe_error.tex` |

### Supplementary

Cross-domain matrices (`domain_matrix_*_erm_s42.png`), per-class recall,
embeddings/UMAP, training curves, error galleries, selective-prediction curves,
the full 18-figure dataset characterisation.

**Do not use anything in `outputs/figures/superseded/`.** Those 26 files were
correct for the batch-16 run that produced them and would contradict your
tables. The directory has a README naming what replaced each.

## B4. Claims you can make, with wording

> "Quadratic weighted kappa falls by only 0.019 on held-out DDR while expected
> calibration error doubles (0.077 → 0.138), showing that discrimination and
> calibration do not degrade together under domain shift."

> "A temperature fitted exclusively on source-domain validation data reduces
> target ECE below the source-domain value on two of four held-out datasets, but
> leaves 42% and 120% excess miscalibration on the remaining two."

> "Relative to models trained on the target domain and evaluated on identical
> images and averaged over three seeds on both sides, cross-domain deployment
> costs 0.138 QWK on DDR (95% CI [0.095, 0.176]) and 0.291 on EyePACS
> (95% CI [0.238, 0.331])."

> "Across three seeds, no domain-generalization method examined improved on
> empirical risk minimisation for target-domain QWK."

## B5. Claims you must NOT make

| Do not claim | Why |
|---|---|
| A mean LODO gap | The four targets span −0.483 to +0.063. The mean (−0.122) describes none of them. |
| That APTOS transfers *better* than in-domain | QWK rose (+0.063) but macro-F1 fell (−0.089). APTOS's label distribution is wider, and QWK normalises by expected disagreement. **A prior-shift artifact.** |
| That cross-domain beats in-domain on IDRiD | On matched images: +0.078, CI [−0.091, +0.260]. Spans zero. 102 test images. |
| That EyePACS's collapse is domain shift | Confounded three ways: shift, the smallest training pool (11,841), and **input resolution**. The 224 px in-domain reference is 0.709; the same protocol at 512 px reaches **0.8004** on the same test images, so part of the apparent damage was the pipeline, not the domain. |
| That EyePACS's 0.709 is a label-noise ceiling | **Falsified 2026-08-22.** 512 px reaches 0.8004, +0.0914 with CI [+0.0694, +0.1142]. This was claimed in Phase 6 §1 and Phase 7 §4 and is **withdrawn**; the corrections are in place. How much label noise remains is unmeasured. |
| Any 224 px number differenced against a 512 px number | Resolution moves EyePACS's QWK by +0.091 and its severe-error rate by −37% relative. Differencing across resolutions charges that to domain shift. `run_in_domain.py` refuses; `in_domain_results.csv` is keyed on `image_size`. |
| That Deep CORAL worsens calibration | Single-seed claim (+0.071); across 3 seeds it is +0.029 against pooled SD 0.033. **Withdrawn.** |
| Any effect < 0.032 QWK at one seed | That is ERM's measured across-seed SD. |
| That combining clinical datasets improves generalization | **Falsified in Phase 7.** One source matches or beats three on all four targets. The 3-source pools were 68-89% EyePACS, so LODO measured *which* source, not *how many*. |
| That single-source is *better* than multi-source | Also unsupported. Three of four deltas are inside the seed SD, and single-source variance is unmeasured at one seed. The supportable claim is "no advantage", not "worse". |
| That abstention rescues a miscalibrated model | Error-detection AUROC is 0.657-0.723. Abstaining on 30% of cases cuts error by only 17-33%, and works worst on EyePACS where it is needed most. |
| Novelty for the components | Ordinal CORAL, Deep CORAL, MixStyle and temperature scaling are all prior work. The contribution is the protocol, the measurement, and the falsification. |

## B6. The CORAL naming trap

The paper uses two unrelated methods called CORAL. Reviewers *will* be confused
if you are not explicit. Recommended: introduce each with its full name and
citation on first use, then use distinct short names throughout —
**"ordinal CORAL (Cao et al., 2020)"** and **"Deep CORAL (Sun & Saenko, 2016)"**
— and never the bare word "CORAL" after that. Add a footnote at first
co-occurrence noting the collision is coincidental.

## B7. What a reviewer will ask for that you do not have

| Gap | Cost | Priority |
|---|---|---|
| **A second backbone** | ~2 h (ConvNeXt-Tiny) | **High** — "is this a DenseNet artifact?" is the first question. |
| ~~Single-source external~~ | done | **COMPLETE** (Phase 7). Bounded the EyePACS size confound at ~1/3 size, ~2/3 shift — and falsified the multi-source premise. |
| Seeds on in-domain runs | ~1 h | Medium — the deployment-cost deltas carry the in-domain model's unmeasured variance. |
| Multiple-comparison correction | free | Medium — `paired_bootstrap_difference` returns `"note": "uncorrected for multiple comparisons"`. Apply Holm–Bonferroni before calling anything significant. |
| DINOv2 / foundation-model baseline | ~2 h | Medium — reviewers increasingly expect one. |
| Hyperparameter sensitivity | high | Low — but state plainly that no search was performed, on any domain. |

The first two are cheap and would materially strengthen the paper.

## B8. Reproducibility package

Everything needed is present:

- **Code** — `src/` + drivers, 271 tests.
- **Configs** — exact settings, with unknown-key rejection so a config cannot
  silently disagree with the run it describes.
- **Registry** — `outputs/experiment_registry.csv`, one row per run with config,
  metrics, paths, timings, VRAM, epochs. **This is the authoritative record**:
  its key is the experiment id, which encodes protocol, backbone, resolution,
  method and seed. Summary tables are derived; where they disagree, trust it.
- **Per-image predictions** — every claim recomputable without a GPU.
- **Consistency audit** — `python audit_consistency.py` re-derives every summary
  table and exported LaTeX table from the registry and the saved predictions,
  and exits non-zero on any disagreement. 600 checks, about five seconds, no
  GPU. **Run it before quoting a number in the manuscript.** It exists because
  two results were silently overwritten on 2026-08-22 and neither failed
  loudly — see A8. Both bugs are in `tests/test_audit_consistency.py` as
  regression cases, so the audit is itself checked against the failures it was
  written for.

Before release: refresh `README.md` (A8), add a LICENSE, add a data-access
statement (the four datasets are public but each has its own terms — DDR, APTOS
2019 via Kaggle, IDRiD via IEEE DataPort, EyePACS via Kaggle), and state that
raw images are not redistributed.

For a checklist-style venue, `docs/` already answers most of the reproducibility
questions. Point at it directly.

## B9. Venue guidance

**Medical imaging (best fit).** MICCAI — strong fit for the protocol rigour and
clinical framing; note its early-year deadline and page limit, and that a
negative result needs the diagnostic contribution foregrounded. MIDL — very
receptive to careful negative results and calibration work. *Medical Image
Analysis* and *IEEE TMI* — journals, no page pressure, room for the full audit;
the data-provenance findings alone justify the length.

**Clinical/translational.** *npj Digital Medicine*, *Ophthalmology Science*,
*Translational Vision Science & Technology* — reachable if you lead with the
severe-error rate and the deployment-cost framing rather than the DG methodology.

**ML venues.** NeurIPS Datasets & Benchmarks is a genuine fit given the audit and
the leakage findings. Main-track ML venues are a harder sell without a proposed
method that wins.

**My recommendation:** MIDL or *Medical Image Analysis*. Both reward exactly what
this project has — a documented protocol, an honest negative result, and a
calibration finding with a clinical consequence. Add the ConvNeXt backbone and
single-source-external runs first; they close the two obvious review gaps for
about 3.5 hours of compute.

## B10. Writing conventions specific to this project

- **One sign convention.** Use Δ = target − source (or LODO − in-domain)
  everywhere, so negative always means worse. Two bugs in this codebase came
  from mixing conventions between a table and the sentence describing it.
- **Always print n with a CI.** Test sets range from 102 to 35,108 images. An
  interval without its n is uninterpretable.
- **Report QWK and macro-F1 together.** They disagree under prior shift, and the
  disagreement is a finding, not an inconsistency to hide.
- **Never average over a partial matrix.** `analyse_lodo.py` refuses to, by
  design.
- **Say NOT RUN, not nothing.** An absent row and a bad row look identical once
  a mean is taken.
