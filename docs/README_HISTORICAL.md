# Mid-project README — HISTORICAL RECORD

> **HISTORICAL PROJECT RECORD — NOT AUTHORITATIVE FOR THE FINAL MANUSCRIPT.**
> This is the README the repository carried while the work was in progress. It
> is preserved verbatim rather than deleted, because it documents the staged
> plan the project actually followed and the reasoning behind several design
> choices.
>
> **It is out of date in ways that matter.** It describes training cells as
> "stubs", quotes three-seed headline results, and uses the retired "two-bar"
> terminology. The programme is now closed and the evidence frozen.
>
> Current state: [`../README.md`](../README.md),
> [`JBHI_EVIDENCE_FREEZE.md`](JBHI_EVIDENCE_FREEZE.md), and
> [`README.md`](README.md) (documentation index).

## 1. Research objective

Diabetic-retinopathy (DR) classifiers routinely report strong in-domain accuracy,
but deployment means running on images from a clinic the model has never seen.
This project asks two questions under a rigorous protocol:

1. **How much do accuracy *and calibration* degrade on a completely unseen
   dataset?** Calibration is treated as a first-class outcome, not an afterthought —
   a model that is confidently wrong on a new clinic's images is worse than one
   that knows it is uncertain.
2. **Do ordinal learning, domain-generalization methods, and post-hoc
   calibration actually reduce that degradation?** Including the possibility that
   they do not. Negative results are reportable results.

DR severity is graded on the 5-point ICDR scale (0 No DR → 4 Proliferative DR),
which is **ordinal**, so ordinal-aware objectives are evaluated alongside plain
5-class cross-entropy.

## 2. Experimental design

Four public datasets, each treated as a distinct clinical domain:

| Domain ID | Dataset | Images | Role |
|---|---|---|---|
| 0 | DDR | 12,424 | source / target |
| 1 | APTOS 2019 | 3,504 | source / target |
| 2 | IDRiD | 507 | source / target |
| 3 | EyePACS | 35,108 *(label-verified)* | source / target |

Counts are **after** removing 265 byte-identical duplicates (see
[`docs/PHASE2_DATA_REPORT.md`](docs/PHASE2_DATA_REPORT.md)). Only EyePACS exposes
patient ids (17,561 patients); it is split at patient level, the others at image
level — a documented dataset limitation.

**Protocols**

- *In-domain*: train, validate and test within one dataset.
- *Single-source external validation*: train on one dataset, test on the other three.
- *Leave-one-domain-out (LODO)*: train on three, test on the fourth. Four runs.

**The held-out target domain is sacred.** It never influences training,
hyperparameters, augmentation choice, early stopping, temperature scaling, λ
selection, or model selection. Validation data comes only from source domains.
If a proposed analysis would violate this, the correct response is to refuse it
and explain the right protocol.

**Methods.** ERM baseline → ordinal objective (CORAL/CORN-style) → Deep CORAL
alignment → MixStyle → temperature scaling → selective prediction. A combined
method is considered *only* if the baselines justify it (see
[Naming](#naming-collision-two-different-corals)).

## 3. Dataset setup

Datasets live outside the repo. Their locations — and the verified facts about
each — are in [`configs/paths.yaml`](configs/paths.yaml).

**Read [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) before touching the
data.** A Phase-1 audit of `D:\DB` found four problems that would silently
invalidate every downstream result:

- **`Dataset 1` is not raw EyePACS.** It is a third-party derivative that also
  contains **100% of the APTOS dataset** (3,662/3,662 ids). Used naively, a LODO
  run holding out APTOS would train on its entire test set.
- **50,070 of its EyePACS labels are corrupted.** Verification against an
  independent copy of the Kaggle `trainLabels.csv` found one single grade-3 image
  among 50,070, where ~1,247 are expected. The domain is restricted to the
  **35,108 images whose labels agree 100.0000%** with the reference.
- **Its provided train/val/test folders leak** — 1,231 base photographs and
  14,244 patients appear in more than one split.
- **It ships baked-in augmented duplicates**, so augmented copies of test images
  sit in the training folders.

These are handled by explicit, asserted exclusion rules in `configs/paths.yaml`,
not by hoping the loader gets it right.

To point the project at a different machine, edit **only** `configs/paths.yaml`.
Nothing else hard-codes a data path.

## 4. Environment setup

Hardware target: **RTX 5060 Laptop (8 GB VRAM), 16 GB RAM, Windows 11**.

### 4.1 Install PyTorch first — the CUDA build matters

The RTX 5060 is Blackwell, **compute capability sm_120**. Kernels for sm_120 ship
only in **CUDA 12.8 and newer** PyTorch builds. A plain `pip install torch` gives
you a CPU-only wheel on Windows; an older CUDA wheel imports fine and reports
`cuda.is_available() == True`, then dies on the first kernel launch with
`no kernel image is available for execution on the device`.

```powershell
cd "D:\Research Code\dr_domain_generalization"
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CUDA 12.8 build -- required for sm_120
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

`src/utils/hardware.py` detects both failure modes explicitly and, in
`assert_cuda_ready()`, **refuses to fall back to CPU** — this study is not
feasible on CPU, so silence would be worse than a crash.

### 4.2 Verify

Run Cells 1–4 of `research_pipeline.py`. Cell 2 prints the full hardware report
and launches a real CUDA kernel as a smoke test.

## 5. How to run

`research_pipeline.py` is a **VS Code Python Interactive** script. Each `# %%`
marker is a cell; press *Run Cell* (Ctrl+Enter) as you would in Colab. Nothing
expensive runs on import, and no cell trains a model unless you run it.

| Cells | Phase | Cost | What it does |
|---|---|---|---|
| 1–4 | 1 | seconds | Imports, GPU diagnostics, config, seeding + fingerprint |
| 5–8 | 1 | ~3–5 min | Directory discovery, metadata, label mapping, **EyePACS label verification (7b)**, provenance audit |
| 9–13 | 2 | ~10 min | Manifest, deduplication, splits, **leakage audit**, image statistics, 18 figures, dataloaders. Cell 13 needs torch |
| 13b | 3 | ~30 min once | **Pre-resized image cache.** Turns a ~5 img/s raw loader into ~1340 img/s; training becomes GPU-bound |
| 14–16 | 3 | GPU: ~20 min | Model + batch-size probe, sanity/overfit check, ERM training |
| 17–18b | 3 | minutes | Evaluation, temperature scaling, generalization gap, figures, registry |
| 19–29 | 4 | GPU hours | Ordinal, Deep CORAL, MixStyle, full LODO, bootstrap, selective prediction |
| 30–32 | 5 | minutes | Figures, tables, summary report |

Cells 14+ are currently **stubs**. Running one prints what it will do and what it
is waiting on. No stub fabricates numbers or writes result files.

### Staged development

Following the plan in the brief: **Stage A** sanity check on a DDR+APTOS subset at
224×224 → **Stage B** single-source domain shift → **Stage C** methodology on
DDR+APTOS→IDRiD (computationally manageable) → **Stage D** full LODO with EyePACS.

## 6. Folder structure

```
dr_domain_generalization/
├── research_pipeline.py        # interactive driver (35 `# %%` cells)
├── run_lodo.py                 # the four leave-one-domain-out experiments
├── run_in_domain.py            # in-domain ceilings + deployment-cost comparison
├── run_single_source.py        # the 4x4 cross-domain matrix
├── run_method_comparison.py    # Stage-C six-method ablation
├── analyse_lodo.py             # single-seed LODO analysis
├── analyse_lodo_seeds.py       # multi-seed LODO (two-bar criterion)
├── analyse_seeds.py            # Stage-C multi-seed
├── analyse_selective.py        # risk-coverage / abstention
├── regenerate_figures.py       # rebuild figures from saved predictions
├── configs/                    # paths, baseline, domain_generalization, experiments
├── data_external/
│   └── eyepacs_trainLabels.csv #   reference labels (~500 KB, Cell 7b)
├── docs/
│   ├── DATA_PROVENANCE.md            # Phase-1 audit  <-- read this first
│   ├── PHASE2_DATA_REPORT.md         # manifest, duplicates, splits, leakage
│   ├── PHASE3_BASELINE_REPORT.md     # first baseline (superseded)
│   ├── PHASE4_METHOD_COMPARISON.md   # six methods x 3 seeds
│   ├── PHASE5_LODO_REPORT.md         # the four LODO experiments
│   ├── PHASE6_IN_DOMAIN_REPORT.md    # in-domain ceilings, deployment cost
│   ├── PHASE7_CROSS_DOMAIN_MATRIX.md # the 4x4 matrix, size vs shift
│   ├── PHASE8_BACKBONE_COMPARISON.md # ConvNeXt-Tiny vs DenseNet121
│   ├── PHASE9_RESOLUTION_REPORT.md   # 512 px vs 224 px
│   └── PROJECT_HANDBOOK.md/.html     # where everything is + paper guide
├── src/
│   ├── data/       aptos.py, augmentations.py, cache.py, ddr.py
│   │               deduplicate.py, eyepacs.py, eyepacs_labels.py, idrid.py
│   │               image_stats.py, inspect.py, leakage.py, loaders.py
│   │               preprocessing.py, provenance.py, schema.py, splits.py
│   │               unified_dataset.py
│   ├── models/     backbones.py, mixstyle.py
│   ├── losses/     classification.py, deep_coral_alignment.py
│   │               ordinal_coral_loss.py
│   ├── training/   checkpointing.py, early_stopping.py, methods.py
│   │               trainer.py
│   ├── evaluation/ bootstrap.py, calibration.py, embeddings.py, evaluate.py
│   │               metrics.py, selective_prediction.py
│   ├── visualization/ calibration_figures.py, dataset_figures.py
│   │                  domain_figures.py, error_figures.py, feature_figures.py
│   │                  performance_figures.py, pipeline_diagram.py, style.py
│   │                  training_figures.py
│   ├── reporting/  summary.py, tables.py
│   └── utils/      config.py, hardware.py, io.py, logging.py, registry.py,
│                    seed.py
├── outputs/        checkpoints, logs, tables, predictions, figures, embeddings, reports
└── tests/         11 files, 271 tests
```

`src/` is the tested library; the top-level `run_*` and `analyse_*` scripts are
thin drivers that compose it. Nothing important lives only in the pipeline file.

## 7. Reproducibility

`src/utils/seed.py` seeds Python, NumPy, PyTorch CPU and CUDA, and records a
`RunFingerprint` (seed, Python/package/CUDA versions, GPU model, driver, git
commit) saved with every run to `outputs/reports/`.

`deterministic=True` is available but costs roughly 10–30% throughput because it
disables cuDNN autotuning. Default is `deterministic=False` with a fixed seed;
determinism is enabled for tests and for anything that must be bit-reproducible.

## 8. Memory budget (8 GB VRAM)

Mixed precision throughout; gradient accumulation where the effective batch size
must be preserved; evaluation under `inference_mode`; predictions moved to CPU
immediately.

**Measured** on the RTX 5060 (8 GB), DenseNet121, 224px, AMP, batch 16:
peak **1.13 GB** allocated — far below the card's limit, so there is ample room
for 384px or larger batches. `estimate_memory()` probes real peak usage rather
than guessing.

| Resolution | Starting batch | Fallback |
|---|---|---|
| 224×224 | 16 (measured 1.13 GB) | 8 |
| 384×384 | 8 | 4, then 2 + gradient accumulation |

Throughput after caching: loader 1343 img/s (2 workers) vs GPU 217 img/s
(DenseNet121) / 317 img/s (ConvNeXt-Tiny) — **GPU-bound**, so raising
`num_workers` will not make training faster.

**Windows note:** DataLoader workers use `spawn`, which re-imports the main
module. Any standalone script using `num_workers > 0` must put its work behind
`if __name__ == "__main__":` or it will appear to hang. VS Code interactive cells
are unaffected. `build_loaders()` warns when it detects the risky combination.

Batch size is **recorded, never silently changed** mid-experiment. With 16 GB
system RAM (~3.5 GB free when audited), images load lazily and `num_workers=4` is
an upper bound to validate, not a default to assume.

## 9. Naming collision: two different "CORAL"s

The literature has two unrelated methods with the same name. This project keeps
them in separate files with unambiguous names:

| File | Method | What it does |
|---|---|---|
| `src/losses/ordinal_coral_loss.py` | **CORAL ordinal regression** (Cao et al.) | Rank-consistent ordinal classification via K−1 binary tasks with shared weights. Exploits `0 < 1 < 2 < 3 < 4`. |
| `src/losses/deep_coral_alignment.py` | **Deep CORAL** (Sun & Saenko) | **CORrelation ALignment** — matches second-order feature covariance across *source domains*. A domain-generalization objective. |

They solve different problems and are never interchangeable. Cells 20 and 21
respectively.

## 10. Experiment naming convention

```
{protocol}_{sources}__{target}_{backbone}_{method}_s{seed}
e.g.  lodo_ddr-aptos-eyepacs__idrid_convnext-tiny_mixstyle-ordinal_s42
      indomain_ddr__ddr_densenet121_erm_s42
```

Every finished run appends one row to `outputs/experiment_registry.csv` with its
config, seed, domains, backbone, loss, hyperparameters, best checkpoint and all
test metrics. Rows are appended; results are never overwritten.

## 11. Troubleshooting CUDA

| Symptom | Cause | Fix |
|---|---|---|
| `torch.version.cuda is None` | CPU-only wheel | Reinstall from the cu128 index (§4.1) |
| `no kernel image is available` | CUDA < 12.8 build on sm_120 | Same |
| `cuda.is_available()` False, `nvidia-smi` works | Wheel/driver mismatch, or `CUDA_VISIBLE_DEVICES=""` | Check the env var, reinstall |
| OOM at 384×384 | 8 GB VRAM | Batch 8 → 4 → 2 + gradient accumulation; record the change |
| Workers crash / RAM exhausted | 16 GB shared with everything | `num_workers` 4 → 2; close other applications |

Cell 2 diagnoses all of these and prints an actionable message.

## 11b. Headline results (3 seeds — read the caveats)

Full leave-one-domain-out, DenseNet121, batch 32, seeds 42/1/2.
Detail: [`docs/PHASE5_LODO_REPORT.md`](docs/PHASE5_LODO_REPORT.md).

| Held out | n_test | target QWK | target ECE | after T | severe |
|---|---|---|---|---|---|
| DDR | 12,424 | 0.7383 ± 0.0055 | 0.1296 ± 0.0083 | **0.0472** | 0.1598 |
| APTOS | 3,504 | 0.8590 ± 0.0054 | 0.1239 ± 0.0137 | **0.0397** | 0.0628 |
| IDRiD | 507 | 0.7413 ± 0.0244 | 0.2294 ± 0.0151 | 0.1213 | 0.1177 |
| EyePACS | 35,108 | 0.4147 ± 0.0081 | 0.2846 ± 0.0074 | 0.1729 | 0.2407 |

**Do not quote a mean over these four.** The range is −0.483 to +0.063 against
source validation; no target is near the mean.

**Validation performance does not predict generalization.** On IDRiD the three
seeds agree to 0.0009 on source validation and differ by 0.0244 on the target —
a ratio of 28×. Model selection may only use source validation under this
protocol, so selection is close to blind with respect to what matters.

**The cost of cross-domain deployment**, against in-domain models on identical
test images, **three seeds on both sides, paired seed-to-seed**
([`docs/PHASE6_IN_DOMAIN_REPORT.md`](docs/PHASE6_IN_DOMAIN_REPORT.md)):

| Domain | in-domain | LODO mean | Δ | Δ SD | verdict |
|---|---|---|---|---|---|
| DDR | 0.8702 ± 0.0086 | 0.7327 ± 0.0038 | **−0.1376** | 0.0119 | REAL (both bars) |
| APTOS | 0.9085 ± 0.0097 | 0.8625 ± 0.0103 | −0.0460 | 0.0127 | CI spans zero |
| IDRiD | 0.5846 ± 0.0191 | 0.6803 ± 0.0504 | +0.0957 | 0.0695 | CI spans zero |
| EyePACS | 0.7063 ± 0.0139 | 0.4153 ± 0.0082 | **−0.2910** | 0.0202 | REAL (both bars) |

Severe errors on the same matched images: DDR 0.0788 → 0.1631 (**+107%**),
EyePACS 0.0938 → 0.2396 (**+155%**), both clearing the two-bar criterion.

## 11c. Method comparison (Stage C, 3 seeds)

`DDR + APTOS → unseen IDRiD`, n=507.
Detail: [`docs/PHASE4_METHOD_COMPARISON.md`](docs/PHASE4_METHOD_COMPARISON.md).

| Method | Target QWK | Target ECE | severe |
|---|---|---|---|
| **ERM** | **0.7235 ± 0.0320** | 0.2973 ± 0.0304 | **0.1065** |
| Ordinal | 0.6545 ± 0.0111 | **0.0610 ± 0.0068** | 0.2156 |
| Deep CORAL | 0.6743 ± 0.0158 | 0.3264 ± 0.0347 | 0.1374 |
| MixStyle | 0.6617 ± 0.0197 | 0.3122 ± 0.0229 | 0.1486 |
| MixStyle + Ordinal | 0.6596 ± 0.0170 | 0.0632 ± 0.0091 | 0.2110 |
| Deep CORAL + Ordinal | 0.6778 ± 0.0068 | 0.0745 ± 0.0182 | 0.2032 |

- **Every alternative is worse than ERM on target QWK**, clearing both the
  across-seed SD and the paired bootstrap.
- **The ordinal calibration win comes with class collapse.** ECE improves ~10×
  the seed noise, but grades 1 and 3 collapse and severe-error rate doubles.
  Aggregate calibration must never be reported without per-class recall beside it.
- **Deep CORAL's calibration penalty was withdrawn.** A single-seed claim
  (+0.071) did not survive three seeds (+0.029 against pooled SD 0.033).

## 11d. Cross-domain matrix (single-source, seed 42)

Target QWK, rows = trained on, columns = tested on; diagonal is in-domain.
Detail: [`docs/PHASE7_CROSS_DOMAIN_MATRIX.md`](docs/PHASE7_CROSS_DOMAIN_MATRIX.md).

| train \ test | DDR | APTOS | IDRiD | EyePACS | n_train |
|---|---|---|---|---|---|
| **DDR** | *0.876* | 0.782 | 0.600 | 0.372 | 8,697 |
| **APTOS** | 0.568 | *0.909* | 0.723 | 0.433 | 2,809 |
| **IDRiD** | 0.516 | 0.597 | *0.588* | 0.236 | 335 |
| **EyePACS** | 0.731 | 0.856 | 0.756 | *0.709* | 24,574 |

**Multi-source training buys nothing.** One source matches three on all four
targets; every difference is inside the seed SD. The three-source pools were
68–89% EyePACS, so the LODO numbers largely measure *which* source was used,
not *how many*.

**The lowest-scoring dataset is the best source.** EyePACS reaches only 0.709 on
its own data at 224 px, yet trained on EyePACS and tested on DDR it scores
**0.731, above its own validation score of 0.712** (both 224 px). A source's own
score is a poor guide to its worth as training data.

> **Corrected 2026-08-22.** This was previously headed *"the worst-labelled
> dataset is the best source"* and explained by label noise. **The 512 px re-run
> reaches 0.8004 on the same test images** (+0.0914, CI [+0.0694, +0.1142]), so
> resolution — not label quality — set the 0.709. The observation stands; the
> label-noise mechanism is withdrawn. See Phase 6 §0a.

**Resolution was the binding constraint on EyePACS.** At 512 px the in-domain
model reaches 0.8004 QWK and cuts the severe-error rate from 0.0919 to 0.0575,
a 37% relative reduction on the metric that matters clinically. Every other
number in this README is a 224 px measurement; the 512 px matrix is running.

## 11e. Selective prediction

Abstention is not a rescue. Error-detection AUROC across the four unseen
domains is 0.657–0.723; handing a clinician the least-confident 30% of cases
cuts the automated error rate by only 17–33%, and works *worst* on EyePACS
where it is needed most. Temperature scaling is monotonic, so these numbers are
identical before and after calibration.

## 11f. Foundation model: the protocol decides the answer

RETFound (ViT-L/16, MAE, ~904,170 colour fundus photographs) against
`vit_large_patch16_224.mae` — **the checkpoint RETFound's own args name as its
initialisation**. Same architecture, same 303.3 M parameters, same objective,
same splits. The intervention is the additional retinal-domain MAE pretraining
stage itself, not dataset identity alone.

**Frozen features, linear probe, 5 seeds**
([`docs/PHASE12_FOUNDATION_MODEL.md`](docs/PHASE12_FOUNDATION_MODEL.md)):

| Target | RETFound | ImageNet-MAE | Δ | Δ/SD | verdict |
|---|---|---|---|---|---|
| DDR | 0.5130 | **0.5756** | +0.0626 | 2.46× | ImageNet better |
| APTOS | 0.4829 | **0.5963** | +0.1134 | 2.07× | ImageNet better |
| IDRiD | 0.6792 | 0.6363 | −0.0429 | 1.25× | CI spans zero |

**Both fine-tuned identically, last 4 of 24 blocks**
([`docs/PHASE13_FINETUNE.md`](docs/PHASE13_FINETUNE.md)):

| Target | RETFound | ImageNet-MAE | Δ | Δ/SD | verdict |
|---|---|---|---|---|---|
| DDR (n=5) | 0.6966 | 0.7183 | +0.0217 | 0.82× | within seed noise |
| APTOS (n=5) | 0.8261 | 0.8334 | +0.0073 | 0.42× | within seed noise |
| IDRiD (n=5) | **0.7676** | 0.7252 | −0.0423 | 1.13× | **RETFound better** |

**The two protocols disagree.** The frozen probe shows a large gap on two of
three targets; fine-tuning shows none, and reverses on the third. Linear probing
is the standard cheap benchmark for a foundation model, and here it does not
predict the model's behaviour when used as intended.

EyePACS is excluded as a target throughout: it is **in RETFound's pretraining
corpus**, so using it would report leakage as generalization.
`assert_target_not_pretrained` raises rather than warns, in both
`run_retfound_probe.py` and `run_lodo.py`.

## 12. Honesty policy

Non-negotiable for this project:

- **Nothing is fabricated** — no placeholder accuracy, QWK, AUROC, CI, timing or
  significance value ever appears in a result file.
- Unrun work is labelled `NOT RUN` / `PENDING`, as the Cell 9+ stubs are.
- Measured facts and assumptions are kept visibly separate; `docs/DATA_PROVENANCE.md`
  cites how each number was obtained and re-derives it in code.
- Known limitations (no patient IDs for DDR/APTOS/IDRiD, the EyePACS reference
  labels being a corroborated *secondary* source, 5 grade-1 test images in IDRiD)
  are stated in the paper, not buried.
- The combined method is not called novel unless an ablation and a literature
  review support it.
