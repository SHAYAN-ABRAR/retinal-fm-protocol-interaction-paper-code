# Beyond In-Domain Accuracy: Calibrated Domain Generalization for Reliable Diabetic Retinopathy Grading

> **Status: Phase 4 — method comparison complete on one target domain, one seed.**
> Data provenance, label verification, manifest, deduplication, splits, passing
> leakage audit, preprocessing, image cache, three backbones, training loop,
> metrics, calibration, bootstrap CIs, selective prediction, ordinal CORAL,
> Deep CORAL, MixStyle, representation analysis, LaTeX tables, and the experiment
> registry. **158 tests pass.**
>
> **Six methods trained** on DDR + APTOS → unseen IDRiD.
> **No method beat the ERM baseline on target QWK** — all five alternatives are
> significantly worse by paired bootstrap. See
> [`docs/PHASE4_METHOD_COMPARISON.md`](docs/PHASE4_METHOD_COMPARISON.md).
>
> **Not a paper result yet:** one seed, one target domain, no hyperparameter
> search. Multi-seed runs are in progress. Every unrun item is marked `NOT RUN`.
> See [Honesty policy](#honesty-policy).

---

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
├── research_pipeline.py        # interactive driver (# %% cells)
├── configs/paths.yaml          # dataset locations + verified counts + exclusion rules
├── data_external/              # small reference files fetched by the pipeline
│   └── eyepacs_trainLabels.csv #   (~500 KB, downloaded by Cell 7b)
├── docs/
│   ├── DATA_PROVENANCE.md      # Phase-1 audit findings  <-- read this first
│   └── PHASE2_DATA_REPORT.md   # manifest, duplicates, splits, leakage
├── src/
│   ├── data/       inspect.py, provenance.py, eyepacs_labels.py       (Phase 1)
│   │               schema.py, ddr.py, aptos.py, idrid.py, eyepacs.py,
│   │               unified_dataset.py, deduplicate.py, splits.py,
│   │               leakage.py, preprocessing.py, augmentations.py,
│   │               image_stats.py                                     (Phase 2)
│   ├── models/     backbones.py, ordinal_head.py, mixstyle.py, domain_generalization.py
│   ├── losses/     classification.py, ordinal_coral_loss.py, deep_coral_alignment.py,
│   │               calibration.py
│   ├── training/   trainer.py, early_stopping.py, scheduler.py, checkpointing.py
│   ├── evaluation/ metrics.py, calibration.py, selective_prediction.py,
│   │               bootstrap.py, statistics.py
│   ├── visualization/ style.py, dataset_figures.py  (Phase 2); training_,
│   │               performance_, calibration_, feature_figures.py  (Phase 3+)
│   └── utils/      seed.py, logging.py, hardware.py, io.py
├── outputs/        checkpoints, logs, tables, predictions, figures, embeddings, reports
└── tests/
```

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

## 11b. Current result (single run — read the caveats)

`DDR + APTOS → IDRiD`, DenseNet121, seed 42, 95% bootstrap CIs (2,000 resamples):

| Metric | Source validation | Unseen IDRiD | Gap |
|---|---|---|---|
| QWK | 0.8787 [0.8618, 0.8948] | 0.6823 [0.6177, 0.7383] | **−0.196** |
| Macro F1 | 0.6683 [0.6337, 0.7012] | 0.4323 [0.3870, 0.4752] | −0.236 |
| AUROC (macro) | 0.9443 [0.9359, 0.9522] | 0.8672 [0.8473, 0.8862] | −0.077 |
| **ECE** | 0.0797 [0.0686, 0.0948] | **0.3456 [0.3049, 0.3875]** | **+0.266 (4.3×)** |

Calibration degrades far more than discrimination. Temperature scaling fitted on
**source validation only** recovers 75% of in-domain ECE but only 35% on the
unseen domain (0.346 → 0.226) — a single in-domain scalar cannot absorb
domain-induced miscalibration. Selective prediction works but weakly off-domain
(error-detection AUROC 0.654 against 0.5 chance).

Note that IDRiD differs from the sources in **case mix** as well as appearance
(32.9% no-DR vs ~50%), so label shift and covariate shift are confounded in this
gap. Full detail and limitations: [`docs/PHASE3_BASELINE_REPORT.md`](docs/PHASE3_BASELINE_REPORT.md).

## 11c. Method comparison (Stage C, single seed — read the caveats)

`DDR + APTOS → unseen IDRiD`, DenseNet121, batch 32, seed 42, n=507.

| Method | Target QWK | ΔQWK vs ERM | Target ECE | Grade-3 recall |
|---|---|---|---|---|
| **ERM** | **0.753** | — | 0.265 | **0.242** |
| Ordinal | 0.665 | −0.088 ✓ | **0.059** | 0.011 |
| Deep CORAL | 0.665 | −0.088 ✓ | 0.335 | 0.066 |
| MixStyle | 0.643 | −0.110 ✓ | 0.302 | 0.066 |
| MixStyle + Ordinal | 0.662 | −0.091 ✓ | 0.063 | 0.011 |
| Deep CORAL + Ordinal | 0.671 | −0.082 ✓ | 0.071 | 0.011 |

✓ = paired-bootstrap CI excludes zero. Three findings that matter:

- **Every alternative is significantly worse than ERM** on target QWK and macro F1.
- **The ordinal calibration win is an artefact of class collapse.** ECE drops
  4.5×, but grade-3 recall falls from 0.242 to 0.011 — grade 3 is severe NPDR,
  the urgent-referral threshold. Aggregate calibration must never be reported
  without per-class recall beside it.
- **Deep CORAL did what it optimises and it wasn't enough.** Silhouette-by-domain
  fell 3.4×, but a linear probe still recovers the domain at 0.981 — *identical
  to ERM*. Covariance alignment does not remove linearly decodable domain
  information.

Also: source-fitted temperature scaling **worsened** target calibration for the
ordinal models (0.059 → 0.113), because they were under-confident in-domain and
already calibrated out-of-domain.

Full analysis and limitations: [`docs/PHASE4_METHOD_COMPARISON.md`](docs/PHASE4_METHOD_COMPARISON.md).

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
