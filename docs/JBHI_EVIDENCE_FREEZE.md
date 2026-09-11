# JBHI evidence freeze

**Frozen 2026-09-11.** The experimental programme is closed. This document
pins the scientific evidence the manuscript may draw on, in a form a reader can
recompute rather than take on trust.

> **No target result will be used to trigger any additional experiment after
> this freeze.** No further training will be run: no IDRiD full fine-tuning, no
> additional seeds, no additional foundation model, no additional domain
> generalisation method, no hyperparameter search, no new resolution or
> adaptation-method runs. The DDR + APTOS full fine-tuning replication is the
> final experimental evidence.

No completed scientific result is altered by this freeze. Everything below
describes artifacts that already existed.

## 1. Commit and verification

| | |
|---|---|
| evidence commit | `15a50dce4d9aa3875fd9339871a2dc60ba73977b` ("APTOS replication complete: the DDR interaction replicates") |
| what came after | derived tables, figures and documentation only — **no experimental result** |
| regenerate the manifest | `python export_evidence_freeze.py` |

The evidence commit is the one at which all 110 authoritative runs and the
pre-registered analysis existed. Commits after it add the manuscript handoff
package: master tables, figures, claim map and these documents. None of them
runs training or changes a prediction file, and the hashes below pin that.

## 2. Hashes

Full manifest: `outputs/tables/JBHI_EVIDENCE_MANIFEST.csv` — 126 artifacts
(1 registry + 15 authoritative tables + 110 target-test prediction files),
each with SHA256 and byte length.

| artifact | SHA256 (first 32) | bytes |
|---|---|---|
| `outputs/experiment_registry.csv` | `63e81d615084812a231afc599dfbb4f1…` | 447,521 |
| `JBHI_MASTER_RESULTS.csv` | `bbab954c7a24047b722ee71028d57645…` | 3,091 |
| `JBHI_PRIMARY_INTERACTION.csv` | `ffd5937e23b8e3c9a3aac946d4a158e6…` | 961 |
| `two_domain_interaction_holm.csv` | `872ae44294f630001c6394c8c1d322ac…` | 707 |

**Prediction set.** 110 target-test prediction files, 125.8 MB. Combined
digest — SHA256 over the sorted member hashes:

```
913aecf7487e88422bc3d513bc55a93bca8d74a4b4ff961019823dd76427d487
```

Predictions are the artifacts that matter: every statistic in this study is
recomputed from them, not from checkpoints. A run whose predictions, history,
config, report and registry row survive is fully reproducible as a *result*
after its weights are deleted, which is why checkpoint pruning was safe.

## 3. Run accounting

`outputs/tables/JBHI_RUN_ACCOUNTING.csv` — one row per authoritative run.

| held-out domain | frozen probe | partial FT | full FT | total |
|---|---|---|---|---|
| DDR | 20 (10 seeds × 2 models) | 10 (5 × 2) | 10 (5 × 2) | 40 |
| APTOS | 20 | 10 | 10 | 40 |
| IDRiD | 20 | 10 | **0 — not run** | 30 |
| **total** | **60** | **30** | **20** | **110** |

**110/110 COMPLETE.** Total training time across authoritative runs:
**134.3 h**.

IDRiD has no full fine-tuning row. That experiment was never run, is not
imputed anywhere, and no claim depends on it. IDRiD contributes frozen and
partial evidence only.

Executions that were abandoned — operator pauses and machine sleeps — are
recorded in `APTOS_EXECUTION_LOG.md`. None contributed to a registered result
and none was abandoned because of what it showed.

## 4. Datasets

| | DDR | APTOS 2019 | IDRiD | EyePACS (verified subset) |
|---|---|---|---|---|
| role | held-out target / source | held-out target / source | held-out target / source | **source only** |
| labelled images | 12,522 | 3,662 | 516 | 35,108 of 85,178 present |
| grades | 0–4 | 0–4 | 0–4 | 0–4 |
| patient IDs | none usable | none | none | yes |

Split sizes actually used, identical across every run of a given target:

| held-out | train | val | test |
|---|---|---|---|
| DDR | 27,718 | 5,677 | **12,424** |
| APTOS | 33,606 | 7,201 | **3,504** |
| IDRiD | 36,080 | 7,472 | **507** |

Integrity rules enforced in code, not assumed — full detail in
`DATA_PROVENANCE.md`:

- **The EyePACS-derivative folder physically contains all 3,662 APTOS images.**
  The loader keeps only `^\d+_(left|right)$` filenames and rejects every 12-hex
  APTOS id (`src/data/eyepacs.py:99,144`). Without this, holding out APTOS would
  have trained on 100% of its own test set. This is the single most important
  integrity guard in the study.
- The derivative's provided train/val/test folders leak (1,231 base photographs
  and 14,244 of 43,565 patients appear in more than one split). They are
  discarded entirely and images are re-split at patient level.
- `augmented_resized_V2/` is on `forbidden_paths` and never read.
- EyePACS labels come from a third-party derivative, so 35,108 were verified
  against an independent copy of the official labels at 100.0000% agreement;
  the 50,070 unverifiable images were excluded.

## 5. Checkpoint identities

| arm | identity |
|---|---|
| ImageNet-MAE | `timm` `vit_large_patch16_224.mae` — the exact checkpoint RETFound was initialised from |
| RETFound-CFP | `RETFound_mae_natureCFP.pth`, 3,952,489,221 bytes |
| RETFound SHA256 | `e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485` |
| loaded | 294 tensors, 100.0% of parameters, asserted at every run start |
| trainable under full FT | 303,306,757 parameters, asserted at runtime in all 20 full-FT runs |

Both arms are ViT-L/16 at 224 px. The comparison is a **lineage intervention**:
the same architecture, the same starting weights, differing only by RETFound's
additional retinal-domain MAE continuation pretraining.

## 6. Hypotheses, as finally defined

**Primary.** The protocol-by-initialisation interaction

```
Δ_protocol(s) = QWK_ImageNet-MAE(s) − QWK_RETFound(s)      within a protocol
I_full(s)     = Δ_full(s) − Δ_frozen(s)                    per seed
```

tested per held-out domain over the five common seeds, two-sided, on QWK.
Family for multiplicity adjustment: **exactly two members**, DDR and APTOS —
the only domains with matched full fine-tuning.

**Secondary.** The model comparison under full fine-tuning within each domain;
the partial-vs-frozen and full-vs-partial interactions; all non-QWK metrics.

**Descriptive.** Per-protocol comparisons at frozen and partial depth;
source-validation optimisation behaviour; every IDRiD result.

QWK is the primary metric and is not replaced by another that reads better.

## 7. Inferential framework and its vocabulary

Fixed for the manuscript. The phrase "two-bar rule" is **not** used.

| instrument | role |
|---|---|
| crossed seed × case bootstrap, 2,000 replicates, rng seed 7 | **uncertainty interval**. Not a test: built around the empirical estimate, not under H₀, so its tail mass is not a calibrated *p*-value and is never Holm-adjusted |
| paired seed-level *t*-test on per-seed effects | **formal inferential test** |
| Holm | **multiplicity adjustment**, across the two domain-specific interaction tests only |
| exact sign-flip permutation | **distribution-free sensitivity analysis**; floor 2/2⁵ = 0.0625 at five seeds, so it cannot reach 0.05 at this sample size for any effect size |
| seed SD, sign agreement | **descriptive stability diagnostics** |

An effect whose multiplicity-adjusted seed-level *p* < 0.05 **and** whose
crossed interval excludes zero is described as *"statistically supported under
the study's inferential framework"*. Where the two disagree, the formal test
governs and the effect is **not** supported — this occurs for the DDR full-FT
model comparison, whose interval excludes zero while *p* = 0.1035.

## 8. Seed sets

| protocol | seeds | n |
|---|---|---|
| frozen linear probe | 42, 1, 2, 3, 4, 5, 6, 7, 8, 9 | 10 |
| partial FT (last 4/24) | 42, 1, 2, 3, 4 | 5 |
| full FT (all encoder parameters) | 42, 1, 2, 3, 4 | 5 |

**Ten-seed frozen estimates and five-seed adaptation estimates are never mixed
without labelling.** Any protocol interaction uses the common five seeds only,
because the contrast is paired within seed. `JBHI_MASTER_RESULTS.csv` carries a
`seed_set` column (`all-10` / `common-5`) on every row and reports both for the
frozen protocol.

## 9. Software and hardware

| | |
|---|---|
| Python | 3.14.3 |
| PyTorch | 2.9.1+cu128 |
| timm | 1.0.28 |
| NumPy / pandas / SciPy / scikit-learn | 2.5.2 / 3.0.5 / 1.18.0 / 1.9.0 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB) |
| OS | Windows 11 (10.0.26200) |
| determinism | `deterministic = False` on every full-FT run — cuDNN autotuning on, deterministic algorithms off |

**Runs are not bit-reproducible and do not claim to be.** Measured directly: an
abandoned execution and its replacement, identical command and seed, both from
scratch, differed at epoch 0 (source-validation train loss 0.7844 vs 0.7856).
Reproducing this study reproduces the distribution over seeds, not individual
numbers. The jitter enters both arms identically and cannot manufacture a
difference between them; it slightly widens seed-level spread, which is
conservative. See `APTOS_EXECUTION_LOG.md`.

## 10. Study chronology — stated as it happened

This ordering matters for how the two-domain family is described, and it is
**not** a prospectively specified two-domain design.

1. The **DDR** full fine-tuning experiment was specified and its protocol frozen
   (`FULL_FINETUNE_PROTOCOL.md`), then run.
2. The **DDR interaction was observed**: I_full = −0.1141, 5/5 seeds negative.
3. **Only then** was APTOS specified as a confirmatory replication. The protocol
   (`APTOS_FULL_FINETUNE_REPLICATION_PROTOCOL.md`) was committed **before the
   first APTOS full-FT run**, and the analysis script
   (`analyse_aptos_replication.py`) was written and rehearsed while seven of the
   ten runs still did not exist.
4. The APTOS **downstream recipe was copied unchanged from DDR and frozen before
   any APTOS full-FT target outcome was inspected**. No APTOS-specific tuning
   was permitted or performed.
5. APTOS **reproduced the direction and the interaction**: I_full = −0.0992,
   5/5 seeds negative.
6. For final presentation, **Holm adjustment is applied across the two
   domain-specific interaction tests**.

> The two-domain family was **not** specified before DDR was observed. DDR is
> the originating result and APTOS is its confirmatory replication. The Holm
> adjustment across the two is applied for conservative final reporting, not as
> evidence of a pre-planned two-domain design. The manuscript must describe it
> this way.

Target blindness during APTOS execution was maintained and is auditable: no
APTOS full-FT metric was computed until the tenth run registered, and the
analysis code predates seven of the ten runs in the git history.

## 11. Gates passed at freeze

| gate | result |
|---|---|
| test suite | 414 passed |
| `audit_consistency.py` | PASS, 3,088 checks |
| `audit_resumed_runs.py` | exit 0 — no reported model affected |
| `audit_experiment_identity.py` | 9 pre-existing conflicts, **none involving any full-FT run** |
| `paper/check_numbers.py` | no unsupported value in the manuscript |
| `paper/check_report_numbers.py` | both reports trace fully to generator CSVs |
| `paper/check_latex.py` | clean across all included files |
