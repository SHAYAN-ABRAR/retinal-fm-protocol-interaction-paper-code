# JBHI pre-submission gap analysis

> **HISTORICAL PROJECT RECORD — NOT AUTHORITATIVE FOR THE FINAL MANUSCRIPT.**
> See [`JBHI_EVIDENCE_FREEZE.md`](JBHI_EVIDENCE_FREEZE.md) and
> [`JBHI_DRAFTING_MANIFEST.md`](JBHI_DRAFTING_MANIFEST.md) for the frozen final
> state. Statements below were true at the time they were written and may have
> been superseded — including older test and audit counts, the retired
> "two-bar" terminology, and novelty framing that has since been narrowed.


**Date:** 2026-08-30 · **Repo state:** 278 runs, 273 COMPLETE, 5 DIVERGED · 328 tests · 2786 audit checks
**Central claim under audit:** *Does linear probing predict fine-tuned cross-domain
performance of a retinal foundation model?* — a lineage-matched comparison of
RETFound against the exact ImageNet-MAE checkpoint it was initialised from.

Every row below was checked against the registry or the source before being
written. Three items in the incoming review are **already resolved**; one is
**understated**; one is **not fixable with compute at all**. Those are called out
rather than silently actioned, because acting on a stale premise is how a queue
burns GPU hours on a solved problem.

---

## 0. Items already resolved (no action)

| # | Claim in review | Actual state |
|---|---|---|
| 1 | "IDRiD partial FT has fewer seeds. Finish it." | **Done.** Partial FT is 5 seeds on DDR, APTOS *and* IDRiD. Frozen probing is **10** seeds on all three. Verified from `finetune_comparison.csv` and `linear_probe_comparison_vit_large_mae_in1k.csv`. |
| 14 | "corrupted references such as `Fig.~[corrupted]ef{...}`" | **Fixed 2026-08-30** (commit `6bb07ee`). Three `\ref` commands had been corrupted by an editing layer reading `\r` as a carriage return. Zero remain; verified by scanning for lines beginning with a bare escape fragment. |
| 1b | "extend the central comparison to 10 seeds if reasonable" | **Frozen arm is already at 10.** Partial FT at 10 would cost ~40 h; see Q3 below. |

---

## 1. Gap table

| # | Issue | Why a JBHI reviewer cares | Current evidence | Required action | Experiment? | GPU | Priority | Files |
|---|---|---|---|---|---|---|---|---|
| **4** | ✅ **RESOLVED 2026-08-31, and the first write-up of it was wrong.** Six 224 px/batch-16 runs (EyePACS, IDRiD, seeds 42/1/2) close the missing cell. **EyePACS:** resolution +0.0865, crossed interval excludes zero, seed-level paired *t*-test survives Holm (0.0296) — **established**. **IDRiD:** resolution +0.0421, crossed interval positive, 3/3 seeds same direction, but the seed-level *t*-test does **not** clear the inferential threshold (0.0876) — **suggestive/underpowered, not established**. Batch survives on neither domain, though it is about half the IDRiD point estimate. An earlier version of this row claimed resolution survived Holm on *both* domains; that rested on Holm-correcting the crossed bootstrap's tail mass, which is not a calibrated *p*-value (see `src/evaluation/seed_inference.py`). No isolated resolution effect is claimed for DDR or APTOS, which have no 224/b16 arm. See Phase 9 §1b and `analyse_configuration.py`. Original entry: **Resolution effect is confounded with batch size.** 224 px runs at batch 32; 512 px runs at batch 16. There is **no 224 px / batch 16 run in the registry** (verified: 27 runs at 224/32, 12 at 512/16, zero at 224/16). | The paper claims resolution beats every DG method (+0.0967 QWK). As run, that comparison changes two variables. A reviewer can invalidate the claim in one sentence. | `resolution_comparison.csv`; Phase 9 | Run 224 px at batch 16 on the two strongest targets, paired seeds. Then either re-state the effect as resolution-only or as "resolution and its batch-size consequence jointly". | **Yes** | ~4 h | **P0** | `run_lodo.py`, `analyse_resolution.py`, Phase 9, README, `main.tex`, fig3 |
| **2** | **"Fine-tuning" means last 4 of 24 blocks.** Full FT never run. | The headline is that protocol changes the ranking. "Your fine-tuning wasn't fine-tuning" attacks the headline directly. | Phase 13; VRAM measured at **5.67 GB** for full ViT-L FT with gradient checkpointing at batch 16 — it *is* feasible here | (a) Rename to **partial fine-tuning** everywhere — free. (b) Run full FT on ≥1 target to show the conclusion survives the strongest adaptation. | **Yes** (b) | ~19 h (DDR, 3 seeds, both models) | **P0** | Phases 12–13, README, `main.tex`, `analyse_finetune.py` |
| **5** | **Two-bar rule is ad hoc as primary inference.** | A statistics reviewer will not accept "Δ > one seed SD" as a significance criterion. It has no calibrated error rate. | `analyse_*.py` all use it; the paired bootstrap is single-level | Implement a **hierarchical bootstrap** (resample seeds, then cases within seed). Demote the two-bar rule to a robustness diagnostic. Holm–Bonferroni across the hypothesis family. | No | 0 | **P0** | `src/evaluation/bootstrap.py`, all `analyse_*.py`, all phase docs |
| **8** | **Size-vs-shift decomposition extrapolates.** Largest *measured* subsample is **8,879**; the in-domain budget is **24,574**. That is **2.77× beyond the largest measured point** — the review says 2.08×, which uses the full 11,841 pool, itself never measured as a sweep point. | Presenting an extrapolated fit as a causal decomposition ("6% size / 94% shift") is the kind of overreach that draws a reject. | `subsample_sweep.csv` (0.25/0.50/0.75), `subsample_decomposition.csv` | **Cannot be fixed with compute.** The LODO source pool is capped at 11,841 by construction; 24,574 is unreachable in this protocol. Reframe as a fitted sensitivity model with slope CI, and shade the extrapolated region. | **No — impossible** | 0 | **P0** | `analyse_subsample.py`, Phase 9, README, `main.tex` |
| **10** | **Reproducibility bundle.** `outputs/predictions/`, `checkpoints/`, `logs/`, `embeddings/` are all gitignored, so a clean clone **cannot** run the prediction-level audit. | README implies audits pass on a clean clone. They do not. | `.gitignore` lines 5–8 | Split audit into **source-only** and **full-artifact** modes; publish a prediction bundle (Zenodo/Release) with checksums and a fetch script; correct the README. | No | 0 | **P0** | `.gitignore`, `audit_consistency.py`, README, new `fetch_artifacts.py` |
| **3** | **Matched recipe may disadvantage one initialisation.** Both get lr 1e-4. | "You didn't tune RETFound" is a fair objection to a null. | Phase 13 §4 states this as a limitation | Add a **source-validation-only** LR selection from a small fixed grid, per model. Never inspect target test. | **Yes** | ~10 h (narrow) | **P1** | `run_lodo.py`, new `run_lr_selection.py` |
| **6** | **Calibration underused.** ECE/NLL/Brier/AURC exist in the registry but the manuscript has no calibration section. | JBHI is a health-informatics venue; reliability under shift is squarely in scope, and the data already exists. | `test_ece`, `test_nll`, `test_brier`, `test_auroc_macro`; `risk_coverage_curves_erm.csv` | Write the section; add probe-vs-FT calibration comparison. Mostly analysis of existing runs. | No | 0 | **P1** | new `analyse_calibration_protocol.py`, `main.tex` |
| **7** | **No referable-DR outcome.** | Clinicians read sensitivity/specificity at a referral threshold, not QWK. | `analyse_severe_error.py` mentions referable; no generated table | Add referable-DR (threshold pre-registered, not tuned on target) with sens/spec/AUROC/AUPRC as **secondary**. | No | 0 | **P1** | new `analyse_referable.py`, `main.tex` |
| **9** | **Provenance not a formal artifact.** `docs/DATA_PROVENANCE.md` exists but is prose, not a generated table. | The contamination findings are a genuine contribution and currently unciteable. | `docs/DATA_PROVENANCE.md`, Phase 2 | Emit a provenance table from structured metadata; state observable facts only, no accusations. | No | 0 | **P1** | new `export_provenance_table.py` |
| **11** | **No literature/novelty audit.** | "First/novel" without a survey is a reject risk. | `docs/LITERATURE_NOVELTY_AUDIT.md` **missing** | Create it with the requested columns before any novelty wording enters the manuscript. | No | 0 | **P1** | new `docs/LITERATURE_NOVELTY_AUDIT.md` |
| **13** | **No CLAIM / TRIPOD+AI checklists.** | Increasingly expected for clinical-AI submissions. | Both **missing** | Create both with status/section/gap per item. | No | 0 | **P1** | new `paper/CLAIM_CHECKLIST.md`, `paper/TRIPOD_AI_CHECKLIST.md` |
| **15** | **Ethics wording too strong.** Manuscript says approval "was not required". | An assertion about institutional requirements the authors have not confirmed. | `main.tex` Data Availability | Soften to secondary-analysis wording; per-dataset licence table; flag for author confirmation. | No | 0 | **P0 (free)** | `main.tex` |
| **12** | **Story hierarchy.** Five results presented as near-equals. | Reads as five papers in one. | `main.tex` | Subordinate DG/resolution/calibration/provenance to the protocol question. Retitle. | No | 0 | **P1** | `main.tex` |
| **14b** | **No compile gate.** | Undefined refs and missing figures currently ship silently. | `check_numbers.py` exists; no LaTeX build | Add a build script failing on undefined refs, missing citations/figures, LaTeX errors. | No | 0 | **P0 (free)** | new `paper/build.sh` |

---

## 2. Two corrections to the incoming review

**Item 1 is already done.** Partial FT is at five seeds on all three targets and
frozen probing at ten. Spending the proposed compute here would buy nothing.

**Item 8 understates the extrapolation and proposes an impossible fix.** The
largest measured subsample is 8,879, not 11,841 — the latter is the full pool
and was never a sweep point. The true extrapolation to the 24,574-image
in-domain budget is **2.77×**. More sweep points cannot close it: the
leave-one-domain-out source pool for EyePACS *is* 11,841 images, so 24,574 is
unreachable without changing the protocol. Option (A) in the review is not
available; option (B), reframing, is the only honest route.

---

## 3. Proposed minimal queue

Everything in §1 marked P0-free is writing and analysis, not compute, and can
proceed immediately without touching the GPU.

Only two P0 items need GPU. Ranked by threat-to-headline per GPU-hour:

### Q1 — Resolution/batch control · ~4 h · **highest value**

```
python run_lodo.py --targets eyepacs --seeds 42,1,2,3,4 --batch-size 16
python run_lodo.py --targets idrid   --seeds 42,1,2,3,4 --batch-size 16
```

Gives the missing 224 px / batch 16 cell on the two strongest resolution
targets. Cheapest possible removal of a confound that currently invalidates a
headline sentence.

**No prerequisite.** Batch size already enters the experiment id
(`erm-b32` vs `erm-b16`, verified), so these runs cannot collide with or
overwrite the existing 224 px results. Ready to launch as written.

### Q2 — Full fine-tuning on one target · ~19 h

```
python run_lodo.py --backbone vit_large_mae_in1k --targets ddr --seeds 42,1,2 \
    --trainable-blocks 24 --batch-size 16 --learning-rate 1e-4 --grad-checkpointing
python run_lodo.py --backbone retfound_cfp       --targets ddr --seeds 42,1,2 \
    --trainable-blocks 24 --batch-size 16 --learning-rate 1e-4 --grad-checkpointing
```

Measured at 5.67 GB, 0.446 s/step, so it fits. Answers "your fine-tuning wasn't
fine-tuning" on the target with the largest test set.

**Prerequisite, confirmed:** neither `run_lodo.py` nor `src/models/backbones.py`
contains any gradient-checkpointing support (grep: 0 hits in both). The flag
above does not exist. `trainable_blocks=24` is also untested end-to-end — the
5.67 GB figure came from a standalone probe, not from this code path. Both must
be implemented and VRAM re-verified before Q2 is launched, or it will OOM
several hours in.

### Q3 — deliberately NOT proposed

- Partial FT to 10 seeds (~40 h): the five-seed nulls are already reported as
  nulls; more seeds cannot make a null more null.
- Source-validation-tuned LR (Q, ~10 h): valuable (item 3) but P1. It
  strengthens a limitation that is currently *stated*, rather than fixing a
  claim that is currently *wrong*.
- Full FT on all three targets (~57 h): one target establishes whether the
  conclusion survives; three is confirmation at triple cost.

**Total proposed GPU: ~23 h**, both items with code prerequisites that must land
first.

---

## 4. Explicitly not doing

- No existing result is modified, re-run or overwritten.
- No experiment is removed from the registry; the 5 `DIVERGED` rows stay.
- No novelty wording enters the manuscript before item 11 exists.
- No protocol change without a registry-visible id change.

---

## 5. Revised status after the free P0 pass (2026-08-30)

| # | Item | Was | Now |
|---|---|---|---|
| 5 | Crossed seed×case bootstrap | P0 open | **DONE** — `src/evaluation/crossed_bootstrap.py`, `analyse_crossed.py`, 6 tests. Changed 3 of 6 verdicts. |
| 2a | Rename to partial fine-tuning | P0 open | **DONE** — `adaptation_mode()`; registry records `partial_finetune_4` |
| 2b | True `--full-finetune` mode | P0 open | **DONE (code)** — proven to leave 0 tensors frozen; `--trainable-blocks 24` leaves 6 |
| 4 | Registry configuration bug | not in original table | **DONE** — 90 NaN rows backfilled; 6 tests pin id↔record agreement |
| 8 | Size-vs-shift reframing | P0 open | **DONE (README)** — reframed as fitted sensitivity model, 2.77× extrapolation stated. Phase 9 + manuscript still to update |
| 10 | Source-only vs full-artifact audit | P0 open | **PARTIAL** — README now states which audit a clean clone can run. Bundle + fetch script still to build |
| 15 | Ethics wording | P0 open | **DONE** — no longer asserts approval was not required |
| 14b | LaTeX build gate | P0 open | **DONE** — `paper/build.sh`; fails on orphaned commands, missing files, numeric drift. LaTeX checks report SKIPPED (pdflatex absent), not passed |
| 11 | Literature novelty audit | P1 | **PROMOTED TO P0, DONE** — `docs/LITERATURE_NOVELTY_AUDIT.md` |
| 8b | RETFound provenance | not in original table | **DONE** — published corpus is 90.2% MEH-MIDAS + 9.8% EyePACS; `"unknown"` → `"not_in_published_corpus"` |

### Still open

| # | Item | Priority | Needs |
|---|---|---|---|
| 4 | Resolution/batch confound | **P0** | Q1, ~4 h GPU |
| 2c | Full FT evidence | **P0** | smoke test, then 5 paired seeds |
| 8c | Phase 9 + manuscript reframing | P0 | writing |
| 10b | Prediction bundle + fetch script | P0 | packaging |
| 6 | Calibration section | P1 | analysis of existing runs |
| 7 | Referable-DR outcomes | P1 | analysis |
| 9 | Provenance table generator | P1 | writing |
| 13 | CLAIM / TRIPOD+AI checklists | P1 | writing |
| 3 | Source-validation-tuned recipe | P1 | ~10 h GPU |

---

## 6. Status

**STOPPED for approval before any GPU run**, as instructed. The free P0 work
(items 5, 8, 10, 15, 14b) can begin immediately on request — it is the larger
share of the risk and costs nothing.
