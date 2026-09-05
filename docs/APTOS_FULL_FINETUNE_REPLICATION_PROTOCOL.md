# APTOS full fine-tuning — confirmatory replication protocol

**Pre-registered 2026-09-06, committed before the first APTOS full-FT run.**

This is a **confirmatory external-domain replication**, performed *after* the
DDR full fine-tuning result was frozen and *before* any APTOS full-FT target
result is inspected. It is not exploratory and it is not a tuning exercise.

The DDR result being replicated (frozen, `docs/FULL_FINETUNE_FINAL_REPORT.md`):

| | |
|---|---|
| mean interaction I_full | −0.1141 |
| crossed 95% CI | [−0.1548, −0.0728] |
| paired seed-level *t*-test *p* | 0.0066 |
| sign agreement | 5/5 |

## The recipe is not retuned for APTOS

Every setting below is copied from the frozen DDR experiment. **No APTOS-specific
tuning is permitted, before or after inspecting source-validation behaviour.**
Observing that one arm converges earlier on APTOS, or that the budget fits one
model better, is *not* grounds to change anything — the fixed budget is what
makes this a control rather than a comparison of tuning effort.

| | |
|---|---|
| models | `vit_large_mae_in1k`, `retfound_cfp` |
| target (held out) | **APTOS** |
| source pool | DDR + IDRiD + EyePACS |
| seeds | **42, 1, 2, 3, 4** |
| trainable | full encoder + head |
| intended trainable parameters | **303,306,757** (asserted at runtime) |
| image size | 224 px |
| physical batch | 16 |
| accumulation steps | 1 |
| effective batch | **16** |
| gradient checkpointing | on |
| optimiser | AdamW |
| learning rate | 1e-4 |
| weight decay | 1e-4 |
| warmup | 1 epoch |
| schedule | cosine |
| maximum epochs | 20 |
| checkpoint selection | source-validation QWK |
| early stopping | source-validation QWK, patience 6 |

**APTOS labels are never used** for model selection, early stopping,
hyperparameter modification, or calibration. Temperature is fitted on source
validation only.

## Primary hypothesis

The primary endpoint is **not** simply ImageNet-MAE versus RETFound under full
fine-tuning. It is the same protocol interaction tested on DDR:

```
D_frozen(s) = QWK_ImageNet_frozen(s)  - QWK_RETFound_frozen(s)
D_full(s)   = QWK_ImageNet_fullFT(s)  - QWK_RETFound_fullFT(s)
I_full(s)   = D_full(s) - D_frozen(s)
```

over the five matched seeds 42, 1, 2, 3, 4.

**Hypothesis:** the relative ImageNet-MAE versus RETFound advantage changes when
moving from frozen representation evaluation to matched full fine-tuning.

**Two-sided.** The direction is *not* specified from the DDR result. A one-sided
test justified by an earlier result on a different domain would convert a
replication into a confirmation-seeking exercise.

**QWK is the primary metric** and is not replaced if another metric reads
better.

## Final multiplicity family

For the manuscript, the full-vs-frozen interaction tests on **DDR** and
**APTOS** form a **two-domain primary family**. After APTOS completes, report
per domain:

- raw paired seed-level *t*-test *p*;
- Holm-adjusted *p* across the two domains;
- crossed seed × case 95% CI, separately per domain;
- exact sign-flip sensitivity;
- the five individual seed interactions.

The crossed bootstrap supplies **uncertainty only**. Its tail mass is not a
calibrated *p*-value — the distribution is built around the empirical estimate
rather than under H0 — and must not be used to manufacture one.

**No pooled two-domain *p*-value.** Pooling two domains with different test-set
sizes and different shift structures would report a number that describes
neither. A forest plot showing the two domain-specific estimates with their
intervals is the honest presentation.

**No equivalence claims from a null.** A non-significant interaction on APTOS
bounds the effect loosely; it does not demonstrate that the protocols agree.

## Target provenance

RETFound's published colour-fundus pretraining corpus is enumerated in the
source publication as **815,468 MEH-MIDAS + 88,702 EyePACS = 904,170**
photographs (Zhou et al., Nature 2023).

**APTOS is not in that corpus**, so APTOS is not a published pretraining-overlap
target and is admissible as a held-out domain. The claim rests on the
publication, not on the checkpoint, which carries no manifest of what it saw.

**Stated transparently:** EyePACS is one of the downstream **source** domains in
the APTOS leave-one-domain-out training pool, and EyePACS was also part of
RETFound's unsupervised pretraining corpus. This is **not target leakage** — the
held-out domain is APTOS, and no APTOS image was seen in pretraining or
training. But it does mean RETFound has previously seen, without labels, images
from one of the three source domains. That is part of the retinal-pretraining
intervention being measured rather than a confound to be removed, and it applies
identically to the frozen and full-FT arms, so it cannot generate the
interaction. It is documented because a reader should not have to discover it.

The same is true of the frozen DDR experiment already reported, where EyePACS
was likewise a source domain.

## Execution

One seed and one model per process, sequentially, never concurrently. After
each: wait for actual process exit, verify COMPLETE, audit, prune `last.pt`
only once every safety guard passes, retain `best_qwk.pt`.

```
42 ImageNet -> 42 RETFound -> 1 ImageNet -> 1 RETFound -> 2 ImageNet
-> 2 RETFound -> 3 ImageNet -> 3 RETFound -> 4 ImageNet -> 4 RETFound
```

**Target blindness.** Until all ten runs complete, no interim APTOS QWK, no
statement of which model leads, no interim mean difference, interaction,
*p*-value or favourable-sign count is reported. `run_lodo.py` computes target
metrics internally; they do not influence execution. Technical failures may be
investigated freely. No scientific protocol change is permitted for any reason.

## Whatever it shows, it is kept

If APTOS confirms DDR, that is reported as replication. If it disagrees, that is
reported as domain heterogeneity. If it reverses, the reversal is reported.
**No APTOS run is hidden, excluded or re-run because its result is
inconvenient.** A run is re-run only for a demonstrated implementation fault,
and any such re-run is recorded with its reason.

## Stop condition

After the APTOS final analysis the experimental programme stops. No IDRiD full
fine-tuning, no additional seeds, no further foundation model, no further DG
method, no hyperparameter tuning.

## Preconditions, verified before launch

| | |
|---|---|
| free disk | **51 GB** (target ≥ 45 GB) |
| corrupted-run replacements | complete; `audit_resumed_runs.py` exits 0 |
| tests | 414 passed |
| audit_consistency | PASS, 2978 checks |
| identity audit | 9 pre-existing conflicts, none involving full FT |
| check_numbers / check_report_numbers / LaTeX gate | clean |
