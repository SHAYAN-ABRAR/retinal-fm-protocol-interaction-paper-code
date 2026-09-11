# APTOS full fine-tuning — confirmatory replication, final report

**Analysis run 2026-09-11, after all ten runs completed and before any number
below was seen.** The analysis script, both LaTeX generators and the report gate
were written and rehearsed on 2026-09-07, while seven of the ten runs did not
exist. Every figure here is read from a generator CSV in `outputs/tables/`; none
is transcribed from console output.

Protocol: `APTOS_FULL_FINETUNE_REPLICATION_PROTOCOL.md` (pre-registered before
the first run). Execution history: `APTOS_EXECUTION_LOG.md`.

## The headline

The protocol interaction found on DDR **replicates on APTOS**: same direction,
similar magnitude, all five seeds agreeing, and it survives Holm correction
across the two-domain family.

**Linear probing overstates the ImageNet-MAE initialisation's advantage relative
to matched full fine-tuning — on both held-out domains.**

## Primary family: the full-versus-frozen interaction

I_full(s) = [QWK_ImageNet − QWK_RETFound]_full − [QWK_ImageNet − QWK_RETFound]_frozen

Holm-corrected across exactly these two members. Source:
`two_domain_interaction_holm.csv`.

| held out | n test | mean I_full | crossed 95% CI | sign | *p* | *p* Holm | *p* flip |
|---|---|---|---|---|---|---|---|
| DDR | 12,424 | −0.1141 | [−0.1548, −0.0728] | 5/5 | 0.0066 | 0.0132 | 0.0625 |
| APTOS | 3,504 | −0.0992 | [−0.1595, −0.0397] | 5/5 | 0.0370 | 0.0370 | 0.0625 |

**Both established** under the two-bar rule: the seed-level test survives Holm
*and* the crossed interval excludes zero, on each domain separately.

Per-seed interactions, all ten negative:

| seed | DDR | APTOS |
|---|---|---|
| 42 | −0.0688 | −0.2064 |
| 1 | −0.1696 | −0.0475 |
| 2 | −0.1201 | −0.0240 |
| 3 | −0.0590 | −0.1286 |
| 4 | −0.1529 | −0.0895 |

The DDR column was recomputed here from its own saved predictions and asserted
equal to the frozen `full_finetune_primary.csv` before being used — worst drift
**2.8e-17**. The replicated result is the one that was frozen, not a drifted
recomputation.

**No pooled two-domain *p*-value.** The test sets differ in size (12,424 vs
3,504) and in shift structure; a pooled value would describe neither. The two
estimates with their intervals are the presentation.

The sign-flip column sits at its floor of 0.0625 on both domains — the smallest
attainable two-sided value at five seeds (2/2⁵). It cannot reach 0.05 at this
sample size for any effect size, so it is reported as sensitivity, never as the
test.

## Primary model comparison under full fine-tuning

ImageNet-MAE minus RETFound, QWK, five paired seeds. Source:
`aptos_full_finetune_per_seed.csv`, `aptos_full_finetune_primary.csv`.

| seed | ImageNet-MAE | RETFound | Δ |
|---|---|---|---|
| 42 | 0.8075 | 0.8221 | −0.0146 |
| 1 | 0.8111 | 0.7931 | +0.0180 |
| 2 | 0.8400 | 0.8085 | +0.0315 |
| 3 | 0.7933 | 0.7930 | +0.0003 |
| 4 | 0.8098 | 0.8035 | +0.0063 |

| | |
|---|---|
| mean paired Δ | **+0.0083** |
| SD | 0.0175 |
| crossed 95% CI | [−0.0095, +0.0258] |
| paired *t*-test *p* | 0.3489 |
| sign-flip *p* | 0.3125 |
| sign agreement | 4/5 |

**No difference demonstrated between the two initialisations under matched full
fine-tuning on APTOS.** This is **not** a demonstration of equivalence: at five
seeds the interval above is how tightly the effect is bounded, and it admits
differences up to roughly ±0.026 QWK.

This mirrors DDR, where the same comparison also failed to demonstrate a
difference (mean −0.0439, CI [−0.0861, −0.0103], *p* = 0.1035, 4/5).

## Adaptation depth

Descriptive context for the interaction, not a second family of tests.
Uncorrected. Source: `aptos_adaptation_depth.csv`.

| protocol | ImageNet-MAE | RETFound | Δ | crossed 95% CI | sign | *p* |
|---|---|---|---|---|---|---|
| frozen linear probe | 0.5872 | 0.4796 | +0.1075 | [+0.0623, +0.1536] | 5/5 | 0.0121 |
| partial (last 4 blocks) | 0.8334 | 0.8261 | +0.0073 | [−0.0107, +0.0234] | 4/5 | 0.4007 |
| full fine-tuning | 0.8123 | 0.8040 | +0.0083 | [−0.0095, +0.0258] | 4/5 | 0.3489 |

This is the interaction in plain view. Under a frozen encoder ImageNet-MAE leads
by **+0.1075** QWK; once the backbone can adapt — partially or fully — the gap
collapses to **+0.0073** and **+0.0083**, neither distinguishable from zero. The
ranking a linear probe produces does not survive adaptation.

Note also that both initialisations gain enormously from adaptation on APTOS
(RETFound 0.4796 → 0.8040), so this is not a ceiling effect masking a
difference: the protocols differ in what they rank, not merely in headroom.

## Secondary outcomes under full fine-tuning

Uncorrected and exploratory; QWK is the pre-specified primary endpoint and none
of these replaces it. Source: `aptos_secondary_outcomes.csv`.

| metric | ImageNet-MAE | RETFound | Δ | SD | *p* | sign |
|---|---|---|---|---|---|---|
| QWK (primary) | 0.8123 | 0.8040 | +0.0083 | 0.0175 | 0.3489 | 4/5 |
| macro F1 | 0.4858 | 0.4713 | +0.0144 | 0.0232 | 0.2381 | 4/5 |
| balanced accuracy | 0.5057 | 0.5033 | +0.0024 | 0.0186 | 0.7873 | 3/5 |
| severe-error rate | 0.0973 | 0.1103 | −0.0130 | 0.0146 | 0.1178 | 4/5 |
| MAE grade | 0.3941 | 0.4264 | −0.0324 | 0.0445 | 0.1789 | 4/5 |
| within-1-grade | 0.9027 | 0.8897 | +0.0130 | 0.0146 | 0.1178 | 4/5 |
| macro AUROC | 0.8049 | 0.7808 | +0.0241 | 0.0179 | 0.0394 | 4/5 |
| ECE | 0.0628 | 0.0754 | −0.0126 | 0.0080 | 0.0246 | 5/5 |
| NLL | 0.9838 | 1.0163 | −0.0325 | 0.0937 | 0.4817 | 4/5 |

Two entries have *p* < 0.05 — macro AUROC and ECE, both favouring ImageNet-MAE.
**Neither is claimed.** They are uncorrected members of a nine-metric
exploratory set, where roughly one such value is expected by chance alone, and
the pre-specified endpoint is QWK, which shows no difference. Promoting either
would be exactly the substitution the protocol forbids.

## Source-validation behaviour

The held-out target plays no part in this table. Source:
`aptos_full_finetune_source_validation.csv`.

| | ImageNet-MAE | RETFound |
|---|---|---|
| best epoch | 18.0 ± 1.0 | **9.2 ± 2.6** |
| early stopped | **0/5** | **5/5** |
| overfit ratio (final ÷ min val loss) | 1.14 ± 0.02 | **2.50 ± 0.32** |
| total GPU runtime | 28.9 h | 23.3 h |

The DDR asymmetry reproduces exactly. RETFound converges at roughly half the
epoch, early-stops in every run while ImageNet-MAE early-stops in none, and
overfits the source pool more than twice as hard. Across both domains that is
**10/10 RETFound runs early-stopping and 0/10 ImageNet-MAE runs**.

Checkpoint selection is on best source-validation QWK, so this does not
compromise the reported models. It does mean one fixed budget is not equally
well matched to the two initialisations, which belongs in the limitations
rather than being discovered by a reader. **No budget, patience or schedule was
changed for any run**, which is what makes this a control rather than a
comparison of tuning effort.

## What this does and does not establish

**Establishes.** On two independent held-out domains, the relative standing of
the two initialisations changes significantly between frozen evaluation and
matched full fine-tuning, in the same direction, with every one of ten seed-level
interactions negative. Frozen linear-probe performance is not a reliable proxy
for fine-tuned cross-domain performance for this pair of checkpoints.

**Does not establish.** That the two initialisations are equivalent under full
fine-tuning. Both domains fail to demonstrate a difference there, and a null at
five seeds is a weak bound, not an equivalence claim. Nor does it establish
anything about other foundation models, other target domains, other adaptation
budgets, or RETFound in the in-domain settings it was designed for.

**Scope.** Two initialisations, one architecture (ViT-L/16), one adaptation
budget, two held-out domains, five seeds each. EyePACS is a source domain in
both leave-one-domain-out pools and was also part of RETFound's unsupervised
pretraining corpus — documented in the protocol. That is not target leakage (no
APTOS or DDR image was seen in pretraining or training), and it applies
identically to the frozen and full-FT arms, so it cannot generate the
interaction.

## Provenance

| | |
|---|---|
| runs | 10 APTOS full-FT (+ 10 frozen, 10 partial already banked) |
| seeds | 42, 1, 2, 3, 4 |
| trainable parameters | 303,306,757 (asserted at runtime in all 10) |
| APTOS test set | 3,504 images, identical across all 30 runs |
| DDR test set | 12,424 images, identical across all 30 runs |
| bootstrap | 2,000 crossed seed × case replicates, rng seed 7 |
| tests | 414 passed |
| audit_consistency | PASS, 3,088 checks |
| audit_resumed_runs | exit 0, no reported model affected |
| identity audit | 9 pre-existing conflicts, **none involving any full-FT run** |
| LaTeX gate | clean |

## Stop condition

Per the pre-registered protocol, the experimental programme **stops here**. No
IDRiD full fine-tuning, no additional seeds, no further foundation model, no
further DG method, no hyperparameter tuning.
