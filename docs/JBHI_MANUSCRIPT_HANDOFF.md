# JBHI manuscript handoff

**For whoever drafts the final IEEE JBHI manuscript.** The experimental
programme is closed. Everything below is frozen evidence; nothing here requires
or recommends further computation.

> **NO FURTHER EXPERIMENTS ARE RECOMMENDED BEFORE MANUSCRIPT DRAFTING.**

---

## 1. Freeze commit

| | |
|---|---|
| evidence commit | `15a50dce4d9aa3875fd9339871a2dc60ba73977b` — all 110 runs and the pre-registered analysis |
| handoff-package commit | `4e220b2411ad…` — tables, figures, docs; **no experimental result** |
| working tree | clean |
| registry SHA256 | `63e81d615084812a231afc599dfbb4f1dc41b756cd8a8f7ce3f20799da3a6192` |
| prediction-set digest | `913aecf7487e88422bc3d513bc55a93bca8d74a4b4ff961019823dd76427d487` |
| manifest | `outputs/tables/JBHI_EVIDENCE_MANIFEST.csv` — 126 artifacts hashed |
| regenerate | `python export_evidence_freeze.py` |

The prediction-set digest is identical before and after the handoff package was
built, which is the check that the package added documentation and not results.

Full detail: **`docs/JBHI_EVIDENCE_FREEZE.md`**.

---

## 2. Run accounting

| held out | frozen probe | partial FT | full FT | total |
|---|---|---|---|---|
| DDR | 20 | 10 | 10 | 40 |
| APTOS | 20 | 10 | 10 | 40 |
| IDRiD | 20 | 10 | **0 — not run** | 30 |
| **total** | **60** | **30** | **20** | **110** |

**110/110 COMPLETE. 134.3 h of training.** Seeds: 10 for the frozen probe
(42,1,2,3,4,5,6,7,8,9), 5 for both adaptation protocols (42,1,2,3,4).

IDRiD has no full fine-tuning. No value is imputed for it anywhere, and the
interaction is therefore a **two-domain** result.

Per-run detail: `outputs/tables/JBHI_RUN_ACCOUNTING.csv`. Execution history
including abandoned executions: `docs/APTOS_EXECUTION_LOG.md`.

---

## 3. Primary finding

**Moving from frozen linear probing to matched full fine-tuning changes the
relative QWK difference between ImageNet-MAE and RETFound — statistically
supported on two independent held-out domains, in the same direction, with all
ten seed-level interactions negative.**

The mechanism in plain view (APTOS): under a frozen encoder ImageNet-MAE leads
by **+0.1075** QWK; once the encoder can adapt the apparent advantage collapses
to **+0.0073** (partial) and **+0.0083** (full), neither distinguishable from
zero. Both arms gain enormously from adaptation (RETFound 0.4796 → 0.8040), so
this is not a ceiling effect.

**No difference is demonstrated between the two initialisations under matched
full fine-tuning on either domain. That is a null, not an equivalence claim.**

---

## 4. Primary interaction table

`outputs/tables/JBHI_PRIMARY_INTERACTION.csv` / `.tex`

I_full(s) = Δ_full(s) − Δ_frozen(s), where Δ = QWK(ImageNet-MAE) − QWK(RETFound)

| held out | n test | seeds | Δ frozen | Δ full | **I_full** | crossed 95% CI | *p* | **Holm *p*** | *p* flip | sign |
|---|---|---|---|---|---|---|---|---|---|---|
| DDR | 12,424 | 5 | +0.0702 | −0.0439 | **−0.1141** | [−0.1548, −0.0728] | 0.0066 | **0.0132** | 0.0625 | 5/5 |
| APTOS | 3,504 | 5 | +0.1075 | +0.0083 | **−0.0992** | [−0.1595, −0.0397] | 0.0370 | **0.0370** | 0.0625 | 5/5 |

Per-seed interactions (all negative): DDR −0.0688, −0.1696, −0.1201, −0.0590,
−0.1529; APTOS −0.2064, −0.0475, −0.0240, −0.1286, −0.0895.

**No pooled two-domain *p*-value** — the test sets differ in size and shift
structure. The sign-flip column sits at its floor (2/2⁵ = 0.0625) and is
sensitivity only.

---

## 5. Master protocol table

`outputs/tables/JBHI_MASTER_RESULTS.csv` / `.tex`

| held out | protocol | seed set | n | ImageNet-MAE | RETFound | Δ | crossed 95% CI | *p* | sign |
|---|---|---|---|---|---|---|---|---|---|
| DDR | frozen | all-10 | 10 | 0.5756 | 0.5130 | +0.0626 | [+0.0440, +0.0824] | 0.0000 | 10/10 |
| DDR | frozen | common-5 | 5 | 0.5805 | 0.5103 | +0.0702 | [+0.0468, +0.0978] | 0.0057 | 5/5 |
| DDR | partial | common-5 | 5 | 0.7183 | 0.6966 | +0.0217 | [−0.0004, +0.0438] | 0.1401 | 3/5 |
| DDR | full | common-5 | 5 | 0.6209 | 0.6648 | −0.0439 | [−0.0861, −0.0103] | 0.1035 | 4/5 |
| APTOS | frozen | all-10 | 10 | 0.5963 | 0.4829 | +0.1134 | [+0.0787, +0.1493] | 0.0001 | 10/10 |
| APTOS | frozen | common-5 | 5 | 0.5872 | 0.4796 | +0.1075 | [+0.0623, +0.1536] | 0.0121 | 5/5 |
| APTOS | partial | common-5 | 5 | 0.8334 | 0.8261 | +0.0073 | [−0.0107, +0.0234] | 0.4007 | 4/5 |
| APTOS | full | common-5 | 5 | 0.8123 | 0.8040 | +0.0083 | [−0.0095, +0.0258] | 0.3489 | 4/5 |
| IDRiD | frozen | all-10 | 10 | 0.6363 | 0.6792 | **−0.0429** | [−0.0888, +0.0037] | 0.0034 | 9/10 |
| IDRiD | frozen | common-5 | 5 | 0.6275 | 0.6701 | −0.0426 | [−0.0959, +0.0131] | 0.0497 | 5/5 |
| IDRiD | partial | common-5 | 5 | 0.7254 | 0.7676 | −0.0421 | [−0.0879, +0.0044] | 0.0653 | 5/5 |

**Two things to notice.**

1. **Frozen rows appear twice.** Ten-seed estimates are descriptive; only the
   `common-5` rows may be differenced against an adaptation protocol, because
   the interaction is paired within seed. Never mix them.
2. **IDRiD reverses the frozen direction** — RETFound leads there. The
   frozen-probe advantage is *not* a universal ImageNet-MAE advantage, and no
   sentence should imply otherwise.

---

## 6. Novelty matrix

**A claim has been retracted.** "First comparison of RETFound against the
ImageNet-MAE checkpoint from which it originated" is **wrong** and must not
appear. RETFound (Nature 2023) already compares against SSL-ImageNet — its own
starting point — with the same architecture and matched fine-tuning, including
cross-dataset DR evaluation.

**What may be claimed:**

> A within-lineage evaluation of how downstream adaptation protocol changes the
> estimated value of RETFound's additional retinal-domain MAE continuation
> pretraining relative to its SSL-ImageNet starting point, with a direct
> protocol-by-initialization interaction test replicated across independent
> held-out DR domains.

| | pairing | protocols | external DR | interaction? |
|---|---|---|---|---|
| RETFound 2023 | **RETFound vs SSL-ImageNet** | fine-tuning only | yes | **no** |
| RETFound-Green 2025 | different models | probe **and** full FT | yes | no |
| Pre-training data effects 2026 | different pretraining cohorts | probe **and** full FT, 5 seeds | yes | no |
| Frozen calibration benchmark 2026 | different families | **frozen only** | yes | no |
| Label efficiency 2026 | different architectures | full FT | yes | no |
| **This work** | **one lineage step** | **frozen / partial / full** | **yes, LODO** | **yes** |

A fresh search for the exact conjunction found nothing, so use **"to our
knowledge"**, never "first". Detail: `docs/LITERATURE_NOVELTY_AUDIT.md`.

---

## 7. Literature additions required

| ref | venue | identifier | why it must appear |
|---|---|---|---|
| RETFound | *Nature* 2023 | `10.1038/s41586-023-06555-x` | closest prior art; establishes the lineage pair and the SSL-ImageNet baseline |
| RETFound-Green | *Nat. Commun.* 2025 | `10.1038/s41467-025-62123-z` | probe and full FT both appear, but model identity varies with protocol |
| Pre-training data effects | *Nat. Commun.* 17:3309 (2026) | `10.1038/s41467-026-70077-z`, PMID 41764179 | nearest methodological neighbour: both protocols, five seeds, external sets |
| Frozen-transfer calibration benchmark | *Front. Med.* 2026 | `10.3389/fmed.2026.1815982`, PMID 42078464 | **already reports RETFound below an ImageNet baseline under frozen external transfer** (0.697 vs 0.745) and argues MAE frozen features need nonlinear adaptation |
| Label efficiency | *Lancet Digit. Health* 2026, online 28 Aug 2026 | PMID 42665469 (preprint arXiv:2501.12016) | **ImageNet-pretrained models comparable to RETFound after full FT on ocular tasks** — directly relevant to our full-FT null |

Plus He et al. (MAE) for the `vit_large_patch16_224.mae` lineage, and the four
dataset sources.

**BibTeX is not pre-written.** Fields are recorded above for a human to verify
against publisher records rather than fabricated into `refs.bib`.

---

## 8. Figure inventory

`outputs/figures/jbhi_final/` — PNG at 300 dpi and vector PDF for each.

| figure | file | content |
|---|---|---|
| 1 | `fig1_study_design` | the lineage intervention: ImageNet-MAE → retinal MAE continuation → RETFound, then three protocols, then LODO evaluation. Makes explicit that this is one lineage step, not two unrelated architectures |
| 2 | `fig2_adaptation_depth` | paired per-seed Δ at frozen / partial / full for DDR and APTOS, with per-seed trajectories and a zero line. The ~0.10 attenuation is visible without implying equivalence |
| 3 | `fig3_primary_interaction` | forest plot of the two interactions with crossed intervals, labelled so that negative = the ImageNet-MAE advantage is reduced after adaptation |
| 4 | `fig4_source_validation` | optimisation behaviour: best epoch and loss inflation, 10/10 vs 0/10 early stopping. **Descriptive only — no causal annotation.** First to cut if space is short |

---

## 9. Table inventory

| table | file | role |
|---|---|---|
| **primary** | `JBHI_PRIMARY_INTERACTION.csv` / `.tex` | the two-domain interaction — likely a central table |
| **master** | `JBHI_MASTER_RESULTS.csv` / `.tex` | every protocol × domain, with seed sets labelled |
| claim map | `final_claim_evidence.csv` | 12 claims with allowed/forbidden wording |
| manifest | `JBHI_EVIDENCE_MANIFEST.csv` | 126 SHA256 hashes |
| accounting | `JBHI_RUN_ACCOUNTING.csv` | 110 runs |
| environment | `JBHI_ENVIRONMENT.csv` | versions, GPU, checkpoint hashes |
| per-domain | `full_finetune_*.csv`, `aptos_*.csv` | the frozen per-domain analyses the reports rest on |
| source validation | `*_source_validation.csv` | optimisation behaviour |
| secondary `.tex` | `table_two_domain_interaction.tex`, `table_aptos_adaptation_depth.tex` | alternative renderings of the same numbers |

---

## 10. Remaining author-only questions

Nothing below can be resolved from code or data.

- Author list, order, affiliations, corresponding author
- Funding statement (or explicit none)
- **Institutional ethics determination wording** — four public, de-identified,
  previously published datasets; the institution must supply the exemption or
  not-human-subjects wording
- Conflict-of-interest statement
- Author contributions (CRediT)
- **Code availability** — release or not, licence, DOI (Zenodo pin recommended)
- **Data availability** — pointers and per-dataset access conditions; confirm
  redistribution is not implied
- Final JBHI formatting: template version, page/figure limits, supplement as
  separate PDF
- Preprint policy

Tracked in `paper/OPEN_ITEMS.md`, which was replaced because it described
completed experiments as "running now".

---

## 11. Test and audit results

| gate | result |
|---|---|
| test suite | **414 passed** |
| `audit_consistency.py` | **PASS — 3,090 checks**, no disagreement |
| `audit_resumed_runs.py` | **exit 0** — no reported model affected |
| `audit_experiment_identity.py` | 9 pre-existing conflicts, **none involving any full-FT run** |
| `paper/check_numbers.py` | no unsupported value in the manuscript |
| `paper/check_report_numbers.py` | **3 documents, 0 failing** — both final reports and the claim map trace fully to generator CSVs |
| `paper/check_latex.py` | clean across all included files |
| table regeneration | **5 authoritative tables byte-identical** on re-run |
| claim-map consistency | every four-decimal value traces to an authoritative table |

---

## 12. Writing constraints carried forward

Read `docs/FINAL_CLAIM_EVIDENCE_MAP.md` before drafting. The essentials:

**Vocabulary.** "Two-bar rule" is retired. Crossed bootstrap = *uncertainty
interval* (never a test). Paired seed-level *t*-test = *formal inferential
test*. Holm = *multiplicity adjustment*, across the two interaction tests only.
Sign-flip = *distribution-free sensitivity analysis*. Seed SD and sign count =
*descriptive stability diagnostics*. Qualifying effects are **"statistically
supported under the study's inferential framework"**.

**Attenuation, not reversal.** APTOS keeps the mean ordering ImageNet-MAE >
RETFound at every depth. Write "the magnitude of the apparent ImageNet-MAE
advantage under frozen probing is strongly attenuated when the encoder is
allowed to adapt". Do **not** write "the ranking reverses" or "does not
survive" as a two-domain conclusion. A mean reversal may be described for DDR
descriptively and must not be generalised.

**Chronology.** DDR was specified, frozen, run, and **observed** first. Only
then was APTOS specified as a confirmatory replication, with its recipe copied
unchanged and its analysis written before any APTOS target outcome was
inspected. Holm across the two is applied for conservative final reporting.
**Do not describe DDR+APTOS as a prospectively specified family.**

**Forbidden throughout.** "RETFound is superior under full fine-tuning"; "the
models are equivalent after fine-tuning"; "linear probing reverses the ranking
on every domain"; "this generalizes to all retinal foundation models"; any
claim from the two uncorrected secondary metrics; any implication of target
leakage from the EyePACS exposure; any imputed IDRiD full-FT value.

**Do not rewrite from `paper/main.tex`.** It is historical scaffolding with
stale framing. Draft from this package.

---

> **NO FURTHER EXPERIMENTS ARE RECOMMENDED BEFORE MANUSCRIPT DRAFTING.**
