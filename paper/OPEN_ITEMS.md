# Manuscript handoff checklist

**Replaced 2026-09-11.** The previous contents of this file described
experiments as "running now" that completed weeks ago (IDRiD fine-tune seeds 3
and 4; frozen probes seeds 5–9). It was stale enough to mislead anyone
inferring project status from it, and has been discarded rather than patched.

> **Do not infer current status from any other document in `paper/`.**
> `paper/main.tex` is historical scaffolding with stale framing and is **not**
> being patched section by section. The authoritative state is
> `docs/JBHI_EVIDENCE_FREEZE.md`.

## Experiments — ALL COMPLETE, programme CLOSED

| item | status |
|---|---|
| frozen linear probe, 3 domains × 2 initialisations × 10 seeds | **complete** (60 runs) |
| partial FT (last 4/24), 3 domains × 2 × 5 seeds | **complete** (30 runs) |
| full FT, DDR, 2 × 5 seeds | **complete** (10 runs) |
| full FT, APTOS confirmatory replication, 2 × 5 seeds | **complete** (10 runs) |
| DG method comparison | complete (supplement) |
| resolution / batch decomposition | complete (supplement) |
| calibration, selective prediction | complete (supplement) |
| **IDRiD full FT** | **will not be run — programme closed** |
| additional seeds / models / DG methods / hyperparameter search | **will not be run — programme closed** |

110 authoritative runs, 110/110 COMPLETE, 134.3 h training. No further GPU
work is planned or recommended.

## Artifacts — COMPLETE

| item | status |
|---|---|
| authoritative master table | `outputs/tables/JBHI_MASTER_RESULTS.csv` / `.tex` |
| primary interaction table | `outputs/tables/JBHI_PRIMARY_INTERACTION.csv` / `.tex` |
| claim–evidence map | `docs/FINAL_CLAIM_EVIDENCE_MAP.md`, `outputs/tables/final_claim_evidence.csv` |
| four main figures (PNG 300 dpi + PDF) | `outputs/figures/jbhi_final/` |
| evidence freeze with hashes | `docs/JBHI_EVIDENCE_FREEZE.md` |
| dataset / contamination table | `docs/JBHI_DATASET_PROVENANCE.md` |
| novelty audit (corrected) | `docs/LITERATURE_NOVELTY_AUDIT.md` |
| main-vs-supplement placement | `docs/JBHI_CONTENT_PLACEMENT.md` |

## Remaining — AUTHOR DECISIONS ONLY

None of these can be resolved from the code or the data. Every one needs a
human.

- [ ] **Author list and order**
- [ ] **Affiliations** for each author
- [ ] **Corresponding author** — name, email, postal address
- [ ] **Funding statement** — grant numbers, or an explicit "no funding" line
- [ ] **Institutional ethics determination** — the exact wording for a study
      using only previously published, de-identified public datasets (DDR,
      APTOS 2019, IDRiD, EyePACS). Typically an IRB-exemption or
      not-human-subjects-research statement; the institution must supply the
      form of words.
- [ ] **Conflict-of-interest statement** for every author
- [ ] **Code availability** — whether this repository is released, under what
      licence, and at what URL/DOI (Zenodo archive recommended for a commit
      pin)
- [ ] **Data availability** — pointers to the four public datasets plus their
      individual access conditions; confirm redistribution is not implied
- [ ] **Author contributions** (CRediT taxonomy)
- [ ] **Final JBHI formatting** — IEEE template version, page and figure
      limits, reference style, whether the supplement is a separate PDF
- [ ] **Preprint policy** — whether to post, and where

## Not open items

These are decided and recorded; do not reopen them during drafting:

- statistical framework and its vocabulary → `JBHI_EVIDENCE_FREEZE.md` §7
- which claims may be made and in what words → `FINAL_CLAIM_EVIDENCE_MAP.md`
- the novelty positioning, and the retracted "first comparison" claim →
  `LITERATURE_NOVELTY_AUDIT.md` §0
- the study chronology and how the two-domain family must be described →
  `JBHI_EVIDENCE_FREEZE.md` §10
- what goes in the supplement → `JBHI_CONTENT_PLACEMENT.md`
