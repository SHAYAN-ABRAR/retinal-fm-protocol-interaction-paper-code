# Documentation index

**The scientific programme is closed and the evidence is frozen.** This index
exists so that nobody reads a superseded phase report as the current
conclusion.

If you are drafting the manuscript, start with
[`JBHI_DRAFTING_MANIFEST.md`](JBHI_DRAFTING_MANIFEST.md).

---

## Authoritative final manuscript sources

These are current. Where any other document disagrees with one of these, these
win.

### Start here

| document | what it settles |
|---|---|
| [`JBHI_MANUSCRIPT_HANDOFF.md`](JBHI_MANUSCRIPT_HANDOFF.md) | the single entry point: freeze commit, accounting, primary finding, inventories, author-only items |
| [`JBHI_DRAFTING_MANIFEST.md`](JBHI_DRAFTING_MANIFEST.md) | which artifact each manuscript section must draw on, plus the claim, chronology and terminology guardrails |
| [`JBHI_EVIDENCE_FREEZE.md`](JBHI_EVIDENCE_FREEZE.md) | hashes, run accounting, datasets, checkpoints, hypotheses, seed sets, environment, **study chronology** |

### Claims, results and wording

| document | what it settles |
|---|---|
| [`FINAL_CLAIM_EVIDENCE_MAP.md`](FINAL_CLAIM_EVIDENCE_MAP.md) | every candidate claim with its evidence, **allowed wording and forbidden stronger wording** |
| [`FULL_FINETUNE_FINAL_REPORT.md`](FULL_FINETUNE_FINAL_REPORT.md) | the DDR full fine-tuning result, in full |
| [`APTOS_FULL_FINETUNE_FINAL_REPORT.md`](APTOS_FULL_FINETUNE_FINAL_REPORT.md) | the APTOS confirmatory replication, in full |
| [`LITERATURE_NOVELTY_AUDIT.md`](LITERATURE_NOVELTY_AUDIT.md) | the corrected novelty position, the comparison matrix, and the full search record |

### Data, figures and presentation

| document | what it settles |
|---|---|
| [`JBHI_DATASET_PROVENANCE.md`](JBHI_DATASET_PROVENANCE.md) | the manuscript dataset table, contamination disclosure, and the APTOS-in-EyePACS guard |
| [`JBHI_FIGURE_REVIEW.md`](JBHI_FIGURE_REVIEW.md) | per-figure purpose, claim supported, placement, redundancy, reviewer-misreading notes |
| [`JBHI_CONTENT_PLACEMENT.md`](JBHI_CONTENT_PLACEMENT.md) | main paper versus supplement |
| [`JBHI_PAGE_BUDGET.md`](JBHI_PAGE_BUDGET.md) | page allocation and what must not be cut to reach it |
| [`DATASET_CITATION_NOTES.md`](DATASET_CITATION_NOTES.md) | bibliography verification status and unresolved citation items |

### Protocols and execution history (current, and still authoritative)

| document | what it settles |
|---|---|
| [`FULL_FINETUNE_PROTOCOL.md`](FULL_FINETUNE_PROTOCOL.md) | the DDR full-FT protocol, pre-registered |
| [`APTOS_FULL_FINETUNE_REPLICATION_PROTOCOL.md`](APTOS_FULL_FINETUNE_REPLICATION_PROTOCOL.md) | the APTOS replication protocol, committed before the first APTOS run |
| [`APTOS_EXECUTION_LOG.md`](APTOS_EXECUTION_LOG.md) | every execution, every interruption, and the determinism caveat |
| [`WINDOWS_TORCH_POLICY_BLOCKER.md`](WINDOWS_TORCH_POLICY_BLOCKER.md) | the application-control incident, **RESOLVED** |
| [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) | the full Phase-1 data audit — still the authoritative provenance detail |
| [`EXPERIMENT_IDENTITY_AUDIT.md`](EXPERIMENT_IDENTITY_AUDIT.md) | generated; config/identity conflicts |

### Outside `docs/`

| path | what it settles |
|---|---|
| `paper/refs_jbhi_verified.bib` | the verified bibliography (28 entries) |
| `paper/jbhi_manuscript.tex` | the manuscript scaffold — draft here, **not** in `main.tex` |
| `paper/CLAIM_2024_CHECKLIST.md` | CLAIM 2024 compliance audit, 44 items |
| `paper/STARD_AI_APPLICABILITY.md` | STARD-AI applicability; why it is a secondary cross-check |
| `paper/OPEN_ITEMS.md` | author-only outstanding items |
| `release/README_REPRODUCTION.md` | the reproducibility bundle (unpublished) |
| `outputs/tables/JBHI_*.csv` | the authoritative result tables |

---

## Historical / superseded project records

**Retained deliberately. Not authoritative.** Each now carries a banner saying
so. They record what was believed at the time, and reading one as the current
conclusion is the specific mistake this index exists to prevent.

| document | superseded by | what changed |
|---|---|---|
| [`JBHI_GAP_ANALYSIS.md`](JBHI_GAP_ANALYSIS.md) | the freeze and the final reports | a planning document; its gaps are closed, and it still describes the retired "two-bar rule" as the inferential framework |
| [`PHASE13_FINETUNE.md`](PHASE13_FINETUNE.md) | `FULL_FINETUNE_FINAL_REPORT.md` | "fine-tuning" there means **partial** FT (last 4 of 24 blocks); full FT came later |
| [`PHASE12_FOUNDATION_MODEL.md`](PHASE12_FOUNDATION_MODEL.md) | `JBHI_MASTER_RESULTS.csv` | frozen-probe framing predating the interaction analysis |
| [`PHASE11_DDR_REPLICATION.md`](PHASE11_DDR_REPLICATION.md) | the final reports | DG-method replication, not the foundation-model result |
| [`PHASE10_DG_METHOD_COMPARISON.md`](PHASE10_DG_METHOD_COMPARISON.md) | supplement | DG benchmark; a null, and not the manuscript's subject |
| [`PHASE9_RESOLUTION_REPORT.md`](PHASE9_RESOLUTION_REPORT.md) | `configuration_decomposition.csv`, figure S3 | the confounded "resolution explains everything" framing has been decomposed and narrowed |
| [`PHASE8_BACKBONE_COMPARISON.md`](PHASE8_BACKBONE_COMPARISON.md) | supplement | architecture comparison |
| [`PHASE7_CROSS_DOMAIN_MATRIX.md`](PHASE7_CROSS_DOMAIN_MATRIX.md) | supplement | cross-domain matrix |
| [`PHASE6_IN_DOMAIN_REPORT.md`](PHASE6_IN_DOMAIN_REPORT.md) | supplement | in-domain runs |
| [`PHASE5_LODO_REPORT.md`](PHASE5_LODO_REPORT.md) | `JBHI_MASTER_RESULTS.csv` | early LODO results under the retired criterion |
| [`PHASE4_METHOD_COMPARISON.md`](PHASE4_METHOD_COMPARISON.md) | supplement | early method comparison |
| [`PHASE3_BASELINE_REPORT.md`](PHASE3_BASELINE_REPORT.md) | supplement | baselines |
| [`PHASE2_DATA_REPORT.md`](PHASE2_DATA_REPORT.md) | `JBHI_DATASET_PROVENANCE.md` | data report; the provenance detail lives on in `DATA_PROVENANCE.md` |
| [`PROJECT_HANDBOOK.md`](PROJECT_HANDBOOK.md) | this index + the freeze | describes the repository as it was mid-project |
| [`FULL_FINETUNE_SMOKE_TESTS.md`](FULL_FINETUNE_SMOKE_TESTS.md) | — | pre-flight checks for the full-FT runs; of historical interest only |
| [`PARTIAL_FT_CHECKPOINT_RETENTION.md`](PARTIAL_FT_CHECKPOINT_RETENTION.md) | — | storage engineering record |
| `paper/main.tex` | `paper/jbhi_manuscript.tex` | **historical scaffolding with stale framing; do not draft from it** |

### Three things the historical documents get wrong

If you read them, know these in advance:

1. **"Two bars for every claim"** is retired. The framework is: crossed
   bootstrap = uncertainty interval; paired seed-level *t*-test = formal test;
   Holm = multiplicity adjustment; sign-flip = sensitivity; seed SD and sign
   count = descriptive diagnostics.
2. **Older test and audit counts** (328 tests, 2786 checks, and similar) are
   superseded by **438 tests and 3090 audit checks**.
3. **Novelty framing** in the older documents is too strong. RETFound's own
   paper already compares against an SSL-ImageNet baseline; the contribution is
   the within-lineage **protocol interaction**, and the wording is "to our
   knowledge", never "first".
