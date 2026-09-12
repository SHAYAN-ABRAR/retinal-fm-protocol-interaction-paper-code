# JBHI page budget

Target: an **8-page IEEE double-column regular paper**.

> ## Eight pages is an ECONOMY TARGET, not a scientific hard limit
>
> It is a planning figure for keeping the paper within the length JBHI allows
> before overlength charges — **not** a constraint the science must be trimmed
> to fit.
>
> - **Do not remove essential methods or statistics to hit eight pages.** If
>   methodology has to go, the cut is wrong; move secondary evidence to the
>   supplement instead. A reviewer who cannot tell how the splits were made,
>   what "full fine-tuning" means here, or which seeds went into which estimate
>   cannot evaluate the paper.
> - **Verify the current official JBHI page limit and charge schedule directly
>   in the IEEE Author Portal immediately before submission.** The figure used
>   here must not be taken on trust, and in particular **must not be inherited
>   from a BHI conference special-issue rule** — that is a different track with
>   a different limit, and this is a regular JBHI submission.
> - **If an extra page materially improves methodological clarity, flag it for
>   the author** rather than cutting necessary detail. Overlength charges are a
>   cost decision for the author to make knowingly, not a reason to under-report
>   the method.

## Allocation

| section | target | notes |
|---|---|---|
| Abstract | ~0.25 | must carry the primary numbers **and** the explicit non-claim |
| I. Introduction | 0.8–1.0 | the gap is the untested assumption, not a missing model |
| II. Related Work | 0.6–0.8 | five verified references carry this; see below |
| III. Methods + IV. Experimental Design | 1.7–2.0 | **do not compress below this** |
| V. Results | 1.7–2.0 | three tables and four figures live here |
| VI. Discussion + VII. Limitations | 1.0–1.2 | limitations L1–L4 are mandatory |
| VIII. Conclusion | ~0.25 | no new claim |
| References | remainder | 28 verified entries |
| **total** | **~8** | |

Figures and tables consume roughly 1.5–2.0 pages of that total when set at
two-column width, which is already assumed in the Results allocation.

## What must not be cut to save space

These are the items a methods reviewer will look for first, and each is short:

1. **The seed sets.** Ten seeds frozen, five for both adaptation protocols, and
   the statement that interactions use the common five. One sentence.
2. **The EyePACS/APTOS integrity control.** The EyePACS folder physically
   contains all 3,662 APTOS images and the loader rejects them by filename.
   Two sentences. Without it an APTOS-held-out result is not evaluable.
3. **The EyePACS pretraining disclosure.** EyePACS is in RETFound's published
   corpus and is a source domain here; not target leakage. Two sentences.
4. **The statistical vocabulary.** Which instrument is the test, which is the
   interval, where Holm applies, and that the formal test governs when they
   disagree. Three or four sentences in Methods E.
5. **The chronology.** DDR first and observed, APTOS specified afterwards as a
   confirmatory replication. Two sentences; omitting it would misrepresent the
   design.
6. **Limitations L1–L4.** IDRiD has no full-FT arm; the two-domain family is a
   post-hoc pairing; EyePACS exposure; no demographics available.
7. **The full-FT null, stated as a null.** Including the DDR case where the
   interval and the test disagree.

## What belongs in the supplement

Detailed in `JBHI_CONTENT_PLACEMENT.md`. In page-budget terms, moving these out
is what buys the eight pages:

- the domain-generalisation method benchmark (a different question, null result)
- the resolution and batch-size decomposition (S3)
- extended calibration (S4), risk–coverage (S5), confusion detail (S6)
- optimisation behaviour (S1)
- dataset profiling (S2)
- full provenance audit detail, checkpoint retention, run accounting
- per-class metrics and seed-instability case studies

## Figure and table budget

**Main paper — four figures, three tables:**

| asset | width | approx. cost |
|---|---|---|
| Fig. 1 study design | two-column | 0.30 page |
| Fig. 2 adaptation depth | two-column | 0.30 |
| Fig. 3 interaction forest | two-column | 0.22 |
| Fig. 5 full-FT paired QWK | two-column | 0.30 |
| TABLE I datasets | two-column | 0.25 |
| TABLE II master results | two-column | 0.35 |
| TABLE III primary interaction | two-column | 0.20 |

`fig4_seed_interaction_slopes` starts in the supplement and is the natural
promotion candidate if space allows; it is also the first main-paper figure to
demote if space is short.

## On merging Tables II and III

**Recommendation: keep them separate.** Merging was considered and is not
advised, for two reasons that are about correctness rather than layout:

1. **They use different seed sets.** Table II reports frozen at ten seeds *and*
   at the common five; Table III is common-five only. A merged table would
   either drop the ten-seed rows or place two seed bases in one column block —
   exactly the confusion the `seed_set` column exists to prevent.
2. **They have different inferential status.** Table II's per-protocol
   comparisons are descriptive and uncorrected; Table III is the
   Holm-adjusted primary family. A single table implies a single status.

No scientific content changes either way — this is a presentation
recommendation only, as requested.

## If the paper runs long

In order, do these before cutting methodology:

1. move Fig. 2 or Fig. 5 to the supplement (not Fig. 3 — it is the result)
2. compress Related Work to a single subsection
3. shorten the Discussion, keeping all of Limitations
4. tighten Table II to the common-five rows, keeping the ten-seed rows in the
   supplement with a pointer

## If the paper runs short

Promote `fig4_seed_interaction_slopes` from the supplement — it shows the
pairing directly and strengthens the primary result without adding a claim.
