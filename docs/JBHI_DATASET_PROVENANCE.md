# Dataset provenance and contamination — manuscript table

Condensed from `DATA_PROVENANCE.md` (full Phase-1 audit) into the form a
manuscript needs. Every rule below is enforced in code, not assumed.

## The table

| dataset | country / source | role | raw count | retained | grading scale | patient IDs | split unit | RETFound pretraining relationship | known overlap issue | action taken |
|---|---|---|---|---|---|---|---|---|---|---|
| **DDR** | China (Sun Yat-sen / public release) | held-out target **and** source domain | 12,522 labelled | 12,522 | ICDR 0–4 | none usable (middle filename field is an image counter, 7,101 singleton groups) | image | **not** in the published RETFound CFP corpus | none | used as released; grade 5 "ungradable" already absent |
| **APTOS 2019** | India (Aravind, Kaggle) | held-out target **and** source domain | 3,662 | 3,662 | ICDR 0–4 | none (anonymous 12-hex ids) | image | **not** in the published RETFound CFP corpus | **all 3,662 images physically present inside the EyePACS-derivative folder** | EyePACS loader rejects every 12-hex filename; APTOS images enter only as the APTOS domain |
| **IDRiD** | India (Nanded) | held-out target **and** source domain | 516 | 516 | ICDR 0–4 | none | image | **not** in the published RETFound CFP corpus | none | used as released; only 5 test images of grade 1, so per-class grade-1 claims are avoided |
| **EyePACS** (verified subset) | USA (Kaggle/EyePACS) | **source domain only — never a held-out target** | 85,178 present in the derivative | **35,108** | ICDR 0–4 | yes | **patient** (both eyes kept together) | **included in RETFound's published CFP pretraining corpus** (88,702 EyePACS images within 904,170 total) | provided splits leak; augmented duplicates baked in; labels from a third-party derivative | labels verified against official labels at 100.0000% agreement, 50,070 unverifiable images excluded; provided splits discarded and re-split at patient level; `augmented_resized_V2/` on `forbidden_paths` |

## Splits actually used

Identical across every run of a given held-out target, and asserted at load time.

| held out | train | val | test |
|---|---|---|---|
| DDR | 27,718 | 5,677 | **12,424** |
| APTOS | 33,606 | 7,201 | **3,504** |
| IDRiD | 36,080 | 7,472 | **507** |

Target labels are used for **nothing** except final scoring: not for model
selection, not for early stopping, not for temperature calibration. Temperature
is fitted on source validation only.

## The contamination statement, in the words the manuscript should use

> **EyePACS appears in the published RETFound colour-fundus pretraining corpus.
> APTOS and DDR do not appear in that corpus.**
>
> EyePACS remains a **source** domain when DDR or APTOS is the held-out target.
> RETFound has therefore previously seen, without labels, images from one of the
> source domains. **This is not target leakage**: no held-out target image
> appears in RETFound's pretraining corpus or in any training split. It is part
> of the retinal-pretraining intervention being measured rather than a confound
> to be removed, and — decisively for the primary analysis — it applies
> identically to the frozen and the fine-tuned arms, so it cannot generate the
> protocol interaction.

Two further points of transparency:

- The claim about corpus membership rests on the **RETFound publication**
  (815,468 MEH-MIDAS + 88,702 EyePACS = 904,170 colour fundus photographs), not
  on the checkpoint, which carries no manifest of what it saw.
- **IDRiD is never a source domain for itself**, and the same EyePACS exposure
  applies when IDRiD is held out.

## The one guard that actually protects the primary result

The folder mapped to the EyePACS domain physically contains **100% of the APTOS
domain** — all 3,662 images, verified by id overlap. Without a filter, holding
out APTOS would have trained on the entirety of its own test set.

**Enforced:** the EyePACS loader keeps only `^\d+_(left|right)$` filenames and
rejects every 12-hex APTOS id (`src/data/eyepacs.py:99,144`). This is asserted
at load time, not assumed.

This is the single most consequential integrity control in the study and should
be stated explicitly in the manuscript's data section — a reader evaluating an
APTOS-held-out result needs to know it was checked.

## Other integrity findings, disclosed

| finding | scale | action |
|---|---|---|
| provided EyePACS train/val/test folders leak | 1,231 base photographs and 14,244 of 43,565 patients in more than one split | splits discarded entirely; pooled and re-split at patient level, both eyes together |
| augmented duplicates baked into files on disk | 3,662 `GF-`/`-GF` files; `augmented_resized_V2/` holds 143,669 further-augmented files with minority classes inflated ~4× | `augmented_resized_V2/` on `forbidden_paths`, never read; augmentation belongs in the training transform |
| EyePACS labels from third-party folder names | 85,178 images | 35,108 verified against official labels (100.0000% agreement, class distribution within 0.02 pp on every grade); 50,070 unverifiable excluded |
| no patient IDs in DDR, APTOS, IDRiD | three of four domains | image-level splitting, documented as a limitation; patient-level splitting is impossible for these releases |
