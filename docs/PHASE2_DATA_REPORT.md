# Phase 2 — Manifest, deduplication, splits, leakage

> **HISTORICAL PROJECT RECORD — NOT AUTHORITATIVE FOR THE FINAL MANUSCRIPT.**
> See [`JBHI_EVIDENCE_FREEZE.md`](JBHI_EVIDENCE_FREEZE.md) and
> [`JBHI_DRAFTING_MANIFEST.md`](JBHI_DRAFTING_MANIFEST.md) for the frozen final
> state. Statements below were true at the time they were written and may have
> been superseded — including older test and audit counts, the retired
> "two-bar" terminology, and novelty framing that has since been narrowed.


**Date:** 2026-08-19
**Reproduce with:** `research_pipeline.py`, Cells 9–13
**Artefacts:** `outputs/reports/unified_manifest_split.csv`, `outputs/reports/phase2_leakage_audit.json`, `outputs/figures/*.png`

---

## 1. The corpus

Four domains, one manifest, 51,543 images after cleaning.

| Domain ID | Domain | Images | Train | Val | Test | Patients | Split unit |
|---|---|---|---|---|---|---|---|
| 0 | DDR | 12,424 | 8,697 | 1,865 | 1,862 | — | image (stratified) |
| 1 | APTOS 2019 | 3,504 | 2,809 | 341 | 354 | — | vendor split kept |
| 2 | IDRiD | 507 | 335 | 70 | 102 | — | vendor split, val carved from train |
| 3 | EyePACS | 35,108 | 24,574 | 5,266 | 5,268 | 17,561 | **patient** |

Grade distributions are held within ~1 percentage point across train/val/test in
every domain by stratified allocation.

---

## 2. Exact duplicates — found and removed *before* splitting

Content hashing (size-prefiltered, so only ~10k of 51,808 files needed hashing)
found **224 groups of byte-identical images under different filenames**:

| Domain | Groups touched | Images removed |
|---|---|---|
| APTOS | 123 | 158 |
| DDR | 95 | 98 |
| IDRiD | 6 | 9 |
| EyePACS | 0 | 0 |

Before removal, **89 of those groups straddled a split boundary** (44 train↔test,
40 train↔val, 5 test↔val). A model would have been trained on a file and
evaluated on a byte-identical copy of it.

### 33 groups had contradictory labels

Byte-identical images carrying **different grades**. Examples:

```
aptos/train/1c9c583c10bf.png  grade 0   ==   aptos/test/ea15a290eb96.png  grade 1
aptos/train/2df07eb5779f.png  grade 4   ==   aptos/train/b91ef82e723a.png grade 2
aptos/train/2f7789c1e046.png  grade 4   ==   aptos/train/a8e88d4891c4.png grade 3
```

The ground truth contradicts itself for these images. **Policy applied:**

- *Consistent* group (191 of them) → keep one representative (lexicographically
  smallest `image_id`, so the choice is deterministic), drop the rest. 197 images.
- *Conflicting* group (33) → **drop the whole group**. We cannot know which label
  is right, and keeping one arbitrarily injects known-bad supervision. 68 images.

Total removed: 265 of 51,808 (0.5%). This is a reportable data-quality finding
about APTOS and DDR, not just housekeeping.

---

## 3. Leakage audit — PASS

Run on the constructed splits (`src/data/leakage.py`), after deduplication:

| Check | train∩val | train∩test | val∩test |
|---|---|---|---|
| `image_id` overlap | 0 | 0 | 0 |
| Patient overlap | 0 | 0 | 0 |
| Byte-identical content | 0 | 0 | 0 |

Also: 0 duplicated manifest rows, 0 shared file paths, 0 groups spanning two
domains.

100 bare filenames repeat (IDRiD's `IDRiD_001.jpg` in both split folders). This
is harmless **because** `image_id` is namespaced as
`<domain>/<source_split>/<filename>`; a manifest keyed on bare filename would
have silently merged those pairs and mislabelled one of each.

The audit is tested against injected leakage — `tests/test_phase2.py` plants a
duplicated id, a patient straddling splits, and a byte-identical copy, and
asserts each is caught. A leakage check that has never been shown to fail proves
nothing.

### Documented limitations

1. **Patient ids exist only for EyePACS.** DDR, APTOS and IDRiD publish no
   patient linkage, so they are split at image level and same-patient images
   could in principle span splits. This is undetectable and unpreventable from
   the released data — a limitation of the datasets, not the protocol, and it
   goes in the paper.
2. **Content hashing finds byte-identical files only.** The same photograph
   re-encoded or resized has different bytes and would not be flagged.
   Near-duplicate detection would need perceptual hashing, which trades
   exactness for recall.

Supporting evidence for limitation 1's severity: in EyePACS, where the check
*is* possible, **87.3% of patients carry the same grade in both eyes**. Splitting
those at image level would leak a near-duplicate label.

---

## 4. Preprocessing

Deterministic pipeline: retina localisation → square padding → resize.
Aspect ratio is never stretched, because a squashed fundus changes lesion shape
and apparent optic-disc size.

The background threshold (0.20) was **chosen from a sweep**, not by taste.
Median retained area by threshold:

| thr | 0.05 | 0.10 | 0.15 | **0.20** | 0.25 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|---|---|
| DDR | 96.1 | 96.0 | 95.7 | **95.7** | 95.6 | 95.5 | 95.5 | 95.5 |
| APTOS | 100.0 | 100.0 | 100.0 | **99.9** | 99.6 | 99.5 | 99.4 | 99.3 |
| IDRiD | 97.8 | 90.4 | 80.7 | **79.7** | 79.6 | 79.6 | 79.5 | 79.5 |
| EyePACS | 100.0 | 100.0 | 100.0 | **100.0** | 100.0 | 100.0 | 99.9 | 99.9 |

Below 0.15 the result swings 7–10 points because JPEG ringing in IDRiD's black
surround clears the cut. From 0.20 every domain is flat to within 0.3 points per
step, so 0.20 is the first point of the stable plateau with margin on both sides.

**An honest caveat:** only IDRiD carries substantial black border. The other
three ship images already close to tightly cropped, and APTOS is bimodal (most
need no crop, a minority need a large one). Border removal eliminates one cheap
confound; it does **not** harmonise the domains, and claiming otherwise would
overstate it.

---

## 5. Domain shift, measured before any model

Median values over a seeded, grade-stratified sample of 400 images per domain,
computed **after** the retina crop (measuring brightness over black border would
mostly measure how much border a dataset ships):

| Domain | Megapixels | Aspect | Brightness | Contrast | R | G | B |
|---|---|---|---|---|---|---|---|
| DDR | 0.26 | 1.00 | 75.3 | 44.4 | 112.0 | 65.2 | 27.0 |
| APTOS | 3.15 | 1.33 | 75.3 | 34.9 | 121.2 | 63.4 | 16.6 |
| IDRiD | 12.21 | 1.51 | 84.4 | 37.2 | 140.0 | 68.4 | 17.5 |
| EyePACS | 0.97 | 1.08 | 80.9 | 43.5 | 109.1 | 73.5 | **51.0** |

Two differences stand out:

- **Resolution spans 47×** (DDR 0.26 MP → IDRiD 12.21 MP).
- **EyePACS' blue channel is ~3× APTOS'** (51.0 vs 16.6). Fundus images are
  normally blue-poor; EyePACS' is markedly less so, which points to a different
  camera/processing pipeline.

A PCA of these hand-crafted statistics (`15_image_statistics_pca.png`) separates
domains far more cleanly than it separates DR grades. **This is descriptive
only.** It shows the datasets differ in basic appearance; it does *not* show that
a trained model relies on those differences. The learned-embedding analysis in
Phase 4 is what addresses that, and the figure carries the caveat on its face.

Note also **label shift**: grade prevalence differs sharply across domains
(no-DR is 32.9% in IDRiD but 73.5% in EyePACS; severe DR is 18.0% in IDRiD but
1.9% in DDR). Covariate shift and label shift are confounded here, and any
cross-domain performance drop will reflect both.

---

## 6. Figures produced

18 figures at 300 DPI in `outputs/figures/`:

| # | File | Shows |
|---|---|---|
| 01 | `01_images_per_dataset` | Domain sizes (log scale — they span 70×) |
| 02–05 | class distribution: counts, normalised, heatmap, stacked | Label shift |
| 06 | `06_split_composition` | Train/val/test per domain |
| 07 | `07_grade_examples_by_domain` | 4×5 grid of real images |
| 08–12 | width, height, aspect, brightness, contrast | Geometry + photometry |
| 13–14 | RGB statistics, box plots | Appearance shift |
| 15 | `15_image_statistics_pca` | Domain separation (with caveat) |
| 16 | `16_preprocessing_examples` | Crop box + before/after per domain |
| 17 | `17_augmentation_examples` | Training augmentation draws |
| 18 | `18_data_quality_summary` | Unreadable images + duplicate removal |

0 unreadable images in the 1,600-image sample.

---

## 7. What is ready, and what is next

Ready: unified manifest, deterministic splits, passing leakage audit,
preprocessing, augmentation (verified against albumentations 2.0.8), figures,
and the experiment-protocol builder with the target-isolation assertion.

The four leave-one-domain-out experiments are already assembled and their sizes
known:

```
lodo_aptos-eyepacs-idrid__ddr      train 27,718  val 5,677  test 12,424
lodo_ddr-eyepacs-idrid__aptos      train 33,606  val 7,201  test  3,504
lodo_aptos-ddr-eyepacs__idrid      train 36,080  val 7,472  test    507
lodo_aptos-ddr-idrid__eyepacs      train 11,841  val 2,276  test 35,108
```

**Not yet done — Phase 3:** torch must be installed (Cell 13 is the first cell
that needs it), then backbones, the training loop, and metrics.

**No model has been trained. No performance number exists anywhere in this
repository.**
