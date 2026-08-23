# Data provenance and integrity — Phase 1 findings

**Audit date:** 2026-08-19
**Data root:** `D:\DB`
**Reproduce with:** `research_pipeline.py`, Cells 5–8
**Machine-readable output:** `outputs/reports/phase1_provenance_audit.json`

Every number below was measured by walking the actual filesystem. None is copied
from a dataset paper or a Kaggle description. The audits in
`src/data/provenance.py` re-measure all of them, so nothing here has to be taken
on trust.

---

## 1. Folder → domain mapping

The folders in `D:\DB` are named `Dataset 1`…`Dataset 4` and do **not** follow the
domain numbering used by the study. The mapping was established from filename
patterns, CSV contents and label counts.

| Physical folder | Domain | Domain ID | Images | Evidence used to identify it |
|---|---|---|---|---|
| `Dataset 4` | DDR | 0 | 12,522 labelled | Label counts `6266/630/4477/236/913` match the official DDR gradable subset exactly |
| `Dataset 2` | APTOS 2019 | 1 | 3,662 | 12-hex `id_code`s; total is exactly the APTOS 2019 training set |
| `Dataset 3` | IDRiD | 2 | 516 | `B. Disease Grading` with `IDRiD_nnn` names, 413 train / 103 test |
| `Dataset 1` | EyePACS *(derivative — see §3)* | 3 | **35,108 verified** (of 85,178 present) | `<n>_left` / `<n>_right` filenames |

---

## 2. Clean domains

### DDR (Domain 0) — `Dataset 4`

- `DR_grading.csv`: 12,522 rows, columns `id_code,diagnosis`; `id_code` **includes**
  the `.jpg` extension.
- Grades 0–4 only. **DDR's official grade 5 ("ungradable") is already absent** —
  the row count matches the official gradable count exactly. Verified by counting,
  not assumed.
- Class distribution: `0:6266  1:630  2:4477  3:236  4:913`.
- **Two files on disk have no CSV row:** `007-7449-601.jpg`, `007-7447-601.jpg`.
  They are excluded explicitly.
- Images are heterogeneous: many are already downsampled to 512×512, the rest
  range up to 3456×3456.
- **No official train/valid/test split is preserved** — everything sits in one flat
  folder, so splits must be constructed here.
- ⚠️ **Patient IDs are not recoverable.** Two filename families exist:
  7,101 of the form `NNN-NNNN-NNN` and 5,423 17-digit timestamps. Grouping the
  first family by its `(site, patient)` prefix yields **7,101 groups of exactly one
  image each**, so the middle field is an image counter, not a patient key. This is
  a documented limitation, not something to paper over.

### APTOS 2019 (Domain 1) — `Dataset 2`

- Cleanest of the four. Pre-split 2,930 / 366 / 366 = 3,662.
- Split id overlap: **0**. Every CSV row has an image; every image has a row.
- Native resolutions vary widely (819×614 → 4288×2848); many are 1050×1050.
- ⚠️ **No patient IDs.** Anonymous 12-hex ids with no eye or patient linkage, so
  patient-level splitting is impossible for this domain. Documented limitation.

### IDRiD (Domain 2) — `Dataset 3`

- Only `B. Disease Grading` is used. `A. Segmentation` and `C. Localization` are
  ignored (they are a subset of the same eyes with pixel-level annotations).
- Uniform 4288×2848 — the only domain with a single native resolution.
- Target column is `Retinopathy grade`. The training CSV also carries nine empty
  padding columns and a `Risk of macular edema ` column **whose name ends in a
  space**. That column is *not* the DR grade and is not used as the target.
- ⚠️ **Filenames collide across splits.** Both the training and testing folders
  begin at `IDRiD_001.jpg`; **103 filenames are reused**. Any index keyed on bare
  filename silently merges two different images and mislabels one. Image ids must
  be namespaced as `<domain>/<split>/<filename>`.
- ⚠️ Only 516 images total, and only 5 test images of grade 1. External-test
  metrics on this domain will have wide confidence intervals — bootstrap CIs are
  mandatory, and per-class claims about grade 1 should be avoided.

---

## 3. `Dataset 1` is **not** EyePACS — four problems

`Dataset 1` is a third-party derivative (`dr_unified_v2` plus
`augmented_resized_V2`), not the raw Kaggle EyePACS release. Four problems make
it unusable in its shipped form. §3A–3C are handled by exclusion rules; §3D was
resolved by verifying the labels against an independent reference, which shrank
the domain to its trustworthy core.

### 3A. It contains 100% of the APTOS domain

| Measurement | Value |
|---|---|
| Unique EyePACS-style images | 85,178 |
| Unique APTOS-style images | 3,662 |
| APTOS domain size (`Dataset 2`) | 3,662 |
| **Overlapping ids** | **3,662 (100.0%)** |

Every single APTOS image is physically present inside the folder mapped to the
EyePACS domain. Treating `Dataset 1` as "the EyePACS domain" would put identical
photographs in two supposedly different domains, and a leave-one-domain-out
experiment holding out APTOS would train on 100% of its own test set.

**Rule enforced:** APTOS images come *only* from Domain 1. The EyePACS loader
keeps `^\d+_(left|right)$` and rejects every 12-hex filename — asserted, not
assumed.

### 3B. The provided train/val/test folders leak

| Overlap | train∩val | train∩test | val∩test |
|---|---|---|---|
| Base photographs | 563 | 575 | 93 |
| **EyePACS patients** | **6,748** | **6,679** | **817** |

1,231 base photographs and 14,244 of 43,565 patients appear in more than one
split. Two mechanisms: augmented `-GF` copies of an image landing in a different
split from their original, and left/right eyes of one patient being split apart.

(These counts cover all 85,178 images shipped in the folder. §3D restricts the
domain to the 35,108 verified ones, but that restriction does *not* repair the
splits — they are discarded regardless.)

**Rule enforced:** the provided split folders are discarded entirely
(`ignore_provided_splits: true`). Images are pooled and re-split at patient level,
keeping both eyes of a patient together.

### 3C. Augmented duplicates are baked in

- 3,662 files carry `GF-` / `-GF` augmentation markers inside `dr_unified_v2`
  (exactly one per APTOS image — the augmentation was applied to APTOS only).
- `augmented_resized_V2/` is a *further* augmented pool (`-600`, `-ALL` suffixes,
  all 600×600), 143,669 files, where minority classes are inflated ~4×.

**Rule enforced:** `augmented_resized_V2` is on the `forbidden_paths` list and is
never read. Augmentation belongs in the training transform, where it can be kept
out of validation and test — not baked into files on disk.

### 3D. Label verification — resolved, and it changed the domain

EyePACS grades come from **parent folder names in a third-party derivative**, the
weakest provenance of the four domains. Rather than accept that as a limitation,
the folder labels were checked against an independent copy of the Kaggle
`trainLabels.csv` (`src/data/eyepacs_labels.py`, Cell 7b). The check split the
domain cleanly in two.

**35,108 images — labels confirmed, 100.0000% agreement.**

| Grade | Verified subset | Published official EyePACS |
|---|---|---|
| 0 | 73.49% (25,802) | 73.48% |
| 1 | 6.94% (2,438) | 6.96% |
| 2 | 15.06% (5,288) | 15.07% |
| 3 | 2.48% (872) | 2.49% |
| 4 | 2.02% (708) | 2.01% |

Every one of the 35,108 overlapping ids matches, and the distribution reproduces
the published official statistics to within 0.02 percentage points on every
grade. These labels are correct, not merely plausible.

**50,070 images — labels corrupted, excluded.** These are the EyePACS *test*
portion, absent from `trainLabels.csv`. Their folder-derived distribution is:

```
{0: 39541,  1: 1457,  2: 7865,  3: 1,  4: 1206}
                                  ^^^
```

**One single grade-3 image among 50,070**, where ~1,247 are expected. A deficit
that large is not a sampling artefact — those labels are broken. Grade 1 is also
roughly half its expected count, with a matching excess in grade 0.

**Decision: the EyePACS domain is the 35,108 verified images (17,561 patients).**
This is a strict improvement, not a compromise:

- EyePACS goes from *weakest* to *fully verified* label provenance.
- The class distribution now matches official EyePACS instead of deviating from it.
- Training cost drops 59% — material on an 8 GB laptop GPU.
- It remains by far the largest domain (35,108 vs DDR's 12,522).

`outputs/reports/eyepacs_verified_index.csv` is the authoritative index. The
Phase-2 loader reads **only** that file and never walks the raw folder tree, so
the exclusion cannot be bypassed by accident.

Also measured: **87.3% of patients have the same grade in both eyes.** That is
direct empirical justification for patient-level splitting — splitting these at
image level would leak a near-duplicate label between train and test.

#### Provenance chain (stated in the paper)

```
Kaggle "Diabetic Retinopathy Detection" (official; 35,126 train rows)
  -> tanlikesmath/diabetic-retinopathy-resized  (35,108 rows; 18 unreadable
     images dropped, remainder resized to max 1024 px)
  -> reference CSV used here (SHA-256 c1b284d4...ffbce3e2)
```

The reference file is a **secondary** source — the official competition file
needs Kaggle authentication plus manual acceptance of the competition rules,
which cannot be automated. It is therefore corroborated three independent ways
before being trusted, and `verify_reference_file()` asserts all three on every run:

1. Three independent Hugging Face mirrors publish the identical 35,108-row set.
2. Its class distribution matches published official EyePACS statistics.
3. It agrees 100% with the folder labels of a separately produced derivative.

Two artefacts built by different people through different pipelines agreeing on
all 35,108 labels is stronger evidence than either source alone. The 18-row gap
versus the official 35,126 is the well-known set of unreadable training images
dropped in redistribution, and is disclosed rather than smoothed over.

*If you later obtain Kaggle credentials*, re-running Cell 7b against the official
`trainLabels.csv` would remove the secondary-source caveat entirely. The
comparison code needs no changes — only the file it points at.

#### A useful by-product: `Dataset 1` is identified

`CDHAI/EyePACS` on Hugging Face has splits of exactly 115,241 / 14,227 / 14,201 —
an exact match to this machine's `augmented_resized_V2`. `Dataset 1` is that
dataset, which pins down its origin and explains both the APTOS pooling and the
baked-in augmentation.

---

## 4. Label semantics across domains

All CSV-labelled domains use the ICDR 5-grade scale with identity mapping:

| Grade | Meaning |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

No domain required re-encoding. The explicit mapping — including the EyePACS
provenance caveat — is written to `outputs/reports/label_mapping.json` by Cell 7.

---

## 5. Patient-level splitting: what is and is not possible

| Domain | Patient IDs | Splitting strategy |
|---|---|---|
| DDR | ❌ not recoverable | Stratified by grade at image level; limitation documented |
| APTOS | ❌ none exist | Uses the provided disjoint split; limitation documented |
| IDRiD | ❌ none | Uses the official train/test split |
| EyePACS | ✅ `<patient>_<eye>` (17,561 patients) | **Patient-level**, both eyes kept in the same split. 87.3% of patients share a grade across eyes, so this is essential, not cosmetic |

Only EyePACS supports true patient-level splitting; it is also the only domain
where the failure to do so would matter at scale (17,561 patients x 2 eyes).
For the other three, image-level stratified splitting is the honest ceiling and
must be stated as a limitation in the paper rather than glossed over.

---

## 6. Disk and memory notes

| Folder | Size |
|---|---|
| `Dataset 1` | 22 GB (`dr_unified_v2` 18 GB, `augmented_resized_V2` ~4 GB) |
| `Dataset 2` | 8.1 GB |
| `Dataset 3` | 966 MB |
| `Dataset 4` | 3.1 GB |
| **Total** | **~34 GB** |

`D:` has ~105 GB free, so no dataset copying or re-encoding is required; images
are read lazily from their current locations.

Measured at audit time: **15.6 GB RAM total, 3.5 GB available (77% in use)**. That
headroom is tight for 4 DataLoader workers on 384×384 crops. Close other
applications before long runs, and treat `num_workers=4` as an upper bound to
validate rather than a default to assume.
