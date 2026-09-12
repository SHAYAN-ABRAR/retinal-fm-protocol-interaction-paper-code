# Citation notes — verification status and unresolved items

Companion to `paper/refs_jbhi_verified.bib`. Records **how each entry was
verified**, and which ones a human must still confirm. Nothing in the bib file
was written from memory alone; where memory and the authoritative record
disagreed, the record won and the correction is noted below.

**Verified 2026-09-12** against Crossref (`api.crossref.org/works/<doi>`), NCBI
E-utilities (PubMed `esummary`), and publisher pages.

## Summary

| | |
|---|---|
| entries | **26** |
| with a verified DOI | **14** (every journal article) |
| without a DOI | 12 — conference proceedings, one preprint, two dataset releases |
| duplicate keys | none |
| duplicate DOIs | none |
| unbalanced braces | none |
| unresolved placeholders inside the bib | none |

## Fully verified against Crossref or PubMed

Author lists, year, volume, issue, pages/article number and DOI all confirmed:

| key | source used |
|---|---|
| `zhou2023retfound` | Crossref — **95 authors**; first 11 listed verbatim, remainder as `others` |
| `engelmann2025retfoundgreen` | Crossref — 2 authors, *Nat. Commun.* 16(1):6862 |
| `zhou2026pretrainingdata` | Crossref — 28 authors, *Nat. Commun.* 17(1):3309 |
| `poyrazer2026frozen` | Crossref — 3 authors, *Front. Med.* 13:1815982 |
| `yew2026labelefficiency` | Crossref + PubMed esummary (PMID 42665469) — 29 authors |
| `cao2020coralordinal` | Crossref — *Pattern Recognit. Lett.* 140:325–331 |
| `cohen1968kappa` | Crossref — *Psychol. Bull.* 70(4):213–220 |
| `brier1950verification` | Crossref — *Mon. Weather Rev.* 78(1):1–3 |
| `li2019ddr` | Crossref — 6 authors, *Inf. Sci.* 501:511–522 |
| `porwal2018idrid` | Crossref — 7 authors, *Data* 3(3):25 |
| `mongan2020claim` | Crossref — *Radiol. Artif. Intell.* 2(2):e200029 |
| `tejani2023claimupdate` | Crossref — 7 authors, *Nat. Mach. Intell.* 5(9):950–951 |
| `collins2024tripodai` | Crossref — 34 authors, *BMJ* 385:e078378 |
| `deng2009imagenet` | Crossref — CVPR 2009, DOI 10.1109/CVPR.2009.5206848 |

### Corrections the verification forced

These were wrong in a first draft written partly from memory and were fixed
against the authoritative record. They are listed because they show the check
was real:

- **`zhou2023retfound`** — drafted with 17 authors as though that were the
  whole list. Crossref reports **95**. Now 11 verbatim + `others`.
- **`yew2026labelefficiency`** — several given names were guessed from PubMed
  initials and were wrong: `Chen, Yang` → **Yibing**; `Yang, Guang Dong` →
  **Gabriel Dawei**; `Zhou, Joey` → **Jun**; `Yew, Samantha M. E.` →
  **Samantha Min Er**; plus punctuation on `Chee, Miao-li`, `Tai, E Shyong`,
  `Tham, Yih Chung`.
- **`tejani2023claimupdate`** — `Mongan, John T.` → **John**;
  `Kahn Jr., Charles E.` → **Kahn, Charles E.**

`porwal2018idrid`, `li2019ddr` and `collins2024tripodai` matched the record
exactly as drafted.

## Venue-limited entries — no DOI exists

These are correct but carry only the fields their venue actually issues. **No
volume, page range or DOI has been invented for any of them.**

| key | venue | why no DOI |
|---|---|---|
| `he2022mae` | CVPR 2022 | open-access proceedings |
| `dosovitskiy2021vit` | ICLR 2021 | ICLR does not mint DOIs |
| `sun2016deepcoral` | ECCV 2016 Workshops | workshop proceedings |
| `zhou2021mixstyle` | ICLR 2021 | as above |
| `sagawa2020groupdro` | ICLR 2020 | as above |
| `gulrajani2021domainbed` | ICLR 2021 | as above |
| `arjovsky2019irm` | arXiv:1907.02893 | preprint; never formally published |
| `guo2017calibration` | ICML 2017 (PMLR 70) | PMLR does not mint DOIs |
| `naeini2015ece` | AAAI 2015 | AAAI proceedings |
| `geifman2017selective` | NeurIPS 2017 | NeurIPS proceedings |

**Action for the author:** `guo2017calibration`, `naeini2015ece` and
`geifman2017selective` carry a `note` asking for a check against the published
proceedings. They were **not** found in Crossref, so their page numbers are not
included. If the target venue's style requires page numbers, take them from
PMLR / the AAAI digital library / the NeurIPS proceedings directly.

## Datasets without a peer-reviewed publication

Two of the four datasets have **no dataset paper**. They are cited as `@misc`
with the organiser, year and competition URL, which is what those releases
actually carry.

### EyePACS (`eyepacs2015kaggle`)

Released for the Kaggle *Diabetic Retinopathy Detection* competition (2015),
sponsored by the California Healthcare Foundation with data from EyePACS.
There is no accompanying peer-reviewed dataset descriptor.

- **Terms:** competition rules govern use. Confirm the current rules permit
  research use and publication of derived results before submission.
- **In this study:** used as a **source domain only**, never a held-out target.
  Only the 35,108 label-verified images were used; see
  `JBHI_DATASET_PROVENANCE.md`.

### APTOS 2019 (`aptos2019kaggle`)

Released for the Kaggle *APTOS 2019 Blindness Detection* competition, organised
by the Asia Pacific Tele-Ophthalmology Society with images from Aravind Eye
Hospital. No peer-reviewed dataset descriptor.

- **Terms:** competition rules govern use; confirm as above.
- **In this study:** one of the two primary held-out target domains.

**If a reviewer or the journal requires a formal citation for either**, the
options are (a) cite the competition page as above, or (b) cite a paper that
formally describes the release — but do not attribute a descriptor paper that
does not exist. Option (a) is what the bib currently does.

## Items that remain UNCERTAIN — author action required

1. **"CLAIM 2024" could not be verified.** A Crossref search for a 2024 CLAIM
   update in *Radiology: Artificial Intelligence* returned **no such record**.
   What does exist and is verified: the original CLAIM (Mongan et al., 2020)
   and an update announcement (Tejani et al., *Nat. Mach. Intell.*, 2023). The
   bibliography contains **both**. If the intended reference is a 2024 revision,
   the author must supply its DOI; it has not been invented here.
2. **Page numbers for the three non-Crossref proceedings entries** (Guo, Naeini,
   Geifman), if the journal style demands them.
3. **Kaggle competition terms** for EyePACS and APTOS, confirmed current at
   submission time.
4. **`yew2026labelefficiency` volume and issue.** The article was online ahead
   of print (28 August 2026) at verification time and carries article number
   101031 with no volume assigned. Re-check before submission in case it has
   since been paginated.

## What to do if a field cannot be verified

Leave it out. A BibTeX entry missing an optional field is a minor style issue;
an entry with a confident, wrong volume number is a citation error that a
reviewer can catch and that undermines the rest of the bibliography.
