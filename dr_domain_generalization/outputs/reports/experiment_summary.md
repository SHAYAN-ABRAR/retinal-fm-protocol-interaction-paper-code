# Experiment summary

*Generated 2026-08-19 23:43 UTC from `outputs/experiment_registry.csv`. Every number below is read from a file; none is typed in by hand.*

## Scope

- **19 completed run(s)** in the registry.
- **3 seed(s).**
- Protocols present: ['lodo']
- Target domains evaluated: ['idrid']
- Backbones: ['densenet121']

## 1. How large is the domain-generalization gap?

| Experiment | Target | Val QWK | Test QWK | Gap | Val ECE→Test ECE |
|---|---|---|---|---|---|
| `erm-none` | idrid | 0.8787 | 0.6823 | **-0.1964** | → 0.3456 |
| `erm` | idrid | 0.8797 | 0.7529 | **-0.1268** | → 0.2647 |
| `deep_coral` | idrid | 0.8794 | 0.6646 | **-0.2148** | → 0.3352 |
| `mixstyle` | idrid | 0.8791 | 0.6433 | **-0.2358** | → 0.3019 |
| `ordinal` | idrid | 0.8625 | 0.6646 | **-0.1979** | → 0.0594 |
| `mixstyle_ordinal` | idrid | 0.8513 | 0.6623 | **-0.1890** | → 0.0628 |
| `deep_coral_ordinal` | idrid | 0.8556 | 0.6707 | **-0.1849** | → 0.0707 |
| `erm` | idrid | 0.8794 | 0.6894 | **-0.1900** | → 0.3247 |
| `ordinal` | idrid | 0.8591 | 0.6563 | **-0.2029** | → 0.0685 |
| `deep_coral` | idrid | 0.8806 | 0.6657 | **-0.2148** | → 0.3557 |
| `mixstyle` | idrid | 0.8774 | 0.6825 | **-0.1949** | → 0.2963 |
| `mixstyle_ordinal` | idrid | 0.8632 | 0.6751 | **-0.1881** | → 0.0725 |
| `deep_coral_ordinal` | idrid | 0.8647 | 0.6783 | **-0.1865** | → 0.0585 |
| `erm` | idrid | 0.8806 | 0.7281 | **-0.1525** | → 0.3027 |
| `ordinal` | idrid | 0.8533 | 0.6426 | **-0.2107** | → 0.0552 |
| `deep_coral` | idrid | 0.8833 | 0.6925 | **-0.1909** | → 0.2881 |
| `mixstyle` | idrid | 0.8862 | 0.6591 | **-0.2271** | → 0.3385 |
| `mixstyle_ordinal` | idrid | 0.8467 | 0.6414 | **-0.2053** | → 0.0544 |
| `deep_coral_ordinal` | idrid | 0.8569 | 0.6843 | **-0.1725** | → 0.0943 |

Mean QWK gap across runs: **-0.1938**.

Note the validation figure is the *best* source-validation score, selected on that split, so it is itself mildly optimistic. The gap is a lower bound on the drop a clinic would see.

## 2. Does calibration degrade more than accuracy?

| Experiment | Test QWK | Test ECE | Test NLL | Test Brier |
|---|---|---|---|---|
| `erm-none` | 0.6823 | 0.3456 | 2.1400 | 0.8148 |
| `erm` | 0.7529 | 0.2647 | 1.6672 | 0.7102 |
| `deep_coral` | 0.6646 | 0.3352 | 1.9436 | 0.7568 |
| `mixstyle` | 0.6433 | 0.3019 | 2.0841 | 0.7761 |
| `ordinal` | 0.6646 | 0.0594 | 1.6030 | 0.6379 |
| `mixstyle_ordinal` | 0.6623 | 0.0628 | 1.6031 | 0.6388 |
| `deep_coral_ordinal` | 0.6707 | 0.0707 | 1.5996 | 0.6261 |
| `erm` | 0.6894 | 0.3247 | 2.1200 | 0.7728 |
| `ordinal` | 0.6563 | 0.0685 | 1.6058 | 0.6325 |
| `deep_coral` | 0.6657 | 0.3557 | 1.9568 | 0.7837 |
| `mixstyle` | 0.6825 | 0.2963 | 1.8703 | 0.7416 |
| `mixstyle_ordinal` | 0.6751 | 0.0725 | 1.6062 | 0.6362 |
| `deep_coral_ordinal` | 0.6783 | 0.0585 | 1.5901 | 0.6321 |
| `erm` | 0.7281 | 0.3027 | 1.7627 | 0.7390 |
| `ordinal` | 0.6426 | 0.0552 | 1.6093 | 0.6382 |
| `deep_coral` | 0.6925 | 0.2881 | 1.7903 | 0.7246 |
| `mixstyle` | 0.6591 | 0.3385 | 2.0037 | 0.8065 |
| `mixstyle_ordinal` | 0.6414 | 0.0544 | 1.6045 | 0.6340 |
| `deep_coral_ordinal` | 0.6843 | 0.0943 | 1.5853 | 0.6249 |

## 3. Do the methods help?

Baseline: `erm` on idrid (test QWK 0.7281, ECE 0.3027).

| Method | Test QWK | vs baseline | Test ECE | vs baseline |
|---|---|---|---|---|
| `erm-none` | 0.6823 | WORSE (-0.0458) | 0.3456 | WORSE (+0.0430) |
| `erm` | 0.7529 | better (+0.0248) | 0.2647 | better (-0.0380) |
| `deep_coral` | 0.6646 | WORSE (-0.0636) | 0.3352 | WORSE (+0.0326) |
| `mixstyle` | 0.6433 | WORSE (-0.0848) | 0.3019 | within noise (-0.0007, |diff| < 0.02) |
| `ordinal` | 0.6646 | WORSE (-0.0636) | 0.0594 | better (-0.2432) |
| `mixstyle_ordinal` | 0.6623 | WORSE (-0.0658) | 0.0628 | better (-0.2399) |
| `deep_coral_ordinal` | 0.6707 | WORSE (-0.0574) | 0.0707 | better (-0.2320) |
| `erm` | 0.6894 | WORSE (-0.0387) | 0.3247 | WORSE (+0.0220) |
| `ordinal` | 0.6563 | WORSE (-0.0718) | 0.0685 | better (-0.2342) |
| `deep_coral` | 0.6657 | WORSE (-0.0624) | 0.3557 | WORSE (+0.0531) |
| `mixstyle` | 0.6825 | WORSE (-0.0456) | 0.2963 | within noise (-0.0064, |diff| < 0.02) |
| `mixstyle_ordinal` | 0.6751 | WORSE (-0.0530) | 0.0725 | better (-0.2301) |
| `deep_coral_ordinal` | 0.6783 | WORSE (-0.0499) | 0.0585 | better (-0.2442) |
| `ordinal` | 0.6426 | WORSE (-0.0855) | 0.0552 | better (-0.2474) |
| `deep_coral` | 0.6925 | WORSE (-0.0357) | 0.2881 | within noise (-0.0145, |diff| < 0.02) |
| `mixstyle` | 0.6591 | WORSE (-0.0690) | 0.3385 | WORSE (+0.0358) |
| `mixstyle_ordinal` | 0.6414 | WORSE (-0.0867) | 0.0544 | better (-0.2483) |
| `deep_coral_ordinal` | 0.6843 | WORSE (-0.0438) | 0.0943 | better (-0.2084) |

*Noise floor: QWK differences below 0.02 and ECE differences below 0.02 are reported as 'within noise'. These thresholds come from the width of the bootstrap intervals on the smallest test split, not from a significance test.*

## 4. Computational cost

| Method | Params (M) | Epochs | Train (s) | Peak VRAM (GB) |
|---|---|---|---|---|
| `erm-none` | 6.96 | 20 | 1111.0 | 1.12 |
| `erm` | 6.96 | 19 | 766.5 | 2.12 |
| `deep_coral` | 6.96 | 20 | 1067.5 | 2.23 |
| `mixstyle` | 6.96 | 20 | 1222.4 | 2.42 |
| `ordinal` | 6.95 | 20 | 795.5 | 2.20 |
| `mixstyle_ordinal` | 6.95 | 20 | 1218.3 | 2.34 |
| `deep_coral_ordinal` | 6.95 | 20 | 1073.6 | 2.21 |
| `erm` | 6.96 | 20 | 1062.3 | 2.12 |
| `ordinal` | 6.95 | 20 | 1067.7 | 2.20 |
| `deep_coral` | 6.96 | 20 | 1081.0 | 2.29 |
| `mixstyle` | 6.96 | 20 | 1098.0 | 2.29 |
| `mixstyle_ordinal` | 6.95 | 20 | 917.2 | 2.34 |
| `deep_coral_ordinal` | 6.95 | 20 | 808.1 | 2.28 |
| `erm` | 6.96 | 20 | 788.6 | 2.23 |
| `ordinal` | 6.95 | 20 | 789.1 | 2.20 |
| `deep_coral` | 6.96 | 20 | 804.2 | 2.21 |
| `mixstyle` | 6.96 | 20 | 895.3 | 2.42 |
| `mixstyle_ordinal` | 6.95 | 20 | 901.0 | 2.42 |
| `deep_coral_ordinal` | 6.95 | 20 | 804.9 | 2.21 |

## 5. Not run

- **Target domains not evaluated:** ['aptos', 'ddr', 'eyepacs']. The leave-one-domain-out matrix is incomplete.
- **Backbones not trained:** ['convnext_tiny', 'dinov2_vits14'].

## 6. Standing limitations

These hold regardless of which experiments finish, and belong in the paper:

- **Patient-level splitting is possible only for EyePACS.** DDR, APTOS and IDRiD publish no patient identifiers, so same-patient images could span splits in those domains. Undetectable and unpreventable from the released data.
- **Label shift is confounded with covariate shift.** The domains differ in case mix as well as appearance (no-DR ranges from 32.9% in IDRiD to 73.5% in EyePACS), so a cross-domain drop reflects both. A DG method targeting covariate shift would not fix the prior shift.
- **EyePACS labels come from a corroborated secondary source**, not the official Kaggle file, and the domain is restricted to the 35,108 images whose labels were verified.
- **Duplicate removal discarded 33 groups with contradictory ground truth** (byte-identical images labelled differently). Those labels were unusable, but their existence suggests residual label noise in the remaining data.
- **Selective prediction is not evidence of clinical readiness.** Deferred cases still require a clinician, and the abstention threshold would have to be fixed prospectively.
