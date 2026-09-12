# Windows Application Control blocked PyTorch — **RESOLVED**

**Status: RESOLVED, 2026-09-12 06:30 AM (UTC+06:00).**

Kept as a historical record rather than deleted: the incident explains why one
session reported a partial test result, and the resolution is the evidence that
the partial result was environmental and not a code regression.

## What happened

Partway through the session of 2026-09-11, while the manuscript figure package
was being built, the Windows **Application Control / Smart App Control** policy
on this machine began blocking a PyTorch runtime library. Every import of
`torch` failed with:

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "D:\Research Code\.venv\Lib\site-packages\torch\lib\
torch_global_deps.dll" or one of its dependencies.
```

Nothing in the repository changed to cause this. The same test suite had run
**414/414 green earlier the same day**, and the failure appeared with no
intervening commit touching `torch`, the virtual environment, or any test.

### Effect on the test suite at the time

| | |
|---|---|
| test files that could not be **collected** | 9 |
| further individual tests that **failed** | 11 |
| occurrences of `WinError 4551` in the run | 11 — one per failure |
| tests that still passed (everything not importing torch) | 261 |

Every failure traced to that single DLL. The nine uncollectable files were
`test_checkpoint_policy`, `test_domain_balanced_batches`,
`test_domain_objectives`, `test_gradient_accumulation`, `test_phase3`,
`test_phase4`, `test_resume_best_state`, `test_retfound` and `test_rng_resume`
— all of which import torch at module level.

### What was *not* done

No test was skipped, weakened, deleted or marked `xfail` to obtain a clean
count. The session reported the partial result honestly and classified it as an
operating-system issue rather than working around it. No scientific artifact was
touched: the figure and table generators do not import torch, so the figure
package completed normally while torch was blocked.

## Resolution

The operator disabled the blocking application-control setting and asked for a
full re-verification.

### Re-test, 2026-09-12 06:27–06:33 AM

| | |
|---|---|
| commit tested | `1f5673e` ("Final figure package: five main, six supplementary, all QC'd") |
| working tree | clean before and after |

**The blocked DLL now loads.** Checked directly, before importing torch:

```
path   D:\Research Code\.venv\Lib\site-packages\torch\lib\torch_global_deps.dll
exists True
load   OK -- no Application Control OSError
```

**Environment as re-verified in a fresh process:**

| | |
|---|---|
| Python | 3.14.3 |
| platform | Windows-11-10.0.26200-SP0 |
| torch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| timm | 1.0.28 |
| torch CUDA | 12.8 |
| CUDA available | **True** |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, capability (12, 0), 1 device |
| live check | a 64×64 matmul executed on `cuda:0` |

No package was reinstalled. The imports were already correct; only the policy
was in the way.

### Full-suite result

```
438 passed, 26 warnings in 84.95s (0:01:24)
0 failed · 0 skipped · 0 xfailed · 0 errors
```

438 = the 414 that passed before the blocker + the 24 manuscript-figure tests
added afterwards. Nothing was lost and nothing needed classifying.

The 26 warnings are pre-existing and benign: a singular-axis warning from a
calibration figure drawn with one point, a `requires_grad` scalar-conversion
notice in the IRM penalty test, and 24 `PendingDeprecationWarning`s from
`colormap.set_bad`.

### Audits re-run at the same commit

| gate | result |
|---|---|
| `audit_consistency.py` | PASS — 3,090 checks, no disagreement |
| `audit_resumed_runs.py` | exit 0 — no reported model affected |
| `audit_experiment_identity.py` | exit 0 — 9 pre-existing conflicts, **0 involving any full-FT run** |
| `paper/check_numbers.py` | no unsupported value in the manuscript |
| `paper/check_report_numbers.py` | 4 documents, 0 failing |
| `paper/check_latex.py` | clean across all included files |
| `export_figure_provenance.py` | every figure has PNG and PDF; every declared source exists |
| `tests/test_manuscript_figures.py` | 24 passed |
| table regeneration | 6 authoritative tables byte-identical |
| figure regeneration | 23 files byte-identical |
| claim–evidence consistency | 12 claims, all cross-checks pass |

## No scientific result changed

Verified by hash, before the re-verification and after every generator had been
re-run:

| artifact | before | after |
|---|---|---|
| `experiment_registry.csv` | `63e81d615084812a…` | **unchanged** |
| prediction set (301 files) | `a660a01b20906146…` | **unchanged** |
| figure set (23 files) | `9b97e23066fe1217…` | **unchanged** |
| `JBHI_MASTER_RESULTS.csv` | `bbab954c7a24047b…` | **unchanged** |
| `JBHI_PRIMARY_INTERACTION.csv` | `ffd5937e23b8e3c9…` | **unchanged** |
| `final_claim_evidence.csv` | `a9a46d1699f0a241…` | **unchanged** |
| `two_domain_interaction_holm.csv` | `872ae44294f63000…` | **unchanged** |
| `aptos_full_finetune_primary.csv` | `b565f10a7b59f7ce…` | **unchanged** |
| `full_finetune_primary.csv` | `537b8d1e47242086…` | **unchanged** |

`git status` reported **no modified files** after the entire verification pass,
including after re-running every table and figure generator. Regenerating the
whole analysis from the frozen predictions reproduces the committed bytes
exactly.

Nothing was trained, no experiment was re-run, no prediction was rewritten, no
statistic was recomputed differently, and `paper/main.tex` was not touched. The
evidence freeze in `JBHI_EVIDENCE_FREEZE.md` stands unaltered.

## If it recurs

The signature is unmistakable: `OSError: [WinError 4551]` naming
`torch_global_deps.dll`. It is a host policy decision, not a Python or CUDA
fault, so reinstalling PyTorch will not help. Check Windows Security →
App & browser control → **Smart App Control**, and any managed WDAC policy.
Confirm the fix the same way this one was confirmed — load the DLL directly with
`ctypes.CDLL` before importing torch, which separates a policy block from a
genuine dependency problem.
