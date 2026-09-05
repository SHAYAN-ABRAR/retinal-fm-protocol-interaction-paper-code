# Partial fine-tuning checkpoint retention

**30 checkpoints, 48.50 GB.** Inventory: `PARTIAL_FT_CHECKPOINT_INVENTORY.csv` (one row per file, with SHA256).

## Why deletion is safe here

Every statistical analysis in this project loads **predictions**, not
checkpoints. A run whose predictions, history, config, report and
registry row are intact stays fully reproducible *as a result* once its
weights are gone; only deriving **new** predictions would need a re-run.
Each row below is deleted only after all five of those artifacts are
confirmed present and the saved predictions are re-scored and matched
against the registry's recorded QWK.

## Policy

- **RETAIN** seed 42 for **both** model families in
  **every** target domain -- a matched representative pair per domain, so
  the retained set cannot favour one family.
- **RETAIN** any run whose prediction-level artifacts are incomplete,
  regardless of seed.
- **RETAIN** the corrected replacement for a corrupted run until the
  publication artifact bundle is frozen.
- **DELETE** model weights only. Never predictions, histories, configs,
  registry rows, reports, metric files or provenance.

## Retained

| target | model | seed | status | reason |
|---|---|---|---|---|
| aptos | ImageNet-MAE | 42 | COMPLETE | matched representative pair for this target |
| aptos | RETFound-CFP | 42 | COMPLETE | matched representative pair for this target |
| ddr | ImageNet-MAE | 42 | COMPLETE | matched representative pair for this target |
| ddr | RETFound-CFP | 42 | COMPLETE | matched representative pair for this target |
| idrid | ImageNet-MAE | 42 | COMPLETE | matched representative pair for this target |
| idrid | RETFound-CFP | 42 | COMPLETE | matched representative pair for this target |

**Symmetric across model families: yes**

## Deleted (weights only)

| target | model | seed | classification | prediction audit |
|---|---|---|---|---|
| aptos | ImageNet-MAE | 1 | authoritative | qwk matches (0.822032) |
| aptos | ImageNet-MAE | 2 | authoritative | qwk matches (0.844928) |
| aptos | ImageNet-MAE | 3 | authoritative | qwk matches (0.809081) |
| aptos | ImageNet-MAE | 4 | authoritative | qwk matches (0.846777) |
| aptos | RETFound-CFP | 1 | authoritative | qwk matches (0.816131) |
| aptos | RETFound-CFP | 2 | authoritative | qwk matches (0.819338) |
| aptos | RETFound-CFP | 3 | authoritative | qwk matches (0.829778) |
| aptos | RETFound-CFP | 4 | authoritative | qwk matches (0.836525) |
| ddr | ImageNet-MAE | 1 | authoritative | qwk matches (0.712553) |
| ddr | ImageNet-MAE | 2 | authoritative | qwk matches (0.726434) |
| ddr | ImageNet-MAE | 3 | authoritative | qwk matches (0.697868) |
| ddr | ImageNet-MAE | 4 | authoritative | qwk matches (0.721936) |
| ddr | RETFound-CFP | 1 | authoritative | qwk matches (0.716139) |
| ddr | RETFound-CFP | 2 | authoritative | qwk matches (0.675846) |
| ddr | RETFound-CFP | 3 | authoritative | qwk matches (0.702665) |
| ddr | RETFound-CFP | 4 | authoritative | qwk matches (0.702073) |
| idrid | ImageNet-MAE | 1 | authoritative | qwk matches (0.733983) |
| idrid | ImageNet-MAE | 2 | authoritative | qwk matches (0.744088) |
| idrid | ImageNet-MAE | 3 | corrupted | qwk matches (0.730166) |
| idrid | ImageNet-MAE | 4 | authoritative | qwk matches (0.710530) |
| idrid | RETFound-CFP | 1 | authoritative | qwk matches (0.771716) |
| idrid | RETFound-CFP | 2 | authoritative | qwk matches (0.767169) |
| idrid | RETFound-CFP | 3 | authoritative | qwk matches (0.761797) |
| idrid | RETFound-CFP | 4 | authoritative | qwk matches (0.722880) |

## Space

| | |
|---|---|
| checkpoints retained | 6 (9.70 GB) |
| checkpoints deleted | 24 (38.80 GB) |
| free before | 17.9 GB |
| free after (projected) | 56.7 GB |

