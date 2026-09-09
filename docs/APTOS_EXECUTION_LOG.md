# APTOS full fine-tuning — execution record

The pre-registered protocol (`APTOS_FULL_FINETUNE_REPLICATION_PROTOCOL.md`)
requires that any re-run be recorded with its reason. This is that record. It
covers **execution only**: which processes ran, when, and why any of them was
abandoned.

**No APTOS target metric appears here, and none has been inspected.** Every
number below is a wall-clock time, an epoch count, or a source-validation
budget. Target blindness remains in force until all ten runs exist.

All times are local (UTC+06:00). Registry timestamps are UTC and are converted
here.

## Restart from scratch, never resume

A resumed run is **not** equivalent to an uninterrupted one in this codebase.
`src/data/loaders.py:166-167` builds a `torch.Generator()` seeded from
`loader_config.seed` and never checkpoints it, so a resumed run replays epoch
0's shuffle ordering rather than continuing the sequence. Separately, the
learning-rate scheduler was not restored across a resume until it was fixed
(see `audit_resumed_runs.py`), which silently restarted the cosine.

So every interrupted execution below was **discarded in full** and restarted
from scratch, with its partial checkpoints deleted so that `last.pt` could not
be picked up. Verified: there is no `resuming ... aptos ... ftfull` line in any
log, and `audit_resumed_runs.py` reports the one multi-execution case
(RETFound seed 42) as *"final execution ran from scratch; 2 earlier
execution(s) superseded"*.

## Runs

Order is the pre-registered one: 42 ImageNet, 42 RETFound, 1 ImageNet,
1 RETFound, 2 ImageNet, 2 RETFound, 3 ImageNet, 3 RETFound, 4 ImageNet,
4 RETFound. Never concurrent.

| # | model | seed | status | registered (local) | epochs | train time |
|---|---|---|---|---|---|---|
| 1 | ImageNet-MAE | 42 | COMPLETE | 2026-09-06 04:58 | 20/20 | 5.81 h |
| 2 | RETFound | 42 | COMPLETE | 2026-09-07 03:12 | 14/20 (early stop) | 4.06 h |
| 3 | ImageNet-MAE | 1 | COMPLETE | 2026-09-07 09:02 | 20/20 | 5.77 h |
| 4 | RETFound | 1 | COMPLETE | 2026-09-08 06:58 | 19/20 (early stop) | 5.51 h |
| 5 | ImageNet-MAE | 2 | COMPLETE | 2026-09-09 04:10 | 20/20 | 5.85 h |
| 6 | RETFound | 2 | COMPLETE | 2026-09-09 23:32 | 15/20 (early stop) | 4.37 h |
| 7 | ImageNet-MAE | 3 | RUNNING | started 2026-09-09 23:33 | — | — |
| 8 | RETFound | 3 | NOT RUN | | | |
| 9 | ImageNet-MAE | 4 | NOT RUN | | | |
| 10 | RETFound | 4 | NOT RUN | | | |

Early stopping is on **source**-validation QWK with patience 6. Runs 2, 4 and 6
stopping at 14, 19 and 15 epochs is the recipe behaving as specified, not an
intervention: all three are RETFound runs, and RETFound early-stopped in all
five DDR runs while ImageNet-MAE early-stopped in none and has now run the full
budget in all three APTOS runs too. Nothing about the budget,
the patience or the schedule was changed for either.

## Abandoned executions

Each was discarded whole. None contributed to any registered result; each log
is retained under a name recording its fate.

| log | epochs reached | ended | cause |
|---|---|---|---|
| `ft_aptos_retfound_s42_interrupted_pause.log` | 14 | 2026-09-06 09:04 | stopped at an operator pause |
| `ft_aptos_retfound_s42_hung_standby_2307.log` | 8 | 2026-09-06 21:52 | Modern Standby deadlock |
| `ft_aptos_retfound_s1_stopped_at_pause.log` | 0 | 2026-09-07 09:05 | stopped at an operator pause |
| `ft_aptos_retfound_s1_hung_standby_2042.log` | 4 | 2026-09-07 20:42 | Modern Standby deadlock |
| `ft_aptos_imagenet_s2_stopped_at_pause.log` | 7 | 2026-09-08 09:02 | stopped at an operator pause |
| `ft_aptos_retfound_s2_stopped_at_pause.log` | 15 | 2026-09-09 08:34 | stopped at an operator pause |

**No execution was ever abandoned because of what it showed.** Four were
operator pauses and two were the same platform fault. The protocol's rule
stands: a run is re-run only for a demonstrated implementation fault or an
interruption, never because a result is inconvenient.

## The Modern Standby deadlock

Twice the machine entered Modern Standby mid-epoch and the run did not resume.
It **deadlocks** rather than failing: the main process spins on exactly one
core, the dataloader workers stop accumulating CPU entirely, VRAM stays
allocated and the GPU reports 0% utilisation. Nothing exits, nothing raises,
and no further line is written, so the failure is indistinguishable from a slow
epoch unless the log's age, the held VRAM and the GPU utilisation are read
together.

Measured on the seed-1 occurrence, after the log had been frozen for over four
and a half hours: the main process accumulated 7.97 s of CPU in an 8 s window
(one full core) while both worker processes accumulated 0.00 s. That is the
signature.

That occurrence cost 4 h 45 min of wall clock — the log's last line is at
20:42:02 and the replacement run started at 01:26:36 — on top of the four
epochs that had to be thrown away.

### Mitigation

`keep_awake.py` holds `ES_SYSTEM_REQUIRED` (with away mode where the power
configuration grants it) for the lifetime of a nominated process, and is
started alongside each training run. It deliberately changes **no** system
setting: the request lives in that process and disappears when it exits, so
nothing has to be undone afterwards and the machine's power policy is not
quietly altered beyond the experiment. The display is still allowed to sleep.

Detection is independent of prevention: a monitor watches for a log frozen more
than 1800 s while more than 1000 MiB of VRAM is held and the GPU is under 20%,
which is the conjunction that distinguishes this deadlock from a long epoch.
It is what caught the seed-1 occurrence.

## A restarted run is not bit-identical to the run it replaces

The seed-1 RETFound restart gave a direct measurement of this, because the
abandoned execution and its replacement ran the identical command, at the
identical seed, both from scratch. Their first epochs differ:

| execution | epoch 0 train loss | epoch 0 val loss | epoch 0 val QWK |
|---|---|---|---|
| abandoned (standby deadlock) | 0.7844 | 0.7055 | 0.6055 |
| replacement | 0.7856 | 0.6955 | 0.6244 |

(Source-validation figures. The target is not involved.)

This is expected and is recorded per run: `run_lodo.py:383` sets
`deterministic: False`, so `src/utils/seed.py` leaves
`torch.backends.cudnn.benchmark = True` and deterministic algorithms off. cuDNN
autotunes kernel selection, and with AMP and gradient checkpointing the
reduction order is not fixed. The registry carries `deterministic = False` on
every full-FT row, so the runs do not claim bit-exact reproducibility and none
is implied anywhere.

What it means, stated plainly:

- **"Seed 1" is not a bit-reproducible label.** Re-running the same command
  gives a nearby but different draw. Reproducing this study reproduces the
  *distribution* over seeds, not the individual numbers.
- **It does not bias the comparison.** Both initialisations run under the same
  setting, on the same seeds, on the same images, and the analysis is paired at
  seed level within each protocol — so kernel non-determinism enters both arms
  identically and cannot produce a difference between them.
- **It does inflate the seed-level SD slightly.** The five-seed spread contains
  both genuine seed-to-seed variation and this run-to-run jitter. Since the
  inference is a paired test on per-seed differences, that makes the intervals
  marginally wider, which is conservative rather than favourable.

## Storage

Full fine-tuning writes a 1.21 GB `best_qwk.pt` and a 3.64 GB `last.pt` per
run. `last.pt` is pruned after each run completes and every safety condition in
`prune_checkpoints.py` passes; `best_qwk.pt` is retained. Free space is checked
before each launch.
