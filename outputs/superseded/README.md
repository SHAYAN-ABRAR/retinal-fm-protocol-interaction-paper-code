# Superseded run artifacts

Each subdirectory holds the predictions, history and evaluation report of a run
that was **executed under a known implementation defect** and has been replaced
by a corrected re-run under the same experiment id.

They are kept, not deleted, because the registry is an append-only log and the
provenance of a corrected result includes what it corrected. The registry row
for the original execution also remains.

**Nothing here may be used as a result.** The authoritative artifacts for these
experiment ids are the ones under `outputs/predictions/`, `outputs/logs/` and
`outputs/reports/`, produced by the corrected re-run.

## Contents

### lodo_aptos-ddr-eyepacs__idrid_vit-large-mae-in1k_erm-b16-tb4-lr0.0001_s3

Trained under the scheduler-resume bug (fixed in `92c11e5`): the run was
interrupted at epoch 2 of 20, and because `Trainer.resume()` could not restore
the scheduler, the learning-rate schedule restarted from step zero. It re-ran
warmup and then followed a cosine two epochs behind the intended one. The
selected checkpoint was epoch 11, which falls after the resume, so the reported
model was trained on the wrong schedule.

Detected by `audit_resumed_runs.py`, which reconstructs each run's intended
cosine from its own config and compares it epoch by epoch against the learning
rate actually used.
