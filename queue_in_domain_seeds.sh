#!/usr/bin/env bash
# Run the in-domain experiment at seeds 1 and 2, once the GPU is free.
#
# Why this exists
# ---------------
# Every in-domain run so far is seed 42 only, while the LODO side has three
# seeds. The project's headline clinical claim -- cross-domain deployment costs
# -0.143 QWK on DDR and -0.294 on EyePACS, and more than doubles severe errors
# -- therefore differences a three-seed LODO mean against a ONE-seed in-domain
# point. analyse_lodo_seeds.py already prints that caveat. This closes it.
#
# Cost: ~49 min of GPU per seed (DDR 620s + APTOS 235s + IDRiD 48s +
# EyePACS 2026s), so ~1.6 h for both.
#
# Settings are left at the module defaults on purpose: densenet121, 224 px,
# batch 32, 20 epochs, lr 3e-4, wd 1e-4 -- byte-identical to seed 42's
# registry row. A seed that differs in any other setting is not a second seed,
# it is a different experiment, and averaging the two would be wrong.
#
# The GPU has 8 GB and the 512 px LODO job holds ~5.3 GB of it, so this waits
# for that process to exit rather than competing with it.
#
# Usage:
#     bash queue_in_domain_seeds.sh <winpid-to-wait-for>
#     bash queue_in_domain_seeds.sh 21896

set -u

REPO="/d/Research Code/dr_domain_generalization"
PYTHON="$REPO/.venv/Scripts/python.exe"
LOG="$REPO/outputs/logs/in_domain_seeds_1_2.log"
WAIT_PID="${1:-}"
POLL_SECONDS=120
SETTLE_SECONDS=90

cd "$REPO" || exit 1

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "queued: in-domain seeds 1 and 2, densenet121 / 224px / batch 32 / 20 epochs"

if [ -n "$WAIT_PID" ]; then
  say "waiting for PID $WAIT_PID (the 512px LODO run) to finish"
  while tasklist //FI "PID eq $WAIT_PID" 2>/dev/null | grep -q "^python"; do
    sleep "$POLL_SECONDS"
  done
  say "PID $WAIT_PID has exited"
  # CUDA does not release VRAM the instant the process dies, and starting into
  # a partly-freed card is how a run OOMs on epoch 0 after queueing all night.
  say "letting the GPU settle for ${SETTLE_SECONDS}s"
  sleep "$SETTLE_SECONDS"
fi

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
say "GPU free memory: ${FREE:-unknown} MiB"
if [ -n "${FREE:-}" ] && [ "$FREE" -lt 5000 ]; then
  say "ABORT: fewer than 5000 MiB free; something else is using the GPU."
  say "       Nothing has been run. Re-launch this script when the card is idle."
  exit 1
fi

STATUS=0
for SEED in 1 2; do
  say "=== in-domain seed $SEED starting ==="
  "$PYTHON" run_in_domain.py --seed "$SEED" >>"$LOG" 2>&1
  CODE=$?
  if [ "$CODE" -ne 0 ]; then
    say "!! seed $SEED exited $CODE -- continuing to the next seed"
    STATUS=1
  else
    say "=== in-domain seed $SEED complete ==="
  fi
done

say "re-running the consistency audit"
"$PYTHON" audit_consistency.py >>"$LOG" 2>&1 || {
  say "!! audit reported a disagreement -- read the tail of $LOG before using these results"
  STATUS=1
}

say "queue finished (status $STATUS)"
say "next: python analyse_lodo_seeds.py   # the deployment cost now has a 3-seed reference on both sides"
exit "$STATUS"
