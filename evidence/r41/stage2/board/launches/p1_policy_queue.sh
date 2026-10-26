#!/bin/bash
#==============================================================================
# R41 Stage 2 P1 policy A/B queue on the run-1 100 MHz image, FPGA host only.
#
#   p1_policy_queue.sh COMMIT OUTDIR
#
# Enforces the 15-minute idle rule, verifies the approved 100 MHz run-1 image,
# and runs spec/r41/stage2/board/s2-policy-ab.json via acceptance_queue.py.
# Never programs or resets the card. Never starts unapproved images.
#==============================================================================
set -euo pipefail

C=${1:?commit}
OUTDIR=${2:-$HOME/r41_accept_p1_$C}
ROOT=${S2_ROOT:-$HOME/r41_accept_$C}
PY=${S2_PYTHON:-$HOME/tc_env/torchenv/bin/python}

echo "=== R41 Stage 2 P1 Policy Queue ==="
echo "Commit: $C"
echo "Root: $ROOT"
echo "Out: $OUTDIR"

exec "$PY" "$ROOT/tools/performance_harness/launches/acceptance_queue.py" \
    --root "$ROOT" \
    --out "$OUTDIR" \
    --python "$PY" \
    --manifest "$ROOT/spec/r41/stage2/board/s2-policy-ab.json"
