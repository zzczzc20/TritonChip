#!/bin/bash
#==============================================================================
# R41 Stage 2 P2 locality/pipeline queue on run-1 100 MHz image, FPGA host only.
#
#   p2_locality_queue.sh COMMIT OUTDIR
#
# Enforces the 15-minute idle rule, checks the approved 100 MHz run-1 image,
# and runs spec/r41/stage2/board/s2-locality.json via acceptance_queue.py.
# Never programs or resets the card. Never starts unapproved images.
#==============================================================================
set -euo pipefail

C=${1:?commit}
OUTDIR=${2:-$HOME/r41_accept_p2_$C}
ROOT=${S2_ROOT:-$HOME/r41_accept_$C}
PY=${S2_PYTHON:-$HOME/tc_env/torchenv/bin/python}

echo "=== R41 Stage 2 P2 Locality Queue ==="
echo "Commit: $C"
echo "Root: $ROOT"
echo "Out: $OUTDIR"

exec "$PY" "$ROOT/tools/performance_harness/launches/acceptance_queue.py" \
    --root "$ROOT" \
    --out "$OUTDIR" \
    --python "$PY" \
    --manifest "$ROOT/spec/r41/stage2/board/s2-locality.json"
