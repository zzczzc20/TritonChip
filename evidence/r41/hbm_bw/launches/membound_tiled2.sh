#!/bin/sh
# Non-default TNCA_TILED_POINTWISE=2 gate on the run-1 100 MHz image.
# Lead-only launcher, FPGA host acf3019 only. It never programs, resets or opens
# the shared U280 itself: each experiment goes through
# tools/performance_harness/launches/acceptance_queue.py, which admits only the
# already-approved run-1 image (sha256 30e79769..., UUID 49964b17-...), waits for
# 900 consecutive idle seconds before every experiment, and refuses a card that
# holds another image or another user's context.
#
#   membound_tiled2.sh IMMUTABLE_SOURCE_ROOT OUTPUT_ROOT
#
# IMMUTABLE_SOURCE_ROOT is a `git archive` of the registering commit. The
# interpreter is the torch environment's python, named by its symlink path and
# never resolved (resolving it selects the system python without torch).
# Slices run in order, followed by the original 113-point non-regression suite.
# A slice whose batch records failures keeps them (board_batch.py preserves
# every failed run) and the next experiment still runs. An experiment refused
# before it starts (identity, card in use) stops the launcher. A rerun resumes
# exactly: completed runs are reused, failed runs are rerun only by an explicit
# --retry-failed, never by this script. These are default-change-gate data, not
# bandwidth-target results.
set -eu
if [ "${1:-}" = "--detached" ]; then DETACHED=1; shift; else DETACHED=0; fi
if [ "$#" -ne 2 ]; then
    echo "usage: $0 IMMUTABLE_SOURCE_ROOT OUTPUT_ROOT" >&2
    exit 2
fi
ROOT=$(cd "$1" && pwd)
OUT=$2
PY="$HOME/tc_env/torchenv/bin/python"
cd "$ROOT"
SLICES=$(for z in small medium large; do ls spec/r41/hbm_bw/membound_tiled2/membound-$z-*.json; done | sed 's#.*/##; s#\.json$##')
EXPERIMENTS="$SLICES s2-original-suite"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
if [ "$DETACHED" = 0 ]; then
    for s in $EXPERIMENTS; do
        "$PY" tools/performance_harness/launches/acceptance_queue.py --root "$ROOT" --out "$OUT/$s" \
            --python "$PY" --check-only --manifest "spec/r41/hbm_bw/membound_tiled2/$s.json" > /dev/null
    done
    setsid nohup sh "$ROOT/evidence/r41/hbm_bw/launches/membound_tiled2.sh" --detached "$ROOT" "$OUT" \
        > "$OUT/membound_tiled2.log" 2>&1 < /dev/null &
    exit 0
fi
run() {
    for s in $EXPERIMENTS; do
        echo "$(date -u +%FT%TZ) experiment $s start"
        rc=0
        "$PY" tools/performance_harness/launches/acceptance_queue.py --root "$ROOT" --out "$OUT/$s" \
            --python "$PY" --budget-seconds 10800 --manifest "spec/r41/hbm_bw/membound_tiled2/$s.json" || rc=$?
        echo "$(date -u +%FT%TZ) experiment $s exit $rc"
        if [ "$rc" -ne 0 ] && [ ! -f "$OUT/$s/$s/report.json" ]; then
            echo "$(date -u +%FT%TZ) experiment $s was refused before it ran; stopping"
            return 1
        fi
    done
    echo "$(date -u +%FT%TZ) MEMBOUND-TILED2-DONE (commands completed; per-point results in each batch.json)"
}
run
