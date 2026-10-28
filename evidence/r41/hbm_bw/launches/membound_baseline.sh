#!/bin/sh
# Baseline of the registered R41 memory-bound suite on the run-1 100 MHz image.
# Lead-only launcher, FPGA host acf3019 only. It never programs, resets or opens
# the shared U280 itself: each slice goes through
# tools/performance_harness/launches/acceptance_queue.py, which admits only the
# already-approved run-1 image (sha256 30e79769..., UUID 49964b17-...), waits for
# 900 consecutive idle seconds before every slice, and refuses a card that holds
# another image or another user's context.
#
#   membound_baseline.sh IMMUTABLE_SOURCE_ROOT OUTPUT_ROOT
#
# IMMUTABLE_SOURCE_ROOT is a `git archive` of the registering commit. The
# interpreter is the torch environment's python, named by its symlink path and
# never resolved (resolving it selects the system python without torch).
# Slices run in order; a slice whose batch records failures keeps them
# (board_batch.py preserves every failed run) and the next slice still runs.
# A slice that is refused before it starts (identity, card in use) stops the
# launcher. A rerun resumes exactly: completed runs are reused, failed runs are
# rerun only by an explicit --retry-failed, never by this script.
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
# Every generated slice (tools/r41/membound_manifests.py), small, medium, large.
SLICES=$(for z in small medium large; do ls spec/r41/hbm_bw/membound/membound-$z-*.json; done | sed 's#.*/##; s#\.json$##')
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
# Before detaching: refuse when a slice's registration does not parse.
if [ "$DETACHED" = 0 ]; then
    for s in $SLICES; do
        "$PY" tools/performance_harness/launches/acceptance_queue.py --root "$ROOT" --out "$OUT/$s" \
            --python "$PY" --check-only --manifest "spec/r41/hbm_bw/membound/$s.json" > /dev/null
    done
    setsid nohup sh "$ROOT/evidence/r41/hbm_bw/launches/membound_baseline.sh" --detached "$ROOT" "$OUT" \
        > "$OUT/membound_baseline.log" 2>&1 < /dev/null &
    exit 0
fi
run() {
    for s in $SLICES; do
        echo "$(date -u +%FT%TZ) slice $s start"
        rc=0
        "$PY" tools/performance_harness/launches/acceptance_queue.py --root "$ROOT" --out "$OUT/$s" \
            --python "$PY" --budget-seconds 10800 --manifest "spec/r41/hbm_bw/membound/$s.json" || rc=$?
        echo "$(date -u +%FT%TZ) slice $s exit $rc"
        if [ "$rc" -ne 0 ] && [ ! -f "$OUT/$s/$s/report.json" ]; then
            echo "$(date -u +%FT%TZ) slice $s was refused before it ran; stopping"
            return 1
        fi
    done
    echo "$(date -u +%FT%TZ) MEMBOUND-BASELINE-DONE (commands completed; per-point results in each batch.json)"
}
run
