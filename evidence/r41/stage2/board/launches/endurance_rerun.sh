#!/bin/bash
#==============================================================================
# R41 Stage 2 endurance rerun on the run-1 100 MHz image, FPGA host only.
#
#   S2_LOAD_APPROVED=1 endurance_rerun.sh COMMIT      (COMMIT: the r41-board-fix commit;
#                                                      archive ~/r41s2_board_COMMIT.tar.gz,
#                                                      a `git archive --format=tar.gz` of it)
#
# Why: the first s2-endurance run (r41s2_board_86a3d929cf3d) failed 250 of 250
# sessions. endurance_mixed.py called Batch.run_one without the binding that
# Batch.execute makes, so the clock evidence digest in the batch state was None
# and every record was rejected as 'qualified clock evidence identity'; the 10
# FP8 sessions failed because the harness created the output directory that
# fp8_cases.py refuses to find. Both are fixed on r41-board-fix. See
# evidence/r41/stage2/board/results.md.
#
# Rules (the same as board_queue.sh):
#   - the card must show no context, no outstanding exec and no client for
#     15 consecutive minutes first (S2_IDLE_MINUTES);
#   - the image is loaded only with S2_LOAD_APPROVED=1 (the user's separate
#     approval), through program.sh --check, then --go; never over anyone
#     else's run, never retried, never reset;
#   - the qualified clock evidence of the first Stage 2 run is reused (same
#     image, same undivided clock); it must exist with the registered digest.
#     The loaded UUID and a 100 MHz counter reading are required before the run
#     and are probed again before every session by endurance_mixed.py;
#   - only this script's own children are ever stopped.
# Only s2-endurance is run, into its own directory.
#==============================================================================
set -u
C=${1:?commit}
H=${S2_HOME:-$HOME}
D=$H/r41s2_endurance_$C
A=$H/r41s2_board_$C.tar.gz
TAG=r41s2_run1_c9f25787494a
X=$H/tnca_images/$TAG/tnca_u280.xclbin
XSHA=${S2_TEST_XSHA:-30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de}
UUID=49964b17-e0b4-5bd2-dd40-7fc37b1205d3
QD=$H/tnca_build/$TAG/r41s2-board
CLKSHA=${S2_TEST_CLKSHA:-49868f45b5ced2f662e4f76bbb1bb4a2be166625915d869f20812afcee42f0d7}
SYSROOT=${S2_SYSFS:-/sys/bus/pci/devices}
SYS=$SYSROOT/0000:04:00.1
K=$SYS/kdsstat
U=$SYS/xclbinuuid
F=$SYSROOT/0000:04:00.0/clock_freq_counter.m.28311552/clock_counter_freqs
IDLE_MIN=${S2_IDLE_MINUTES:-15}
POLL=${S2_POLL_SECONDS:-60}
LOAD_WAIT=${S2_LOAD_WAIT_SECONDS:-600}
PY=${S2_PYTHON:-$H/tc_env/torchenv/bin/python}
MANIFEST=s2-endurance
L=$D.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
field() { awk -F: -v f="$1" '$1 ~ f {gsub(/[ \t]/,"",$2); print $2; exit}' $K; }
client() { awk 'f{print $1; exit} /client pid/{f=1}' $K; }
idle() { [ "$(field contexts)" = "0" ] && [ "$(field outstanding)" = "0" ] && [ -z "$(client)" ]; }
uuid() { tr -d '[:space:]' < $U 2>/dev/null; }
mhz() { head -1 $F 2>/dev/null | tr -d '[:space:]'; }
identity_ok() { [ "$(uuid)" = "$UUID" ] && [ "$(mhz)" = "100" ]; }
wait_idle() {
    local n=0
    while [ $n -lt $IDLE_MIN ]; do
        if idle; then n=$((n+1)); else n=0; fi
        sleep $POLL
    done
}
command -v xclbinutil > /dev/null || source ${S2_XRT_SETUP:-/opt/xilinx/xrt/setup.sh} > /dev/null 2>&1

log "endurance rerun start commit $C archive $(sha256sum $A 2>/dev/null | cut -c1-64)"
[ -f $A ] || { log "archive $A missing; nothing run"; exit 3; }
[ "$(sha256sum $X | cut -c1-64)" = "$XSHA" ] || { log "image $X is not $XSHA; nothing run"; exit 3; }
[ -f $QD/qualified_clock.json ] && [ "$(sha256sum $QD/qualified_clock.json | cut -c1-64)" = "$CLKSHA" ] \
    || { log "qualified clock evidence missing or not $CLKSHA; nothing run"; exit 3; }
if [ -e $D/out ]; then log "$D/out exists; refusing to mix with an earlier attempt"; exit 3; fi

# 0. preparation (no card access)
if [ ! -d $D/src ]; then
    mkdir -p $D/src.tmp && tar xzf $A -C $D/src.tmp && mv $D/src.tmp $D/src
fi
cd $D/src
PROGRAM=${S2_PROGRAM:-$D/src/flow/fpga/program.sh}

# 1. the idle rule
wait_idle
log "card idle for $IDLE_MIN consecutive minutes; loaded $(uuid), clock $(mhz) MHz"

# 2. load, only on approval
if [ "$(uuid)" != "$UUID" ]; then
    [ "${S2_LOAD_APPROVED:-}" = 1 ] || { log "image not loaded and S2_LOAD_APPROVED is not 1; nothing loaded"; exit 4; }
    idle || { log "card busy at load time; nothing loaded"; exit 4; }
    bash $PROGRAM --check $X >> $L 2>&1 || { log "program.sh --check refused; nothing loaded"; exit 4; }
    bash $PROGRAM --go $X >> $L 2>&1 || { log "program.sh --go failed"; exit 4; }
    end=$((SECONDS + LOAD_WAIT))
    while [ "$(uuid)" != "$UUID" ] && [ $SECONDS -lt $end ]; do sleep 10; done
    bash $PROGRAM --status $X >> $L 2>&1
fi
log "loaded uuid $(uuid) clock counter $(mhz) MHz"
identity_ok || { log "image or clock not as required (want $UUID at 100 MHz); stopping"; exit 5; }

# 3. the run
idle || { log "card busy before $MANIFEST; stopping"; exit 6; }
log "manifest $MANIFEST start"
$PY tools/performance_harness/experiment.py spec/r41/stage2/board/$MANIFEST.json \
    --root $D/src --out $D/out/$MANIFEST --python $PY >> $D/$MANIFEST.log 2>&1
rc=$?
log "manifest $MANIFEST exit $rc status $(tail -1 $D/$MANIFEST.log)"
log "ENDURANCE-RERUN-DONE exit $rc"
exit $rc
