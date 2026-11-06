#!/bin/bash
#==============================================================================
# R41 Stage 1 board batch on the B40 image (a88ebdf5, 90 MHz), FPGA host only.
#
#   board_queue.sh COMMIT
#
# The user authorized loading the B40 image while the shared U280 is idle
# (2026-10-21). "Idle" here is stricter than program.sh's own gate: kdsstat
# must show no context, no outstanding exec and no client for 15 consecutive
# minutes. Another user's back-to-back runs leave seconds-long gaps, and
# loading in one of them would break their sequence.
#
# Then:
#   1. unpack the git archive of COMMIT (the frozen Stage 1 tree) into a fresh
#      source directory;
#   2. load the image with the archive's own flow/fpga/program.sh: --check,
#      then --go, which detaches the loader;
#   3. wait for the loaded UUID 1bada9db... and a 90 MHz clock counter;
#   4. run the registered manifests in order through experiment.py, each into
#      its own output directory.
#
# Every fpga step probes the UUID, the clock and kdsstat itself and refuses
# when the card changed. If one does, this queue stops and leaves the rest for
# a later run; experiment.py resumes exactly. It never retries a failed load.
#==============================================================================
set -u
C=${1:?commit}
D=$HOME/r41s1_board_$C
A=$HOME/r41s1_$C.tar.gz
X=$HOME/tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin
UUID=1bada9db-78f3-5bc8-7ba6-cce062e7ade8
K=/sys/bus/pci/devices/0000:04:00.1/kdsstat
PY=$HOME/tc_env/torchenv/bin/python
L=$D/queue.log
mkdir -p $D
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
field() { awk -F: -v f="$1" '$1 ~ f {gsub(/[ \t]/,"",$2); print $2; exit}' $K; }
client() { awk 'f{print $1; exit} /client pid/{f=1}' $K; }
idle() { [ "$(field contexts)" = "0" ] && [ "$(field outstanding)" = "0" ] && [ -z "$(client)" ]; }

log "queue start commit $C archive $(sha256sum $A | cut -c1-64)"
if [ "$(cat /sys/bus/pci/devices/0000:04:00.1/xclbinuuid)" != "$UUID" ]; then
    n=0
    while [ $n -lt 15 ]; do
        if idle; then n=$((n+1)); else n=0; fi
        sleep 60
    done
    log "card idle for 15 consecutive minutes; loaded $(cat /sys/bus/pci/devices/0000:04:00.1/xclbinuuid)"
fi

if [ ! -d $D/src ]; then
    mkdir -p $D/src && tar xzf $A -C $D/src
fi
cd $D/src
if [ "$(cat /sys/bus/pci/devices/0000:04:00.1/xclbinuuid)" != "$UUID" ]; then
    bash flow/fpga/program.sh --check $X >> $L 2>&1 || { log "program.sh --check refused; nothing loaded"; exit 3; }
    bash flow/fpga/program.sh --go $X >> $L 2>&1 || { log "program.sh --go failed"; exit 4; }
    for i in $(seq 1 60); do
        [ "$(cat /sys/bus/pci/devices/0000:04:00.1/xclbinuuid)" = "$UUID" ] && break
        sleep 10
    done
    bash flow/fpga/program.sh --status $X >> $L 2>&1
fi
u=$(cat /sys/bus/pci/devices/0000:04:00.1/xclbinuuid)
mhz=$(cat /sys/bus/pci/devices/0000:04:00.0/clock_freq_counter.m.28311552/clock_counter_freqs 2>/dev/null | head -1)
log "loaded uuid $u clock counter $mhz MHz"
[ "$u" = "$UUID" ] && [ "$mhz" = "90" ] || { log "image or clock not as required; stopping"; exit 5; }

for m in anchors original-suite gemm-tile prospective memory-service pipeline-admission tune-qualification; do
    log "manifest board-$m start"
    $PY tools/performance_harness/experiment.py spec/r41/stage1/experiments/board-$m.json \
        --root $D/src --out $D/out/board-$m --python $PY >> $D/board-$m.log 2>&1
    rc=$?
    log "manifest board-$m exit $rc"
    if [ $rc -ne 0 ]; then
        u=$(cat /sys/bus/pci/devices/0000:04:00.1/xclbinuuid)
        if [ "$u" != "$UUID" ] || ! idle; then
            log "card changed or busy after board-$m (uuid $u); stopping queue"; exit 6
        fi
    fi
done
log BOARD-QUEUE-DONE
