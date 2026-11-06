#!/bin/bash
#==============================================================================
# R41 hbm8 follow-on board batch: an 8 GiB window on the already-loaded B40
# image (a88ebdf5, 90 MHz). FPGA host only. This script NEVER loads, resets,
# programs or opens the card itself; experiment.py opens it per step only
# after its own read-only identity/idle probe.
#
#   hbm8_queue.sh COMMIT
#
# Order of gates (each logged to $D/queue.log):
#   1. Wait for the Stage 1 queue (~/r41s1_board_25094197f6e6/queue.log) to log
#      BOARD-QUEUE-DONE, or for its process to have exited.
#   2. The Stage 1 idle rule: kdsstat shows no context, no outstanding exec and
#      no client for 15 consecutive minutes (another user's back-to-back runs
#      leave seconds-long gaps).
#   3. The loaded image is B40 (UUID 1bada9db-...) and the clock counter reads
#      90 MHz. Anything else stops here: nothing is loaded.
#   4. MemAvailable >= 16 GiB (the 8 GiB BO has host backing; transfers are
#      staged in 64 MiB pieces). Waits up to 6 h, re-checking the idle rule.
#   5. Unpack the git archive of COMMIT into a fresh source directory and run
#      spec/r41/hbm8/experiments/board-hbm8.json, then board-hbm8-big.json,
#      each through experiment.py into its own output directory. A failed case
#      stops only its own manifest (preserved); a changed or busy card stops
#      the queue.
#
# Test overrides (defaults are the real host): HBM8_SYSFS, HBM8_STAGE1_LOG,
# HBM8_STAGE1_PATTERN, HBM8_IDLE_MINUTES, HBM8_POLL_SECONDS, HBM8_MEMINFO,
# HBM8_MIN_MEM_KB, HBM8_MEM_WAIT_SECONDS, HBM8_PYTHON, HBM8_HOME.
#==============================================================================
set -u
C=${1:?commit}
H=${HBM8_HOME:-$HOME}
D=$H/r41hbm8_board_$C
A=$H/r41hbm8_$C.tar.gz
UUID=1bada9db-78f3-5bc8-7ba6-cce062e7ade8
SYS=${HBM8_SYSFS:-/sys/bus/pci/devices}
K=$SYS/0000:04:00.1/kdsstat
U=$SYS/0000:04:00.1/xclbinuuid
F=$SYS/0000:04:00.0/clock_freq_counter.m.28311552/clock_counter_freqs
Q=${HBM8_STAGE1_LOG:-$H/r41s1_board_25094197f6e6/queue.log}
# The Stage 1 queue runs as: bash /home/zzhouch/r41s1_board_queue.sh 25094197f6e6
QP=${HBM8_STAGE1_PATTERN:-'^(/usr)?(/bin/)?bash [^ ]*r41s1_board_queue\.sh 25094197f6e6$'}
IDLE_MIN=${HBM8_IDLE_MINUTES:-15}
POLL=${HBM8_POLL_SECONDS:-60}
MEMINFO=${HBM8_MEMINFO:-/proc/meminfo}
MIN_MEM_KB=${HBM8_MIN_MEM_KB:-16777216}
PY=${HBM8_PYTHON:-$H/tc_env/torchenv/bin/python}
L=$D/queue.log
mkdir -p $D
exec 9>$D/queue.lock
flock -n 9 || { echo "another hbm8 queue holds $D/queue.lock" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
field() { awk -F: -v f="$1" '$1 ~ f {gsub(/[ \t]/,"",$2); print $2; exit}' $K; }
client() { awk 'f{print $1; exit} /client pid/{f=1}' $K; }
idle() { [ "$(field contexts)" = "0" ] && [ "$(field outstanding)" = "0" ] && [ -z "$(client)" ]; }
uuid() { cat $U 2>/dev/null; }
mhz() { head -1 $F 2>/dev/null; }
mem_kb() { awk '/^MemAvailable:/ {print $2; exit}' $MEMINFO; }
wait_idle() {
    local n=0
    while [ $n -lt $IDLE_MIN ]; do
        if idle; then n=$((n+1)); else n=0; fi
        sleep $POLL
    done
}
identity_ok() { [ "$(uuid)" = "$UUID" ] && [ "$(mhz)" = "90" ]; }

log "hbm8 queue start commit $C archive $(sha256sum $A 2>/dev/null | cut -c1-64)"
[ -f $A ] || { log "archive $A missing; nothing run"; exit 3; }

# 1. After the Stage 1 queue.
while true; do
    if grep -q BOARD-QUEUE-DONE $Q 2>/dev/null; then
        log "Stage 1 queue logged BOARD-QUEUE-DONE"; break
    fi
    if ! pgrep -u "$(id -u)" -f "$QP" > /dev/null; then
        log "Stage 1 queue is not running and did not log BOARD-QUEUE-DONE; last line: $(tail -1 $Q 2>/dev/null)"
        break
    fi
    sleep $POLL
done

# 2-4. Idle, identity, host memory.
wait_idle
log "card idle for $IDLE_MIN consecutive minutes; loaded $(uuid), clock $(mhz) MHz"
identity_ok || { log "loaded image/clock is not B40 at 90 MHz ($(uuid), $(mhz) MHz); stopping without loading anything"; exit 5; }
memory_deadline=$((SECONDS + ${HBM8_MEM_WAIT_SECONDS:-21600}))
while [ "$(mem_kb)" -lt $MIN_MEM_KB ]; do
    [ $SECONDS -ge $memory_deadline ] && { log "MemAvailable $(mem_kb) kB < $MIN_MEM_KB kB; memory wait budget exhausted; stopping"; exit 7; }
    log "MemAvailable $(mem_kb) kB < $MIN_MEM_KB kB; waiting"
    sleep $((POLL * 5))
    wait_idle
    identity_ok || { log "card changed while waiting for memory ($(uuid), $(mhz) MHz); stopping"; exit 5; }
done
log "MemAvailable $(mem_kb) kB"

# 5. Fresh source tree from the archive; the two registered manifests.
if [ ! -d $D/src ]; then
    mkdir -p $D/src && tar xzf $A -C $D/src
fi
cd $D/src
for m in board-hbm8 board-hbm8-big; do
    identity_ok && idle || { log "card changed or busy before $m ($(uuid), $(mhz) MHz); stopping queue"; exit 6; }
    log "manifest $m start"
    $PY tools/performance_harness/experiment.py spec/r41/hbm8/experiments/$m.json \
        --root $D/src --out $D/out/$m --python $PY >> $D/$m.log 2>&1
    rc=$?
    log "manifest $m exit $rc status $(tail -1 $D/$m.log)"
    if [ $rc -ne 0 ] && ! { identity_ok && idle; }; then
        log "card changed or busy after $m ($(uuid)); stopping queue"; exit 6
    fi
done
log HBM8-QUEUE-DONE
