#!/bin/bash
#==============================================================================
# R41 Stage 2 board batch on the run-1 100 MHz image, FPGA host only.
#
#   board_queue.sh COMMIT          (COMMIT: the r41s2-board commit; archive
#                                   ~/r41s2_board_COMMIT.tar.gz, staged by
#                                   stage_inputs.sh)
#
# Image: ~/tnca_images/r41s2_run1_c9f25787494a/tnca_u280.xclbin, sha256
# 30e79769...4e5de, UUID 49964b17-e0b4-5bd2-dd40-7fc37b1205d3 (xclbinutil
# --info), source c9f2578, profile core-vrf-sync-v5-x8, DATA_CLK 100 MHz.
#
# The user authorized loading THIS image on the shared U280 only
#   - after the Stage 1 queue (r41s1_board_queue.sh 25094197f6e6) and the hbm8
#     queue (hbm8_queue.sh 2205c136...) have finished: their DONE line is in
#     their queue.log, or their process is gone;
#   - when kdsstat shows no context, no outstanding exec and no client for 15
#     consecutive minutes (the board_queue.sh rule of Stage 1);
#   - through program.sh --check, then --go; never over anyone else's run.
#
# Order:
#   0. before waiting (no card access): unpack the archive into a fresh source
#      directory, check the image sha256 and its xclbinutil UUID/DATA_CLK, run
#      the post-route audit of the routed checkpoint once (Vivado, analysis
#      only: timing, route, CDC, actual core placement at 100 MHz) and build the
#      FP8 venv when the inputs were staged (failure only skips the FP8 Hub and
#      Qwen manifests);
#   1. wait for both queues, then for 15 idle minutes;
#   2. program.sh --check, then --go (unless the image is already loaded);
#      wait for the new UUID;
#   3. capture the clock counter (must read 100 MHz) and assemble the qualified
#      clock evidence (clock_qualify.py; refuses on any review mismatch);
#   4. run the registered manifests in order through experiment.py. Before
#      each: the UUID, 100 MHz and an idle card, or stop. s2-identity failing
#      stops the queue; any other failure continues only while the card is
#      still ours and idle. experiment.py resumes exactly on a rerun.
#
# It never retries a failed load, never resets the card and never kills
# anything but its own children.
#==============================================================================
set -u
C=${1:?commit}
H=${S2_HOME:-$HOME}
D=$H/r41s2_board_$C
A=$H/r41s2_board_$C.tar.gz
TAG=r41s2_run1_c9f25787494a
X=$H/tnca_images/$TAG/tnca_u280.xclbin
DCP=$H/tnca_images/$TAG/level0_wrapper_routed.dcp
XSHA=${S2_TEST_XSHA:-30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de}
DCPSHA=${S2_TEST_DCPSHA:-311bd49cdfe5870056c9496804112edc263e6adc2ae9caeecd7a409967045215}
UUID=49964b17-e0b4-5bd2-dd40-7fc37b1205d3
QD=$H/tnca_build/$TAG/r41s2-board
SYSROOT=${S2_SYSFS:-/sys/bus/pci/devices}
SYS=$SYSROOT/0000:04:00.1
K=$SYS/kdsstat
U=$SYS/xclbinuuid
F=$SYSROOT/0000:04:00.0/clock_freq_counter.m.28311552/clock_counter_freqs
Q1=${S2_STAGE1_LOG:-$H/r41s1_board_25094197f6e6/queue.log}
P1=${S2_STAGE1_PATTERN:-'^(/usr)?(/bin/)?bash [^ ]*r41s1_board_queue\.sh 25094197f6e6$'}
Q2=${S2_HBM8_LOG:-$H/r41hbm8_board_2205c136f38066e568e1a2262c42289d2cb5e745/queue.log}
P2=${S2_HBM8_PATTERN:-'^(/usr)?(/bin/)?bash [^ ]*hbm8_queue\.sh 2205c136f38066e568e1a2262c42289d2cb5e745$'}
IDLE_MIN=${S2_IDLE_MINUTES:-15}
POLL=${S2_POLL_SECONDS:-60}
LOAD_WAIT=${S2_LOAD_WAIT_SECONDS:-600}
PY=${S2_PYTHON:-$H/tc_env/torchenv/bin/python}
PROGRAM=${S2_PROGRAM:-}
VIVADO=${S2_VIVADO:-/tools/Xilinx/Vivado/2023.2/bin/vivado}
MEMINFO=${S2_MEMINFO:-/proc/meminfo}
AUDIT_MIN_KB=${S2_AUDIT_MIN_KB:-25165824}
FP8IN=$H/r41s2_fp8_inputs
FP8ENV=$H/r41s2_fp8env
MANIFESTS=${S2_MANIFESTS:-"s2-identity s2-anchors s2-fp8-cases s2-fp8-blockwise s2-hbm8 s2-hbm8-big s2-original-suite s2-fp8-hub s2-qwen-fp8 s2-endurance"}
CRITICAL="s2-identity"
L=$D/queue.log
mkdir -p $D
exec 9>$D/queue.lock
flock -n 9 || { echo "another stage 2 board queue holds $D/queue.lock" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
field() { awk -F: -v f="$1" '$1 ~ f {gsub(/[ \t]/,"",$2); print $2; exit}' $K; }
client() { awk 'f{print $1; exit} /client pid/{f=1}' $K; }
idle() { [ "$(field contexts)" = "0" ] && [ "$(field outstanding)" = "0" ] && [ -z "$(client)" ]; }
uuid() { tr -d '[:space:]' < $U 2>/dev/null; }
mhz() { head -1 $F 2>/dev/null | tr -d '[:space:]'; }
mem_kb() { awk '/^MemAvailable:/ {print $2; exit}' $MEMINFO; }
identity_ok() { [ "$(uuid)" = "$UUID" ] && [ "$(mhz)" = "100" ]; }
wait_idle() {
    local n=0
    while [ $n -lt $IDLE_MIN ]; do
        if idle; then n=$((n+1)); else n=0; fi
        sleep $POLL
    done
}
wait_queue() {   # LOG DONE-LINE PATTERN NAME
    while true; do
        if grep -q "$2" "$1" 2>/dev/null; then log "$4 queue logged $2"; return; fi
        if ! pgrep -u "$(id -u)" -f "$3" > /dev/null; then
            log "$4 queue is not running and did not log $2; last line: $(tail -1 "$1" 2>/dev/null)"; return
        fi
        sleep $POLL
    done
}
command -v xclbinutil > /dev/null || source ${S2_XRT_SETUP:-/opt/xilinx/xrt/setup.sh} > /dev/null 2>&1

log "stage 2 board queue start commit $C archive $(sha256sum $A 2>/dev/null | cut -c1-64)"
[ -f $A ] || { log "archive $A missing; nothing run"; exit 3; }
[ "$(sha256sum $X | cut -c1-64)" = "$XSHA" ] || { log "image $X is not $XSHA; nothing run"; exit 3; }
info=$(xclbinutil --info --input $X 2>/dev/null)
xu=$(echo "$info" | awk '/UUID \(xclbin\)/ {print $3; exit}')
xc=$(echo "$info" | awk '/Name:[ \t]*DATA_CLK/ {f=1} f && /Frequency:/ {print $2; exit}')
log "xclbinutil: UUID $xu DATA_CLK $xc MHz"
[ "$xu" = "$UUID" ] && [ "$xc" = "100" ] || { log "xclbin UUID/clock not as registered; nothing run"; exit 3; }

# 0. preparation (no card access)
if [ ! -d $D/src ]; then
    mkdir -p $D/src.tmp && tar xzf $A -C $D/src.tmp && mv $D/src.tmp $D/src
fi
cd $D/src
PROGRAM=${PROGRAM:-$D/src/flow/fpga/program.sh}
if [ ! -f $QD/audit/actual_cores.rpt ]; then
    [ "$(sha256sum $DCP | cut -c1-64)" = "$DCPSHA" ] || { log "routed checkpoint is not $DCPSHA; stopping"; exit 3; }
    while [ "$(mem_kb)" -lt $AUDIT_MIN_KB ]; do log "MemAvailable $(mem_kb) kB < $AUDIT_MIN_KB kB before the audit; waiting"; sleep $((POLL * 5)); done
    mkdir -p $QD && rm -rf $QD/audit.tmp
    log "post-route audit start (Vivado, analysis only)"
    ( cd $QD && nice -n 10 $VIVADO -mode batch -nojournal -log $QD/audit.vivado.log \
        -source $D/src/flow/fpga/audit_r40_operating_clock.tcl -tclargs $DCP $QD/audit.tmp 100 placement ) > $QD/audit.stdout 2>&1
    if grep -q R40_OPERATING_CLOCK_ANALYSIS_DONE $QD/audit.stdout && [ -f $QD/audit.tmp/actual_cores.rpt ]; then
        mv $QD/audit.tmp $QD/audit; log "post-route audit done"
    else
        log "post-route audit failed ($QD/audit.stdout); the clock cannot be qualified; nothing loaded"; exit 3
    fi
fi
if [ -f $FP8IN/STAGED ] && [ ! -x $FP8ENV/bin/python ]; then
    log "building the FP8 venv"
    if $PY -m venv --system-site-packages $FP8ENV.tmp >> $D/fp8env.log 2>&1 \
       && $FP8ENV.tmp/bin/python -m pip install --no-input transformers==5.20.2 kernels==0.17.2 accelerate==1.15.0 \
            huggingface_hub==1.33.0 safetensors==0.8.0 >> $D/fp8env.log 2>&1 \
       && $FP8ENV.tmp/bin/python -c "import torch, triton, transformers, kernels; assert torch.__version__.startswith('2.14.0') and triton.__version__ == '3.8.0'" >> $D/fp8env.log 2>&1; then
        mv $FP8ENV.tmp $FP8ENV; log "FP8 venv ready"
    else
        log "FP8 venv build failed ($D/fp8env.log); s2-fp8-hub and s2-qwen-fp8 will be skipped"
    fi
fi

# 1. wait for the earlier queues, then the idle rule
wait_queue "$Q1" BOARD-QUEUE-DONE "$P1" "Stage 1"
wait_queue "$Q2" HBM8-QUEUE-DONE "$P2" "hbm8"
wait_idle
log "card idle for $IDLE_MIN consecutive minutes; loaded $(uuid), clock $(mhz) MHz"

# 2. load
if [ "$(uuid)" != "$UUID" ]; then
    idle || { log "card busy at load time; nothing loaded"; exit 4; }
    bash $PROGRAM --check $X >> $L 2>&1 || { log "program.sh --check refused; nothing loaded"; exit 4; }
    bash $PROGRAM --go $X >> $L 2>&1 || { log "program.sh --go failed"; exit 4; }
    end=$((SECONDS + LOAD_WAIT))
    while [ "$(uuid)" != "$UUID" ] && [ $SECONDS -lt $end ]; do sleep 10; done
    bash $PROGRAM --status $X >> $L 2>&1
fi
log "loaded uuid $(uuid) clock counter $(mhz) MHz"
identity_ok || { log "image or clock not as required (want $UUID at 100 MHz); stopping"; exit 5; }

# 3. clock evidence, captured on this load
mkdir -p $QD
if [ ! -f $QD/qualified_clock.json ]; then
    cap=$QD/board_clock_after_load.json
    [ -f $cap ] || $PY tools/performance_harness/clock_qualify.py capture --image $X --uuid $UUID \
        --sysfs $SYSROOT --out $cap >> $L 2>&1 || { log "clock capture refused; stopping"; exit 5; }
    $PY tools/performance_harness/clock_qualify.py assemble --image $X --audit $QD/audit --capture $cap \
        --out $QD/qualified_clock.json.tmp >> $L 2>&1 \
        && mv $QD/qualified_clock.json.tmp $QD/qualified_clock.json \
        || { log "clock qualification refused; stopping"; exit 5; }
fi
log "qualified clock $(sha256sum $QD/qualified_clock.json | cut -c1-64)"

# 4. manifests
for m in $MANIFESTS; do
    case $m in s2-fp8-hub|s2-qwen-fp8)
        [ -x $FP8ENV/bin/python ] || { log "manifest $m skipped: FP8 venv/inputs not prepared"; continue; } ;;
    esac
    identity_ok && idle || { log "card changed or busy before $m ($(uuid), $(mhz) MHz); stopping queue"; exit 6; }
    log "manifest $m start"
    $PY tools/performance_harness/experiment.py spec/r41/stage2/board/$m.json \
        --root $D/src --out $D/out/$m --python $PY >> $D/$m.log 2>&1
    rc=$?
    log "manifest $m exit $rc status $(tail -1 $D/$m.log)"
    if [ $rc -ne 0 ]; then
        case " $CRITICAL " in *" $m "*) log "critical manifest $m failed; stopping queue"; exit 7 ;; esac
        { identity_ok && idle; } || { log "card changed or busy after $m ($(uuid), $(mhz) MHz); stopping queue"; exit 6; }
    fi
done
log S2-BOARD-QUEUE-DONE
