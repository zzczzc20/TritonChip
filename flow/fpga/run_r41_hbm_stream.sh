#!/usr/bin/env bash
# One immutable full implementation. Run only after the lead allocates a
# numbered full-run slot. This script never opens, resets or programs a card.
set -euo pipefail
# Quarantined (evidence/r41/hbm_bw/AUDIT.md Q1): the 32-PC stream CU is a
# constructive exception (doc/engineering_discipline.md 2.2.4). No image with
# it may be built without the author's written approval of that unit.
echo 'run_r41_hbm_stream.sh is quarantined: the stream CU needs written author approval (see evidence/r41/hbm_bw/AUDIT.md)' >&2
exit 3
build=${1:?usage: run_r41_hbm_stream.sh FRESH_BUILD_ROOT IMAGE_CONFIG RUN_NUMBER}
config=${2:?image configuration required}
run_number=${3:?approved full-run number required}
case "$run_number" in 2|3|4|5) ;; *) echo 'full-run number must be 2..5' >&2; exit 2;; esac
root=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$config" ] && [ ! -e "$build" ] || { echo 'configuration missing or build directory exists' >&2; exit 2; }
# Refuse an incompatible core XML/layout before allocating a run directory.
R41_HBM_IMAGE_CONFIG="$config" tclsh <<'TCL'
source $::env(R41_HBM_IMAGE_CONFIG)
foreach {name expected} {HBM_STRIPED 1 AXI_DATA_W 256 MEM_PORTS 3} {
    if {![dict exists $TNCA_IMAGE_PARAMS $name] || [dict get $TNCA_IMAGE_PARAMS $name] != $expected} {
        puts stderr "stream image configuration requires $name=$expected"
        exit 2
    }
}
TCL
if [ -f "$HOME/x86_link_env.sh" ]; then
    minimum_available=20
else
    minimum_available=38
fi
[ "$(free -g | awk '/^Mem:/{print $7}')" -ge "$minimum_available" ] || { echo "full run needs at least $minimum_available GiB available" >&2; exit 2; }
mkdir -p "$build/build"
exec >"$build/build/runner.log" 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$build/build/attempt.exit"' EXIT
printf '%s/5\n' "$run_number" > "$build/build/full_run_number"
cd "$root"
set +u
if [ -f "$HOME/x86_link_env.sh" ]; then
    source "$HOME/x86_link_env.sh" >/dev/null 2>&1
else
    source /tools/Xilinx/Vivado/2023.2/settings64.sh >/dev/null 2>&1
    source /tools/Xilinx/Vitis/2023.2/settings64.sh >/dev/null 2>&1
    source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
fi
set -u
export TNCA_BOARD=u280 VPP_JOBS=2
export LINK_EXTRA='--vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_SpreadSLLs'
export TNCA_IMAGE_CONFIG="$config"
date -u +%FT%TZ > "$build/build/package.started_utc"
vivado -mode batch -nojournal -nolog -source flow/fpga/package_xo.tcl \
    -tclargs "$root" "$build/core_xo" xcu280-fsvh2892-2L-e multicore-hbm > "$build/build/core_package.log" 2>&1
env -u TNCA_IMAGE_CONFIG vivado -mode batch -nojournal -nolog -source flow/fpga/package_xo.tcl \
    -tclargs "$root" "$build/stream_xo" xcu280-fsvh2892-2L-e stream-hbm > "$build/build/stream_package.log" 2>&1
export TNCA_STREAM_XO="$build/stream_xo/tnca_stream.xo"
sha256sum "$build/core_xo/tnca_u50.xo" "$TNCA_STREAM_XO" > "$build/build/xo.sha256"
date -u +%FT%TZ > "$build/build/link.started_utc"
bash flow/fpga/link.sh "$build/core_xo" > "$build/build/link.stdout" 2>&1
