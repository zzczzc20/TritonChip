#!/bin/bash
# Run only in a disposable source copy on the x86 host. Never uses a card.
set -euo pipefail
src=${1:?isolated source directory}
out=${2:?fresh output directory}
image=${3:?image config in the isolated source}
[ ! -e "$out" ] || { echo 'output already exists' >&2; exit 2; }
mkdir -p "$out"
exec >"$out/runner.log" 2>&1
trap 'rc=$?; date -u +%FT%TZ > "$out/end.utc"; printf "%s\n" "$rc" > "$out/exit"' EXIT
set +u
source ~/x86_link_env.sh
set -u
if [ -x "$HOME/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls" ]; then
  export XILINX_HLS="$HOME/Xilinx/Vitis_HLS/2023.2"
  export PATH="$XILINX_HLS/bin:$PATH"
fi
cd "$src"
export TNCA_BOARD=u280 VPP_JOBS=2 TNCA_IMAGE_CONFIG="$image"
export LINK_EXTRA='--vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_SpreadSLLs'
# Wait for memory headroom and at most one existing tool process. This is
# conservative when a single Vivado job has multiple worker processes.
wait_slot() {
  while :; do
    available=$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    active=$(ps -eo args | awk '/[/]unwrapped\/lnx64.o\/vivado/ {n++} END {print n+0}')
    links=$(ps -eo args | awk '/(^|\/)v\+\+([[:space:]]|$)/ {n++} END {print n+0}')
    printf '%s available_mib=%s existing_vivado_processes=%s links=%s\n' "$(date -u +%FT%TZ)" "$available" "$active" "$links"
    if [ "$available" -ge 20480 ] && [ "$active" -le 1 ] && [ "$links" -eq 0 ]; then return; fi
    sleep 60
  done
}
wait_slot
# These substitutions are confined to this disposable copy. The original
# placement script is still sourced and its expected hierarchy refusal logged.
python3 tools/r41/x86_link_dryrun/make_tiny.py rtl/board/tnca_multicore_hbm_xo.v rtl/board/tnca_multicore_hbm_xo.v
cp flow/fpga/place_r40_multicore.tcl flow/fpga/place_r40_multicore.real.tcl
cp tools/r41/x86_link_dryrun/tiny_place_hook.tcl flow/fpga/place_r40_multicore.tcl
sha256sum rtl/board/tnca_multicore_hbm_xo.v flow/fpga/place_r40_multicore.real.tcl flow/fpga/place_r40_multicore.tcl fpga/u280_multicore_hbm.cfg
printf 'PACKAGE_START %s\n' "$(date -u +%FT%TZ)"
python3 tools/r41/x86_link_dryrun/resource_watch.py "$out/package-memory.log" \
  taskset -c 0-5 vivado -mode batch -source flow/fpga/package_xo.tcl \
  -tclargs "$src" "$out/xo" xcu280-fsvh2892-2L-e multicore-hbm >"$out/package.log" 2>&1
printf 'PACKAGE_END %s\n' "$(date -u +%FT%TZ)"
sha256sum "$out/xo/tnca_u50.xo"
wait_slot
printf 'LINK_START %s\n' "$(date -u +%FT%TZ)"
python3 tools/r41/x86_link_dryrun/resource_watch.py "$out/link-memory.log" \
  taskset -c 0-5 bash flow/fpga/link.sh "$out/xo" >"$out/link.stdout" 2>&1
printf 'LINK_END %s\n' "$(date -u +%FT%TZ)"
