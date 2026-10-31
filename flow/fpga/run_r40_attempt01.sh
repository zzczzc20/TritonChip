#!/bin/bash
# One immutable, fresh-directory implementation. Never retries or loads a board.
set -euo pipefail
R40_ATTEMPT_ROOT=${1:?usage: run_r40_attempt01.sh FRESH_BUILD_ROOT [IMAGE_CONFIG]}
R40_SOURCE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
R40_IMAGE_CONFIG=${2:-"$R40_SOURCE_ROOT/evidence/r40/v2/convergence/attempt01/image.tcl"}
[ -f "$R40_IMAGE_CONFIG" ] || { echo 'image configuration missing' >&2; exit 2; }
[ ! -e "$R40_ATTEMPT_ROOT" ] || { echo 'refusing existing build root' >&2; exit 2; }
mkdir -p "$R40_ATTEMPT_ROOT/build"
exec >"$R40_ATTEMPT_ROOT/build/runner.log" 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$R40_ATTEMPT_ROOT/build/attempt.exit"' EXIT
cd "$R40_SOURCE_ROOT"
set +u
# On the x86 host the tools are installed under the user's home and its
# environment is sourced before this runner. Under set -e a missing FPGA-host
# setup script would otherwise abort before packaging.
for setup in /tools/Xilinx/Vivado/2023.2/settings64.sh \
             /tools/Xilinx/Vitis/2023.2/settings64.sh /opt/xilinx/xrt/setup.sh; do
    if [ -f "$setup" ]; then source "$setup" >/dev/null 2>&1; fi
done
set -u
export TNCA_IMAGE_CONFIG="$R40_IMAGE_CONFIG"
export TNCA_BOARD=u280 VPP_JOBS=2
# Preserve the prior placement strategy; the hypothesis concerns RTL/geometry.
export LINK_EXTRA='--vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_SpreadSLLs'
date -u +%FT%TZ > "$R40_ATTEMPT_ROOT/build/package.started_utc"
vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs "$R40_SOURCE_ROOT" "$R40_ATTEMPT_ROOT/xo" xcu280-fsvh2892-2L-e multicore-hbm > "$R40_ATTEMPT_ROOT/build/package.log" 2>&1
sha256sum "$R40_ATTEMPT_ROOT/xo/tnca_u50.xo" > "$R40_ATTEMPT_ROOT/build/xo.sha256"
date -u +%FT%TZ > "$R40_ATTEMPT_ROOT/build/link.started_utc"
bash flow/fpga/link.sh "$R40_ATTEMPT_ROOT/xo" > "$R40_ATTEMPT_ROOT/build/link.stdout" 2>&1
