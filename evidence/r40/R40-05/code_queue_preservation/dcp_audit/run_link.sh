#!/bin/bash
set -u
R40_BUILD_ROOT=/home/zzhouch/tnca_build/u280_r40ownership_preserved09
cd "$R40_BUILD_ROOT/src" || exit 2
export TNCA_BOARD=u280 VPP_JOBS=2
export LINK_EXTRA='--vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_SpreadSLLs'
bash flow/fpga/link.sh "$R40_BUILD_ROOT/xo"
R40_LINK_RC=$?
printf '%s\n' "$R40_LINK_RC" > "$R40_BUILD_ROOT/build/link.exit"
exit "$R40_LINK_RC"
