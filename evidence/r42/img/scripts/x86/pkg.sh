#!/bin/bash
# pkg.sh SRCDIR ARM : XO packaging dry run (flow/fpga/package_xo.tcl, multicore-hbm) of an image of the study, as run_r40_attempt01.sh does as its first step; no link, no board
source ~/x86_link_env.sh >/dev/null 2>&1
src=$1; arm=$2; out=$HOME/r42img/pkg_$(basename $src)_$arm
[ -e $out ] && { echo exists; exit 1; }
mkdir -p $out; cd $src
TNCA_IMAGE_CONFIG=$src/evidence/r42/img/images/image_$arm.tcl TNCA_BOARD=u280 nice -n 5 vivado -mode batch -nojournal -log $out/package.log -source flow/fpga/package_xo.tcl -tclargs $src $out/xo xcu280-fsvh2892-2L-e multicore-hbm > $out/vivado.stdout 2>&1
echo "$(date -u +%FT%TZ) package $(basename $src) $arm exit $?" >> ~/r42img/chain.log
grep -E "^TNCA_(KERNEL_XML|PKG_FILES|PKG_HEADERS|PKG_ERROR)" $out/package.log | head -6; grep -E "^TNCA_IMAGE_PARAM (TENSOR_LINE_HALF|MTX_PIPE|MAC_F32|MAC_DP|HBM_STRIPED|AXI_DATA_W|TENSOR_LINE)=" $out/package.log; echo "critical warnings: $(grep -c 'CRITICAL WARNING' $out/package.log)"; ls -la $out/xo | tail -3
