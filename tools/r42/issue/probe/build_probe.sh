#!/bin/bash
# R42 issue probe: build the core bench in four VRF/issue variants (run on GB10, from the tree root).
#   run2   the R41 run-2 core machine (make core-run2): banked VRF, VRF_PREFETCH 1, VRF_BYPASS 1
#   nopf   the same with VRF_PREFETCH 0
#   nobyp  the same with VRF_BYPASS 0
#   behav  the behavioural VRF (X reads the file itself), VALU_LATE 2 / TENSOR_MUL_PIPE 1 as in run2
# usage: build_probe.sh OUTDIR [extra iverilog -P options...]
set -e
O=$1; shift
mkdir -p $O
BASE="-P tb_tnca_core.VRF_BYPASS=1 -P tb_tnca_core.FE_NOBUBBLE=1 -P tb_tnca_core.VRF_PREFETCH=1 -P tb_tnca_core.RCTX_SWITCH=2 \
-P tb_tnca_core.LSU_INFLIGHT=2 -P tb_tnca_core.FPU_OVERLAP=1 -P tb_tnca_core.RSLICE=1 -P tb_tnca_core.NUM_RCTX=4 \
-P tb_tnca_core.RCTX_W=2 -P tb_tnca_core.PHYS_W=8 -P tb_tnca_core.HAS_TENSOR_XPOSE=2 -P tb_tnca_core.RPAIR=1 \
-P tb_tnca_core.MTX_ABUF=1 -P tb_tnca_core.BAR_N=8 -P tb_tnca_core.BAR_W=3 -P tb_tnca_core.BAR_EPOCH_W=38 \
-P tb_tnca_core.MTX_VRF_REG=1 -P tb_tnca_core.FMA_IMPL=1 -P tb_tnca_core.MAC_DP=1 -P tb_tnca_core.MAC_CUTS=29"
IV="iverilog -g2005 -I rtl/common -y rtl/core -y rtl/common -y rtl/gen -y tb/common"
$IV -P tb_tnca_core.USE_BANKED_VRF=1 $BASE "$@" -o $O/tb_run2.vvp tb/core/tb_tnca_core.v &
$IV -P tb_tnca_core.USE_BANKED_VRF=1 $BASE -P tb_tnca_core.VRF_PREFETCH=0 "$@" -o $O/tb_nopf.vvp tb/core/tb_tnca_core.v &
$IV -P tb_tnca_core.USE_BANKED_VRF=1 $BASE -P tb_tnca_core.VRF_BYPASS=0 "$@" -o $O/tb_nobyp.vvp tb/core/tb_tnca_core.v &
$IV -P tb_tnca_core.USE_BANKED_VRF=1 $BASE -P tb_tnca_core.VRF_FWD=1 "$@" -o $O/tb_fwd.vvp tb/core/tb_tnca_core.v &
$IV -P tb_tnca_core.USE_BANKED_VRF=0 -P tb_tnca_core.VALU_LATE=2 -P tb_tnca_core.TENSOR_MUL_PIPE=1 $BASE "$@" -o $O/tb_behav.vvp tb/core/tb_tnca_core.v &
wait
ls -la $O/*.vvp
# the pipeline-monitor bench (+pipetrace), run2 machine only
$IV -P tb_tnca_core.USE_BANKED_VRF=1 $BASE "$@" -o $O/tb_pt_run2.vvp tools/r42/issue/probe/tb_probe_core.v
$IV -P tb_tnca_core.USE_BANKED_VRF=1 $BASE -P tb_tnca_core.VRF_FWD=1 "$@" -o $O/tb_pt_fwd.vvp tools/r42/issue/probe/tb_probe_core.v
