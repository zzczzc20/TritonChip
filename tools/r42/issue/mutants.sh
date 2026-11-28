#!/bin/bash
# R42 issue: does the lockstep see a wrong forward?  Three single-point mutants of tnca_core (VRF_FWD=1 builds), each run against the unmutated VRF_FWD 0 core
# on the first programs of a lockstep set.     mutants.sh SNAPSHOT OUTDIR NPROGS
SNAP=$1; OUT=$2; N=${3:-30}
mkdir -p $OUT; cd $SNAP; source ~/tc_env/env.sh
python3 tools/r42/issue/lockstep_gen.py $OUT/progs $N 301 > /dev/null
BASE="-P tb_tnca_core.VRF_BYPASS=1 -P tb_tnca_core.FE_NOBUBBLE=1 -P tb_tnca_core.VRF_PREFETCH=1 -P tb_tnca_core.RCTX_SWITCH=2 -P tb_tnca_core.LSU_INFLIGHT=2 -P tb_tnca_core.FPU_OVERLAP=1 -P tb_tnca_core.RSLICE=1 -P tb_tnca_core.NUM_RCTX=4 -P tb_tnca_core.RCTX_W=2 -P tb_tnca_core.PHYS_W=8 -P tb_tnca_core.HAS_TENSOR_XPOSE=2 -P tb_tnca_core.RPAIR=1 -P tb_tnca_core.MTX_ABUF=1 -P tb_tnca_core.BAR_N=8 -P tb_tnca_core.BAR_W=3 -P tb_tnca_core.BAR_EPOCH_W=38 -P tb_tnca_core.MTX_VRF_REG=1 -P tb_tnca_core.FMA_IMPL=1 -P tb_tnca_core.MAC_DP=1 -P tb_tnca_core.MAC_CUTS=29 -P tb_tnca_core.USE_BANKED_VRF=1"
IV="iverilog -g2005 -I rtl/common"
# reference: unmutated, VRF_FWD 0
$IV -y rtl/core -y rtl/common -y rtl/gen -y tb/common $BASE -o $OUT/ref.vvp tb/core/tb_tnca_core.v
i=0
for M in "wn_full_one" "swap_source_c" "hold_forwarded"; do
  rm -rf $OUT/rtl_$M; mkdir -p $OUT/rtl_$M; cp rtl/core/*.v $OUT/rtl_$M/
  f=$OUT/rtl_$M/tnca_core.v
  case $M in
    wn_full_one)    sed -i "s/wire wn_full = vrf_wb_freeze ? w_full : (&nw_lane_en);/wire wn_full = 1'b1;/" $f ;;
    swap_source_c)  sed -i "s/wire \[VW-1:0\] of_v2 = of_fwd\[2\] ?/wire [VW-1:0] of_v2 = of_fwd[1] ?/" $f ;;
    hold_forwarded) sed -i "s/of_got <= of_got | of_arrive_rsp;   \/\/ a forwarded source is not held (see of_v0)/of_got <= of_got | of_arrive;/" $f ;;
  esac
  cmp -s rtl/core/tnca_core.v $f && echo "MUTANT $M: sed changed nothing" >> $OUT/mutants.log
  $IV -y $OUT/rtl_$M -y rtl/common -y rtl/gen -y tb/common $BASE -P tb_tnca_core.VRF_FWD=1 -o $OUT/$M.vvp tb/core/tb_tnca_core.v
  python3 tools/r42/issue/lockstep_run.py $OUT/progs $OUT/ref.vvp $OUT/$M.vvp $OUT/$M.tsv 1,2 > $OUT/$M.log 2>&1
  echo "MUTANT $M: $(tail -1 $OUT/$M.log)" >> $OUT/mutants.log
done
echo MUTANTS_DONE >> $OUT/mutants.log
