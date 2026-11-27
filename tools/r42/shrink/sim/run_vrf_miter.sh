#!/bin/bash
# run_vrf_miter.sh TREE OUT [JOBS]   (GB10) -- tb_vrf_miter: tnca_vrf_banked shipped (XBAR 0, VLD_RAM 0) against XBAR 1/2 and VLD_RAM 1, several seeds and traffic mixes.
set -u
TREE=$1; OUT=$2; JOBS=${3:-3}
source ~/tc_env/env.sh
mkdir -p $OUT
cd $TREE
cat > $OUT/matrix.txt <<'M'
x1_s1|tb.XB=1 tb.SEED=1 tb.CYCLES=400000
x1_s2_rdy|tb.XB=1 tb.SEED=2 tb.CYCLES=400000 tb.P_RDY=60
x1_s3_small|tb.XB=1 tb.SEED=3 tb.CYCLES=400000 tb.NREG=24 tb.P_WR=70
x2_s4|tb.XB=2 tb.SEED=4 tb.CYCLES=400000
x2_s5_rdy|tb.XB=2 tb.SEED=5 tb.CYCLES=400000 tb.P_RDY=60
x2_s6_small|tb.XB=2 tb.SEED=6 tb.CYCLES=400000 tb.NREG=24 tb.P_WR=70
v1_s7|tb.XB=0 tb.VLDB=1 tb.SEED=7 tb.CYCLES=400000
v1_s8_small|tb.XB=0 tb.VLDB=1 tb.SEED=8 tb.CYCLES=400000 tb.NREG=24 tb.P_WR=30
v1x1_s9|tb.XB=1 tb.VLDB=1 tb.SEED=9 tb.CYCLES=400000
v1x2_s10|tb.XB=2 tb.VLDB=1 tb.SEED=10 tb.CYCLES=400000 tb.P_RDY=70
v1x2_s11_small|tb.XB=2 tb.VLDB=1 tb.SEED=11 tb.CYCLES=400000 tb.NREG=24 tb.P_WR=30
M
run_one() {
    IFS='|' read -r name ov <<< "$1"
    vvp_bin=$OUT/${name}.vvp
    vl=""; case "$ov" in *VLDB=1*) vl='-DVLDA=,.VLD_RAM(0) -DVLDB_=,.VLD_RAM(VLDB)';; esac
    iverilog -g2012 -I rtl/common $vl -o $vvp_bin -s tb tools/r42/shrink/sim/tb_vrf_miter.v rtl/core/tnca_vrf_banked.v rtl/core/tnca_rf_bank.v $(for o in $ov; do echo -P $o; done) > $OUT/${name}.compile 2>&1
    nice -n 19 vvp $vvp_bin > $OUT/${name}.log 2>&1
    rm -f $vvp_bin
}
export -f run_one; export OUT
cat $OUT/matrix.txt | xargs -P $JOBS -I{} bash -c 'run_one "{}"'
grep -h "VRF MITER" $OUT/*.log | sort > $OUT/summary.txt
echo VRF_MITER_DONE >> $OUT/summary.txt
