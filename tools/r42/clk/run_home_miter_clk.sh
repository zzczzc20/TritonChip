#!/bin/bash
# run_home_miter_clk.sh TREE OUT [JOBS]   (GB10) -- tools/r42/shrink/sim/tb_home_pipe_miter.v: instance A = the shipped RTL (CLKFIX 0, DATA_RAM 0), instance B = DATA_RAM:AUX_RAM and CLKFIX of the
# candidate; every output compared every cycle (mem_req_addr only while mem_req_valid). Env: CFS="2" (CLKFIX values of B), COMBOS="0:0 2:2" (DATA_RAM:AUX_RAM of B), CYC=cycles per run, REF=1 with REFFILE=<path of origin/R42's rtl/nuca/tnca_home_pipe.v> (A = that file).
set -u
TREE=$1; OUT=$2; JOBS=${3:-3}
source ~/tc_env/env.sh
mkdir -p $OUT; cd $TREE
# REF=1: instance A is the module as of origin/R42 (renamed tnca_home_pipe_ref), taken from the git object database of $REFREPO (default $TREE/.git, else the tree's own copy at REFFILE)
if [ -n "${REFFILE:-}" ]; then sed 's/module tnca_home_pipe\b/module tnca_home_pipe_ref/' $REFFILE > $OUT/tnca_home_pipe_ref.v; else echo '' > $OUT/tnca_home_pipe_ref.v; fi
cat > $OUT/matrix.txt <<'M'
img_s1|tb.SEED=1 tb.CYCLES=300000
img_s3|tb.SEED=3 tb.CYCLES=300000
img_hot_s4|tb.SEED=4 tb.CYCLES=300000 tb.HOT=1 tb.NLINES=512
img_hot_s5|tb.SEED=5 tb.CYCLES=300000 tb.HOT=1 tb.NLINES=48 tb.P_WR=60
img_small_s6|tb.SEED=6 tb.CYCLES=300000 tb.NLINES=96
img_big_s7|tb.SEED=7 tb.CYCLES=300000 tb.NLINES=3000
img_slow_rsp_s8|tb.SEED=8 tb.CYCLES=300000 tb.P_RSP=15 tb.P_MEM=25 tb.MAXLAT=60
img_fast_s9|tb.SEED=9 tb.CYCLES=300000 tb.P_RSP=100 tb.P_MEM=100 tb.P_MRSP=100 tb.MAXLAT=3 tb.P_REQ=100
img_amo_s10|tb.SEED=10 tb.CYCLES=300000 tb.P_AMO=45 tb.P_WR=50 tb.NLINES=128 tb.HOT=1
img_maint_s11|tb.SEED=11 tb.CYCLES=300000 tb.P_MAINT=15 tb.NLINES=200
tiny_s14|tb.SEED=14 tb.CYCLES=300000 tb.MSHR_N=4 tb.MSHR_W=2 tb.TARGET_N=8 tb.TARGET_W=3 tb.RQ_N=4 tb.RQ_W=2 tb.WBQ_N=2 tb.HOT=1 tb.NLINES=300
mid_s15|tb.SEED=15 tb.CYCLES=300000 tb.MSHR_N=8 tb.MSHR_W=3 tb.TARGET_N=16 tb.TARGET_W=4 tb.RQ_N=8 tb.RQ_W=3 tb.NLINES=400
sets128_s16|tb.SEED=16 tb.CYCLES=300000 tb.SETS=128 tb.SET_W=7 tb.NLINES=2400
ways4_s18|tb.SEED=18 tb.CYCLES=300000 tb.WAYS=4 tb.WAY_W=2 tb.NLINES=300
M
run_one() {
    IFS='|' read -r name ov <<< "$1"
    for cf in ${CFS:-2}; do for combo in ${COMBOS:-0:0 2:2}; do
        dr=${combo%:*}; ar=${combo#*:}; tag=${name}_cf${cf}_d${dr}a${ar}
        iverilog -g2012 -I rtl/common -o $OUT/$tag.vvp -s tb tools/r42/shrink/sim/tb_home_pipe_miter.v rtl/nuca/tnca_home_pipe.v $OUT/tnca_home_pipe_ref.v $([ "${REF:-0}" = 1 ] && echo -DREFA) -P tb.DR=$dr -P tb.AR=$ar -P tb.CF=$cf $(for o in $ov; do echo -P $o; done) > $OUT/$tag.compile 2>&1
        nice -n 19 vvp $OUT/$tag.vvp > $OUT/$tag.log 2>&1
        rm -f $OUT/$tag.vvp
    done; done
}
export -f run_one; export OUT CFS COMBOS REF
[ -n "${CYC:-}" ] && sed -i "s/tb.CYCLES=[0-9]*/tb.CYCLES=$CYC/" $OUT/matrix.txt
cat $OUT/matrix.txt | xargs -P $JOBS -I{} bash -c 'run_one "{}"'
grep -H "^MITER" $OUT/*.log | sort > $OUT/summary.txt
echo MITER_MATRIX_DONE >> $OUT/summary.txt
