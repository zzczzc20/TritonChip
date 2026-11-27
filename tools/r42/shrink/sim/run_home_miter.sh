#!/bin/bash
# run_home_miter.sh TREE OUT [JOBS]   (GB10) -- tb_home_pipe_miter over a matrix of structural and traffic parameters, DR 1 and 2.
# One line of the matrix = "name|iverilog -P overrides". Every run is nice 19; at most JOBS (default 3) at once. Results: OUT/<name>_d<DR>a<AR>.log; env COMBOS="1:0 2:0 0:1 0:2 1:1 2:2" (DATA_RAM:AUX_RAM of instance B)
set -u
TREE=$1; OUT=$2; JOBS=${3:-3}
source ~/tc_env/env.sh
mkdir -p $OUT
cd $TREE
cat > $OUT/matrix.txt <<'M'
img_s1|tb.SEED=1 tb.CYCLES=600000
img_s2|tb.SEED=2 tb.CYCLES=600000
img_s3|tb.SEED=3 tb.CYCLES=600000
img_hot_s4|tb.SEED=4 tb.CYCLES=600000 tb.HOT=1 tb.NLINES=512
img_hot_s5|tb.SEED=5 tb.CYCLES=600000 tb.HOT=1 tb.NLINES=48 tb.P_WR=60
img_small_s6|tb.SEED=6 tb.CYCLES=600000 tb.NLINES=96
img_big_s7|tb.SEED=7 tb.CYCLES=600000 tb.NLINES=3000
img_slow_rsp_s8|tb.SEED=8 tb.CYCLES=600000 tb.P_RSP=15 tb.P_MEM=25 tb.MAXLAT=60
img_fast_s9|tb.SEED=9 tb.CYCLES=600000 tb.P_RSP=100 tb.P_MEM=100 tb.P_MRSP=100 tb.MAXLAT=3 tb.P_REQ=100
img_amo_s10|tb.SEED=10 tb.CYCLES=600000 tb.P_AMO=45 tb.P_WR=50 tb.NLINES=128 tb.HOT=1
img_maint_s11|tb.SEED=11 tb.CYCLES=600000 tb.P_MAINT=15 tb.NLINES=200
img_nofill0_s12|tb.SEED=12 tb.CYCLES=400000 tb.NOFILL=0
img_chain0_s13|tb.SEED=13 tb.CYCLES=400000 tb.CHAIN=0
tiny_s14|tb.SEED=14 tb.CYCLES=600000 tb.MSHR_N=4 tb.MSHR_W=2 tb.TARGET_N=8 tb.TARGET_W=3 tb.RQ_N=4 tb.RQ_W=2 tb.WBQ_N=2 tb.HOT=1 tb.NLINES=300
mid_s15|tb.SEED=15 tb.CYCLES=600000 tb.MSHR_N=8 tb.MSHR_W=3 tb.TARGET_N=16 tb.TARGET_W=4 tb.RQ_N=8 tb.RQ_W=3 tb.NLINES=400
sets128_s16|tb.SEED=16 tb.CYCLES=600000 tb.SETS=128 tb.SET_W=7 tb.NLINES=2400
sets128_hot_s17|tb.SEED=17 tb.CYCLES=600000 tb.SETS=128 tb.SET_W=7 tb.NLINES=1500 tb.HOT=1
ways4_s18|tb.SEED=18 tb.CYCLES=600000 tb.WAYS=4 tb.WAY_W=2 tb.NLINES=300
M
run_one() {
    IFS='|' read -r name ov <<< "$1"
    for combo in ${COMBOS:-1:0 2:0 0:1 0:2 1:1 2:2}; do
        dr=${combo%:*}; ar=${combo#*:}
        vvp_bin=$OUT/${name}_d${dr}a${ar}.vvp
        iverilog -g2012 -I rtl/common -o $vvp_bin -s tb tools/r42/shrink/sim/tb_home_pipe_miter.v rtl/nuca/tnca_home_pipe.v -P tb.DR=$dr -P tb.AR=$ar $(for o in $ov; do echo -P $o; done) > $OUT/${name}_d${dr}a${ar}.compile 2>&1
        nice -n 19 vvp $vvp_bin > $OUT/${name}_d${dr}a${ar}.log 2>&1
        rm -f $vvp_bin
    done
}
export -f run_one; export OUT COMBOS
cat $OUT/matrix.txt | xargs -P $JOBS -I{} bash -c 'run_one "{}"'
grep -h "^MITER" $OUT/*_d?a?.log | sort > $OUT/summary.txt
echo MITER_MATRIX_DONE >> $OUT/summary.txt
