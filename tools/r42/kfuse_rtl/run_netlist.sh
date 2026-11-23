#!/bin/bash
# run_netlist.sh NAME G W NVEC [GENERICS...]  (x86 host, from the tree root)
# Synthesises tnca_kfuse_lane (OOC) to a functional netlist and simulates it with xsim on NVEC vectors of a
# vector file made by kfuse_rtl_vectors.py (pass VEC=path). The bench is the same tb_kfuse_lane.v (-d NETLIST).
set -u
NAME=$1; G=$2; W=$3; NV=$4; shift 4
source ~/x86_link_env.sh
ROOT=$(cd $(dirname $0)/../../.. && pwd)
OUTROOT=${OUTROOT:-$HOME/r42_kfuse_out}
OUT=$OUTROOT/$NAME
rm -rf $OUT; mkdir -p $OUT
R=$ROOT/rtl
FILES="$R/core/tnca_kfuse_lane.v $R/gen/tnca_mac_b0_xls.v $R/gen/tnca_fmad_norm_xls.v $R/gen/tnca_fma_round_xls.v"
( cd $OUT && OOC_GENERICS="G=$G W=$W $*" nice -n 5 vivado -mode batch -nojournal -nolog -source $ROOT/tools/r42/kfuse_rtl/netlist.tcl \
    -tclargs $OUT $FILES > $OUT/vivado.log 2>&1 )
( head -n $((NV/4)) ${VEC:?VEC}; tail -n $((NV - NV/4)) $VEC ) > $OUT/vec.hex     # a quarter from the directed prefix, the rest random
LAT=${LAT:?LAT}; LATF=${LATF:?LATF}
cd $OUT
GLBL=$(dirname $(which vivado))/../data/verilog/src/glbl.v
xvlog netlist.v $GLBL > xvlog.log 2>&1
xvlog -sv -d NETLIST -d USE_NETLIST -d G=$G $ROOT/tools/r42/kfuse_rtl/tb_kfuse_lane.v >> xvlog.log 2>&1
xelab -L unisims_ver -timescale 1ns/1ps -generic_top G=$G -generic_top W=$W -generic_top NMAX=$((NV+10)) tb_kfuse_lane glbl -s sim > xelab.log 2>&1
xsim sim -runall -testplusarg VEC=$OUT/vec.hex -testplusarg N=$NV -testplusarg LAT=$LAT -testplusarg LATF=$LATF -testplusarg MAXERR=20 > xsim.log 2>&1
grep -E "RESULT|MISMATCH|ERROR" xsim.log | head -25
