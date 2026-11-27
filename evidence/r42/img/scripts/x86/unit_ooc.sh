#!/bin/bash
# unit_ooc.sh NAME SRCDIR GENERIC=VALUE... : tnca_matrix out of context, 10 ns, SLR0 pblock (flow/fpga/screen_r42_unit.tcl of SRCDIR)
set -u
name=$1 src=$2; shift 2
source ~/x86_link_env.sh >/dev/null 2>&1
d=$HOME/r42img/unit/$name; [ -e "$d" ] && { echo "exists $d"; exit 1; }
mkdir -p "$d"; cd "$d"
ls $src/rtl/common/*.v $src/rtl/core/*.v $src/rtl/gen/*.v > files.txt
start=$(date +%s)
nice -n 5 vivado -mode batch -nojournal -log unit.log -source $HOME/r42img/lmf-a099878e/src/flow/fpga/screen_r42_unit.tcl -tclargs tnca_matrix $src/rtl/common files.txt out VLEN=16 ELEN=32 ACC_N=16 ACC_W=4 ROWS=16 ROW_W=4 BAR_W=3 FMA_LAT=2 FMA_IMPL=1 MAC_DP=1 MAC_CUTS=29 ABUF=1 "$@" > /dev/null 2>&1
echo "$(date -u +%FT%TZ) unit $name exit $? $(( $(date +%s) - start )) s" >> ~/r42img/chain.log
