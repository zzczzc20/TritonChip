#!/bin/bash
# run_cell_census.sh NAME DCP   (x86 host) -- cell_census.tcl on a synthesized checkpoint, 4 threads, nice 5, results in ~/r42_shrink_out/NAME
set -u
NAME=$1; DCP=$2
source ~/x86_link_env.sh
ROOT=$(cd $(dirname $0)/../../.. && pwd)
OUT=$HOME/r42_shrink_out/$NAME
mkdir -p $OUT
cd $OUT && nice -n 5 vivado -mode batch -nojournal -log census.log -source $ROOT/tools/r42/shrink/cell_census.tcl -tclargs $DCP $OUT/census.csv > /dev/null 2>&1
echo "exit $?" >> $OUT/done.txt
