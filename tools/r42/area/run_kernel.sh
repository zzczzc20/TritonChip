#!/bin/bash
# run_kernel.sh NAME IMAGE_TCL   (x86 host, tree root ~/r42_area with rtl/ and flow/)
# Kernel synthesis (tools/r42/area/synth_kernel.tcl: whole three-tile kernel, -flatten_hierarchy $FLATTEN (default none), 4 threads) of the
# tree at the image parameters IMAGE_TCL (a file under tools/r42/area/images). Results in $OUTROOT/NAME.
set -u
NAME=$1; IMG=$2
source ~/x86_link_env.sh
ROOT=$(cd $(dirname $0)/../../.. && pwd)
OUTROOT=${OUTROOT:-$HOME/r42_area_out}
OUT=$OUTROOT/$NAME
n=$(pgrep -fc 'unwrapped/lnx64.o/vivado -mode batch' || true)
if [ "${n:-0}" -ge 2 ]; then echo "two Vivado jobs already running, not starting $NAME"; exit 3; fi
mkdir -p $OUTROOT
echo "$(date -u +%FT%TZ) start $NAME kernel image=$IMG" >> $OUTROOT/run.log
( cd $OUTROOT && FLATTEN=${FLATTEN:-none} nice -n 5 vivado -mode batch -nojournal -log $NAME.synth.log -source $ROOT/tools/r42/area/synth_kernel.tcl -tclargs $ROOT $ROOT/tools/r42/area/images/$IMG $OUT > /dev/null 2>&1 )
echo "$(date -u +%FT%TZ) end $NAME exit $?" >> $OUTROOT/run.log
