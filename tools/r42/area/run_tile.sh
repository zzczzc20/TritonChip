#!/bin/bash
# run_tile.sh NAME TILE_SYNTH_DCP SLR   (x86 host, tree root ~/r42_area; PERIOD env, default 10)
# One tile screen (tile_screen.tcl) at PERIOD ns, 6 threads, nice 5, results in $OUTROOT/NAME. Refuses to start when two
# Vivado jobs run.
set -u
NAME=$1; DCP=$2; SLR=${3:-0}
source ~/x86_link_env.sh
ROOT=$(cd $(dirname $0)/../../.. && pwd)
OUTROOT=${OUTROOT:-$HOME/r42_area_out}
PERIOD=${PERIOD:-10}
OUT=$OUTROOT/$NAME
n=$(pgrep -fc 'unwrapped/lnx64.o/vivado -mode batch' || true)
if [ "${n:-0}" -ge 2 ]; then echo "two Vivado jobs already running, not starting $NAME"; exit 3; fi
echo "$(date -u +%FT%TZ) start $NAME tile dcp=$DCP slr=$SLR period=$PERIOD" >> $OUTROOT/run.log
mkdir -p $OUTROOT
( cd $OUTROOT && nice -n 5 vivado -mode batch -nojournal -log $NAME.vivado.log -source $ROOT/tools/r42/area/tile_screen.tcl -tclargs $DCP $SLR $PERIOD $OUT 6 > /dev/null 2>&1 )
echo "$(date -u +%FT%TZ) end $NAME exit $?" >> $OUTROOT/run.log
