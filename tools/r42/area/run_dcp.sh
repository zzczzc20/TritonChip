#!/bin/bash
# run_dcp.sh NAME DCP LIMIT_NS   (x86 host, tree root ~/r42_area)
# Read-only timing census of a routed checkpoint (dcp_timing.tcl); never writes a checkpoint. 6 threads, nice 5.
set -u
NAME=$1; DCP=$2; LIM=${3:-3.5}
source ~/x86_link_env.sh
ROOT=$(cd $(dirname $0)/../../.. && pwd)
OUTROOT=${OUTROOT:-$HOME/r42_area_out}
OUT=$OUTROOT/$NAME
n=$(pgrep -fc 'unwrapped/lnx64.o/vivado -mode batch' || true)
if [ "${n:-0}" -ge 2 ]; then echo "two Vivado jobs already running, not starting $NAME"; exit 3; fi
mkdir -p $OUTROOT
echo "$(date -u +%FT%TZ) start $NAME dcp=$DCP limit=$LIM" >> $OUTROOT/run.log
( cd $OUTROOT && nice -n 5 vivado -mode batch -nojournal -log $NAME.vivado.log -source $ROOT/tools/r42/area/dcp_timing.tcl -tclargs $DCP $OUT $LIM 6 > /dev/null 2>&1 )
echo "$(date -u +%FT%TZ) end $NAME exit $?" >> $OUTROOT/run.log
