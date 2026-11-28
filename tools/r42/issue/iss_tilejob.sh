#!/bin/bash
# iss_tilejob.sh NAME "ENVS" DCP SLR   (x86 host) -- tools/r42/area/tile_screen.tcl on a synthesized tnca_tile checkpoint (DCP relative to ~/r42_issue_out) at PERIOD ns,
# pinned to one SLR pblock. ENVS e.g. "PERIOD=6.67". Optimistic like every OOC tile screen: no shell, no other tile, no SLR crossing.
set -u
NAME=$1; ENVS=$2; DCP=$3; SLR=${4:-0}
source ~/x86_link_env.sh
ROOT=$HOME/r42_issue
OUTROOT=$HOME/r42_issue_out
OUT=$OUTROOT/$NAME
PERIOD=10
for e in $ENVS; do case $e in PERIOD=*) PERIOD=${e#PERIOD=};; esac; done
echo "$(date -u +%FT%TZ) start $NAME tile screen dcp=$DCP slr=$SLR period=$PERIOD" >> $OUTROOT/run.log
( cd $OUTROOT && nice -n 5 vivado -mode batch -nojournal -log $NAME.vivado.log -source $ROOT/tools/r42/area/tile_screen.tcl -tclargs $OUTROOT/$DCP $SLR $PERIOD $OUT ${THREADS:-4} > /dev/null 2>&1 )
echo "$(date -u +%FT%TZ) end $NAME exit $?" >> $OUTROOT/run.log
