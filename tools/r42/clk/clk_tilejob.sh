#!/bin/bash
# clk_tilejob.sh NAME "ENVS" DCP SLR   (x86 host, tree ~/r42_clk) -- tools/r42/area/tile_screen.tcl on a synthesized tnca_tile checkpoint
# (DCP relative to ~/r42_clk_out, or absolute) at PERIOD ns, pinned to one SLR pblock. ENVS e.g. "PERIOD=5.6". Optimistic like every OOC tile screen: no shell,
# no other tile, no SLR crossing. THREADS env (default 6).
set -u
NAME=$1; ENVS=$2; DCP=$3; SLR=${4:-0}
source ~/x86_link_env.sh
ROOT=$HOME/r42_clk
OUTROOT=$HOME/r42_clk_out
OUT=$OUTROOT/$NAME
PERIOD=10
for e in $ENVS; do case $e in PERIOD=*) PERIOD=${e#PERIOD=};; esac; done
case $DCP in /*) ;; *) DCP=$OUTROOT/$DCP;; esac
echo "$(date -u +%FT%TZ) start $NAME tile screen dcp=$DCP slr=$SLR period=$PERIOD" >> $OUTROOT/run.log
( cd $OUTROOT && nice -n 5 vivado -mode batch -nojournal -log $NAME.vivado.log -source $ROOT/tools/r42/area/tile_screen.tcl -tclargs $DCP $SLR $PERIOD $OUT ${THREADS:-6} > /dev/null 2>&1 )
echo "$(date -u +%FT%TZ) end $NAME exit $?" >> $OUTROOT/run.log
