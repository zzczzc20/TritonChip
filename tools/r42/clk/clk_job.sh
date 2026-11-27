#!/bin/bash
# clk_job.sh NAME "ENVS" TOP "FILES" "GENERICS"   (x86 host, tree ~/r42_clk) -- one OOC run of tools/r42/area/ooc.tcl (synthesis, or synthesis+P&R without SYNTH_ONLY).
# FILES: comma-separated paths (globs allowed) relative to the tree root. ENVS e.g. "PERIOD=5.6 SYNTH_ONLY=1 FLATTEN=none" (or "-"). THREADS env (default 6).
set -u
NAME=$1; ENVS=$2; TOP=$3; FILES=$4; GEN=$5
[ "$ENVS" = "-" ] && ENVS=""
source ~/x86_link_env.sh
ROOT=$HOME/r42_clk
OUTROOT=$HOME/r42_clk_out
OUT=$OUTROOT/$NAME
mkdir -p $OUT
PERIOD=10
for e in $ENVS; do case $e in PERIOD=*) PERIOD=${e#PERIOD=};; esac; done
FL=""
for f in ${FILES//,/ }; do FL="$FL $(ls $ROOT/$f)"; done
echo "$(date -u +%FT%TZ) start $NAME top=$TOP generics=$GEN env=$ENVS" >> $OUTROOT/run.log
( cd $OUT && env $ENVS OOC_GENERICS="$GEN" INCDIRS="$ROOT/rtl/common $ROOT/rtl/noc" nice -n 5 vivado -mode batch -nojournal -nolog -source $ROOT/tools/r42/area/ooc.tcl -tclargs $TOP $PERIOD xcu280-fsvh2892-2L-e $OUT ${THREADS:-6} $FL > $OUT/vivado.log 2>&1 )
echo "$(date -u +%FT%TZ) end $NAME exit $?" >> $OUTROOT/run.log
