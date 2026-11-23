#!/bin/bash
# usage: arch_all.sh OUTDIR LISTFILE JOBS
OUTDIR=$1; LIST=$2; JOBS=$3
rm -f $HOME/r42/arch_out/$OUTDIR.done
cat $LIST | xargs -P $JOBS -I{} $HOME/r42/arch_out/arch_pt.sh $OUTDIR {}
echo ALLDONE >> $HOME/r42/arch_out/$OUTDIR.done
