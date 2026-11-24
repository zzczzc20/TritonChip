#!/bin/bash
# usage: xf_rtlq.sh JOBFILE NPAR   each line: OUTNAME POINT [VAR=VAL ...]   single-core RTL runs (xf_pt.sh), NPAR at a time
JF=$1; N=${2:-2}
cat $JF | xargs -P $N -L 1 $HOME/r42/xfer_out/xf_pt.sh $HOME/r42/xfer
echo RTLQ_DONE
