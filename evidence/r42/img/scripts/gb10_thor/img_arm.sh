#!/bin/bash
# img_arm.sh ARM SETNAME POINT... : one arm of the combined-image study on the three-core stand-in (rtl-3c), records in ~/r42/img_out/mc/ARM[_SETNAME]
#   ARM: A (R42 head defaults, merged tree), A0 (the origin/R42 tree itself), L, M, F, LM, LF, LMF. Elaborations are cached (R42_VVP_CACHE).
ARM=$1; TAG=$2; shift 2
case $ARM in
  A|A0) IMG=base; PROF=core-vrf-sync-v5-x8-g2c4-dp;;
  L)   IMG=l;   PROF=core-r42-line32;;
  M)   IMG=m;   PROF=core-r42;;
  F)   IMG=f;   PROF=core-r42-fp32;;
  LM)  IMG=lm;  PROF=core-r42;;
  LF)  IMG=lf;  PROF=core-r42-fp32;;
  LMF) IMG=lmf; PROF=core-r42-img;;
  *) echo bad arm; exit 2;;
esac
TREE=img_mc; [ $ARM = A0 ] && TREE=img_base
export R42_VVP_CACHE=$HOME/r42/img_out/vvpcache
O=$HOME/r42/img_out/mc/${ARM}_${TAG}
exec $HOME/r42/img_out/img_mc.sh $TREE $O $HOME/r42/img_mc/tools/r42/images/r42img_${IMG}_sim.tcl $PROF - "$@"
