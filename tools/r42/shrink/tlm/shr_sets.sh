#!/bin/bash
# shr_sets.sh CFGNAME SET [SET...]  (GB10) -- the sets of the 145-point model bundle for configuration ~/r42/shrink_out/cfg/dse_<CFGNAME>.cfg with the head software of the dse worker
# (tree ~/r42/dse), output tag head_dse_<CFGNAME>_<set>. Sets: tsm hv px xd xp (id lists and proxy module of the dse worker, read only).
# env: BIN (default the dse worker's tnca-chip-v5), JOBS (default 3), CORES (default nx*ny), SEEDTAG (reuse compile caches of ~/r42/dse_out/tlm/<SEEDTAG>_<set>, default head_dse_base)
CFGN=$1; shift
TREE=$HOME/r42/dse
DOUT=$HOME/r42/dse_out
BIN=${BIN:-$DOUT/tnca-chip-v5}
PM=$TREE/tools/r42/dse_proxies.py
SEEDTAG=${SEEDTAG:-head_dse_base}
for S in "$@"; do
  case $S in
    tsm) IDS=tsm77.txt; P="";;
    hv)  IDS=hv.txt; P="";;
    xd)  IDS=xd.txt; P="";;
    px)  IDS=px.txt; P=$PM;;
    xp)  IDS=xp.txt; P=$PM;;
    *) echo "unknown set $S"; continue;;
  esac
  SD=""; [ -d $DOUT/tlm/${SEEDTAG}_$S ] && SD=$DOUT/tlm/${SEEDTAG}_$S
  SEED=$SD JOBS=${JOBS:-3} $HOME/r42/shrink/tools/r42/shrink/tlm/shr_run.sh $TREE $HOME/r42/shrink_out/cfg/dse_$CFGN.cfg head_dse_${CFGN}_$S $BIN $DOUT/ids/$IDS $P
  echo "SET $S of $CFGN done"
done
echo SETS_DONE $CFGN
