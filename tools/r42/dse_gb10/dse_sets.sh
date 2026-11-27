#!/bin/bash
# DSE worker: run the sets of the 145-point model bundle for one (tree, configuration) on GB10 (domain: SystemC/TLM model).
# usage: dse_sets.sh TREEDIR TREETAG CFGNAME SET [SET...]       sets: tsm hv px pxv xd xp  (ids/*.txt; px/pxv/xp use the proxy module)
# env: BIN (default tnca-chip-v5), SEEDTAG (reuse the compile caches of <SEEDTAG>_<set> runs of the same tree), CORES (default nx*ny), JOBS
TREE=$1; TT=$2; CFGN=$3; shift 3
OUTD=$HOME/r42/dse_out
BIN=${BIN:-$OUTD/tnca-chip-v5}
PM=$HOME/r42/dse/tools/r42/dse_proxies.py
for S in "$@"; do
  case $S in
    tsm) IDS=tsm77.txt; P="";;
    hv)  IDS=hv.txt; P="";;
    xd)  IDS=xd.txt; P="";;
    px)  IDS=px.txt; P=$PM;;
    pxv) IDS=pxv.txt; P=$PM;;
    xp)  IDS=xp.txt; P=$PM;;
    *) echo "unknown set $S"; continue;;
  esac
  SD=""; [ -n "$SEEDTAG" ] && [ -d $OUTD/tlm/${SEEDTAG}_$S ] && SD=$OUTD/tlm/${SEEDTAG}_$S
  SEED=$SD JOBS=${JOBS:-3} $HOME/r42/dse/tools/r42/dse_gb10/dse_run.sh $TREE $OUTD/cfg/$CFGN.cfg ${TT}_${CFGN}_$S $BIN $OUTD/ids/$IDS $P
  echo "SET $S of ${TT}_$CFGN done"
done
echo SETS_DONE
