#!/bin/bash
# usage: ks_rtl.sh PROFILE OUTNAME POINTID...   single-core RTL simulation of ArchBench points from the kfuse-speed tree (default knobs)
PROFILE=$1; OUTN=$2; shift 2
cd $HOME/r42/kfuse-speed || exit 2
rm -rf python/tnca/data
source $HOME/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
PY=$HOME/tc_env/torchenv/bin/python
O=$HOME/r42/kfuse-speed_out/rtl; mkdir -p $O
$PY -c "import tnca,tnca_cc,tnca_asm;print('tnca from',tnca.__file__,tnca_cc.__file__,tnca_asm.__file__)"
for ID in "$@"; do
  T=$(date +%s)
  nice -n 19 $PY -m archbench.run --platform tnca --profile $PROFILE --point "$ID" --counters --out $O/${OUTN}_$(echo $ID | tr '/' '_').jsonl > $O/${OUTN}_$(echo $ID | tr '/' '_').log 2>&1
  echo "$ID rc=$? seconds=$(( $(date +%s) - T ))"
done
echo RTL_DONE
