#!/bin/bash
# img_1c.sh TREE PROFILE TIER JOBS OUTNAME [archbench.run args]: ArchBench on the single-core RTL simulator (rtl-1c), default knobs,
# records to ~/r42/img_out/c1/OUTNAME.jsonl (r42_run.sh of the shared scripts, with an output directory of its own)
TREE=$1; PROFILE=$2; TIER=$3; JOBS=$4; OUT=$5; shift 5
cd ~/r42/$TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
PY=$HOME/tc_env/torchenv/bin/python
mkdir -p ~/r42/img_out/c1
$PY -c "import tnca,tnca_cc,tnca_asm;print('tnca from',tnca.__file__,tnca_cc.__file__,tnca_asm.__file__)"
nice -n 19 $PY -c "import sys;from tnca.cli import sim_main;sys.argv=['tnca-sim','prepare','--profile','$PROFILE'];sim_main()" 2>&1 | tail -2
rm -f ~/r42/img_out/c1/$OUT.jsonl
TIERARG=""; [ "$TIER" != "-" ] && TIERARG="--tier $TIER"
nice -n 19 $PY -m archbench.run --platform tnca --profile $PROFILE $TIERARG --jobs $JOBS --out ~/r42/img_out/c1/$OUT.jsonl "$@"
echo DONE $?
