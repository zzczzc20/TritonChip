#!/bin/bash
# img_gates.sh TREE LOGDIR TARGET... : make targets of the merged RTL, one log each, summary.txt
cd ~/r42/$1 || exit 2; L=$2; shift 2; mkdir -p $L
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/tools/asm
for t in "$@"; do
  nice -n 19 make $t > $L/$t.log 2>&1; echo "make $t exit $? $(date -u +%H:%M:%S)" | tee -a $L/summary.txt
done
echo GATES_DONE | tee -a $L/summary.txt
