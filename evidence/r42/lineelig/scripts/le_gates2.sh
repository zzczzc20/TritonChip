#!/bin/bash
# le_gates2.sh TREE LOGDIR : the RTL lanes of R42_COMMON.md plus this branch's lanes, default parameters, final tree
cd ~/r42/$1 || exit 2; L=$2; mkdir -p $L
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/tools/asm
for t in unit core-g2c4 core-run2 core-mmadp core-abuf16 r41-xpose8-rtl r38-xpose-rtl; do
  nice -n 19 make $t > $L/$t.log 2>&1; echo "make $t exit $?" | tee -a $L/summary.txt
done
nice -n 19 python3 tools/r41/tensor_line_tests.py > $L/tensor_line_tests.log 2>&1; echo "tensor_line_tests exit $?" | tee -a $L/summary.txt
nice -n 19 python3 tools/r40/tensor_bounds_tests.py > $L/tensor_bounds_tests.log 2>&1; echo "tensor_bounds_tests exit $?" | tee -a $L/summary.txt
nice -n 19 python3 tools/r42/lineelig_tests.py mutants --seeds 3 > $L/mutants.log 2>&1; echo "mutants exit $?" | tee -a $L/summary.txt
echo GATES_DONE | tee -a $L/summary.txt
