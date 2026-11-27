#!/bin/bash
# le_final.sh TREE LOGDIR : every mode of tools/r42/lineelig_tests.py on the final test bench
cd ~/r42/$1 || exit 2; L=$2; mkdir -p $L
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/tools/asm
for m in directed core; do nice -n 19 python3 tools/r42/lineelig_tests.py $m > $L/$m.log 2>&1; echo "$m exit $?" | tee -a $L/summary.txt; done
nice -n 19 python3 tools/r42/lineelig_tests.py miter --seeds 3 > $L/miter.log 2>&1; echo "miter exit $?" | tee -a $L/summary.txt
nice -n 19 python3 tools/r42/lineelig_tests.py lock --ref ~/r42/lineelig_out/tnca_tensor_ref.v --seeds 3 > $L/lock.log 2>&1; echo "lock exit $?" | tee -a $L/summary.txt
nice -n 19 python3 tools/r42/lineelig_tests.py mutants --seeds 3 > $L/mutants.log 2>&1; echo "mutants exit $?" | tee -a $L/summary.txt
echo FINAL_DONE | tee -a $L/summary.txt
