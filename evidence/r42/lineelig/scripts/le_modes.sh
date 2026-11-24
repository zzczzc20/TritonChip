#!/bin/bash
# le_modes.sh TREE SEEDS : the unit modes of tools/r42/lineelig_tests.py
cd ~/r42/$1 || exit 2
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/tools/asm
for m in directed core; do nice -n 19 python3 tools/r42/lineelig_tests.py $m; done
nice -n 19 python3 tools/r42/lineelig_tests.py miter --seeds $2
nice -n 19 python3 tools/r42/lineelig_tests.py lock --ref ~/r42/lineelig_out/tnca_tensor_ref.v --seeds $2
echo MODES_DONE
