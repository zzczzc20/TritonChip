#!/bin/bash
set -u
source ~/tc_env/env.sh
cd ~/tritonchip-r40-convergence-85fa7d8 || exit 2
export PYTHONPATH=python:compiler/python:tools/asm
export CUDA_VISIBLE_DEVICES= TNCA_BANK_ALLOC=0 TNCA_WINDOW_REUSE=0 TNCA_PACKED_WINDOW_UNPACK=0
~/tc_env/torchenv/bin/python -u tools/r40/contract_tests.py --reference /tmp/r40_contract_reference_frozen.json
r40_contract_rc=$?
printf '%s\n' "$r40_contract_rc" > /tmp/r40_contract85.exit
exit "$r40_contract_rc"
