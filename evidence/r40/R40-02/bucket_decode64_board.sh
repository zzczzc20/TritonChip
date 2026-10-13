#!/bin/bash
set -euo pipefail
cd /home/zzhouch/tritonchip-r40-board-timed
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
export PATH=/opt/xilinx/xrt/bin:$PATH
export TNCA_BACKEND=fpga-u280 TNCA_PROFILE=core-vrf-sync-v5 TNCA_HBM_BYTES=8589934592 TNCA_DEADLINE=4000000000 TNCA_HOST_TIMEOUT=172800
export TNCA_BACKEND_ARGS='{"xclbin":"/home/zzhouch/tnca_build/u280_r40ram/tnca_u280.xclbin","image_profile":"core-vrf-sync-v5"}'
export TNCA_RCTX_SPMD=0
export TORCHINDUCTOR_CACHE_DIR=$PWD/sim/build/r40/bucket64_board_cache
mkdir -p sim/build/r40
/home/zzhouch/tc_env/torchenv/bin/python -u tools/r40/decode_residency_tests.py --steps 64 --bucket
