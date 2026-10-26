#!/bin/bash
set -euo pipefail
cd /home/zzhouch/tnca_srcR40sdpa_f2
source /opt/xilinx/xrt/setup.sh >/dev/null
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=python:compiler/python:tools/asm
export TNCA_PROFILE=core-vrf-sync-v5 TNCA_BACKEND=fpga-u280
export TNCA_BACKEND_ARGS='{"xclbin":"/home/zzhouch/tnca_build/u280_r40hybrid/xo/tnca_u280.xclbin","image_profile":"core-vrf-sync-v5","device_clock":true,"heap_base":524288}'
exec /home/zzhouch/tc_env/torchenv/bin/python tools/r40/online_attention_tests.py --out /tmp/r40_sdpa_board_inventory_hbm.json
