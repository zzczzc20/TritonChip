#!/bin/bash
set -euo pipefail
cd /home/zzhouch/tnca_srcR40native2b670
source /opt/xilinx/xrt/setup.sh >/dev/null
export CUDA_VISIBLE_DEVICES= PYTHONPATH=python:compiler/python:tools/asm
export TNCA_PROFILE=core-vrf-sync-v5 TNCA_BACKEND=fpga-u280
export TNCA_BACKEND_ARGS='{"xclbin":"/home/zzhouch/tnca_build/u280_r40hybrid/xo/tnca_u280.xclbin","image_profile":"core-vrf-sync-v5","device_clock":true,"hbm_bytes":268435456}'
export ARCHBENCH_BLOCK_ID=native-functional-after-publication
mkdir -p /tmp/r40_native77_after_publication
exec /home/zzhouch/tc_env/torchenv/bin/python -u -m archbench.run --platform tnca --backend fpga-u280 --profile core-vrf-sync-v5 --backend-args "$TNCA_BACKEND_ARGS" --tier tiny small medium --jobs 1 --set TNCA_TORCH_DEVICE=native --set TNCA_RCTX_ROLES=0 --warmup 1 --timed 1 --out /tmp/r40_native77_after_publication/results.jsonl
