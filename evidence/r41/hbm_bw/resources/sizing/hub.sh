#!/bin/bash
# hub.sh NAME TREE PROFILE : compile-only Hub FP8 kernels for one profile
cd ~/r41bw_aud; n=$1; t=$2; p=$3; rm -rf hub_$n
source ~/tc_env/env.sh >/dev/null 2>&1
export HF_HOME=$HOME/flux-hf HF_HUB_OFFLINE=0 OMP_NUM_THREADS=2 TORCH_EXTENSIONS_DIR=$HOME/r41-sw-work/val/ext PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=$PWD/hub_$n/triton TORCHINDUCTOR_CACHE_DIR=$PWD/hub_$n/ind
TNCA_PROFILE=$p ~/flux-venv/bin/python -I fp8_hub_compile_only.py --tree $t --dump hub_$n/dump > hub_$n.log 2>&1
echo "exit $?" >> hub_$n.log
