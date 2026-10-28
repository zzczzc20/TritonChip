#!/bin/bash
# usage: rtlrun.sh NAME PROFILE BACKEND CORES -- cmd args...
NAME=$1; PROFILE=$2; BACKEND=$3; CORES=$4; shift 5
O=$HOME/r41-p3/runs/$NAME; mkdir -p $O
exec systemd-run --user --unit=p3-$NAME -p MemoryMax=${MEMMAX:-16G} --working-directory=$HOME/r41-p3/src \
  bash -c "source \$HOME/tc_env/env.sh >/dev/null 2>&1; exec /usr/bin/env HF_HOME=\$HOME/flux-hf HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES= QWEN_FP8_VIEW_DIR=\$HOME/r41-p3/snapview \
  TNCA_BACKEND=$BACKEND TNCA_PROFILE=$PROFILE TNCA_IEEE_DIV=1 TNCA_TORCH_DEVICE=native TNCA_INDEX_OPT=1 TNCA_TRITON=1 \
  TNCA_HOST_TIMEOUT=${HOST_TIMEOUT:-14400} TRITON_CACHE_DIR=$O/triton TORCHINDUCTOR_CACHE_DIR=$O/inductor \
  taskset -c $CORES bash -c '\"\$@\" > $O/run.log 2>&1; echo exit=\$? >> $O/run.log' _ \"\$@\"" _ "$@"
