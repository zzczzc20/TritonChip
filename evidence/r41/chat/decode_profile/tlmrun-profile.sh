#!/bin/bash
# usage: tlmrun-profile.sh NAME PROFILE CORES -- cmd args...   (GB10; tree in ~/r41-fp8prof/src)
# Same environment as evidence/r41/chat/p3/tlmrun-default.sh (default TNCA settings; model/harness settings
# identical), with run directories under ~/r41-fp8prof/runs.
NAME=$1; PROFILE=$2; CORES=$3; shift 4
O=$HOME/r41-fp8prof/runs/$NAME; mkdir -p $O
cd $HOME/r41-fp8prof/src
exec systemd-run --user --unit=fp8prof-$NAME -p MemoryMax=${MEMMAX:-16G} --working-directory=$HOME/r41-fp8prof/src \
  /usr/bin/env HF_HOME=$HOME/flux-hf HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES= QWEN_FP8_VIEW_DIR=$HOME/r41-p3/snapview TNCA_BACKEND=systemc-tlm-grid \
  TNCA_PROFILE=$PROFILE TNCA_TLM_BINARY=$HOME/r41-p3/build-tlm/tnca-chip TNCA_TLM_MEMORY_BYTES=0x200000000 \
  TNCA_TLM_OUT=$O/tlm-out TNCA_DUMP_DIR=$O/dumps TNCA_KERNEL_DUMP=1 TNCA_DRAM_PATH=fast TNCA_TLM_UNTIMED_COLD=0 \
  TNCA_DEADLINE=40000000000 \
  TNCA_HOST_TIMEOUT=${HOST_TIMEOUT:-3600} TRITON_CACHE_DIR=$O/triton TORCHINDUCTOR_CACHE_DIR=$O/inductor \
  taskset -c $CORES bash -c "\"\$@\" > $O/run.log 2>&1; echo exit=\$? >> $O/run.log" _ "$@"
