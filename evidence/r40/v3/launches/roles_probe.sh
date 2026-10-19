cd ~/tnca_src_r40_v3x2
export TORCH_EXTENSIONS_DIR=$HOME/.cache/tnca/r40_conditional_6bd07b7_extensions MAX_JOBS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export TNCA_OPERATING_CLOCK_EVIDENCE=$HOME/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json TNCA_DEADLINE=4000000000 TNCA_HOST_TIMEOUT=600
export PYTHONPATH=$PWD:$PWD/python:$PWD/compiler/python:$PWD/tools/asm
A='{"xclbin": "/home/zzhouch/tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin", "image_profile": "core-vrf-sync-v5", "core_count": 3, "memory_ports": 3, "hbm_dma": true, "hbm_bytes": 4294967296, "device_clock": true}'
K="--set TNCA_TORCH_DEVICE=native --set TNCA_TRANSFER_POLICY=per-kernel --set TNCA_ACTIVE_CORES=1 --set TNCA_RTL_MEM_PORTS=1 --set TNCA_RTL_OUTSTANDING=1 --set TNCA_MAX_BLOCK=16 --set TNCA_GEMM_TILE=16,16,16 --set TNCA_SDPA=decomposed --set TNCA_SUBMIT=single"
for P in 1 2; do
  timeout 600 ~/tc_env/torchenv/bin/python -u -m archbench.run --platform tnca --backend fpga-u280-multicore --backend-args "$A" --profile core-vrf-sync-v5 \
    --point op.gemv/tiny/fp16-k64-n64 op.gemv/medium/fp16-k320-n448 --jobs 1 --warmup 0 --timed 1 --timeout 600 \
    $K --set TNCA_RCTX_ROLES=2 --set TNCA_PIPELINE=$P --out ~/tnca_v3_probe/roles2_p$P.jsonl
  echo "pipeline=$P exit=$?"
done
