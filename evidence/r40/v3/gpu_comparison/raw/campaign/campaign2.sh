#!/bin/sh
# R40-10 item 4 GPU campaign, gated: one point at a time, only while no other user holds the GPU.
. ~/tc_env/env.sh
SRC=~/r40v3_gpu
R=~/r40v3_gpu_runs
cd $SRC
exec $TORCH_PYTHON -u tools/r40/v3_gpu_campaign.py --raw $R/raw \
  --stage boundary:inductor-cudagraph:$R/points_unique.txt:boundary_inductor-cudagraph.jsonl \
  --stage archbench:inductor-cudagraph:$SRC/points_existing.txt:archbench_inductor-cudagraph_existing.jsonl \
  --stage archbench:inductor-cudagraph:$SRC/points_hbm-core.txt:archbench_inductor-cudagraph_hbm-core.jsonl \
  --stage archbench:inductor-cudagraph:$SRC/points_large.txt:archbench_inductor-cudagraph_large.jsonl \
  --stage boundary:inductor-triton-bigsm:$R/points_unique.txt:boundary_inductor-triton-bigsm.jsonl \
  --stage archbench:inductor-triton:$SRC/points_existing.txt:archbench_inductor-triton_existing.jsonl \
  --stage archbench:inductor-triton:$SRC/points_hbm-core.txt:archbench_inductor-triton_hbm-core.jsonl \
  --stage archbench:inductor-triton:$SRC/points_large.txt:archbench_inductor-triton_large.jsonl \
  --stage boundary:inductor-triton:$R/points_unique.txt:boundary_inductor-triton.jsonl
