#!/bin/sh
# Frozen inductor-triton stages again, hermetic compile caches (the first pass hit bigsm FX-graph cache entries).
. ~/tc_env/env.sh
SRC=~/r40v3_gpu
R=~/r40v3_gpu_runs
cd $SRC
exec $TORCH_PYTHON -u tools/r40/v3_gpu_campaign.py --raw $R/raw --fresh-cache \
  --stage archbench:inductor-triton:$SRC/points_existing.txt:archbench_inductor-triton_existing.jsonl \
  --stage archbench:inductor-triton:$SRC/points_hbm-core.txt:archbench_inductor-triton_hbm-core.jsonl \
  --stage archbench:inductor-triton:$SRC/points_large.txt:archbench_inductor-triton_large.jsonl \
  --stage boundary:inductor-triton:$R/points_unique.txt:boundary_inductor-triton.jsonl
