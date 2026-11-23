#!/bin/bash
# Run the whole-layer study (kfuse_layer.py): one process per line of the job list on stdin, NPROC at a time.
#   echo "POINT SEED ATTN CFG1,CFG2,..." | kfuse_layer_run.sh OUTDIR
# Each job appends to OUTDIR/<point>_attn<ATTN>_s<SEED>_<n>.jsonl (n: the chunk index, so chunks do not collide).
set -u
OUT=${1:?outdir}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/xl_tree:$ROOT/tools/r42"
PY=${PY:-$HOME/tc_env/torchenv/bin/python}
run_one() {
  point=$1; seed=$2; attn=$3; cfgs=$4; n=$5
  f="$OUT/$(echo "$point" | tr / _)_attn${attn}_s${seed}_$n"
  [ -f "$f.done" ] && exit 0
  rm -f "$f.jsonl"
  nice -n 19 $PY "$ROOT/tools/r42/kfuse_layer.py" --point "$point" --seed "$seed" --attn "$attn" --cfgs "$cfgs" \
      --refdir "$OUT" --out "$f.jsonl" > "$f.log" 2>&1 && touch "$f.done"
}
export -f run_one; export ROOT OUT PY
nl -w1 -s' ' | xargs -P ${NPROC:-4} -L 1 bash -c 'run_one $1 $2 $3 $4 $0'
