#!/bin/bash
# Run every (format/point, seed) job of kfuse_study.py, 4 processes at a time, nice 19, one thread each.
#   kfuse_run_all.sh OUTDIR [SEEDS="0 1 2"] [FILTER_REGEX]
# Longest jobs (largest K) first. Existing OUTDIR/*.jsonl of a finished job are skipped (a job is finished when its
# .done marker exists). Run from the tree root; PYTHONPATH is set here (xl_tree provides `archbench`).
set -u
OUT=${1:?outdir}; SEEDS=${2:-"0 1 2"}; FILTER=${3:-.}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/xl_tree:$ROOT/tools/r42"
PY=${PY:-$HOME/tc_env/torchenv/bin/python}
JOBS=$(mktemp)
for j in $($PY "$ROOT/tools/r42/kfuse_study.py" --list | grep -E "$FILTER"); do
  K=$(echo "$j" | sed -E 's/.*[kl]([0-9]+)$/\1/')
  for s in $SEEDS; do echo "$K $j $s"; done
done | sort -rn | awk '{print $2, $3}' > "$JOBS"
run_one() {
  j=$1; s=$2; f="$OUT/$(echo "$j" | tr / _)_s$s"
  [ -f "$f.done" ] && exit 0
  rm -f "$f.jsonl"
  nice -n 19 $PY "$ROOT/tools/r42/kfuse_study.py" --job "$j" --seed "$s" --out "$f.jsonl" > "$f.log" 2>&1 && touch "$f.done"
}
export -f run_one; export ROOT OUT PY
xargs -P 4 -L 1 bash -c 'run_one $0 $1' < "$JOBS"
rm -f "$JOBS"
