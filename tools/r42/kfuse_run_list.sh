#!/bin/bash
# Run an explicit list of kfuse_study.py jobs, 4 processes at a time, nice 19, one thread each.
#   kfuse_run_list.sh OUTDIR "JOB1 JOB2 ..." "SEEDS" [extra kfuse_study.py args, e.g. --only-ws -6,-4,-2]
set -u
OUT=${1:?outdir}; JOBS=${2:?jobs}; SEEDS=${3:-"0 1 2"}; shift 3; EXTRA="$*"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/xl_tree:$ROOT/tools/r42"
PY=${PY:-$HOME/tc_env/torchenv/bin/python}
run_one() {
  j=$1; s=$2; f="$OUT/$(echo "$j" | tr / _)_s$s"
  [ -f "$f.done" ] && exit 0
  rm -f "$f.jsonl"
  nice -n 19 $PY "$ROOT/tools/r42/kfuse_study.py" --job "$j" --seed "$s" --out "$f.jsonl" $EXTRA > "$f.log" 2>&1 && touch "$f.done"
}
export -f run_one; export ROOT OUT PY EXTRA
for j in $JOBS; do for s in $SEEDS; do echo "$j $s"; done; done | xargs -P ${NPROC:-4} -L 1 bash -c 'run_one $0 $1'
