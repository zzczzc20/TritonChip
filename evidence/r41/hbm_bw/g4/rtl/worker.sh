#!/bin/bash
# worker.sh W NW : takes every NW-th job starting at W
w=$1; nw=$2
cd ~/r41-g4rtl-2ce25c8 && source ~/tc_env/env.sh >/dev/null 2>&1
i=0
while read m p; do
  if [ $((i % nw)) -eq $w ]; then
    set_args=(); [ "$m" != 0 ] && set_args=(--set TNCA_TILED_POINTWISE=$m)
    ulimit -v 16000000
    ~/tc_env/torchenv/bin/python tools/r41/link_budget_rtl.py --point $p "${set_args[@]}" --out ~/r41-g4-rtl/mode$m.w$w.jsonl >> ~/r41-g4-rtl/w$w.log 2>&1
  fi
  i=$((i+1))
done < ~/r41-g4-rtl/jobs.txt
echo done >> ~/r41-g4-rtl/w$w.log
