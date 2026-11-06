#!/bin/bash
w=$1; nw=$2; T=$3
cd $T && source ~/tc_env/env.sh >/dev/null 2>&1
i=0
while read img m p; do
  if [ $((i % nw)) -eq $w ]; then
    set_args=(); [ "$m" != 0 ] && set_args=(--set TNCA_TILED_POINTWISE=$m)
    ~/tc_env/torchenv/bin/python tools/r41/link_budget_rtl.py --image evidence/r41/hbm_bw/g4/images/image_$img.tcl --profile core-vrf-sync-v5-x8-g2 --point $p "${set_args[@]}" --out ~/r41-g4-rtl5/$img.mode$m.w$w.jsonl >> ~/r41-g4-rtl5/w$w.log 2>&1
  fi
  i=$((i+1))
done < ~/r41-g4-rtl5/jobs.txt
echo done >> ~/r41-g4-rtl5/w$w.log
