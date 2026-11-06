#!/bin/bash
w=$1; nw=$2; T=$3; D=~/r41-c4
cd $T && source ~/tc_env/env.sh >/dev/null 2>&1
i=0
while read arm img prof p; do
  if [ $((i % nw)) -eq $w ]; then
    ~/tc_env/torchenv/bin/python tools/r41/link_budget_rtl.py --image $img --profile $prof --point $p --out $D/$arm.w$w.jsonl >> $D/w$w.log 2>&1
    echo "job $arm $p rc=$?" >> $D/w$w.log
  fi
  i=$((i+1))
done < $D/jobs.txt
echo done >> $D/w$w.log
