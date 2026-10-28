#!/bin/bash
# R41 G2 sizing experiment: compile audits of the scratch variant trees, two at a time (3 jobs each).
cd ~/r41bw_aud
while pgrep -u $(id -u) -f "label g2-usage" >/dev/null; do sleep 30; done
run() { n=$1; t=t_$n; ~/tc_env/torchenv/bin/python $t/tools/performance_harness/default_change_audit.py run \
  --source-root $t --software-tree $(cat $t/TREE) --label $n --out run-$n --jobs 3 \
  --torch-extensions-dir ~/r41s1-audit/ext --profile core-vrf-sync-$n > run-$n.log 2>&1; }
run g2v0-ctx2 & run g2v1-ctx2-acc8 & wait
run g2v2-ctx2-bar8 & run g2v3-ctx2-bar4 & wait
echo done > variants.done
