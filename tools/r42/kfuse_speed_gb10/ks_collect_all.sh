#!/bin/bash
# collect every finished tlm/<name> that has no collected json yet or is newer: usage ks_collect_all.sh [NAME...]
cd $HOME/r42/kfuse-speed
source $HOME/tc_env/env.sh
PY=$HOME/tc_env/torchenv/bin/python
O=$HOME/r42/kfuse-speed_out
mkdir -p $O/collected
NAMES="$@"
[ -z "$NAMES" ] && NAMES=$(ls $O/tlm)
for N in $NAMES; do
  [ -f $O/tlm/$N/results.json ] || continue
  $PY tools/r42/kfuse_speed_collect.py $O/tlm/$N $O/collected/$N.json > /dev/null && echo collected $N
done
