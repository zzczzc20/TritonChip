#!/bin/bash
# run_queue.sh QUEUEFILE  (x86 host, tree root): starts the jobs of QUEUEFILE one after the other, never more than two
# Vivado jobs at once. Line format:  NAME|ENV (e.g. "SYNTH_ONLY=1 FLATTEN=none PERIOD=5" or "-")|TOP|GENERICS
cd $(cd $(dirname $0)/../../.. && pwd)
count() { pgrep -fc 'unwrapped/lnx64.o/vivado -mode batch' || true; }
while IFS='|' read -r NAME ENVS TOP GEN; do
    [ -z "$NAME" ] && continue
    while [ "$(count)" -ge 2 ]; do sleep 10; done
    [ "$ENVS" = "-" ] && ENVS=""
    ( env $ENVS setsid nohup bash tools/r42/kfuse_rtl/run_ooc.sh $NAME $TOP $GEN > /dev/null 2>&1 < /dev/null & )
    sleep 8
done < $1
