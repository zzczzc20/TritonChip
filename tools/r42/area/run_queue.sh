#!/bin/bash
# run_queue.sh QUEUEFILE  (x86 host, tree root): starts the jobs of QUEUEFILE one after the other, never more than two
# Vivado jobs on the host at once (other workers' jobs count). A host-wide flock makes the check-and-start of concurrent
# queue processes atomic. Line format:  NAME|ENV (e.g. "SYNTH_ONLY=1 FLATTEN=none PERIOD=5" or "-")|TOP|GENERICS
cd $(cd $(dirname $0)/../../.. && pwd)
count() { pgrep -fc 'unwrapped/lnx64.o/vivado -mode batch' || true; }
while IFS='|' read -r NAME ENVS TOP GEN; do
    [ -z "$NAME" ] && continue
    [ "$ENVS" = "-" ] && ENVS=""
    (
        flock 9
        while [ "$(count)" -ge 2 ]; do sleep 10; done
        case $NAME in tile_*) R=run_tile.sh;; dcp_*) R=run_dcp.sh;; kern_*) R=run_kernel.sh;; *) R=run_ooc.sh;; esac
        ( env $ENVS setsid nohup bash tools/r42/area/$R $NAME $TOP $GEN > /dev/null 2>&1 < /dev/null 9>&- & )
        sleep 15
    ) 9>/tmp/r42_area_launch2.lock
done < $1
