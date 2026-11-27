#!/bin/bash
# shr_queue.sh QUEUE [MAXJOBS=2]   (x86 host) -- starts the lines of QUEUE one after the other:  NAME|ENVS|TOP|FILES|GENERICS
# Counts only the jobs of this queue (PID files in ~/r42_shrink_out/.pids; `kill -0` is an existence test, no signal), at most MAXJOBS at once.
# Reads the stop file ~/r42_shrink_out/STOP between jobs: create it to stop launching (running jobs finish). A NAME whose output directory exists is skipped.
Q=$1; MAX=${2:-2}
OUTROOT=$HOME/r42_shrink_out; PIDS=$OUTROOT/.pids; STOP=$OUTROOT/STOP
mkdir -p $PIDS
mine() { n=0; for f in $PIDS/*.pid; do [ -f "$f" ] || continue; p=$(cat $f); if kill -0 $p 2>/dev/null; then n=$((n+1)); else rm -f $f; fi; done; echo $n; }
while IFS='|' read -r NAME ENVS TOP FILES GEN; do
    [ -z "$NAME" ] && continue
    case $NAME in \#*) continue;; esac
    if [ -d $OUTROOT/$NAME ]; then echo "$(date -u +%T) skip $NAME (exists)"; continue; fi
    while true; do
        if [ -f $STOP ]; then echo "$(date -u +%T) stop file present, not starting $NAME"; exit 0; fi
        [ "$(mine)" -lt "$MAX" ] && break
        sleep 15
    done
    setsid nohup bash $HOME/r42_shrink/tools/r42/shrink/shr_job.sh "$NAME" "$ENVS" "$TOP" "$FILES" "$GEN" > /dev/null 2>&1 < /dev/null &
    echo $! > $PIDS/$NAME.pid
    echo "$(date -u +%T) started $NAME pid $!"
    sleep 5
done < $Q
echo "$(date -u +%T) queue $Q: all started"
