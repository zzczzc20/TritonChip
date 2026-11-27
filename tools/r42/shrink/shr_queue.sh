#!/bin/bash
# shr_queue.sh QUEUE [MAXJOBS=2]   (x86 host) -- starts the lines of QUEUE one after the other:  NAME|ENVS|TOP|FILES|GENERICS
# Counts only the jobs of this queue (PID files in ~/r42_shrink_out/.pids; `kill -0` is an existence test, no signal), at most MAXJOBS (default 1) at once and only while MemAvailable >= MINAVAIL_MB (default 16000).
# Reads the stop file ~/r42_shrink_out/STOP between jobs: create it to stop launching (running jobs finish). A NAME whose output directory exists is skipped.
Q=$1; MAX=${2:-1}
MINAVAIL_MB=${MINAVAIL_MB:-16000}      # do not start a job while MemAvailable is below this (a synthesis of a RAM-heavy unit grows to ~10 GB in child processes; other workers share the host)
OUTROOT=$HOME/r42_shrink_out; PIDS=${PIDS_DIR:-$OUTROOT/.pids}; STOP=${STOP_FILE:-$OUTROOT/STOP}
mkdir -p $PIDS
mine() { n=0; for f in $PIDS/*.pid; do [ -f "$f" ] || continue; p=$(cat $f); if kill -0 $p 2>/dev/null; then n=$((n+1)); else rm -f $f; fi; done; echo $n; }
while IFS='|' read -r NAME ENVS TOP FILES GEN; do
    [ -z "$NAME" ] && continue
    case $NAME in \#*) continue;; esac
    if [ -d $OUTROOT/$NAME ]; then echo "$(date -u +%T) skip $NAME (exists)"; continue; fi
    while true; do
        if [ -f $STOP ]; then echo "$(date -u +%T) stop file present, not starting $NAME"; exit 0; fi
        av=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
        [ "$(mine)" -lt "$MAX" ] && [ "$av" -ge "$MINAVAIL_MB" ] && break
        sleep 15
    done
    case $NAME in
        tscr_*) JOB=shr_tilejob.sh; set -- "$NAME" "$ENVS" "$TOP" "$FILES";;       # TOP = checkpoint relative to ~/r42_shrink_out, FILES = SLR
        *)      JOB=shr_job.sh;     set -- "$NAME" "$ENVS" "$TOP" "$FILES" "$GEN";;
    esac
    setsid nohup bash $HOME/r42_shrink/tools/r42/shrink/$JOB "$@" > /dev/null 2>&1 < /dev/null &
    echo $! > $PIDS/$NAME.pid
    echo "$(date -u +%T) started $NAME pid $!"
    sleep 5
done < $Q
echo "$(date -u +%T) queue $Q: all started"
