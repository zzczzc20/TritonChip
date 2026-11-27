#!/bin/bash
# clk_queue.sh QUEUE [MAXJOBS=2]   (x86 host, tree ~/r42_clk) -- starts the lines of QUEUE one after the other:  NAME|ENVS|TOP|FILES|GENERICS
# Own jobs only: PID files in ~/r42_clk_out/.pids (`kill -0` is an existence test, no signal is ever sent), at most MAXJOBS at once (R42-clk limit: 2) and only
# while MemAvailable >= MINAVAIL_MB (default 16000). Reads the stop file ~/r42_clk_out/STOP between jobs: create it to stop launching (running jobs finish).
# A NAME whose output directory exists is skipped. NAME prefix tscr_ = tile screen (TOP = synth.dcp path relative to ~/r42_clk_out or absolute, FILES = SLR number).
Q=$1; MAX=${2:-2}
MINAVAIL_MB=${MINAVAIL_MB:-16000}
OUTROOT=$HOME/r42_clk_out; PIDS=${PIDS_DIR:-$OUTROOT/.pids}; STOP=${STOP_FILE:-$OUTROOT/STOP}
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
        tscr_*) JOB=clk_tilejob.sh; set -- "$NAME" "$ENVS" "$TOP" "$FILES";;
        *)      JOB=clk_job.sh;     set -- "$NAME" "$ENVS" "$TOP" "$FILES" "$GEN";;
    esac
    setsid nohup bash $HOME/r42_clk/tools/r42/clk/$JOB "$@" > /dev/null 2>&1 < /dev/null &
    echo $! > $PIDS/$NAME.pid
    echo "$(date -u +%T) started $NAME pid $!"
    sleep 5
done < $Q
echo "$(date -u +%T) queue $Q: all started"
