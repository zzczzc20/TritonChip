#!/bin/bash
# c2_queue.sh QUEUE [MAXJOBS=2]   (x86 host, tree ~/r42_clk2) -- starts the lines of QUEUE one after the other:  NAME|ENVS|TOP|FILES|GENERICS
# Own jobs only: PID files in ~/r42_clk2_out/.pids (`kill -0` is an existence test, no signal is ever sent), at most MAXJOBS at once (per queue process; run one queue for tile screens, one for units) and only
# while MemAvailable >= MINAVAIL_MB (default 12000). Reads the stop file ~/r42_clk2_out/STOP between jobs: create it to stop launching (running jobs finish).
# A NAME whose output directory exists is skipped. NAME prefix tscr_ = tile screen (TOP = synth.dcp path relative to ~/r42_clk2_out or absolute, FILES = SLR number).
Q=$1; MAX=${2:-1}
MINAVAIL_MB=${MINAVAIL_MB:-12000}
OUTROOT=${C2OUT:-$HOME/r42_clk2_out}; PIDS=${PIDS_DIR:-$OUTROOT/.pids}; STOP=${STOP_FILE:-$OUTROOT/STOP}
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
        tscr_*) JOB=c2_tilejob.sh; set -- "$NAME" "$ENVS" "$TOP" "$FILES";;
        *)      JOB=c2_job.sh;     set -- "$NAME" "$ENVS" "$TOP" "$FILES" "$GEN";;
    esac
    setsid nohup bash ${C2ROOT:-$HOME/r42_clk2}/tools/r42/clk2/$JOB "$@" > /dev/null 2>&1 < /dev/null &
    echo $! > $PIDS/$NAME.pid
    echo "$(date -u +%T) started $NAME pid $!"
    sleep 5
done < $Q
echo "$(date -u +%T) queue $Q: all started"
