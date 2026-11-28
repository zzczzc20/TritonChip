#!/bin/bash
# iss_queue.sh QUEUE [MAXJOBS=2]   (x86 host) -- starts the lines of QUEUE one after the other:  NAME|ENVS|TOP|FILES|GENERICS
# Counts only the jobs of this queue (PID files in ~/r42_issue_out/.pids; `kill -0` is an existence test, no signal), at most MAXJOBS (default 1) at once and only while MemAvailable >= MINAVAIL_MB (default 16000).
# R42 issue additions: a job whose NAME ends in _syn (kernel-scale synthesis: up to 10 helper processes of about 2 GB) starts only while no foreign Vivado
# job runs and at most one of this queue's jobs runs (counted, never signalled), a job named tscr_* only while the memory gate holds. FOREIGN_OK (default 0) is the number of foreign Vivado jobs tolerated for a _syn job.
# Reads the stop file ~/r42_issue_out/STOP between jobs: create it to stop launching (running jobs finish). A NAME whose output directory exists is skipped.
Q=$1; MAX=${2:-1}
MINAVAIL_MB=${MINAVAIL_MB:-16000}      # do not start a job while MemAvailable is below this (a synthesis of a RAM-heavy unit grows to ~10 GB in child processes; other workers share the host)
OUTROOT=$HOME/r42_issue_out; PIDS=${PIDS_DIR:-$OUTROOT/.pids}; STOP=${STOP_FILE:-$OUTROOT/STOP}
mkdir -p $PIDS
mine() { n=0; for f in $PIDS/*.pid; do [ -f "$f" ] || continue; p=$(cat $f); if kill -0 $p 2>/dev/null; then n=$((n+1)); else rm -f $f; fi; done; echo $n; }
while IFS='|' read -r NAME ENVS TOP FILES GEN; do
    [ -z "$NAME" ] && continue
    case $NAME in \#*) continue;; esac
    if [ -d $OUTROOT/$NAME ]; then echo "$(date -u +%T) skip $NAME (exists)"; continue; fi
    while true; do
        if [ -f $STOP ]; then echo "$(date -u +%T) stop file present, not starting $NAME"; exit 0; fi
        av=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
        tot=$(pgrep -fc 'unwrapped/lnx64.o/vivado -mode batch' || true); m=$(mine); foreign=$(( ${tot:-0} - m )); [ $foreign -lt 0 ] && foreign=0
        quiet_ok=1; case $NAME in *_syn) { [ $foreign -le ${FOREIGN_OK:-0} ] && [ "$m" -le 1 ]; } || quiet_ok=0;; esac
        [ "$m" -lt "$MAX" ] && [ "$av" -ge "$MINAVAIL_MB" ] && [ $quiet_ok = 1 ] && break
        sleep 15
    done
    case $NAME in
        tscr_*) JOB=iss_tilejob.sh; set -- "$NAME" "$ENVS" "$TOP" "$FILES";;       # TOP = checkpoint relative to ~/r42_issue_out, FILES = SLR
        *)      JOB=iss_job.sh;     set -- "$NAME" "$ENVS" "$TOP" "$FILES" "$GEN";;
    esac
    setsid nohup bash $HOME/r42_issue/tools/r42/issue/$JOB "$@" > /dev/null 2>&1 < /dev/null &
    echo $! > $PIDS/$NAME.pid
    echo "$(date -u +%T) started $NAME pid $!"
    sleep 5
done < $Q
echo "$(date -u +%T) queue $Q: all started"
