#!/bin/bash
#==============================================================================
# R42 board A/B driver (run on the lead machine; see evidence/r42/board/RECIPE.md).
#
#   board_ab.sh stage    COMMIT DIR                 git archive COMMIT (without evidence/, e7c/) -> FPGA host ~/r42/DIR
#   board_ab.sh register NAME COMMIT DIR [--knob TNCA_X=V ...]
#                                                   write spec/r42/board/NAME/ (commit + push it before `launch`)
#   board_ab.sh check    NAME                       copy the registration to ~/r42/reg/NAME, run --check-only (no card)
#   board_ab.sh launch   NAME                       detached guarded run on the FPGA host (refuses an unpushed
#                                                   registration and an existing output directory)
#   board_ab.sh status   NAME                       last events, queue log tail, card state (read-only)
#   board_ab.sh fetch    NAME                       records -> evidence/r42/board/NAME/records.tgz (small files only)
#
# S (helper directory with fp / gb_rsh) defaults to the R42 session scratchpad; the helpers hold the credentials
# and are never printed or copied. The card rules of evidence/r41/SERVERS.md are enforced on the host by
# tools/r42/board_ab.py and the R41 queue it calls; this script only copies files and starts that runner.
#==============================================================================
set -euo pipefail
S=${S:-/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BRANCH=${R42_BRANCH:-R42}
fp() { "$S/fp" "$1" < /dev/null; }
simple() { [[ "$1" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "not a simple name: $1" >&2; exit 2; }; }
cmd=${1:?command}; shift

case "$cmd" in
stage)
    C=$(git -C "$ROOT" rev-parse "${1:?commit}^{commit}"); D=${2:?dir}; simple "$D"
    fp "test ! -e ~/r42/$D || { echo '~/r42/$D exists; staged trees are never overwritten' >&2; exit 3; }; mkdir -p ~/r42/$D.tmp"
    git -C "$ROOT" archive "$C" -- . ':!evidence' ':!e7c' | "$S/fp" "tar xf - -C ~/r42/$D.tmp"
    fp "cd ~/r42 && echo $C > $D.tmp/SOURCE_COMMIT && mv $D.tmp $D && cd $D && ~/tc_env/torchenv/bin/python -c \"
import importlib.util, json, pathlib
s = importlib.util.spec_from_file_location('p', 'archbench/provenance.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(json.dumps(dict(dir=str(pathlib.Path.cwd()), commit='$C', software_tree=m.software_tree(pathlib.Path.cwd(), refresh=True))))\" | tee ../$D.staged.json"
    ;;
register)
    N=${1:?name}; C=${2:?commit}; D=${3:?dir}; shift 3; simple "$N"; simple "$D"
    mkdir -p "$S/tmp"
    TMPDIR="$S/tmp" python3 "$ROOT/tools/r42/board_ab.py" register --name "$N" --commit "$C" --src-dir "$D" "$@"
    echo "now: git add spec/r42/board/$N && commit && push origin $BRANCH, before 'board_ab.sh launch $N'"
    ;;
check|launch)
    N=${1:?name}; simple "$N"; R=spec/r42/board/$N
    [ -f "$ROOT/$R/queue.json" ] || { echo "no registration $R" >&2; exit 2; }
    if [ "$cmd" = launch ]; then
        git -C "$ROOT" fetch -q origin "$BRANCH"
        [ -z "$(git -C "$ROOT" status --porcelain -- "$R" tools/r42/board_ab.py)" ] \
            || { echo "registration or runner has uncommitted changes" >&2; exit 4; }
        for f in "$R/queue.json" "$R/$N.json" "$R/$N-identity.json" tools/r42/board_ab.py; do
            [ "$(git -C "$ROOT" rev-parse "origin/$BRANCH:$f" 2>/dev/null)" = "$(git -C "$ROOT" hash-object "$ROOT/$f")" ] \
                || { echo "$f is not what origin/$BRANCH holds: push the registration first" >&2; exit 4; }
        done
    fi
    fp "mkdir -p ~/r42/reg/$N ~/r42/q"
    tar -C "$ROOT/$R" -cf - queue.json "$N.json" "$N-identity.json" -C "$ROOT/tools/r42" board_ab.py | "$S/fp" "tar xf - -C ~/r42/reg/$N"
    fp "cd ~/r42/reg/$N && sha256sum *"
    if [ "$cmd" = check ]; then
        fp "cd ~/r42/reg/$N && ~/tc_env/torchenv/bin/python board_ab.py run --reg ~/r42/reg/$N --out ~/r42/q/$N.check --check-only"
    else
        fp "test ! -e ~/r42/q/$N || { echo '~/r42/q/$N exists: results are never overwritten' >&2; exit 3; }; mkdir -p ~/r42/q/$N && cd ~/r42/reg/$N || exit 6; setsid nohup ~/tc_env/torchenv/bin/python -u board_ab.py run --reg ~/r42/reg/$N --out ~/r42/q/$N > ~/r42/q/$N/launch.log 2>&1 < /dev/null & echo \$! > ~/r42/q/$N/runner.pid; sleep 3; kill -0 \$(cat ~/r42/q/$N/runner.pid) && echo runner pid \$(cat ~/r42/q/$N/runner.pid) || { echo 'runner is not alive'; cat ~/r42/q/$N/launch.log; exit 5; }"
    fi
    ;;
status)
    N=${1:?name}; simple "$N"
    fp "cd ~/r42/q/$N 2>/dev/null || { echo 'no output yet'; exit 0; }; date -u; pgrep -u \$(id -u) -af '[b]oard_ab.py run --reg|[a]cceptance_queue.py|[b]oard_batch.py' | cut -c1-160; echo '--- events'; grep -v idle-probe events.jsonl | tail -6 | cut -c1-400; grep idle-probe events.jsonl | tail -1 | cut -c1-300; echo '--- launch.log'; tail -5 launch.log; echo '--- queue.log'; tail -5 queue.log 2>/dev/null; for d in queue/*/steps/*/*/attempt-*; do [ -f \$d/stdout.log ] && { echo \"--- \$d\"; grep -c ' OK ' \$d/stdout.log; grep -c FAIL \$d/stdout.log; tail -2 \$d/stdout.log | cut -c1-200; }; done; echo '--- card'; cat /sys/bus/pci/devices/0000:04:00.1/kdsstat; head -1 /sys/bus/pci/devices/0000:04:00.0/clock_freq_counter*/clock_counter_freqs"
    ;;
fetch)
    N=${1:?name}; simple "$N"; mkdir -p "$ROOT/evidence/r42/board/$N"
    "$S/fp" "cd ~/r42/q/$N && tar czf - --exclude=caches --exclude=bundles --exclude='*_logs' events.jsonl launch.log queue.log program.log* preflight_*/preflight.json queue/*.log queue/DONE queue/*/report.json queue/*/clock.json queue/*/*/batch.json queue/*/*/state.json queue/*/*/formal.jsonl queue/*/*/runs queue/*/steps/*/*/attempt-*/result.json queue/*/steps/*/*/attempt-*/stdout.log queue/*/steps/*/*/attempt-*/stderr.log 2>/dev/null" < /dev/null \
        > "$ROOT/evidence/r42/board/$N/records.tgz" || true
    ls -la "$ROOT/evidence/r42/board/$N/records.tgz"; sha256sum "$ROOT/evidence/r42/board/$N/records.tgz"
    tar tzf "$ROOT/evidence/r42/board/$N/records.tgz" | wc -l
    ;;
*) echo "unknown command $cmd" >&2; exit 2 ;;
esac
