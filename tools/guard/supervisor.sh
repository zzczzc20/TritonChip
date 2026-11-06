#!/bin/sh
# Own lock only; this script never searches for or signals other processes.
set -eu
umask 077
state=/home/ubuntu/.guard
common=$(git -C /home/ubuntu/TritonChip rev-parse --path-format=absolute --git-common-dir)
mkdir -p "$state"
test -f "$state/targets.json" || exit 0
exec 9>"$state/watchdog.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$state/watchdog.pid"
cd /home/ubuntu/TritonChip
exec python3 "$common/guard/watchdog.py" --live --config "$state/targets.json"
