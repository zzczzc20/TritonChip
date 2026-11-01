#!/bin/sh
# Preserve other cron entries. This is separate from hook installation so a
# dry run and an author report can precede live targeting.
set -eu
common=$(git rev-parse --path-format=absolute --git-common-dir)
state=/home/ubuntu/.guard
mkdir -p "$state"
tmp=$(mktemp "$state/cron.XXXXXX")
trap 'rm -f "$tmp"' EXIT HUP INT TERM
(crontab -l 2>/dev/null || :) | awk '!/# tritonchip-guard$/' > "$tmp"
printf '@reboot %s/guard/supervisor.sh >> %s/watchdog.log 2>&1 # tritonchip-guard\n' "$common" "$state" >> "$tmp"
printf '*/5 * * * * %s/guard/supervisor.sh >> %s/watchdog.log 2>&1 # tritonchip-guard\n' "$common" "$state" >> "$tmp"
crontab "$tmp"
