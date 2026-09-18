#!/usr/bin/env bash
# run_test.sh <name> <command>
#
# Runs a simulation and decides pass/fail from its output. A testbench that
# prints neither PASS nor FAIL is a failure: a silent run is not a passing run.
set -uo pipefail
name="$1"; shift
cmd="$1"
out=$(eval "$cmd" 2>&1)
rc=$?
if [ $rc -ne 0 ]; then
    printf '  %-22s ERROR (exit %d)\n' "$name" "$rc"
    echo "$out" | sed 's/^/      /'
    exit 1
fi
if echo "$out" | grep -q '^FAIL'; then
    printf '  %-22s FAIL\n' "$name"
    echo "$out" | sed 's/^/      /'
    exit 1
fi
if ! echo "$out" | grep -q '^PASS'; then
    printf '  %-22s NO RESULT (testbench printed neither PASS nor FAIL)\n' "$name"
    echo "$out" | sed 's/^/      /'
    exit 1
fi
stats=$(echo "$out" | grep -E '^(cycles=|tb_)' | head -1)
printf '  %-22s PASS   %s\n' "$name" "$stats"
exit 0
