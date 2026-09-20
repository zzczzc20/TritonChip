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
# Evidence lines from the bench itself, kept even on a pass. R1-08 asks for
# the seed's effect to be reported rather than asserted -- trace hashes and
# coverage counts -- and a harness that prints them only on failure is a
# harness that hides exactly the numbers the contract wants to see.
# Indented by six, so the gate's result parser (which anchors on two spaces
# followed by a non-space) does not read them as case results.
echo "$out" | grep -E '^(seed=|sent=|issued=|hit=)' | sed 's/^/      /'
exit 0
