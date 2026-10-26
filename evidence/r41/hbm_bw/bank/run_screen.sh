#!/usr/bin/env bash
# Execute on the FPGA host; the slot guard never accesses the card.
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
out="$root/period10"
mkdir -p "$out"
cd "$out"
set +e
"$HOME/r41s2_tools/vivado_slot.sh" --min-gb 20 "$out/guard.log" -- \
    vivado -mode batch -nojournal -nolog -source "$root/flow/fpga/screen_r41_hbm_bank.tcl" \
    -tclargs "$root/rtl" "$out" 10
result=$?
printf '%s\n' "$result" > "$out/exit_status"
exit "$result"
