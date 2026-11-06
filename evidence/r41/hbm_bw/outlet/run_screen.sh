#!/usr/bin/env bash
# Execute on the FPGA host through the shared CPU-only Vivado slot guard.
set -euo pipefail
slots=${1:?usage: run_screen.sh 32|64}
case "$slots" in 32|64) ;; *) exit 2;; esac
root=$(cd "$(dirname "$0")" && pwd)
out="$root/slots$slots"
mkdir -p "$out"
cd "$out"
set +e
"$HOME/r41s2_tools/vivado_slot.sh" --min-gb 8 "$out/guard.log" -- \
    vivado -mode batch -nojournal -nolog -source "$root/screen_r41_hbm_outlet.tcl" \
    -tclargs "$root/tnca_line_axi.v" "$out" "$slots"
result=$?
printf '%s\n' "$result" > "$out/exit_status"
exit "$result"
