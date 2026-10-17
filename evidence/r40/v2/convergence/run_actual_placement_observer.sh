#!/bin/bash
# Read-only follow-up to the already queued post-route audit, never a relink.
set -u
r40_candidate=/home/zzhouch/tnca_build/u280_r40ownership_preserved09
r40_observer=$r40_candidate/build/convergence_observer_8abf527
while [ ! -f "$r40_observer/audit.exit" ]; do
  if [ -f "$r40_candidate/build/link.exit" ] && [ "$(cat "$r40_candidate/build/link.exit")" != 0 ]; then
    printf 'implementation failed; no placement acceptance or retry\n'
    exit 1
  fi
  sleep 30
done
if [ "$(cat "$r40_observer/audit.exit")" != 0 ]; then
  printf 'postroute audit failed; preserve its reports for review\n'
  exit 1
fi
source /tools/Xilinx/Vivado/2023.2/settings64.sh
r40_routed=$r40_candidate/xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp
r40_auditor=/tmp/r40_audit_placed_cores.tcl
r40_report=$r40_candidate/xo/r40_postroute_audit/actual_placed_cores.rpt
vivado -mode batch -nojournal -log "$r40_observer/actual_placement_audit.log" \
  -source "$r40_auditor" -tclargs "$r40_routed" "$r40_report"
r40_placement_rc=$?
printf '%s\n' "$r40_placement_rc" > "$r40_observer/actual_placement.exit"
sha256sum "$r40_auditor" "$r40_routed" "$r40_candidate/xo/tnca_u280.xclbin" > "$r40_observer/actual_placement_inputs.sha256"
exit "$r40_placement_rc"
