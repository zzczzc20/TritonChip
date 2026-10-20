#!/bin/bash
# Observe one existing candidate; generate reports once after successful link.
set -u
r40_candidate=/home/zzhouch/tnca_build/u280_r40ownership_preserved09
r40_software=/home/zzhouch/tnca_srcR40convergence_8abf527
r40_reports=$r40_candidate/build/convergence_observer_8abf527
mkdir "$r40_reports" || exit 2
r40_previous_phase=
while [ ! -f "$r40_candidate/build/link.exit" ]; do
  r40_phase=$(sed -n '/^Phase /h; ${x;p;}' "$r40_candidate/xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log")
  if [ "$r40_phase" != "$r40_previous_phase" ]; then
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    printf '%s\n' "$r40_phase"
    r40_previous_phase=$r40_phase
  fi
  sleep 30
done
r40_link_rc=$(cat "$r40_candidate/build/link.exit")
printf 'link_exit=%s\n' "$r40_link_rc" > "$r40_reports/link_status.txt"
cp "$r40_candidate/xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log" "$r40_reports/implementation.log"
if [ "$r40_link_rc" != 0 ]; then
  printf 'failed implementation retained; no retry performed\n'
  exit "$r40_link_rc"
fi
if [ -d "$r40_candidate/xo/r40_postroute_audit" ]; then
  printf 'audit directory already exists; preserve it and require review\n'
  exit 2
fi
source /tools/Xilinx/Vivado/2023.2/settings64.sh
vivado -mode batch -nojournal -log "$r40_reports/vivado_audit.log" \
  -source "$r40_software/flow/fpga/audit_r40_image.tcl" -tclargs "$r40_candidate/xo"
r40_audit_rc=$?
printf '%s\n' "$r40_audit_rc" > "$r40_reports/audit.exit"
if [ "$r40_audit_rc" = 0 ]; then
  sha256sum "$r40_candidate/xo/tnca_u280.xclbin" > "$r40_reports/image.sha256"
  printf 'reports generated; actual placement, 100MHz timing and CDC require review before board checkpoint\n'
fi
exit "$r40_audit_rc"
