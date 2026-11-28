#!/bin/bash
# R42 issue: copy the small evidence of x86 OOC runs into evidence/r42/issue/ooc/ (utilization, timing summary, worst paths, worst endpoints, per-unit LUT)
#   ooc_fetch.sh S RUN [RUN...]     S = the session scratchpad that holds the vx helper
S=$1; shift
OUT=$(dirname $0)/../../../evidence/r42/issue/ooc
mkdir -p $OUT
for R in "$@"; do
  D=$OUT/$R; mkdir -p $D
  for f in synth_util.rpt synth_util_hier.rpt utilization_routed.rpt timing_routed.rpt timing_placed.rpt; do
    $S/vx "test -f ~/r42_issue_out/$R/$f && cat ~/r42_issue_out/$R/$f" < /dev/null 2>/dev/null | gzip -c > $D/$f.gz
    [ -s $D/$f.gz ] && [ $(stat -c %s $D/$f.gz) -lt 40 ] && rm -f $D/$f.gz
  done
  $S/vx "test -f ~/r42_issue_out/$R/top_paths_routed.rpt && head -400 ~/r42_issue_out/$R/top_paths_routed.rpt" < /dev/null 2>/dev/null | gzip -c > $D/top_paths_routed_head.rpt.gz
  $S/vx "test -f ~/r42_issue_out/$R/endpoints.csv && (head -1 ~/r42_issue_out/$R/endpoints.csv; tail -n +2 ~/r42_issue_out/$R/endpoints.csv | sort -t, -k1 -g | head -300)" < /dev/null 2>/dev/null | gzip -c > $D/worst300_endpoints.csv.gz
  find $D -size -40c -delete
done
