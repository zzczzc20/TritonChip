#!/bin/bash
# collect.sh OUTTGZ  (x86 host): the small reports of every OOC result directory under ~/r42_area_out (utilization, hierarchy,
# timing summary, 30 worst paths, the log tail) plus run.log, as one tar.gz for evidence/r42/area/ooc/.
OUT=${1:?out.tgz}
cd ~/r42_area_out
list=$(ls -d */ | grep -v "^dcp_run3\|^tile0_base_synth" | tr -d /)
tar czf $OUT run.log $(for d in $list; do for f in synth_util.rpt synth_util_hier.rpt synth_summary.rpt route_util.rpt route_util_hier.rpt route_summary.rpt route_paths.rpt utilization.rpt timing.rpt; do [ -f $d/$f ] && echo $d/$f; done; done) 2>/dev/null
ls -la $OUT
