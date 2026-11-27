#!/bin/bash
# collect.sh OUTTGZ  (x86 host): the small reports of every OOC result directory of this branch (~/r42_shrink_out/*/): utilization, hierarchy, timing summary,
# 30 worst paths, plus run.log, as one tar.gz for evidence/r42/shrink/ooc/.
OUT=${1:?out.tgz}
cd ~/r42_shrink_out
list=$(ls -d */ | grep -v "^gen\|^uram_lat\|^tile0_base" | tr -d /)
tar czf $OUT run.log queue_*.log $(for d in $list; do for f in synth_util.rpt synth_util_hier.rpt synth_summary.rpt synth_paths.rpt route_util.rpt route_util_hier.rpt route_summary.rpt route_paths.rpt; do [ -f $d/$f ] && echo $d/$f; done; done) 2>/dev/null
ls -la $OUT
