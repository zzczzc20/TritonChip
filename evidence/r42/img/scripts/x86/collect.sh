#!/bin/bash
# collect.sh OUTDIR : small copies of the reports of every r42img run (kernel synthesis, tile 0, unit screens) and a summary
O=$1; rm -rf $O; mkdir -p $O; cd $HOME/r42img
for d in base idef l-a099878e m-a099878e f-a099878e lf-a099878e lmf-a099878e; do
  [ -d $d/out ] || continue
  mkdir -p $O/$d; cp $d/out/utilization.rpt $O/$d/kernel_utilization.rpt; gzip -f $O/$d/kernel_utilization.rpt
  [ -f $d/synth.log ] && tail -n 40 $d/synth.log | gzip > $O/$d/synth_log_tail.gz
  if [ -f $d/tile0/utilization_routed.rpt ]; then
    cp $d/tile0/utilization_routed.rpt $O/$d/tile0_utilization_routed.rpt; gzip -f $O/$d/tile0_utilization_routed.rpt
    head -n 160 $d/tile0/timing_routed.rpt | gzip > $O/$d/tile0_timing_routed_head.gz
    head -n 4000 $d/tile0/top_paths_routed.rpt | gzip > $O/$d/tile0_top_paths_head.gz
    cp $d/tile0/failing_endpoints.tsv $O/$d/ ; tail -n 5 $d/tile0.log | gzip > $O/$d/tile0_log_tail.gz
  fi
done
for u in unit/*; do [ -d $u/out ] || continue; n=$(basename $u); mkdir -p $O/$n
  for f in routed_util.rpt routed_timing.rpt routed_paths.rpt synth_util.rpt; do [ -f $u/out/$f ] && head -n 3000 $u/out/$f | gzip > $O/$n/$f.gz; done
  grep -h "R42_UNIT_SCREEN" $u/unit.log > $O/$n/wns.txt 2>/dev/null; done
cp chain.log $O/chain.log
for d in pkg_src_88890078_lmf pkg_src_88890078_lf; do [ -f $d/package.log ] && gzip -c $d/package.log > $O/package_${d#pkg_src_88890078_}.log.gz; done
{ echo "# tile 0 (out of context, 10 ns, SLR0 pblock): routed utilization and timing"; bash tile.sh base idef l-a099878e m-a099878e f-a099878e lf-a099878e lmf-a099878e; echo; echo "# worst routed slack per endpoint group of tile 0 (ns; count of endpoints among the most critical 2000)"; bash groups.sh base idef l-a099878e m-a099878e f-a099878e lf-a099878e lmf-a099878e;
  echo; echo "# kernel synthesis (three cores, hierarchical report) and per-core units"; bash units.sh base idef l-a099878e m-a099878e f-a099878e lf-a099878e lmf-a099878e;
  echo; echo "# unit screens (tnca_matrix)"; for u in unit/*; do n=$(basename $u); echo "$n $(grep -h R42_UNIT_SCREEN $u/unit.log 2>/dev/null) $(grep -h -m1 -E '^\| CLB LUTs' $u/out/routed_util.rpt 2>/dev/null | awk -F'|' '{print "LUT",$3}') $(grep -h -m1 -E '^\| CLB Registers' $u/out/routed_util.rpt 2>/dev/null | awk -F'|' '{print "FF",$3}')"; done; } > $O/ooc_summary.txt 2>&1
tar czf $O.tgz -C $(dirname $O) $(basename $O)
ls -la $O.tgz
