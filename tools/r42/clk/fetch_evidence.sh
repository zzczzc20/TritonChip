#!/bin/bash
# fetch_evidence.sh S  -- copy the small result files of the GB10 simulation runs and of the x86 OOC runs into evidence/r42/clk (S = scratchpad holding gb_rsh and vx)
S=$1
OUT=$(dirname $0)/../../../evidence/r42/clk
mkdir -p $OUT/sim $OUT/ooc
for f in run2_base.cycles run2_c0.cycles run2_c1.cycles run2_c4.cycles run2_c8.cycles run2_c256.cycles hyb_base.log hyb_clk.log hyb_clk2.log g4_cf16.log g4_rev2.log g4_rev3.log miter_ref.log miter_ref2.log; do
  $S/gb_rsh x "cat ~/r42/clk_out/$f 2>/dev/null" < /dev/null | gzip -c > $OUT/sim/$f.gz
  [ $(stat -c %s $OUT/sim/$f.gz) -lt 30 ] && rm -f $OUT/sim/$f.gz
done
for d in miter_ref miter_ref2; do
  $S/gb_rsh x "cat ~/r42/clk_out/$d/summary.txt 2>/dev/null | sed 's#.*/##'" < /dev/null | gzip -c > $OUT/sim/${d}_summary.txt.gz
  [ $(stat -c %s $OUT/sim/${d}_summary.txt.gz) -lt 30 ] && rm -f $OUT/sim/${d}_summary.txt.gz
done
for a in base pick s2 c23 idle; do
  $S/gb_rsh x "ls ~/r42/mc/mc7_$a/*.jsonl >/dev/null 2>&1 && tar cz -C ~/r42/mc mc7_$a/*.jsonl" < /dev/null > $OUT/sim/mc7_$a.tgz
  [ $(stat -c %s $OUT/sim/mc7_$a.tgz) -lt 100 ] && rm -f $OUT/sim/mc7_$a.tgz
done
for R in "$@"; do :; done
