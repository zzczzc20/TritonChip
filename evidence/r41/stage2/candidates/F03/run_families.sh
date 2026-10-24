#!/bin/bash
# run_families.sh TAG UNIT -- re-run the family report of fma_ooc.tcl on the
# synthesis checkpoint an earlier run_ooc.sh left (no new synthesis).
set -u
D=$HOME/r41s2-fma
tag=${1:?tag}; unit=${2:?unit}
out=$D/out/$tag/$unit
FAMILIES_ONLY=1 ~/r41s2_tools/vivado_slot.sh $out/families.log -- \
  vivado -mode batch -nojournal -nolog -source $D/fma_ooc.tcl \
    -tclargs $unit 10 xcu280-fsvh2892-2L-e $out $out/${unit}_synth.dcp
echo "$(date -u +%FT%TZ) $tag $unit families exit $?" >> $D/out/$tag/run.log
