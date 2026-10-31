#!/bin/bash
# Unit OOC screen of the FMA datapath (F03/F08/F09), on the FPGA host, through
# ~/r41s2_tools/vivado_slot.sh. Synthesis only at 10 ns, xcu280-fsvh2892-2L-e,
# OOC_SV=1, image parameters (B40 attempt03 image.tcl: the core instantiates
# tnca_vector_fpu at its defaults and tnca_matrix with ABUF=MTX_ABUF=1; the
# FMA at the core's FP_LAT=2). Never touches the card.
#
#   run_ooc.sh TAG [UNIT...]      TAG is a tree under ~/r41s2-fma/
set -u
D=$HOME/r41s2-fma
tag=${1:?tag}; shift
units=${*:-"tnca_vector_fpu tnca_matrix tnca_fp32_fma"}
root=$D/$tag
for unit in $units; do
  case $unit in
    tnca_vector_fpu) G="" ;;
    tnca_matrix)     G="ABUF=1" ;;
    tnca_fp32_fma)   G="LAT=2" ;;
    *)               G="" ;;
  esac
  out=$D/out/$tag/$unit; rm -rf $out; mkdir -p $out
  files=$(cd $root/rtl && ls common/*.v core/*.v gen/*.v nuca/*.v noc/*.v tile/*.v fpga/*.v *.v)
  OOC_TCL=$root/flow/synth/vivado_ooc_timing.tcl OOC_SV=1 OOC_GENERICS="$G" GROUP_SLACK=5 \
    ~/r41s2_tools/vivado_slot.sh $out/vivado.log -- \
    vivado -mode batch -nojournal -nolog -source $D/fma_ooc.tcl \
      -tclargs $unit 10 xcu280-fsvh2892-2L-e $out $root/rtl $files
  echo "$(date -u +%FT%TZ) $tag $unit exit $?" >> $D/out/$tag/run.log
done
echo "$(date -u +%FT%TZ) $tag DONE" >> $D/out/$tag/run.log
