#!/bin/bash
# run_ooc.sh TREE OUTROOT PERIOD TAG GENERICS... -- on the FPGA host, one lane
# OOC (synthesis + place + route) through ~/r41s2_tools/vivado_slot.sh.
# FILES overrides the rtl/ file list (the base tree has no tnca_pmac_lane).
set -u
tree=${1:?tree}; outroot=${2:?out}; period=${3:?period}; tag=${4:?tag}; shift 4
here=$(cd $(dirname $0) && pwd)
out=$outroot/${tag}_p$period; rm -rf $out; mkdir -p $out
files=${FILES:-"core/tnca_pmac_lane.v core/tnca_fpu_multi.v core/tnca_fp32_fma.v gen/tnca_widen_xls.v \
 gen/tnca_fdp_v1_f1_xls.v gen/tnca_fdp_v2l_f1_xls.v gen/tnca_fdp_v2d_f1_xls.v gen/tnca_fdp_f2_xls.v \
 gen/tnca_fdp_v3_f1_xls.v gen/tnca_fdp_v3_f2_xls.v gen/tnca_fma_alignp_xls.v gen/tnca_fma_align_xls.v \
 gen/tnca_fma_norm_xls.v gen/tnca_fma_round_xls.v gen/tnca_fma_l3_xls.v"}
files="$files $here/pmac_ooc_top.v"
cd $out
OOC_GENERICS="$*" ~/r41s2_tools/vivado_slot.sh --min-gb 8 $out/vivado.log -- \
    vivado -mode batch -nojournal -nolog -source $here/lane_ooc.tcl \
    -tclargs $period xcu280-fsvh2892-2L-e $out $tree/rtl $files
echo "$(date -u +%FT%TZ) $tag p$period exit $? ($*)" >> $outroot/run.log
