#!/bin/bash
# fma_impl1_miter.sh OUTDIR N [SEED] -- the F03 FMA miter
# (evidence/r41/stage2/candidates/F03/miter) on tnca_fp32_fma IMPL=1 at
# latency 2 and 3, against the generated FMA before F03 (f3db5c5^, the
# reference of evidence/r41/acceptance/gate_after_rectification). The miter
# instantiates tnca_fp32_fma without IMPL, so the candidate copy has the
# parameter's default set to 1; nothing else differs from this tree.
set -e
OUT=${1:?outdir}; N=${2:?vectors per class}; SEED=${3:-1}
here=$(cd $(dirname $0) && pwd)
root=$(cd $here/../../../../.. && pwd)
mkdir -p $OUT/ref $OUT/new/core $OUT/new/gen
# (the reference needs git; a tree without it takes OUTDIR/ref prepared elsewhere)
[ -s $OUT/ref/tnca_fma_l3_xls.v ] || for f in tnca_fma_xls.v tnca_fma_l3_xls.v; do
    git -C $root show f3db5c5^:rtl/gen/$f > $OUT/ref/$f
done
sed 's/parameter IMPL = 0/parameter IMPL = 1/' $root/rtl/core/tnca_fp32_fma.v > $OUT/new/core/tnca_fp32_fma.v
grep -q 'parameter IMPL = 1' $OUT/new/core/tnca_fp32_fma.v
cp $root/rtl/gen/tnca_fma_align_xls.v $root/rtl/gen/tnca_fma_norm_xls.v $root/rtl/gen/tnca_fma_round_xls.v \
   $root/rtl/gen/tnca_fma_l3_xls.v $root/rtl/gen/tnca_fmad_norm_xls.v $root/rtl/gen/tnca_fmad_l3_xls.v $OUT/new/gen/
bash $root/evidence/r41/stage2/candidates/F03/miter/run_miter.sh $OUT/ref $OUT/new $N $SEED
