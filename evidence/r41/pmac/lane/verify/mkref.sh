#!/bin/bash
# mkref.sh [REV] -- the current lane (tnca_fpu_multi at REV, default 0b0a3cc =
# r41-pmac's base) with every module renamed *_ref, for the FP32-mode miter.
set -e
REV=${1:-0b0a3cc}
here=$(cd $(dirname $0) && pwd)
root=$(git -C $here rev-parse --show-toplevel)
out=$here/ref; rm -rf $out; mkdir -p $out
mods="tnca_fpu_multi tnca_fp32_fma tnca_widen_xls tnca_fma_align_xls tnca_fma_norm_xls tnca_fma_round_xls tnca_fma_l3_xls"
expr=""; for m in $mods; do expr="$expr -e s/\\b$m\\b/${m}_ref/g"; done
for f in core/tnca_fpu_multi.v core/tnca_fp32_fma.v gen/tnca_widen_xls.v gen/tnca_fma_align_xls.v \
         gen/tnca_fma_norm_xls.v gen/tnca_fma_round_xls.v gen/tnca_fma_l3_xls.v; do
    git -C $root show $REV:rtl/$f | sed $expr | grep -v 'LAT_must_be_2_or_3' > $out/$(basename $f .v)_ref.v
done
echo "$REV" > $out/REV
sha256sum $out/*.v
