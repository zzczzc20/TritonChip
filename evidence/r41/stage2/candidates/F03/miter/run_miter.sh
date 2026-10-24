#!/bin/bash
# run_miter.sh REF_GEN_DIR NEW_RTL_DIR N [SEED]
#   REF_GEN_DIR  a directory holding the reference tnca_fma_xls.v and tnca_fma_l3_xls.v
#                (r41s2's rtl/gen: git show r41s2:rtl/gen/tnca_fma{,_l3}_xls.v)
#   NEW_RTL_DIR  the candidate's rtl/ (core/tnca_fp32_fma.v and gen/tnca_fma_*_xls.v)
# Builds fma_miter with Verilator and runs N vectors per class.
set -e
REF=${1:?ref rtl/gen dir}; NEW=${2:?new rtl dir}; N=${3:-1000000}; SEED=${4:-1}
here=$(cd $(dirname $0) && pwd)
wd=$(mktemp -d ${TMPDIR:-/tmp}/fma_miter.XXXX)
sed 's/module tnca_fma_xls(/module tnca_fma_xls_ref(/' $REF/tnca_fma_xls.v > $wd/ref_l2.v
sed 's/module tnca_fma_l3_xls(/module tnca_fma_l3_xls_ref(/' $REF/tnca_fma_l3_xls.v > $wd/ref_l3.v
mkdir -p $wd/new
cp $NEW/gen/tnca_fma_*_xls.v $wd/new/
# Verilator resolves the pins of the untaken g_bad branch (a deliberate
# refusal to elaborate an unsupported LAT); the miter only builds LAT 2 and 3,
# so that one instance line is dropped from the copy it compiles.
grep -v 'LAT_must_be_2_or_3' $NEW/core/tnca_fp32_fma.v > $wd/new/tnca_fp32_fma.v
cp $here/fma_miter.v $here/fma_miter.cpp $wd/
cd $wd
verilator --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module fma_miter \
    fma_miter.v ref_l2.v ref_l3.v new/*.v fma_miter.cpp -j 4 >build.log 2>&1 \
    || { tail -30 build.log; exit 2; }
sha256sum ref_l2.v ref_l3.v new/*.v
./obj_dir/Vfma_miter $N $SEED
