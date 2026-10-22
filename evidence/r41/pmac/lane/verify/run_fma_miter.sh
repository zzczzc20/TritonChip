#!/bin/bash
# run_fma_miter.sh OUT N SEED -- on GB10: F03's miter, regenerated FMA vs 0b0a3cc's.
set -e
source ~/tc_env/env.sh; export PATH=$PATH:$HOME/tc_env/root/usr/bin; export VERILATOR_ROOT=$HOME/tc_env/vlroot
here=$(cd $(dirname $0) && pwd); root=$(cd $here/../../../../.. && pwd)
OUT=${1:?out}; N=${2:-10000000}; SEED=${3:-1}
mkdir -p $OUT; cd $OUT
grep -v 'LAT_must_be_2_or_3' $root/rtl/core/tnca_fp32_fma.v > tnca_fp32_fma.v
G=$root/rtl/gen
verilator --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module fma_miter \
    $here/fma_miter.v tnca_fp32_fma.v $G/tnca_fma_align_xls.v $G/tnca_fma_norm_xls.v $G/tnca_fma_round_xls.v \
    $G/tnca_fma_l3_xls.v $here/ref/tnca_fp32_fma_ref.v $here/ref/tnca_fma_align_xls_ref.v \
    $here/ref/tnca_fma_norm_xls_ref.v $here/ref/tnca_fma_round_xls_ref.v $here/ref/tnca_fma_l3_xls_ref.v \
    $here/fma_miter.cpp -j 4 > build.log 2>&1 || { tail -30 build.log; exit 2; }
sha256sum $G/tnca_fma_*_xls.v $here/ref/tnca_fma_*_xls_ref.v
./obj_dir/Vfma_miter $N $SEED
