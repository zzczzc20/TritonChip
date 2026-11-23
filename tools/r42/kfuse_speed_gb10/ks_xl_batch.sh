#!/bin/bash
# usage: ks_xl_batch.sh TAG VARIANT...   FP8 proxy points through the TLM, one variant after the other
TAG=$1; shift
O=$HOME/r42/kfuse-speed_out
IDS="op.fp8_linear/xl/fp8-m1-k512-n384 op.fp8_linear/xl/fp8-m1-k1024-n384 op.fp8_linear/xl/fp8-m16-k512-n384 op.fp8_linear/xl/fp8-m16-k1024-n384 op.fp8_linear/xl/fp8-m16-k2048-n384 op.fp8_linear/xl/fp8-m48-k1024-n384 op.fp8_linear/xl/fp8-m16-k256-n256 op.fp8_linear/xl/fp8-m1-k256-n256 op.fp8_linear/xl/fp8-m16-k1024-n1152 op.fp8_linear/xl/fp8-m1-k1024-n1152"
for V in "$@"; do
  rm -rf $O/tlm/${TAG}_$V
  $O/ks_tlm_xl.sh $O/cfg/$V.cfg ${TAG}_$V $O/tlm-build2/tnca-chip $IDS
  echo "VARIANT $V done"
done
echo BATCH_DONE
