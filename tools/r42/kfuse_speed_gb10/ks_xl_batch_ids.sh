#!/bin/bash
# usage: ks_xl_batch_ids.sh TAG IDFILE VARIANT...   proxies of the xl tree through the TLM, variants one after the other
TAG=$1; IDF=$2; shift 2
O=$HOME/r42/kfuse-speed_out
for V in "$@"; do
  rm -rf $O/tlm/${TAG}_$V
  $O/ks_tlm_xl.sh $O/cfg/$V.cfg ${TAG}_$V $O/tlm-build3/tnca-chip $(cat $IDF)
  echo "VARIANT $V done"
done
echo BATCH_DONE
