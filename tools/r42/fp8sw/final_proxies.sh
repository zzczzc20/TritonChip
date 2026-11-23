#!/bin/bash
# usage: final_proxies.sh OUTDIR PARALLEL TREE...   the single-program Hub-kernel proxies on the single-core RTL simulator, per tree
# (profile core-vrf-sync-v5-x8-g2c4-dp, default knobs), with counters, the issue probe and the sha256 of every output tensor
# (FP8SW_DIGEST_DIR). K = 256 and 512 give the cost of a K block by difference. Domain: single-core RTL simulation (a screen).
OUT=$1; PAR=$2; shift 2
PROBE=${PROBE_DEVICE:-$HOME/r42/fp8sw_out/probe/device.vvp}
jobs=()
for T in "$@"; do
  for ID in op.fp8_linear/xl/fp8-m16-k256-n128 op.fp8_linear/xl/fp8-m16-k512-n128 op.fp8_linear/xl/fp8-m1-k256-n128 op.fp8_linear/xl/fp8-m1-k512-n128; do
    jobs+=("$T $ID")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -L 1 bash -c '
  T=$0; ID=$1; OUT='"$OUT"'; PROBE='"$PROBE"'
  TAG=$(echo "$ID" | tr "/" "_")
  mkdir -p $OUT/$T
  FP8SW_DIGEST_DIR=$OUT/$T/digest TNCA_DEVICE=$PROBE TNCA_DEVICE_PLUSARGS="+iprobe=$OUT/$T/$TAG.iprobe" \
    bash $HOME/r42/$T/tools/r42/fp8sw/rtl_point.sh $HOME/r42/$T $OUT/$T $ID --counters'
