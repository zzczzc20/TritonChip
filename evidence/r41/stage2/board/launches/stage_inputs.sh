#!/bin/bash
#==============================================================================
# Stage the R41 Stage 2 board batch onto the FPGA host. Run on GB10.
#
#   stage_inputs.sh COMMIT ARCHIVE
#
# ARCHIVE is `git archive --format=tar.gz COMMIT` of the r41s2-board commit.
# Copies to acf3019 (files only; the card is not touched):
#   ~/r41s2_board_COMMIT.tar.gz            the source archive (the queue unpacks it)
#   ~/r41s2_board_queue.sh                 evidence/r41/stage2/board/launches/board_queue.sh from it
#   ~/r41s2_fp8_inputs/hf/hub/...          Qwen3-VL-2B-Instruct-FP8 and kernels-community/finegrained-fp8
#                                          from GB10's HF cache (~/flux-hf)
#   ~/r41s2_fp8_inputs/qwen/vl/ref_fp8     the CPU W8A8 reference of qwen3vl_run.py (teacher forcing)
#   ~/r41s2_fp8_inputs/hub-main            the finegrained-fp8 main checkout (fp8_kernels_cases --group v1)
#   ~/r41s2_fp8_inputs/STAGED              written last, with the sha256 of every staged file
# Without STAGED the queue skips s2-fp8-hub and s2-qwen-fp8 and runs the rest.
#==============================================================================
set -eu
C=${1:?commit}
ARCHIVE=${2:?archive}
FP=${FP:-zzhouch@acf3019.ece.ust.hk}
HF=${HF:-$HOME/flux-hf}
QW=${QW:-$HOME/qwen-work}
HUBMAIN=${HUBMAIN:-$HOME/r41-fp8-kw/hub-main-0fc209e0}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
RS="rsync -a --partial -e"

tmp=$(mktemp -d)
trap 'rm -rf $tmp' EXIT
tar xzf $ARCHIVE -C $tmp evidence/r41/stage2/board/launches/board_queue.sh
$RS "$SSH" $ARCHIVE $FP:r41s2_board_$C.tar.gz
$RS "$SSH" $tmp/evidence/r41/stage2/board/launches/board_queue.sh $FP:r41s2_board_queue.sh
$SSH $FP "mkdir -p r41s2_fp8_inputs/hf/hub r41s2_fp8_inputs/qwen/vl && rm -f r41s2_fp8_inputs/STAGED"
for d in models--Qwen--Qwen3-VL-2B-Instruct-FP8 kernels--kernels-community--finegrained-fp8; do
    $RS "$SSH" $HF/hub/$d $FP:r41s2_fp8_inputs/hf/hub/
done
$RS "$SSH" $QW/out/vl/ref_fp8 $FP:r41s2_fp8_inputs/qwen/vl/
$RS "$SSH" $HUBMAIN/ $FP:r41s2_fp8_inputs/hub-main/
$SSH $FP "cd r41s2_fp8_inputs && find . -type f ! -name STAGED -print0 | sort -z | xargs -0 sha256sum > STAGED.tmp && mv STAGED.tmp STAGED && wc -l STAGED && sha256sum ../r41s2_board_$C.tar.gz"
