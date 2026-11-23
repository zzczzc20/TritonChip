#!/bin/sh
# xl_compile_only.sh OUT JOBS [POINT...]
#
# Compile every kernel of each xl point on profile core-vrf-sync-v5-x8-g2c4-dp without running one (archbench.run
# --compile-only, TNCA_LAUNCH=skip), each point in its own process with fresh Triton and tnca compile caches, so the
# wall seconds are cold compile times. Per point: OUT/records/<id>.jsonl (the archbench record), OUT/logs, OUT/bundles/
# <id>/kernels (the dump of every kernel: .ttir, .tnir, .s), OUT/done.txt (exit code and seconds). Run it from a tree
# with PYTHON (the interpreter), PYTHONPATH naming all three source paths and ARCHBENCH_FINEGRAINED_FP8 set where the `kernels` package is
# absent. Nothing here launches a kernel: the CPU-tensor path, on which a launch's 384 KiB staging rule reports
# `capacity-exceeded` for every large point by construction; that status is not a compile failure.
out=$1; jobs=$2; shift 2
mkdir -p "$out/records" "$out/logs" "$out/bundles" "$out/cc"
if [ $# -eq 0 ]; then
  set -- $("${PYTHON:-python}" -m archbench.run --list --tier xl | awk '$1 ~ /\/xl\// {print $1}')
fi
here=$(cd "$(dirname "$0")" && pwd)
printf '%s\n' "$@" | xargs -P "$jobs" -I{} sh "$here/xl_compile_one.sh" "$out" {}
