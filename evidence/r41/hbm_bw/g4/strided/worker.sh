#!/bin/bash
set -u
worker=$1
root=$2
out=$3
cd "$root" || exit 1
source ~/tc_env/env.sh
mkdir -p "$out"
points=(op.add/membound/fp16-small-strided-r32-c1024 op.clone/membound/fp16-small-strided-r32-c1024)
case "$worker" in
  0) arms=(g2 hybrid nofill slots s2 outlet) ; point=${points[0]} ;;
  1) arms=(g2 hybrid nofill slots s2 outlet) ; point=${points[1]} ;;
  2) arms=(s4 s1 pipe background mshr) ; point=${points[0]} ;;
  3) arms=(s4 s1 pipe background mshr) ; point=${points[1]} ;;
  *) exit 2 ;;
esac
for arm in "${arms[@]}"; do
  printf 'START %s %s %s\n' "$(date -u +%FT%TZ)" "$arm" "$point"
  ~/tc_env/torchenv/bin/python tools/r41/link_budget_rtl.py --image "evidence/r41/hbm_bw/g4/strided/images/$arm.tcl" --profile core-vrf-sync-v5-x8-g2 --point "$point" --out "$out/$arm.w$worker.jsonl" > "$out/$arm.w$worker.log" 2>&1
  rc=$?
  printf 'END %s %s exit=%s\n' "$(date -u +%FT%TZ)" "$arm" "$rc"
done
