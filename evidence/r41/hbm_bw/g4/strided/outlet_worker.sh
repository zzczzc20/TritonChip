#!/bin/bash
set -u
source ~/tc_env/env.sh
cd "$HOME/r41-codex-t9b" || exit 1
out="$HOME/r41-codex-t9b-outlet"
mkdir -p "$out"
point="op.$1/membound/fp16-small-strided-r32-c1024"
for arm in outlet_depth outlet_pipe; do
 { date -u; printf 'arm=%s point=%s\n' "$arm" "$point"; "$HOME/tc_env/torchenv/bin/python" tools/r41/link_budget_rtl.py --image "evidence/r41/hbm_bw/g4/strided/images/$arm.tcl" --profile core-vrf-sync-v5-x8-g2 --point "$point" --out "$out/$arm.$1.jsonl"; echo "exit=$?"; date -u; } > "$out/$arm.$1.log" 2>&1
done
