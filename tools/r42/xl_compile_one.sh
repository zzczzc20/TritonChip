#!/bin/sh
# xl_compile_one.sh OUT POINT : one point of xl_compile_only.sh
out=$1; id=$2
n=$(echo "$id" | tr '/' '_')
start=$(date +%s)
TNCA_COMPILE_CACHE_DIR="$out/cc/$n" nice -n 19 "${PYTHON:-python}" -m archbench.run --platform tnca \
    --profile core-vrf-sync-v5-x8-g2c4-dp --point "$id" --compile-only \
    --bundle-dir "$out/bundles" --out "$out/records/$n.jsonl" > "$out/logs/$n.log" 2>&1
echo "$id exit=$? seconds=$(( $(date +%s) - start ))" >> "$out/done.txt"
rm -rf "$out/cc/$n"
