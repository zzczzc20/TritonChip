#!/bin/sh
# xl_cpu_check.sh OUT_DIR : the smallest point of every xl family, one process each (so the peak memory is per point)
out=$1; mkdir -p "$out"
for id in $("${PYTHON:-python}" tools/r42/xl_cpu_check.py --list-smallest 2>/dev/null | grep '/xl/'); do
  n=$(echo "$id" | tr '/' '_')
  nice -n 19 "${PYTHON:-python}" tools/r42/xl_cpu_check.py --point "$id" --out "$out/$n.json" > "$out/$n.log" 2>&1
done
echo finished > "$out/DONE"
