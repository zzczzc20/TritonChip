#!/bin/bash
# R40-10 item 5: rebuild from a clean source archive and rerun on the board.
#
#   tools/r40/v3_reproduce.sh ARCHIVE.tar.gz OUTDIR [REFERENCE_DIR]
#
# ARCHIVE is `git archive` of the delivery commit. Everything is built fresh:
# the native-device C++ extension, Triton and Inductor caches, the Python
# bytecode. The product tree must equal the v3 delivery identity. Then the
# contract and protocol checks run, and three frozen points run on the loaded
# image (one per family of the suites: HBM streaming, GEMM, decoder). With
# REFERENCE_DIR (a native77/HBM14 run of the delivery identity), each point's
# device cycles are compared to its reference record.
set -euo pipefail
ARCHIVE=$1; OUT=$2; REF=${3:-}
test ! -e "$OUT"
mkdir -p "$OUT"/src "$OUT"/cache/{ext,triton,inductor}
tar xzf "$ARCHIVE" -C "$OUT"/src
cd "$OUT"/src
PY=${TORCH_PYTHON:-$HOME/tc_env/torchenv/bin/python}
export TORCH_EXTENSIONS_DIR="$OUT"/cache/ext TRITON_CACHE_DIR="$OUT"/cache/triton \
       TORCHINDUCTOR_CACHE_DIR="$OUT"/cache/inductor PYTHONDONTWRITEBYTECODE=1 \
       CUDA_VISIBLE_DEVICES= MAX_JOBS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/python:$PWD/compiler/python:$PWD/tools/asm"
{
  echo "archive_sha256 $(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
  "$PY" - <<'EOF'
import json, sys
sys.path.insert(0, 'tools/r40')
from archbench.provenance import software_tree
import contract_v3
tree = software_tree()
want = contract_v3.delivery(json.load(open('doc/r40_v3/plan.json')))['software_tree']
print('software_tree', tree)
print('delivery_tree', want)
assert tree == want, 'clean archive is not the v3 delivery identity'
EOF
  (cd tools/r40 && "$PY" contract_v3.py)
  "$PY" tools/r40/protocol_tests.py | tail -1
} 2>&1 | tee "$OUT"/checks.log
XCLBIN=$HOME/tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin
ARGS='{"xclbin": "'$XCLBIN'", "image_profile": "core-vrf-sync-v5", "core_count": 3, "memory_ports": 3, "hbm_dma": true, "hbm_bytes": 4294967296, "device_clock": true}'
TNCA_OPERATING_CLOCK_EVIDENCE=$HOME/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json \
TNCA_DEADLINE=24000000000 TNCA_HOST_TIMEOUT=7200 \
"$PY" -u -m archbench.run --platform tnca --profile core-vrf-sync-v5 --backend fpga-u280-multicore \
  --backend-args "$ARGS" --jobs 1 --warmup 1 --timed 1 \
  --set TNCA_TORCH_DEVICE=native --set TNCA_TRANSFER_POLICY=resident --set TNCA_ACTIVE_CORES=3 \
  --point op.add/hbm/fp16-n67108864 op.matmul/medium/fp16-m128-k192-n128 model.llama_decode/hbm/fp16-b1-p512 \
  --bundle-dir "$OUT"/bundles --out "$OUT"/points.jsonl 2>&1 | tee "$OUT"/points.log
"$PY" - "$OUT"/points.jsonl "$REF" <<'EOF' | tee "$OUT"/compare.log
import json, sys, glob
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ref = {}
if sys.argv[2]:
    for f in glob.glob(sys.argv[2] + '/**/*.jsonl', recursive=True):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get('status') == 'pass':
                    ref[r['point_id']] = r
ok = True
for r in rows:
    cyc = r['timings']['device-cycles-serial-grid']['median']
    line = f"{r['point_id']} {r['status']} cycles={cyc} err={r['correctness']['max_rel_err']}"
    ok &= r['status'] == 'pass' and not r.get('fallbacks')
    if r['point_id'] in ref:
        rc = ref[r['point_id']]['timings']['device-cycles-serial-grid']['median']
        rel = abs(cyc - rc) / rc
        line += f' reference={rc} rel_diff={rel:.5f}'
        ok &= rel <= 0.002
    print(line)
print('REPRODUCTION', 'PASS' if ok and len(rows) == 3 else 'FAIL')
sys.exit(0 if ok and len(rows) == 3 else 1)
EOF
