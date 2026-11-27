#!/usr/bin/env python3
"""Single-core calibration table: TLM (ideal memory, 1 core) against single-core RTL cycles of the tiny tier (domain: model vs rtl-1c).
    python3 tools/r42/dse_cmp1c.py COLLECTED.json [...]   RTL source: evidence/r42/xfer/tables.txt section 3b (software 44070eb0, config core-vrf-sync-v5-x8-g2c4-dp)
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dse_common as C
RTL = Path('/home/ubuntu/TritonChip/evidence/r42/xfer/tables.txt')
if not RTL.exists():
    RTL = C.ROOT / 'evidence/r42/xfer/tables.txt'
rtl = {}
for l in RTL.read_text().splitlines():
    m = re.match(r'\s+(\S+/tiny/\S+)\s+pass\s+(\d+)', l)
    if m:
        rtl[m.group(1)] = int(m.group(2))
GEMM = {'matmul', 'linear', 'mlp', 'gemv', 'bmm', 'gpt2_block', 'llama_prefill', 'llama_decode'}
# embedding, transpose, cumsum were lowered differently when the RTL numbers were taken (before R42-copy): excluded from the fit
EXCL = {'embedding', 'transpose', 'cumsum'}
for f in sys.argv[1:]:
    c = json.load(open(f))['points']
    g, o, a = [], [], []
    for p, v in rtl.items():
        if p in c and c[p].get('device_cycles') and C.family_of(p) not in EXCL:
            r = c[p]['device_cycles'] / v
            a.append(r); (g if C.family_of(p) in GEMM else o).append(r)
    med = lambda x: sorted(abs(i - 1) for i in x)[len(x) // 2]
    print(f'{Path(f).stem:<28} all n={len(a)} geo {C.geo(a):.3f} med|e| {med(a):.3f} | gemm-shaped geo {C.geo(g):.3f} med|e| {med(g):.3f} | other n={len(o)} geo {C.geo(o):.3f} med|e| {med(o):.3f}')
