#!/usr/bin/env python3
"""r42_cmp.py BASE.jsonl CAND.jsonl : per-point cycle ratio and geomean (computed from the records)."""
import json, math, sys, collections
def load(f):
    d = {}
    for l in open(f):
        r = json.loads(l); d[r['point_id']] = r
    return d
def cyc(r):
    if r.get('status') != 'pass': return None
    t = r.get('time') or {}
    return t.get("device")
B, C = load(sys.argv[1]), load(sys.argv[2])
rows = []; fam = collections.defaultdict(list)
for p in sorted(B):
    b = cyc(B[p]); c = cyc(C[p]) if p in C else None
    if b and c:
        rows.append((p, b, c, b / c)); fam[B[p].get('family', p.split('/')[0])].append(b / c)
    else:
        print(f'MISSING/FAIL {p}: base={B[p].get("status")} cand={(C.get(p) or {}).get("status")}')
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
if '-q' not in sys.argv:
    for p, b, c, s in rows: print(f'{p:62s} {b:>10d} {c:>10d} {s:6.3f}')
print(f'points {len(rows)}/{len(B)}  S_geo {g([r[3] for r in rows]):.4f}  sum-ratio {sum(r[1] for r in rows)/sum(r[2] for r in rows):.4f}  slower>2%: {sum(1 for r in rows if r[3] < 0.98)}')
