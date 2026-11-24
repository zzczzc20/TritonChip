#!/usr/bin/env python3
"""usage: xf_audit_cmp.py CTRL_DIR CAND_DIR   per point: do the kernels' assembly texts of two xf_alldump.sh runs match?"""
import hashlib, os, sys, re, json
ctrl, cand = sys.argv[1:3]
def kernels(d):
    out = {}
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.endswith('.s'):
                t = open(os.path.join(root, f)).read()
                out[f[:-2]] = (hashlib.sha256(t.encode()).hexdigest(), t.count('m.mma'), t.count('gemm_pipe:'), t.count('\n'))
    return out
pts = sorted(p for p in os.listdir(ctrl) if os.path.isdir(os.path.join(ctrl, p)))
same = diff = 0; rows = []
for p in pts:
    a, b = kernels(os.path.join(ctrl, p)), kernels(os.path.join(cand, p)) if os.path.isdir(os.path.join(cand, p)) else {}
    ks = sorted(set(a) | set(b))
    ch = [k for k in ks if a.get(k, (None,))[0] != b.get(k, (None,))[0]]
    if ch:
        diff += 1
        for k in ch:
            ra, rb = a.get(k), b.get(k)
            rows.append((p, k, ra[1:] if ra else None, rb[1:] if rb else None))
    else:
        same += 1
print('points with identical assembly:', same, ' with a changed kernel:', diff)
for p, k, ra, rb in rows:
    print('%-60s %-70s ctrl(mma,pipe,lines)=%s cand=%s' % (p, k[:70], ra, rb))
