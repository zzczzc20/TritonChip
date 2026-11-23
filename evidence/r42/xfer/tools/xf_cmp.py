#!/usr/bin/env python3
"""usage: xf_cmp.py COLLECTED.json BASETAG [TAG...]  per-point cycle ratios (cycles/base) and geometric mean"""
import json, sys, math
c = json.load(open(sys.argv[1])); base = c[sys.argv[2]]
tags = sys.argv[3:] or [t for t in c if t != sys.argv[2]]
pts = sorted(base)
print('%-62s %10s' % ('point', sys.argv[2]) + ''.join(' %14s' % t[:14] for t in tags))
gm = {t: [] for t in tags}
for p in pts:
    row = '%-62s %10s' % (p, base[p][0])
    for t in tags:
        v = c[t].get(p)
        if v is None or v[0] is None or base[p][0] is None or v[1] != 'pass':
            row += ' %14s' % ('FAIL' if v and v[1] != 'pass' else '-')
            continue
        r = v[0] / base[p][0]
        gm[t].append(r)
        row += ' %14.3f' % r
    print(row)
print('%-62s %10s' % ('geomean of (cycles/base) = 1/speedup', '') + ''.join(' %14.3f' % (math.exp(sum(map(math.log, gm[t])) / len(gm[t])) if gm[t] else float('nan')) for t in tags))
