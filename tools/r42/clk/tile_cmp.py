#!/usr/bin/env python3
"""tile_cmp.py PERIOD LABEL1=endpoints1.csv [LABEL2=endpoints2.csv ...]   (domain: OOC tile screens, routed)

Per unit (owner of the endpoint: u_core/<child>, g_home_pipe, ...) and per screen: endpoints failing at PERIOD, and endpoints whose routed path (PERIOD - slack) exceeds 6.0, 6.2, 6.4 and 6.667 ns
(what would still fail at those periods if the paths stayed as routed; the screen only lists endpoints within 1.5 ns of closing). Also the worst slack per unit.
"""
import csv
import sys
from collections import defaultdict

period = float(sys.argv[1])
screens = [a.split('=', 1) for a in sys.argv[2:]]
T = [period, 6.0, 6.2, 6.4, 6.667]


def unit(pin):
    p = pin.split('/')
    if p[0] == 'u_core':
        return 'u_core/' + (p[1] if len(p) > 3 else '(core own)')
    return p[0] if len(p) > 2 else '(tile own)'


data = {}
for lab, path in screens:
    d = defaultdict(lambda: [0] * len(T) + [9.9])
    for r in csv.DictReader(open(path)):
        s = float(r['slack'])
        pl = period - s
        u = unit(r['end'])
        for i, t in enumerate(T):
            if (i == 0 and s < 0) or (i and pl > t):
                d[u][i] += 1
        d[u][-1] = min(d[u][-1], s) if d[u][-1] != 9.9 else s
    data[lab] = d
units = sorted(set().union(*[set(d) for d in data.values()]), key=lambda u: -max(d[u][0] if u in d else 0 for d in data.values()))
print('| unit | ' + ' | '.join('%s fail@%.1f / >6.0 / >6.2 / >6.4 / >6.667 / worst slack' % (lab, period) for lab, _ in screens) + ' |')
print('|---|' + '---|' * len(screens))
tot = {lab: [0] * len(T) for lab, _ in screens}
for u in units:
    cells = []
    for lab, _ in screens:
        v = data[lab].get(u, [0] * len(T) + [0])
        cells.append(' / '.join(str(x) for x in v[:-1]) + ' / %.3f' % v[-1])
        for i in range(len(T)):
            tot[lab][i] += v[i]
    print('| %s | ' % u + ' | '.join(cells) + ' |')
print('| **tile** | ' + ' | '.join(' / '.join(str(x) for x in tot[lab]) for lab, _ in screens) + ' |')
