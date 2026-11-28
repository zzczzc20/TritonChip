#!/usr/bin/env python3
"""mc_table.py DIR_BASE DIR_ARM [DIR_ARM ...]   (GB10 or local; domain: three-core RTL stand-in, run-3 image geometry, simulation)

Per ArchBench point: device cycles of each arm (time.device of the link_budget_rtl record), ratio against the first directory, numerics status, and the geometric mean
over the points that passed in every arm. Directories hold <point>.jsonl as written by tools/r42/clk/clk_mc.sh.
"""
import json
import math
import sys
from pathlib import Path

dirs = [Path(d) for d in sys.argv[1:]]


def load(d):
    r = {}
    for f in sorted(d.glob('*.jsonl')):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            r[j['point']] = (j['status'], j['time']['device'], j['correctness'].get('max_rel_err'))
    return r


R = [load(d) for d in dirs]
pts = sorted(set().union(*[set(r) for r in R]))
print('%-48s' % 'point' + ''.join('  %-22s' % d.name for d in dirs))
logs = [[] for _ in dirs[1:]]
for p in pts:
    row = '%-48s' % p
    base = R[0].get(p)
    for i, r in enumerate(R):
        v = r.get(p)
        if v is None:
            row += '  %-22s' % '-'
            continue
        s = '%s %d' % (v[0], v[1])
        if i and base and base[0] == 'pass' and v[0] == 'pass':
            ratio = v[1] / base[1]
            s += ' (%.4f)' % ratio
            logs[i - 1].append(math.log(ratio))
        row += '  %-22s' % s
    print(row)
for i, l in enumerate(logs):
    if l:
        print('geomean cycles ratio %s / %s over %d points: %.4f' % (dirs[i + 1].name, dirs[0].name, len(l), math.exp(sum(l) / len(l))))
