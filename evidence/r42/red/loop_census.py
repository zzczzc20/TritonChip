#!/usr/bin/env python3
"""loop_census.py DUMP_DIR... : instructions per loop of every dumped kernel (.s), by class.

Reads the assembly the compiler dumped (TNCA_KERNEL_DUMP=1); computes everything from those files.
Classes: mem (v.ld/v.st/gather/scatter, s.ld/s.st), sfu (exp2/rcp/rsqrt/log2 approx), vfp (vf.*),
vint (other v.*), pred (p.*), scalar (s.*, .li), ctl (br, fence, td.load, exit)."""
import re, sys, pathlib

def klass(m):
    if m.startswith(('v.ld', 'v.st', 's.ld', 's.st')): return 'mem'
    if 'approx' in m: return 'sfu'
    if m.startswith('vf.'): return 'vfp'
    if m.startswith('v.'): return 'vint'
    if m.startswith('p.'): return 'pred'
    if m.startswith(('s.', '.li')): return 'scalar'
    return 'ctl'

def census(path):
    labels, cur, rows = [], '(entry)', {}
    order = []
    for line in open(path):
        line = line.split('//')[0].rstrip()
        if not line.strip(): continue
        m = re.match(r'^([A-Za-z_][\w.]*):', line)
        if m:
            cur = m.group(1); continue
        tok = line.split()
        mn = tok[1] if tok[0].startswith('@') else tok[0]
        if cur not in rows: rows[cur] = {}; order.append(cur)
        rows[cur][klass(mn)] = rows[cur].get(klass(mn), 0) + 1
    return order, rows

for d in sys.argv[1:]:
    for s in sorted(pathlib.Path(d).rglob('*.s')):
        order, rows = census(s)
        print(f'{s.parent.parent.name}  {s.stem}')
        for lab in order:
            r = rows[lab]
            print(f'    {lab:18s} total {sum(r.values()):4d}  ' + ' '.join(f'{k}={r[k]}' for k in ('mem','sfu','vfp','vint','pred','scalar','ctl') if k in r))
