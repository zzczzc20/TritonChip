#!/usr/bin/env python3
"""tlm_cmp.py BASE_DIR HIT_DIR  -- (GB10) per-tier geometric mean of device_cycles(hit config) / device_cycles(base) over the collected TLM runs of the sets tsm hv px xd xp
(BASE_DIR = ~/r42/dse_out/col with head_dse_base_<set>.json, HIT_DIR with hit1_<set>.json). Domain: SystemC/TLM model."""
import collections
import json
import math
import sys

base, hit = sys.argv[1], sys.argv[2]


def idx(j):
    if isinstance(j, dict) and 'points' in j:
        j = j['points']
    return j if isinstance(j, dict) else {r['key']: r for r in j}


g = lambda l: math.exp(sum(map(math.log, l)) / len(l))
tier = collections.defaultdict(list)
for s in ('tsm', 'hv', 'px', 'xd', 'xp'):
    a = idx(json.load(open('%s/head_dse_base_%s.json' % (base, s))))
    b = idx(json.load(open('%s/hit1_%s.json' % (hit, s))))
    for k in a:
        if k in b and a[k].get('device_cycles') and b[k].get('device_cycles') and a[k].get('status') == 'pass' and b[k].get('status') == 'pass':
            tier[k.split('/')[1]].append(b[k]['device_cycles'] / a[k]['device_cycles'])
allr = [x for l in tier.values() for x in l]
print('points %d geomean %.4f' % (len(allr), g(allr)))
for t in ('tiny', 'small', 'medium', 'large', 'hbm', 'xl'):
    if tier[t]:
        print('%-7s %3d  %.4f  (min %.4f max %.4f)' % (t, len(tier[t]), g(tier[t]), min(tier[t]), max(tier[t])))
