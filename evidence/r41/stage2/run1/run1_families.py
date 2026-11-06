#!/usr/bin/env python3
"""Group run 1's routed near-critical kernel endpoints (< +0.5 ns) into families.

    run1_families.py REPO NEAR_TSV OUT_JSON

Family keys: tools/r40/group_failing_paths.family (the A3 census key). In-tile
owners: evidence/r41/stage2/integration/screen_families.owner on the tile-relative
names. Out-of-tile paths (either end outside g_tile[*].u_tile) get an area bucket.
"""
import json, re, sys, importlib.util
from collections import defaultdict
repo, tsv, out = sys.argv[1:4]
sys.path.insert(0, repo + '/tools/r40')
from group_failing_paths import family, parse_tsv
spec = importlib.util.spec_from_file_location('sf', repo + '/evidence/r41/stage2/integration/screen_families.py')
sf = importlib.util.module_from_spec(spec); spec.loader.exec_module(sf)
PREFIX = 'level0_i/ulp/tnca_u50_1/inst/device/'
TILE = re.compile(r'g_tile\[(\d+)\]\.u_tile/')
OUT = [  # out-of-tile area buckets, first match on dst then src
    (r'core_idle|tile_idle|quiesc|handoff|host_allow|memory_active', 'quiescence / handoff cone (cross-SLR)'),
    (r'dispatch', 'F06 grid dispatch'),
    (r'u_mesh|router|noc|g_link|link_pipe|core_ni', 'mesh / NoC links'),
    (r'nuca|u_slice|home', 'F07 NUCA / memory home'),
    (r'hbm|axi|outlet|mem_port|bridge|line_axi', 'memory outlets / AXI'),
    (r'control|mmio|tmu', 'control / MMIO'),
]
def area(s, d):
    if not s.startswith(PREFIX) or not d.startswith(PREFIX):
        return 'shell / platform (outside the kernel device)'
    for text in (d, s):
        for pat, name in OUT:
            if re.search(pat, text):
                return name
    return 'other kernel (outside tiles)'
rows = parse_tsv(tsv)
fam = defaultdict(lambda: dict(n=0, failing=0, tns=0.0, worst=None))
for r in rows:
    s, d = r['source'], r['destination']
    ts, td = TILE.search(s), TILE.search(d)
    if ts and td and ts.group(1) == td.group(1):
        rel = lambda p: p[TILE.search(p).end():]
        cls, who, tile = 'in-tile', sf.owner(rel(s), rel(d)), int(ts.group(1))
    else:
        cls, who, tile = 'out-of-tile', area(s, d), None
    key = (cls, who, tile, family(s), family(d))
    f = fam[key]; f['n'] += 1
    if r['slack'] < 0: f['failing'] += 1; f['tns'] += r['slack']
    if f['worst'] is None or r['slack'] < f['worst']['slack']: f['worst'] = r
res = []
for (cls, who, tile, fs, fd), f in fam.items():
    w = f['worst']
    res.append(dict(cls=cls, owner=who, tile=tile, source_family=fs, destination_family=fd, endpoints=f['n'],
                    failing=f['failing'], tns_ns=round(f['tns'], 3), worst_slack_ns=w['slack'], levels=w['levels'],
                    delay_ns=w['delay'], net_ns=w['net'], slr=f"{w.get('start_slr')}->{w.get('end_slr')}"))
res.sort(key=lambda x: x['worst_slack_ns'])
own = defaultdict(lambda: dict(families=0, endpoints=0, worst_slack_ns=9.0, below_0p2=0))
for r in rows:
    pass
for x in res:
    o = own[(x['cls'], x['owner'], x['tile'])]
    o['families'] += 1; o['endpoints'] += x['endpoints']; o['worst_slack_ns'] = min(o['worst_slack_ns'], x['worst_slack_ns'])
summary = [dict(cls=k[0], owner=k[1], tile=k[2], **v) for k, v in sorted(own.items(), key=lambda kv: kv[1]['worst_slack_ns'])]
json.dump(dict(schema='tnca-r41s2-run1-families/1', margin_ns=0.5, endpoints=len(rows), owners=summary, families=res),
          open(out, 'w'), indent=1)
for s in summary:
    print(f"{s['worst_slack_ns']:+.3f} {s['endpoints']:6d} ep {s['families']:4d} fam  {s['cls']:11s} t{s['tile']}  {s['owner']}")
