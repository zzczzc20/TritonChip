#!/usr/bin/env python3
"""Achieved GFLOP/s per point on the run-3 image: demand FLOP (ArchBench demand field, roofline/points.json) / (cycles/1e8).
Old = run-3 image + main software, new = run-3 image + R42 software round 1 (ab-int1). Board cycles."""
import json
ab=json.load(open('evidence/r42/board/ab-int1/comparison.json'))
pts=json.load(open('evidence/r41/stage2/roofline/points.json'))
dem={}
for k in ('tnca100','tnca90'):
    for p in pts[k]: dem.setdefault(p['id'],(p['flop'],p['bytes'],p['dtype']))
rows=[]
for r in ab['rows']:
    if r['point'] not in dem: continue
    f,b,dt=dem[r['point']]
    g_old=f/(r['old_cycles']/1e8)/1e9; g_new=f/(r['new_cycles']/1e8)/1e9
    rows.append((g_new,g_old,r['point'],r['family'],f,b,r['new_cycles']))
rows.sort(reverse=True)
print('top points by achieved GFLOP/s (board, new = R42 round 1; old = run-3 base)')
for g,go,p,fam,f,b,c in rows[:14]:
    print('%7.3f %7.3f  %-60s FLOP %.3g  B %.3g  AI %.1f'%(g,go,p,f,b,f/b))
print('\nbest per family (new)')
best={}
for g,go,p,fam,f,b,c in rows:
    if fam not in best: best[fam]=(g,go,p)
for fam,(g,go,p) in sorted(best.items(),key=lambda kv:-kv[1][0]):
    print('%-18s %7.3f (old %7.3f) %s'%(fam,g,go,p))
