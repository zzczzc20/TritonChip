#!/usr/bin/env python3
"""Q-B: recorded counters of the B39 baseline (spec/r40/b39/{tiny,small,medium}.jsonl: single-core RTL simulation,
profile core-vrf-sync-v5, pre-R41 software). Per family: share of cycles in which an instruction issued (perf_issued),
exclusive stall shares by cause, and perf_x_frontend. Sum over the family's points of that tier."""
import json, collections, sys
R='spec/r40/b39/'
fields=['perf_issued','perf_x_hazard','perf_x_lsu','perf_x_fpu','perf_x_mtx','perf_x_txf','perf_x_fence','perf_x_arrive','perf_x_rsv','perf_x_wb','perf_x_vrf','perf_x_vrfwb','perf_x_frontend']
for tier in ('tiny','small','medium'):
    agg=collections.defaultdict(collections.Counter); n=collections.Counter()
    for l in open(R+tier+'.jsonl'):
        r=json.loads(l); c=r.get('counters')
        if not c or r.get('status')!='ok' and r.get('status') is not None and False: continue
        f=r['family']; n[f]+=1; agg[f]['cyc']+=c['perf_cycles']
        for k in fields: agg[f][k]+=c.get(k,0)
    print('\n== %s tier, %d points; shares of perf_cycles (sum over points)'%(tier,sum(n.values())))
    print('%-18s %2s %10s %6s %5s %5s %5s %5s %5s %5s %6s'%('family','n','cycles','issued','hazrd','lsu','fpu','mtx','txf','fence','frontd'))
    tot=collections.Counter()
    for f in sorted(agg):
        a=agg[f]; cy=a['cyc']
        other=sum(a[k] for k in fields if k not in ('perf_issued','perf_x_hazard','perf_x_lsu','perf_x_fpu','perf_x_mtx','perf_x_txf','perf_x_fence','perf_x_frontend'))
        print('%-18s %2d %10d %5.1f%% %4.1f%% %4.1f%% %4.1f%% %4.1f%% %4.1f%% %4.1f%% %5.1f%%'%(f,n[f],cy,*(100*a[k]/cy for k in ('perf_issued','perf_x_hazard','perf_x_lsu','perf_x_fpu','perf_x_mtx','perf_x_txf','perf_x_fence','perf_x_frontend'))))
        for k in a: tot[k]+=a[k]
    cy=tot['cyc']
    print('%-18s %2d %10d %5.1f%% %4.1f%% %4.1f%% %4.1f%% %4.1f%% %4.1f%% %4.1f%% %5.1f%%'%('ALL',sum(n.values()),cy,*(100*tot[k]/cy for k in ('perf_issued','perf_x_hazard','perf_x_lsu','perf_x_fpu','perf_x_mtx','perf_x_txf','perf_x_fence','perf_x_frontend'))))
