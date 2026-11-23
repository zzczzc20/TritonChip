#!/usr/bin/env python3
"""Markdown tables for doc sec 2.2 from (a) recorded B39 counters and (b) fresh runs. Usage: qb_md.py b39 TIER | fresh DIR [DIR..]"""
import sys, os, json, collections, math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import qb_issue as q
mode=sys.argv[1]
if mode=='b39':
    tier=sys.argv[2]
    R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','spec','r40','b39',tier+'.jsonl')
    agg=collections.defaultdict(collections.Counter); n=collections.Counter()
    for l in open(R):
        r=json.loads(l); c=r['counters']; f=r['family']; n[f]+=1; agg[f]['cyc']+=c['perf_cycles']
        for k in c: agg[f][k]+=c[k]
    print('| family | n | cycles | issued | hazard | lsu | fpu | mtx | txf | fence | barrier wait / front end |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    T=collections.Counter()
    for f in sorted(agg):
        a=agg[f]; cy=a['cyc']
        print('| %s | %d | %d | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% |'%(f,n[f],cy,*(100*a[k]/cy for k in ('perf_issued','perf_x_hazard','perf_x_lsu','perf_x_fpu','perf_x_mtx','perf_x_txf','perf_x_fence','perf_x_frontend'))))
else:
    rows=[]
    for d in sys.argv[2:]:
        for tag,(rec,stream,nl) in q.load(d).items():
            a=q.analyse(rec,stream); a['tag']=tag; a['fam']=rec['family']; a['tier']=rec['tier']; rows.append(a)
    by=collections.defaultdict(list)
    for a in rows: by[(a['tier'],a['fam'])].append(a)
    S=lambda L,k: sum(a[k] for a in L); C=lambda L,k: sum(a['c'][k] for a in L)
    print('| tier | family | n | cycles | issued | hazard | lsu | fpu | mtx | txf | fence | barrier wait / front end | slot A of issued | 2-wide in-order bound | all slot A removed |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for (tier,f),L in sorted(by.items()):
        cy=S(L,'cyc'); n=S(L,'NA')+S(L,'NB')
        P=lambda k: 100*C(L,k)/cy
        print('| %s | %s | %d | %d | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.2fx | %.2fx |'%(tier,f,len(L),cy,P('perf_issued'),P('perf_x_hazard'),P('perf_x_lsu'),P('perf_x_fpu'),P('perf_x_mtx'),P('perf_x_txf'),P('perf_x_fence'),P('perf_x_frontend'),100*S(L,'NA')/n,cy/(cy-S(L,'pairs_ctx')),cy/(cy-S(L,'NA'))))
    for tier in sorted(set(a['tier'] for a in rows)):
        L=[a for a in rows if a['tier']==tier]
        g=lambda k: math.exp(sum(math.log(a['cyc']/(a['cyc']-a[k])) for a in L)/len(L))
        print('| %s | **geometric mean of the per-point bound** | %d | | | | | | | | | | | **%.3fx** | **%.3fx** |'%(tier,len(L),g('pairs_ctx'),g('NA')))
