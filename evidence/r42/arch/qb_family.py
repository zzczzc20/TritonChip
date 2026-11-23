#!/usr/bin/env python3
"""Q-B: per-family table from fresh single-core RTL runs of the R42 head (profile core-vrf-sync-v5-x8-g2c4-dp):
cycle partition (perf counters) and the Amdahl bounds of a two-slot core from the dynamic instruction stream.
Usage: qb_family.py DIR [DIR ...]  (each DIR holds <tag>.jsonl and <tag>.probe from arch_pt.sh)"""
import sys, os, collections, math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import qb_issue as q
rows=[]
for d in sys.argv[1:]:
    for tag,(rec,stream,nl) in q.load(d).items():
        a=q.analyse(rec,stream); a['tag']=tag; a['fam']=rec['family']; a['tier']=rec['tier']; rows.append(a)
by=collections.defaultdict(list)
for a in rows: by[(a['tier'],a['fam'])].append(a)
def S(lst,k): return sum(a[k] for a in lst)
def C(lst,k): return sum(a['c'][k] for a in lst)
print('%-6s %-18s %2s %9s | %6s %6s %6s %6s %6s %6s %6s | %6s | %7s %7s %7s'%('tier','family','n','cycles','issued','hazard','lsu','fpu/mtx','txf','fence','front','A/N','dual_ctx','dual_min','rm_A'))
tot=collections.defaultdict(float)
for (tier,f),L in sorted(by.items(),key=lambda kv:(kv[0][0],kv[0][1])):
    cy=S(L,'cyc'); n=S(L,'NA')+S(L,'NB')
    iss=C(L,'perf_issued'); hz=C(L,'perf_x_hazard'); ls=C(L,'perf_x_lsu'); fm=C(L,'perf_x_fpu')+C(L,'perf_x_mtx'); tx=C(L,'perf_x_txf'); fn=C(L,'perf_x_fence'); fe=C(L,'perf_x_frontend')
    print('%-6s %-18s %2d %9d | %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %5.1f%% | %7.3f %7.3f %7.3f'%(tier,f,len(L),cy,100*iss/cy,100*hz/cy,100*ls/cy,100*fm/cy,100*tx/cy,100*fn/cy,100*fe/cy,100*S(L,'NA')/n,cy/(cy-S(L,'pairs_ctx')),cy/(cy-S(L,'pairs_min')),cy/(cy-S(L,'NA'))))
for tier in sorted(set(a['tier'] for a in rows)):
    L=[a for a in rows if a['tier']==tier]; cy=S(L,'cyc'); n=S(L,'NA')+S(L,'NB')
    print('%-6s %-18s %2d %9d | issued %5.1f%%  A/N %5.1f%%  dual_ctx %.3f dual_min %.3f rm_A %.3f'%(tier,'ALL (cycle-weighted)',len(L),cy,100*C(L,'perf_issued')/cy,100*S(L,'NA')/n,cy/(cy-S(L,'pairs_ctx')),cy/(cy-S(L,'pairs_min')),cy/(cy-S(L,'NA'))))
    # geometric mean over points of the per-point bound (the suite metric is a geomean)
    g=lambda k: math.exp(sum(math.log(a['cyc']/(a['cyc']-a[k])) for a in L)/len(L))
    print('%-6s %-18s %2d %9s | per-point geomean of bound: dual_ctx %.3f dual_min %.3f rm_A %.3f'%(tier,'ALL (geomean)',len(L),'',g('pairs_ctx'),g('pairs_min'),g('NA')))

# ---- instruction mix by major, dynamic share of issued instructions, per family ----
GROUP={'SALU':'salu','CTRL':'ctrl','PRED':'pred','SYS':'sys','LSU':'ldst','ATOM':'ldst','VINT':'vec','VFP':'vec','CVT':'vec','SFU':'vec','SHUF':'vec','RED':'vec',
       'MATRIX':'mtx','ACC':'mtx','TXFER':'txf','TDESC':'txf','THINT':'txf','MBAR':'bar','ROLE':'bar','DEBUG':'bar'}
cols=['salu','ctrl','pred','sys','ldst','vec','mtx','txf','bar']
print('\n%-6s %-18s %9s | '%('tier','family','issued')+' '.join('%6s'%c for c in cols))
agg=collections.defaultdict(collections.Counter)
for a in rows:
    for mj,v in a['bymaj'].items(): agg[(a['tier'],a['fam'])][GROUP.get(mj,'bar')]+=v
for (tier,f),c in sorted(agg.items()):
    n=sum(c.values())
    print('%-6s %-18s %9d | '%(tier,f,n)+' '.join('%5.1f%%'%(100*c[k]/n) for k in cols))
top=collections.defaultdict(collections.Counter)
for a in rows:
    for nm,v in a['byname'].items():
        m=q.mn
        top[a['tier']][nm]+=v
for tier,c in top.items():
    print('\ntop mnemonics, %s tier (all points): '%tier+', '.join('%s %d'%(k,v) for k,v in c.most_common(14)))
