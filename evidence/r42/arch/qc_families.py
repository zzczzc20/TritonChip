#!/usr/bin/env python3
"""Q-C: per-family cycle share, eta and achieved FLOP/s of the 113-point suite.
Inputs (all recorded): evidence/r42/board/ab-int1/comparison.json (old = R41 run-3 image with main software,
new = same image with R42 software round 1), evidence/r42/baseline/b41_board.json (run-3 eta per point),
evidence/r41/stage2/roofline/points.json (demand FLOP and bytes per point).
Derived (not recorded): eta_new = eta_old * old_cycles/new_cycles (the ArchBench bound is a property of the point)."""
import json, math, collections, sys
R='evidence/'
ab=json.load(open(R+'r42/board/ab-int1/comparison.json'))
bb=json.load(open(R+'r42/baseline/b41_board.json'))
pts=json.load(open(R+'r41/stage2/roofline/points.json'))
dem={}; bind={}
for key in ('tnca100','tnca90'):
    for p in pts[key]:
        dem.setdefault(p['id'],(p['flop'],p['bytes'])); bind.setdefault(p['id'],p['binding'])
eta={r['point']:r['run3_eta'] for r in bb['rows']}
fam=collections.OrderedDict()
tot_old=sum(r['old_cycles'] for r in ab['rows']); tot_new=sum(r['new_cycles'] for r in ab['rows'])
for r in ab['rows']:
    f=r['family']; d=fam.setdefault(f,dict(n=0,old=0,new=0,ne=0,noeta=0,lr=0.0,leta_o=0.0,leta_n=0.0,flop=0.0,missing=0))
    d['n']+=1; d['old']+=r['old_cycles']; d['new']+=r['new_cycles']
    d['lr']+=math.log(r['new_over_old'])
    e=eta[r['point']]
    if e is None: d['noeta']=d.get('noeta',0)+1
    else:
        d['leta_o']+=math.log(e); d['leta_n']+=math.log(e*r['old_cycles']/r['new_cycles']); d['ne']=d.get('ne',0)+1
    if r['point'] in dem:
        d['flop']+=dem[r['point']][0]; d['bind_'+bind[r['point']]]=d.get('bind_'+bind[r['point']],0)+1
        g=dem[r['point']][0]/(r['new_cycles']/1e8)/1e9; d['best']=max(d.get('best',0),g)
    else: d['missing']+=1
N=len(ab['rows']); G=sum(d['lr'] for d in fam.values())
print('points',N,'geomean new/old',math.exp(G/N),'speedup',math.exp(-G/N))
print('total cycles old',tot_old,'new',tot_new)
rows=[]
for f,d in fam.items():
    rows.append((f,d['n'],d['old'],d['new'],d['old']/tot_old,d['new']/tot_new,math.exp(-d['lr']/d['n']),
                 (math.exp(d['leta_o']/d['ne']) if d['ne'] else float('nan')),(math.exp(d['leta_n']/d['ne']) if d['ne'] else float('nan')),d['lr']/N))
# share of the geomean log-gain; headroom = log mass still at eta<1 weighted by n
print('%-18s %3s %14s %14s %7s %7s %7s %7s %7s'%('family','n','cyc_old','cyc_new','sh_old','sh_new','speedup','eta_o','eta_n'))
for r in sorted(rows,key=lambda x:-x[5]):
    print('%-18s %3d %14d %14d %6.1f%% %6.1f%% %7.3f %7.4f %7.4f'%(r[0],r[1],r[2],r[3],100*r[4],100*r[5],r[6],r[7],r[8]))
# geomean eta over all points, old and new
NE=sum(d['ne'] for d in fam.values()); lo=sum(d['leta_o'] for d in fam.values())/NE; ln=sum(d['leta_n'] for d in fam.values())/NE
print('geomean eta old',math.exp(lo),'new',math.exp(ln),'over',NE,'points with an eta (',N-NE,'points have no FLOP bound: embedding, transpose, copy)')
# headroom to eta=1 in log space, per family (share of total log(1/eta_new) mass)
tot=sum(-d['leta_n'] for d in fam.values())
print('\nremaining log(1/eta) mass to eta=1 (share of the geomean gap)')
for f,d in sorted(fam.items(),key=lambda kv:-(-kv[1]['leta_n'])):
    print('%-18s n=%2d  share %5.1f%%   (mean log10 gap per point %.2f)'%(f,d['ne'],100*(-d['leta_n'])/tot,(-d['leta_n']/max(d['ne'],1))/math.log(10)))

print('\nbinding term of the ArchBench strict bound (roofline points.json) and best achieved GFLOP/s per family (R42 board A/B)')
cside=0; mside=0
for f,d in fam.items():
    fb=d.get('bind_fp_lane',0); mb=d.get('bind_mem_bytes',0)
    cside+=fb; mside+=mb
    print('%-18s n=%2d  compute-side %d  memory-side %d   best %.3f GFLOP/s'%(f,d['n'],fb,mb,d.get('best',0)))
print('points with a recorded binding: compute-side %d, memory-side %d (of %d)'%(cside,mside,N))
cyc_c=sum(d['new'] for f,d in fam.items() if d.get('bind_fp_lane',0)>d.get('bind_mem_bytes',0))
print('cycle share (new) of families whose points are mostly compute-side: %.1f%%'%(100*cyc_c/tot_new))

# ---- markdown table for doc/r42_arch_headroom.md ----
print('\n### markdown')
print('| family | n | cycles share old | cycles share new | speed-up (geo) | share of the 1.613x log-gain | eta old | eta new | share of remaining log(1/eta) | compute-side pts | best GFLOP/s |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
Gl=-sum(d['lr'] for d in fam.values())
totm=sum(-d['leta_n'] for d in fam.values())
for f,d in sorted(fam.items(),key=lambda kv:-kv[1]['new']):
    e_o=('%.3f'%math.exp(d['leta_o']/d['ne'])) if d['ne'] else '-'
    e_n=('%.3f'%math.exp(d['leta_n']/d['ne'])) if d['ne'] else '-'
    sh=('%.1f%%'%(100*(-d['leta_n'])/totm)) if d['ne'] else '-'
    print('| %s | %d | %.1f%% | %.1f%% | %.2fx | %.1f%% | %s | %s | %s | %d/%d | %.2f |'%(f,d['n'],100*d['old']/tot_old,100*d['new']/tot_new,math.exp(-d['lr']/d['n']),100*(-d['lr'])/Gl,e_o,e_n,sh,d.get('bind_fp_lane',0),d['n'],d.get('best',0)))
