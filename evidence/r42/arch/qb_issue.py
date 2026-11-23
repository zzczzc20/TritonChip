#!/usr/bin/env python3
"""Q-B: issue-bound vs wait-bound, per point and per family, and the Amdahl bound of a two-slot core.
Input: a directory of per-point outputs of arch_pt.sh: <tag>.jsonl (ArchBench record with --counters) and
<tag>.probe (arch_probe.v: one 'I rctx major func' per issued instruction).
Classes (by mnemonic, from isa/tnca-isa-v0.1.json):
  A (slot A, scalar/control side): SALU, PRED, SYS (nop, s.rdspecial, s.wrspecial), CTRL
  B (slot B, everything else): LSU (s.ld/s.st too), ATOM, VINT, VFP, CVT, SFU, SHUF, RED, MATRIX, ACC, TXFER, TDESC, THINT, MBAR, ROLE, DEBUG
Pairing bounds (upper bounds: no register dependence, no structural hazard is checked):
  min(NA,NB)       perfect pairing anywhere in the dynamic stream
  adj_ctx          greedy pairing of adjacent instructions of the same role context (an in-order 2-wide front end)
  adj_all          greedy pairing of adjacent instructions of the issue stream (any contexts)
Saved cycles = number of pairs (one issue cycle each). Bound speedup = cycles / (cycles - pairs)."""
import json, sys, glob, os, collections, math
ROOT=os.path.dirname(os.path.abspath(__file__))
isa=json.load(open(os.path.join(ROOT,'..','..','..','isa','tnca-isa-v0.1.json')))
maj={v:k for k,v in isa['majors'].items()}
mn={}
for i in isa['instructions']: mn[(i['major_code'],i['func'])]=(i['major'],i['mnemonic'])
A_MAJ={'SALU','PRED','SYS','CTRL'}
def klass(major,func):
    m,name=mn.get((major,func),(maj.get(major,'?'),'?'))
    return ('A' if m in A_MAJ else 'B'), m, name
def load(dirn):
    pts={}
    for pf in sorted(glob.glob(os.path.join(dirn,'*.probe'))):
        tag=os.path.basename(pf)[:-6]
        jf=os.path.join(dirn,tag+'.jsonl')
        if not os.path.exists(jf): continue
        rec=None
        for l in open(jf):
            if l.strip(): rec=json.loads(l)
        if not rec or not rec.get('counters'): continue
        stream=[]; launches=0
        for l in open(pf):
            t=l.split()
            if not t: continue
            if t[0]=='I': stream.append((int(t[1]),int(t[2]),int(t[3])))
            elif t[0]=='L': launches+=1
        pts[tag]=(rec,stream,launches)
    return pts
def analyse(rec,stream):
    c=rec['counters']; cyc=c['perf_cycles']; iss=c['perf_issued']
    cnt=collections.Counter(); bymaj=collections.Counter(); byname=collections.Counter()
    for r,m,f in stream:
        k,mj,name=klass(m,f); cnt[k]+=1; bymaj[mj]+=1; byname[name]+=1
    NA,NB=cnt['A'],cnt['B']
    # greedy adjacent pairing, same context / any context
    pend={}; pc=0
    for r,m,f in stream:
        k=klass(m,f)[0]
        p=pend.get(r)
        if p is not None and p!=k: pc+=1; pend[r]=None
        else: pend[r]=k
    pa=0; p=None
    for r,m,f in stream:
        k=klass(m,f)[0]
        if p is not None and p!=k: pa+=1; p=None
        else: p=k
    return dict(cyc=cyc,issued=iss,probe_issued=len(stream),NA=NA,NB=NB,bymaj=bymaj,byname=byname,pairs_min=min(NA,NB),pairs_ctx=pc,pairs_all=pa,c=c)
def family(tag): 
    return tag.split('_')[0].split('.')[1] if tag.startswith('op.') else tag.split('_')[0].split('.')[1]
if __name__=='__main__':
    d=sys.argv[1]
    pts=load(d)
    rows=[]
    for tag,(rec,stream,nl) in pts.items():
        a=analyse(rec,stream); a['tag']=tag; a['launches']=nl; a['dev']=rec['time']['device']; rows.append(a)
    print('%-52s %8s %7s %6s %6s %5s %5s %5s %5s %5s %7s %7s %7s'%('point','cycles','issued','iss%','stall%','hz%','lsu%','mtx%','txf%','fnc%','NA/N','sp_ctx','sp_min'))
    for a in sorted(rows,key=lambda x:x['tag']):
        c=a['c']; cy=a['cyc']
        sp=lambda p: cy/(cy-p)
        print('%-52s %8d %7d %5.1f%% %5.1f%% %4.1f%% %4.1f%% %4.1f%% %4.1f%% %4.1f%% %6.1f%% %7.3f %7.3f'%(a['tag'][:52],cy,a['issued'],100*a['issued']/cy,100*c['perf_stall_total']/cy,100*c['perf_x_hazard']/cy,100*c['perf_x_lsu']/cy,100*c['perf_x_mtx']/cy,100*c['perf_x_txf']/cy,100*c['perf_x_fence']/cy,100*a['NA']/max(1,a['NA']+a['NB']),sp(a['pairs_ctx']),sp(a['pairs_min'])))
    json.dump([{k:(dict(v) if isinstance(v,collections.Counter) else v) for k,v in a.items()} for a in rows],open(os.path.join(d,'qb_summary.json'),'w'))
