#!/usr/bin/env python3
"""Per small-tier point: the kernel whose largest innermost loop is largest, that loop's length, slot-A share and the
slot-A breakdown (static). Markdown rows for doc/r42_arch_headroom.md sec 2.3."""
import sys, os, collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import qb_loops as q
root=sys.argv[1]
best={}
for dp,dn,fn in sorted(os.walk(root)):
    for f in sorted(fn):
        if not f.endswith('.s'): continue
        pt=os.path.relpath(dp,root).split('/')[0].replace('dump_','')
        ins,labels=q.parse(os.path.join(dp,f))
        L,inner=q.loops(ins,labels)
        for a,b,t in inner:
            if t.startswith('sdiv'): continue
            m=q.mix(ins,a,b); N,A=q.row(m)
            k=(pt)
            if k not in best or N>best[k][0]: best[k]=(N,A,m,t,f[:-2])
print('| point (small tier) | hot innermost loop | instr | slot A | inc | cmp | br | const | mov | addr | pred | vector/ldst/mtx/txf/bar |')
print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for pt,(N,A,m,t,k) in sorted(best.items()):
    other=m['vx']+m['vld']+m['vst']+m['sld']+m['sst']+m['mtx']+m['txf']+m['bar']+m['atom']
    print('| %s | `%s` (%s) | %d | %.0f %% | %d | %d | %d | %d | %d | %d | %d | vx %d, ld/st %d, mtx %d, txf %d, bar %d |'%(pt,t,k.split('_',2)[-1][:28],N,100*A/N,m['inc'],m['cmp'],m['br'],m['const'],m['mov'],m['addr'],m['pred'],m['vx'],m['vld']+m['vst']+m['sld']+m['sst']+m['atom'],m['mtx'],m['txf'],m['bar']))
