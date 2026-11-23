#!/usr/bin/env python3
"""Q-B (static): classify the instructions of the innermost loops of compiler-emitted kernels (.s, from
TNCA_KERNEL_DUMP). Usage: qb_loops.py DUMPROOT  -> prints one row per kernel and a family summary.
A loop is a backward branch (br*, target label before the branch). Innermost = contains no other backward-branch loop.
Loops whose label starts with 'sdiv' (the software integer divide, there is no scalar divide in hardware) and the
rctx wrapper are reported separately. '.li' is 4 instructions (movz + 3x movk, tools/asm/tnca_asm.py).
Classes: addr = scalar arithmetic that is not one of the others; inc = s.addi/s.add whose destination is also a source
(induction variable or pointer bump); const = s.movz/s.movk/s.movn/.li; cmp = s.cmp/s.cmpi/s.sel; mov = s.mov or s.addi #0;
br = CTRL; pred = p.*; sys = nop/rdspecial/wrspecial; vld/vst = v.ld/v.st (+gather/scatter); sld/sst = s.ld/s.st;
vx = VINT/VFP/CVT/SFU/SHUF/RED; mtx = m.*/a.*; txf = t.*/td.*/w.*; bar = mb.*; atom = atom.*/fence."""
import re, sys, os, collections, json
ISA=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','isa','tnca-isa-v0.1.json')))
MAJ={i['mnemonic']:i['major'] for i in ISA['instructions']}
def cls(line):
    t=line.split('//')[0].strip()
    if not t or t.endswith(':'): return None,0
    parts=t.replace(',',' ').split()
    while parts and (parts[0].startswith('@')): parts=parts[1:]
    if not parts: return None,0
    op=parts[0].lower()
    if op=='.li': return 'const',4
    if op.startswith('.'): return None,0
    if op in ('br','br.nz','br.z','br.r','call','ret','exit','trap','yield'): return 'br',1
    base=op
    while base not in MAJ and '.' in base: base=base.rsplit('.',1)[0]
    m=MAJ.get(base); op=base
    if m is None: return 'other',1
    if m=='SALU':
        if op in ('s.movz','s.movk','s.movn'): return 'const',1
        if op in ('s.cmp','s.cmpi','s.sel') or op.startswith('s.cmp'): return 'cmp',1
        if op=='s.mov': return 'mov',1
        a=[p for p in parts[1:] if not p.startswith('#')]
        if op in ('s.addi','s.add','s.subi','s.sub') and len(a)>=2 and a[0]==a[1] : return 'inc',1
        if op=='s.addi' and parts[-1] in ('#0',): return 'mov',1
        return 'addr',1
    if m=='CTRL': return 'br',1
    if m=='PRED': return 'pred',1
    if m=='SYS': return 'sys',1
    if m=='LSU':
        return {'v.ld':'vld','v.st':'vst','v.ld.gather':'vld','v.st.scatter':'vst','s.ld':'sld','s.st':'sst'}[op],1
    if m in ('ATOM',): return 'atom',1
    if m in ('VINT','VFP','CVT','SFU','SHUF','RED'): return 'vx',1
    if m in ('MATRIX','ACC'): return 'mtx',1
    if m in ('TXFER','TDESC','THINT'): return 'txf',1
    if m in ('MBAR','ROLE'): return 'bar',1
    return 'other',1
SLOT_A={'addr','inc','const','cmp','mov','br','pred','sys'}
def parse(path):
    ins=[]; labels={}
    for ln in open(path):
        t=ln.split('//')[0].strip()
        if not t: continue
        if re.match(r'^[A-Za-z_][\w.]*:$',t): labels[t[:-1]]=len(ins); continue
        c,n=cls(ln)
        if c is None: continue
        parts=t.replace(',',' ').split()
        while parts and parts[0].startswith('@'): parts=parts[1:]
        tgt=parts[-1] if parts[0].lower() in ('br','br.nz','br.z') else None
        ins.append((c,n,tgt,t))
    return ins,labels
def loops(ins,labels):
    L=[]
    for j,(c,n,tgt,t) in enumerate(ins):
        if c=='br' and tgt in labels and labels[tgt]<=j: L.append((labels[tgt],j,tgt))
    inner=[l for l in L if not any(o!=l and o[0]>=l[0] and o[1]<=l[1] for o in L)]
    return L,inner
def mix(ins,a,b):
    m=collections.Counter()
    for c,n,_,_ in ins[a:b+1]: m[c]+=n
    return m
def row(m):
    N=sum(m.values()); A=sum(v for k,v in m.items() if k in SLOT_A)
    return N,A
if __name__=='__main__':
    root=sys.argv[1]
    for dp,dn,fn in sorted(os.walk(root)):
        for f in sorted(fn):
            if not f.endswith('.s'): continue
            ins,labels=parse(os.path.join(dp,f))
            L,inner=loops(ins,labels)
            pt=os.path.relpath(dp,root).split('/')[0].replace('dump_','')
            print('## %s  %s  static=%d loops=%d inner=%d'%(pt,f[:-2][:60],sum(n for _,n,_,_ in ins),len(L),len(inner)))
            for a,b,t in inner:
                if t.startswith('sdiv'): continue
                m=mix(ins,a,b); N,A=row(m)
                print('   loop %-12s len=%4d slotA=%4.1f%%  %s'%(t,N,100*A/N,' '.join('%s=%d'%(k,v) for k,v in sorted(m.items(),key=lambda kv:-kv[1]))))
