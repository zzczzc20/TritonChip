import csv,collections,re,sys
RAMS={'RAMD64E':64,'RAMD32':32,'RAMS32':32,'RAMS64E':64,'RAM64X1D':64,'RAM32X1D':32,'RAM256X1D':256,'RAM128X1D':128,'RAM64M8':64*8,'RAM32M16':32*16,'RAM64M':64*4,'RAM32M':32*4}
LUTS={'RAMD64E':1,'RAMD32':1,'RAMS32':1,'RAMS64E':1,'RAM64X1D':2,'RAM32X1D':2,'RAM256X1D':8,'RAM128X1D':4,'RAM64M8':8,'RAM32M16':8,'RAM64M':4,'RAM32M':4}
def load(path, prefix):
    g=collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(path)):
        n=r['name']; ref=r['ref']; c=int(r['count'])
        if not n.startswith(prefix): continue
        if ref in RAMS or ref.startswith('FD') or ref in ('SRL16E','SRLC32E','RAMB36E2','RAMB18E2','DSP48E2','URAM288'):
            a=n[len(prefix):].lstrip('/')
            a=re.sub(r'(_rep(__\d+)?|__\d+)$','',a)
            g[a][ref]+=c
    return g
if __name__=='__main__':
    g=load(sys.argv[1], sys.argv[2])
    minff=int(sys.argv[3]) if len(sys.argv)>3 else 16
    rows=[]
    for a,cn in g.items():
        ff=sum(c for r,c in cn.items() if r.startswith('FD'))
        ram=sum(c*LUTS[r] for r,c in cn.items() if r in RAMS)
        bits=sum(c*RAMS[r] for r,c in cn.items() if r in RAMS)
        other={r:c for r,c in cn.items() if not r.startswith('FD') and r not in RAMS}
        rows.append((a,ff,ram,bits,other))
    rows.sort(key=lambda x:-(x[1]+x[2]*4))
    for a,ff,ram,bits,other in rows:
        if ff>=minff or ram>0 or other: print(f"{a[:70]:<70} FF={ff:6d} LUTRAM_LUT={ram:5d} ramcells_bits={bits:7d} {other if other else ''}")
