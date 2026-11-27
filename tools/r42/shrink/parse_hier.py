import re,sys
def parse(path):
    rows=[]
    for line in open(path):
        if not line.startswith('|') or line.startswith('|-') : continue
        c=[x for x in line.rstrip('\n').split('|')[1:-1]]
        if len(c)<11 or c[0].strip()=='Instance': continue
        inst=c[0]; depth=(len(inst)-len(inst.lstrip(' ')))//2
        try: v=[int(x) for x in (c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10])]
        except: continue
        rows.append(dict(depth=depth,inst=inst.strip(),mod=c[1].strip(),lut=v[0],logic=v[1],lutram=v[2],srl=v[3],ff=v[4],r36=v[5],r18=v[6],uram=v[7],dsp=v[8]))
    return rows
def with_paths(rows):
    stack=[]
    for r in rows:
        while len(stack)>r['depth']: stack.pop()
        stack.append(r['inst'])
        r['path']='/'.join(stack)
    return rows
if __name__=='__main__':
    rows=with_paths(parse(sys.argv[1]))
    pref=sys.argv[2]
    for r in rows:
        if pref in r['path']:
            print(f"{r['depth']:2d} {r['path'].replace(pref,'')[:90]:<90} {r['mod'][:34]:<34} {r['lut']:7d} {r['lutram']:6d} {r['srl']:4d} {r['ff']:6d} {r['r36']:3d} {r['r18']:3d} {r['uram']:2d} {r['dsp']:4d}")
