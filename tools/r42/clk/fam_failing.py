import csv,sys,re,collections
path=sys.argv[1]; period=float(sys.argv[2]); unitsel=sys.argv[3]; topn=int(sys.argv[4]) if len(sys.argv)>4 else 25
def unit(pin):
    p=pin.split('/')
    if p[0]=='u_core': return 'u_core/'+(p[1] if len(p)>3 else '(core own)')
    return p[0] if len(p)>2 else '(tile own)'
def fam(pin):
    p=pin.split('/')
    leaf=p[-2] if len(p)>1 else pin
    leaf=re.sub(r'\[\d+\]','[]',leaf)
    leaf=re.sub(r'_rep(licate|lica)?(_\d+)?(__\d+)?$','',leaf)
    leaf=re.sub(r'_replica(_\d+)?','',leaf)
    leaf=re.sub(r'__\d+$','',leaf)
    return leaf
g=collections.defaultdict(list)
for r in csv.DictReader(open(path)):
    if unit(r['end'])!=unitsel and unitsel!='*': continue
    g[(fam(r['start']),fam(r['end']))].append((float(r['slack']),int(r['levels']),float(r['delay']),float(r['net'])))
rows=[]
for k,v in g.items():
    f=[x for x in v if x[0]<0]
    if not f: continue
    rows.append((len(f),min(x[0] for x in v),max(x[1] for x in f),k, sum(x[3] for x in f)/max(sum(x[2] for x in f),1e-9)))
rows.sort(key=lambda x:-x[0])
print('failing endpoints (slack<0 at %.2f) by start->end family; unit %s; total failing %d' % (period,unitsel,sum(r[0] for r in rows)))
for n,ws,lv,k,nf in rows[:topn]:
    print('%6d  ws %.3f  maxlv %2d  net %.0f%%  %s -> %s'%(n,ws,lv,100*nf,k[0],k[1]))
