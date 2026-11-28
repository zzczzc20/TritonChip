import csv,sys,re,collections
path=sys.argv[1]; per=float(sys.argv[2]); thr=float(sys.argv[3]); units=sys.argv[4].split(';'); topn=int(sys.argv[5])
def unit(pin):
    p=pin.split('/')
    if p[0]=='u_core': return 'u_core/'+(p[1] if len(p)>3 else '(core own)')
    return p[0] if len(p)>2 else '(tile own)'
def fam(pin):
    p=pin.split('/')
    leaf=p[-2] if len(p)>1 else pin
    leaf=re.sub(r'\[\d+\]','[]',leaf); leaf=re.sub(r'_replica(_\d+)?','',leaf); leaf=re.sub(r'_rep(_\d+)?(__\d+)?$','',leaf); leaf=re.sub(r'__\d+$','',leaf)
    return leaf
for u in units:
    g=collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if unit(r['end'])!=u: continue
        pl=per-float(r['slack'])
        if pl>thr: g[(fam(r['start']),fam(r['end']))].append((pl,int(r['levels']),float(r['net'])/max(float(r['delay']),1e-9)))
    rows=sorted(g.items(), key=lambda kv:-len(kv[1]))
    print('== %s: endpoints with routed path > %.2f ns: %d in %d families'%(u,thr,sum(len(v) for k,v in rows),len(rows)))
    for k,v in rows[:topn]:
        print('  %5d  max %.2f  lv %2d  net %2.0f%%  %s -> %s'%(len(v),max(x[0] for x in v),max(x[1] for x in v),100*sum(x[2] for x in v)/len(v),k[0],k[1]))
