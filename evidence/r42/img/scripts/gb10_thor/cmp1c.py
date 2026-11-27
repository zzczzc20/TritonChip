import json,sys,math
def load(f):
    r={}
    for l in open(f):
        d=json.loads(l)
        t=d.get('time') or {}
        c=d.get('correctness') or {}
        r[d['point_id']]=(d['status'], t.get('device'), c.get('max_rel_err'), c.get('tolerance'), (d.get('timings') or {}))
    return r
names=sys.argv[1:]
recs=[load(f) for f in names]
pts=sorted(set().union(*[set(r) for r in recs]))
print('%-52s'%'point'+''.join('%14s'%n.split('/')[-1].replace('.jsonl','')[-13:] for n in names))
logs=[[] for _ in names]
for p in pts:
    row='%-52s'%p
    b=recs[0].get(p)
    for i,r in enumerate(recs):
        x=r.get(p)
        if not x: row+='%14s'%'-'; continue
        s='%s'%(x[1] if x[0]=='pass' else x[0])
        if i>0 and b and b[0]=='pass' and x[0]=='pass' and b[1]:
            ratio=x[1]/b[1]; logs[i].append(math.log(ratio)); s+=' %.3f'%ratio
        row+='%14s'%s
    print(row)
for i,n in enumerate(names):
    if i>0 and logs[i]: print('geomean %s / %s: %.3f over %d points'%(n.split('/')[-1],names[0].split('/')[-1],math.exp(sum(logs[i])/len(logs[i])),len(logs[i])))
print('not pass:',[ (n.split('/')[-1],p) for n,r in zip(names,recs) for p,x in r.items() if x[0]!='pass'] or 'none')
