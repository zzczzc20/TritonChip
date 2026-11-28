import csv,sys,re,collections
f=sys.argv[1]; unit=sys.argv[2]
def norm(pin):
    p=pin.split('/')
    p=[re.sub(r'\[\d+\]','[]',x) for x in p]
    return '/'.join(p[2:]) if len(p)>2 else pin
rows=[r for r in csv.DictReader(open(f)) if r['end'].startswith(unit)]
c=collections.defaultdict(lambda:[0,0,9,0])
for r in rows:
    s=float(r['slack'])
    k=(norm(r['start'].rsplit('/',1)[0]),norm(r['end'].rsplit('/',1)[0]))
    d=c[k]; d[1]+=1; d[2]=min(d[2],s); d[3]=max(d[3],int(r['levels']))
    if s<0: d[0]+=1
for k,d in sorted(c.items(), key=lambda kv:kv[1][2]):
    if d[0]>=int(sys.argv[3]): print('%6d/%6d %7.3f lv%2d  %s -> %s'%(d[0],d[1],d[2],d[3],k[0],k[1]))
