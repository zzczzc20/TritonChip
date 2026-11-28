import csv,sys,collections,bisect
def unit(pin):
    p=pin.split('/')
    if p[0]=='u_core': return 'u_core/'+(p[1] if len(p)>3 else '(core own)')
    return p[0] if len(p)>2 else '(tile own)'
tiles={'s0 default tile':'s0/endpoints.csv','s3 shrunk tile':'s3/endpoints.csv'}
for name,f in tiles.items():
    by=collections.defaultdict(list)
    for r in csv.DictReader(open(f)): by[unit(r['end'])].append(float(r['slack']))
    print('\n%s, routed OOC, 5.6 ns constraint (endpoints with slack < 1.5 ns; path = 5.6 - slack)'%name)
    print('| unit | endpoints <1.5 | fail @5.6 | path>5.9 (slack<-0.3) | path>6.2 (<-0.6) | path>6.67 (<-1.07) | worst slack |')
    print('|---|--:|--:|--:|--:|--:|--:|')
    tot=[0]*5
    for u,l in sorted(by.items(), key=lambda kv:min(kv[1])):
        l.sort(); v=[len(l),bisect.bisect_left(l,0),bisect.bisect_left(l,-0.3),bisect.bisect_left(l,-0.6),bisect.bisect_left(l,-1.07)]
        for i in range(5): tot[i]+=v[i]
        print('| %s | %d | %d | %d | %d | %d | %.3f |'%(u,*v,l[0]))
    print('| **tile** | %d | %d | %d | %d | %d | |'%tuple(tot))
