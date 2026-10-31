import json,hashlib,collections,math
from pathlib import Path
home=Path.home();profile=json.load(open(home/'r41_cache_model_lru_c/profile/launches.json'))['launches'];seq=[]
for x in profile:
 p=Path(x['artifact']);key=hashlib.sha256(p.read_bytes()).hexdigest();seq.append((x['step'],key,x['code_bytes']))
for power in ('lru',0,0.25,0.5,0.75,1,1.5):
 cache={};freq=collections.Counter();free=[(0,131072)];counts={};calls=0
 def alloc(size):
  for i,(p,n) in enumerate(free):
   if n>=size:
    free.pop(i)
    if n>size:free.insert(i,(p+size,n-size))
    return p
  return None
 for index,(step,key,size) in enumerate(seq):
  counts.setdefault(step,[0,0,0]);calls+=1
  if calls==256:
   for k in list(freq):
    freq[k]//=2
    if not freq[k]:del freq[k]
   calls=0
  freq[key]+=1
  if key in cache:
   p,n,_=cache[key];cache[key]=(p,n,index);continue
  counts[step][0]+=1;counts[step][1]+=size;pos=alloc(size)
  while pos is None and cache:
   victim=min(cache,key=lambda k:(cache[k][2],) if power=='lru' else (freq[k]/(cache[k][1]**power),cache[k][2]))
   p,n,_=cache.pop(victim);counts[step][2]+=1;free.append((p,n));free.sort();merged=[]
   for b,l in free:
    if merged and merged[-1][0]+merged[-1][1]==b:merged[-1]=(merged[-1][0],merged[-1][1]+l)
    else:merged.append((b,l))
   free=merged;pos=alloc(size)
  if pos is None:raise RuntimeError('too large')
  cache[key]=(pos,size,index)
 print(power,counts)
