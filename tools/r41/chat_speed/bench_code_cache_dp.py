#!/usr/bin/env python3
"""Synthetic code-cache pressure replay plus exact DP compute, no board.

Recorded legacy VL launch order is mapped to unique available DP artifacts by
kernel name (or the closest-size artifact when unavailable). The artifacts
are loaded only, not launched. An independent compiled DP dot is executed and
checked bitwise once after each mapped sequence. This is not full-model TLM.
"""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import random
import re
import struct
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'compiler/test'),str(ROOT/'tools/asm'),str(ROOT/'tools/isa'),str(ROOT/'tools/model'),str(ROOT/'tools/performance_harness'),str(ROOT/'tools/performance_harness/systemc')]

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--rounds',type=int,default=12);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 import triton,fdp,tlm_backend
 from tnca import aot,artifact
 from tnca.runtime import Session
 from tnca.triton_backend import driver
 from test_mma_dp import dot_rows
 tlm_backend.register()
 trace=json.load(gzip.open(a.trace,'rt'))['launches'];trace=[x for x in trace if x['step']==3]
 available=[]
 for path in a.artifacts.rglob('*.tnca'):
  man,segments=artifact.load(path)
  if man['target']['profile']=='core-vrf-sync-v5-x8-g2c4-dp':
   available.append(dict(path=str(path),name=man['identity']['kernel_name'],size=len(segments['text'])))
 normalized=lambda x:re.sub(r'_\d+$','',x)
 mapping={};used=set()
 for item in trace:
  if item['artifact'] in mapping:continue
  candidates=[x for x in available if x['path'] not in used]
  if not candidates:raise RuntimeError('not enough unique DP artifacts')
  def rank(x):
   exact=x['name']==item['kernel'];same=normalized(x['name'])==normalized(item['kernel'])
   return (0 if exact else 1 if same else 2,abs(x['size']-item['code_bytes']),x['path'])
  choice=min(candidates,key=rank);used.add(choice['path']);mapping[item['artifact']]=choice
 seq=[mapping[x['artifact']]['path'] for x in trace]
 compiled=aot.compile_aot(dot_rows,signature={'A':'*fp8e4nv','B':'*fp8e4nv','O':'*fp32','K':'constexpr'},constexprs={'K':32},profile='core-vrf-sync-v5-x8-g2c4-dp',options={'index_opt':1},output=str(a.out/'probe.tnca'))
 rng=random.Random(4181);finite=[0,0x80,1,0x81,0x38,0xb8,0x7e,0xfe]
 av=[rng.choice(finite) for _ in range(16*32)];bv=[rng.choice(finite) for _ in range(32*16)]
 want=fdp.mma_dp_ref('e4m3','e4m3',av,bv,[0]*256,16,16,32)
 rows=[];loads=0;upload=0;orig=Session.load
 def counted(self,*args,**kw):
  nonlocal loads,upload
  m=orig(self,*args,**kw);loads+=1;upload+=m.code_size;return m
 Session.load=counted
 try:
  with Session(profile='core-vrf-sync-v5-x8-g2c4-dp',backend='systemc-tlm-grid',host_timeout=300) as s:
   aa,bb,out=[s.allocate(n) for n in (512,512,1024)];s.upload(aa,bytes(av));s.upload(bb,bytes(bv))
   for i in range(a.rounds):
    ld,up,ev=loads,upload,driver.evicted;t=time.monotonic()
    for path in seq:_,_=driver._load(s,path)
    _,mod=driver._load(s,compiled.path)
    before=s.dev.counters_snapshot();event=s.submit(mod,{'A':aa,'B':bb,'O':out});s.wait(event);after=s.dev.counters_snapshot()
    if not event.ok:raise RuntimeError(str(event.error))
    got=struct.unpack('<256I',s.download(out));bad=sum(x!=y for x,y in zip(got,want))
    instr=sum(after[k]-before.get(k,0) for k in after if k.startswith('core_') and k.endswith('_instructions'))
    row=dict(round=i,phase='cold' if i==0 else 'steady',trace_load_requests=len(seq),probe_launches=1,
      code_loads=loads-ld,code_upload_bytes=upload-up,code_evictions=driver.evicted-ev,
      probe_instructions_per_mac=instr/(16*16*32),probe_device_cycles=event.device_cycles,
      probe_bit_mismatches=bad,probe_elements=256,wall_seconds=time.monotonic()-t)
    rows.append(row);print(json.dumps(row),flush=True)
    if bad:raise RuntimeError('probe differs from exact DP reference')
 finally:Session.load=orig
 result=dict(schema='r41-code-cache-synthetic-ab/1',status='pass',policy=os.getenv('TNCA_CODE_CACHE_POLICY','lru'),
   domain='SystemC/TLM DP; recorded legacy launch order with mapped DP code load requests and exact DP probe; not full-model execution or board',
   profile='core-vrf-sync-v5-x8-g2c4-dp',mapping=mapping,rounds=rows)
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
