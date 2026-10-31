#!/usr/bin/env python3
"""Runtime-history A/B: standard Torch matmul on DP TLM, no board."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm'),str(ROOT/'tools/performance_harness/systemc')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--history',type=int,default=1000);p.add_argument('--reps',type=int,default=1000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 os.environ.update(TNCA_TORCH_DEVICE='native',TNCA_INDEX_OPT='1',TNCA_TRITON='1',TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp',TNCA_BACKEND='systemc-tlm-grid',TRITON_CACHE_DIR=str(a.out/'triton'),TORCHINDUCTOR_CACHE_DIR=str(a.out/'inductor'))
 os.environ.setdefault('TNCA_DEADLINE','40000000000');os.environ.setdefault('TNCA_HOST_TIMEOUT','40000')
 import torch,triton,tlm_backend
 import tnca.torch_device as td
 from tnca.runtime import Session
 from tnca.triton_backend import driver
 tlm_backend.register();td.register();torch.set_num_threads(1)
 x=(torch.arange(256,dtype=torch.int32).reshape(16,16)%7-3).float();w=(torch.arange(256,dtype=torch.int32).reshape(16,16)%5-2).float();want=x@w
 dx,dw=x.to('tnca'),w.to('tnca');f=torch.compile(lambda a,b:a@b,fullgraph=True);s=td.MEMORY.session(0)
 loads=0;orig=Session.load
 def counted(self,*args,**kw):
  nonlocal loads
  r=orig(self,*args,**kw);loads+=1;return r
 Session.load=counted
 try:
  with driver.launch_log() as cold:y=f(dx,dw)
  s.drain();assert torch.equal(y.cpu().view(torch.int32),want.view(torch.int32))
  cold=[e for e in cold if not e.get('skipped')]
  for _ in range(a.history):y=f(dx,dw);s.drain()
  before_history=len(s._events);ev0=driver.evicted;ld0=loads;t=time.monotonic()
  for _ in range(a.reps):y=f(dx,dw);s.drain()
  wall=time.monotonic()-t;assert torch.equal(y.cpu().view(torch.int32),want.view(torch.int32))
  before=s.dev.counters_snapshot();ev1=driver.evicted;ld1=loads
  with driver.launch_log() as log:y=f(dx,dw)
  s.drain();after=s.dev.counters_snapshot();got=y.cpu();bad=int((got.view(torch.int32)!=want.view(torch.int32)).sum())
  measured=[e for e in log if not e.get('skipped')];instr=sum(after[k]-before.get(k,0) for k in after if k.startswith('core_') and k.endswith('_instructions'))
  r=dict(schema='r41-runtime-event-history-ab/1',domain='standard Torch FP32 matmul on uncalibrated SystemC/TLM; not board',
   exact_bit_mismatches=bad,elements=got.numel(),shape=[16,16,16],history_before_timed=before_history,
   repetitions=a.reps,wall_seconds=wall,host_wall_seconds_per_call=wall/a.reps,history_after=len(s._events),
   timed_code_loads=ld1-ld0,timed_code_evictions=ev1-ev0,
   final_call=dict(instructions=instr,instructions_per_mac=instr/4096,device_cycles=sum(e['device_cycles'] for e in measured),
    launches=len(measured),code_loads=loads-ld1,code_evictions=driver.evicted-ev1),cold_launches=len(cold))
  (a.out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
  if bad:raise RuntimeError('bitwise mismatch')
 finally:Session.load=orig
if __name__=='__main__':main()
