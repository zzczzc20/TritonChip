#!/usr/bin/env python3
"""Check asynchronous per-launch profiling against the runtime launch log."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm'),str(ROOT/'tools/models'),str(ROOT/'tools/performance_harness/systemc')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 os.environ.update(TNCA_TORCH_DEVICE='native',TNCA_INDEX_OPT='1',TNCA_TRITON='1',TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp',TNCA_BACKEND='systemc-tlm-grid',TRITON_CACHE_DIR=str(a.out/'triton'),TORCHINDUCTOR_CACHE_DIR=str(a.out/'inductor'))
 import torch,triton,tlm_backend
 import tnca.torch_device as td
 from tnca.triton_backend import driver
 import decode_profile as prof
 tlm_backend.register();td.register();torch.set_num_threads(1)
 x=(torch.arange(256,dtype=torch.int32).reshape(16,16)%7-3).float();w=(torch.arange(256,dtype=torch.int32).reshape(16,16)%5-2).float();want=x@w
 a0,b0=x.to('tnca'),w.to('tnca');fn=torch.compile(lambda a,b:a@b,fullgraph=True)
 prof.install()
 with prof.chat.transfer_meter(True), driver.launch_log() as log:
  prof.STATE['scopes'][-1]['kind']='decode'
  first=fn(a0,b0);second=fn(a0,b0)
 prof.finalize();rows=[r for r in prof.ROWS if not r['skipped']];actual=[r for r in log if not r.get('skipped')]
 if any(r.get('scope_kind')!='decode' or r['step']!=0 for r in rows):raise RuntimeError('profile scope classification mismatch')
 if len(rows)!=len(actual) or not rows:raise RuntimeError('profile launch count differs from the runtime log')
 for r,l in zip(rows,actual):
  if r['kernel']!=l['kernel'] or r['device_cycles']!=l['device_cycles'] or r['event_state']!='SUCCEEDED':raise RuntimeError('profile event mismatch')
  if not r['counters']:raise RuntimeError('per-launch TLM counters are missing')
 got=second.cpu();bad=int((got.view(torch.int32)!=want.view(torch.int32)).sum())
 if bad:raise RuntimeError('output differs bitwise from exact reference')
 if sum(r['code_load_count'] for r in rows)<1:raise RuntimeError('code load was not counted')
 math_rows=rows[-2:]
 if len(math_rows)!=2 or math_rows[0]['kernel']!=math_rows[1]['kernel'] or 'mm' not in math_rows[0]['kernel']:
  raise RuntimeError('cannot identify the two matmul launches')
 result=dict(status='pass',domain='DP SystemC/TLM, not board',exact_bit_mismatches=bad,elements=got.numel(),launches=len(rows),
  code_loads=sum(r['code_load_count'] for r in rows),code_upload_bytes=sum(r['code_upload_bytes'] for r in rows),
  matmul_kernel=math_rows[0]['kernel'],matmul_macs_per_launch=16**3,
  matmul_instructions_per_mac=[sum(v for k,v in r['counters'].items() if k.startswith('core_') and k.endswith('_instructions'))/(16**3) for r in math_rows],
  matmul_device_cycles=[r['device_cycles'] for r in math_rows],skipped_compilation_calls=sum(r['skipped'] for r in prof.ROWS))
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
