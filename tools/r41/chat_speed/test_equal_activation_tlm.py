#!/usr/bin/env python3
"""Check equal-row FP8 staging with differing addresses, masks and scales."""
import argparse,json,os,random,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm'),str(ROOT/'tools/isa'),str(ROOT/'tools/model'),str(ROOT/'tools/performance_harness/systemc')]
import triton
import triton.language as tl

@triton.jit
def kernel(A,W,O,ROW_STRIDE:tl.constexpr,ROW_LIMIT:tl.constexpr,VAR_SCALE:tl.constexpr):
 r=tl.arange(0,16);k=tl.arange(0,32);n=tl.arange(0,16)
 if ROW_LIMIT == 16:
  mask=tl.broadcast_to(k[None,:]<29,(16,32))
 else:
  mask=(r[:,None]<ROW_LIMIT)&(k[None,:]<29)
 x=tl.load(A+r[:,None]*ROW_STRIDE+k[None,:],mask=mask,other=0).to(tl.float32)
 scale=tl.max(tl.abs(x),1)/448.0
 if VAR_SCALE:
  scale=scale*(r.to(tl.float32)+1.0)
 q=(x/tl.maximum(scale[:,None],1e-12)).to(tl.float8e4nv)
 w=tl.load(W+k[:,None]*16+n[None,:])
 z=tl.dot(q,w)
 tl.store(O+r[:,None]*16+n[None,:],z)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 import torch,fdp,tlm_backend
 from tnca import aot,artifact
 from tnca.runtime import Session
 tlm_backend.register();torch.set_num_threads(1);g=torch.Generator().manual_seed(4181)
 data=torch.randn((16,32),generator=g,dtype=torch.bfloat16);weights=(torch.randn((32,16),generator=g)*.1).to(torch.float8_e4m3fn)
 raw_weights=weights.contiguous().view(torch.uint8).numpy().tobytes(); wb=weights.contiguous().view(torch.uint8).reshape(-1).tolist();rows=[]
 cases=((0,16,False,False),(0,7,False,False),(32,16,False,False),(0,16,True,False),(0,16,False,True))
 for stride,limit,var,zero in cases:
  name=f's{stride}_r{limit}_v{int(var)}_z{int(zero)}';src=data.clone()
  if zero:src.zero_()
  x=torch.zeros((16,32),dtype=torch.float32)
  for r in range(limit):x[r,:29]=src[0 if stride==0 else r,:29].float()
  scale=x.abs().amax(dim=1)/448.0
  if var:scale=scale*(torch.arange(16,dtype=torch.float32)+1)
  q=(x/torch.maximum(scale[:,None],torch.tensor(1e-12))).to(torch.float8_e4m3fn)
  av=q.contiguous().view(torch.uint8).reshape(-1).tolist();want=fdp.mma_dp_ref('e4m3','e4m3',av,wb,[0]*256,16,16,32)
  compiled=aot.compile_aot(kernel,signature={'A':'*bf16','W':'*fp8e4nv','O':'*fp32','ROW_STRIDE':'constexpr','ROW_LIMIT':'constexpr','VAR_SCALE':'constexpr'},
   constexprs={'ROW_STRIDE':stride,'ROW_LIMIT':limit,'VAR_SCALE':var},profile='core-vrf-sync-v5-x8-g2c4-dp',options={'index_opt':1},output=str(a.out/(name+'.tnca')))
  man,_=artifact.read_header(compiled.path)
  with Session(profile='core-vrf-sync-v5-x8-g2c4-dp',backend='systemc-tlm-grid',host_timeout=180) as sess:
   module=sess.load(compiled.path);aa,ww,out=[sess.allocate(n) for n in (src.numel()*2,len(raw_weights),1024)]
   sess.upload(aa,src.view(torch.uint8).numpy().tobytes());sess.upload(ww,raw_weights)
   before=sess.dev.counters_snapshot();event=sess.submit(module,{'A':aa,'W':ww,'O':out});sess.wait(event);after=sess.dev.counters_snapshot()
   if not event.ok:raise RuntimeError(str(event.error))
   actual=struct.unpack('<256I',sess.download(out));bad=sum(x!=y for x,y in zip(actual,want));ins=sum(after[k]-before.get(k,0) for k in after if k.startswith('core_') and k.endswith('_instructions'))
   row=dict(name=name,bit_mismatches=bad,elements=256,device_cycles=event.device_cycles,instructions_per_mac=ins/(16*16*32),launches=1,code_loads=1,code_evictions=0,assembly_sha256=man['optional_metadata']['provenance']['assembly_sha256']);rows.append(row);print(json.dumps(row),flush=True)
   if bad:raise RuntimeError(f'{name}: {bad} bit mismatches')
 result=dict(status='pass',domain='DP SystemC/TLM; not board',policy=os.getenv('TNCA_X8_EQUAL_A_STAGE','default'),cases=rows)
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
