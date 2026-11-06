#!/usr/bin/env python3
"""Replay captured CPU reason-prompt attention operands on DP-profile TLM."""
import argparse,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
import chat

def compare(x,y):
 a,b=x.float(),y.float();d=(a-b).abs()
 return dict(equal=bool(torch.equal(x,y)),different=int((x!=y).sum()),elements=x.numel(),max_abs=float(d.max()),rel_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--tree',required=True);p.add_argument('--captures',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if os.environ.get('TNCA_PROFILE')!='core-vrf-sync-v5-x8-g2c4-dp':raise ValueError('DP profile required')
 dev=chat.L.use_tnca(a.tree)
 from tnca.inductor import host_fallbacks
 trace=torch.load(a.captures,map_location='cpu',weights_only=True);outputs={};result={}
 for name in ('legacy_chunk8','grouped_chunk8'):
  row=trace[name][3]; scale=row['scale']
  def attention(q,k,v,mask):return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0,is_causal=False,scale=scale)
  fn=torch.compile(attention,fullgraph=True,dynamic=False); args=[row[k].to(dev) for k in ('q','k','v','mask')]; t0=time.monotonic()
  with torch.no_grad():out,code=chat.run_with_code_capture(lambda:fn(*args)); value=out.cpu().contiguous()
  fallback=host_fallbacks(code) if code else None
  result[name]=dict(seconds=time.monotonic()-t0,query_shape=list(row['q'].shape),compiled_wrapper_fallbacks=fallback,
                    vs_cpu=compare(value,row['out']))
  outputs[name]=value;print(json.dumps(dict(event='path-complete',path=name,result=result[name])),flush=True)
 result['first_query_legacy_vs_grouped']=compare(outputs['legacy_chunk8'][:,:,0,:],outputs['grouped_chunk8'][:,:,0,:])
 result.update(schema='r41-vl-reason-sdpa-tlm/1',domain='standalone SystemC/TLM attention replay, not whole model or board',profile=os.environ['TNCA_PROFILE'])
 a.out.mkdir(parents=True,exist_ok=True);save_file(outputs,str(a.out/'outputs.safetensors'));(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
