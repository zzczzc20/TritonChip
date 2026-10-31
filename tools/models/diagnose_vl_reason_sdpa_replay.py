#!/usr/bin/env python3
"""Replay saved reason-prompt SDPA operands with controlled CPU query shapes."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def cmp(a,b):
 a,b=a.float(),b.float(); d=(a-b).abs()
 return dict(max_abs=float(d.max()),rel_l2=float((a-b).norm()/a.norm().clamp_min(1e-30)),equal=bool(torch.equal(a,b)))

def run(backend,q,k,v,mask,scale):
 with sdpa_kernel(backend):return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0,is_causal=False,scale=scale)

def main():
 p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 t=torch.load(a.captures,map_location='cpu',weights_only=True);old=t['legacy_chunk8'][3];group=t['grouped_chunk8'][3];r={}
 for label,backend in [('flash_cpu',SDPBackend.FLASH_ATTENTION),('math',SDPBackend.MATH)]:
  q1=group['q'][:,:,:1,:].contiguous(); mask1=group['mask'][...,:1,:]; scale=group['scale']
  grouped=run(backend,group['q'],group['k'],group['v'],group['mask'],scale)
  single=run(backend,q1,group['k'],group['v'],mask1,scale)
  old_single=run(backend,old['q'],old['k'],old['v'],old['mask'],old['scale'])
  r[label]=dict(same_kv_shape_one_vs_six=cmp(single[:,:,0,:],grouped[:,:,0,:]),
                old_cache_vs_grouped_cache_single=cmp(old_single[:,:,0,:],single[:,:,0,:]),
                single_vs_observed_old=cmp(single[:,:,0,:],old['out'][:,:,0,:]),
                group_vs_observed_group=cmp(grouped[:,:,0,:],group['out'][:,:,0,:]))
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(domain='CPU replay of captured operands, not TLM or board',results=r),indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
