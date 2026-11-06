#!/usr/bin/env python3
"""Compare captured prompt attention with DP TLM and CPU math SDPA."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.torch import load_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,required=True);p.add_argument('--tlm',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 traces=torch.load(a.captures,map_location='cpu',weights_only=True);outputs=load_file(str(a.tlm)); result={}
 for name in ('legacy_chunk8','grouped_chunk8'):
  row=traces[name][3]
  with sdpa_kernel(SDPBackend.MATH): expected=F.scaled_dot_product_attention(row['q'],row['k'],row['v'],attn_mask=row['mask'],dropout_p=0,is_causal=False,scale=row['scale'])
  got=outputs[name]; diff=(got.float()-expected.float()).abs()
  result[name]=dict(equal=bool(torch.equal(got,expected)),different=int((got!=expected).sum()),elements=got.numel(),max_abs=float(diff.max()),rel_l2=float(diff.norm()/expected.float().norm().clamp_min(1e-30)))
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(domain='SystemC/TLM output versus CPU math SDPA, not board',results=result),indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
