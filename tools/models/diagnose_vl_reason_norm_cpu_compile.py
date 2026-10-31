#!/usr/bin/env python3
"""Compare eager CPU, CPU-Inductor and TLM RMSNorm for a real prompt input."""
import argparse,json,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
import chat

def cmp(x,y):
 d=(x.float()-y.float()).abs();return dict(equal=bool(torch.equal(x,y)),different=int((x!=y).sum()),elements=x.numel(),max_abs=float(d.max()),rel_l2=float(d.norm()/y.float().norm().clamp_min(1e-30)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--norm',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--emulate-precision-casts',action='store_true');a=p.parse_args()
 args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','2','--new-tokens','1']);app=chat.ChatApplication(args);norm=app.model.model.language_model.layers[2].input_layernorm;t=load_file(str(a.norm/'tensors.safetensors'))
 from torch._inductor import config
 start=time.monotonic()
 with config.patch(emulate_precision_casts=a.emulate_precision_casts),torch.no_grad():
  fn=torch.compile(norm,fullgraph=True,dynamic=False); value=fn(t['input']).detach().cpu().contiguous()
 r=dict(schema='r41-vl-reason-norm-cpu-inductor/1',domain='CPU eager/CPU Inductor vs TLM; not board',emulate_precision_casts=a.emulate_precision_casts,seconds=time.monotonic()-start,cpu_inductor_vs_eager=cmp(value,t['reference']),cpu_inductor_vs_tlm=cmp(value,t['output']))
 a.out.mkdir(parents=True,exist_ok=True);save_file(dict(output=value),str(a.out/'output.safetensors'));(a.out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
