#!/usr/bin/env python3
"""Run the reason prompt's captured layer-2 FP8 Q projection on DP-profile TLM."""
import argparse,copy,json,os,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
import chat

def compare(x,y):
 a,b=x.float(),y.float();d=(a-b).abs()
 return dict(equal=bool(torch.equal(x,y)),different=int((x!=y).sum()),elements=x.numel(),max_abs=float(d.max()),rel_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--tree',required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if os.environ.get('TNCA_PROFILE')!='core-vrf-sync-v5-x8-g2c4-dp':raise ValueError('DP profile required')
 dev=chat.L.use_tnca(a.tree);args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--new-tokens','1']);app=chat.ChatApplication(args)
 chat.L.install_fp8_route('official');x=load_file(str(a.input));module=copy.deepcopy(app.model.model.language_model.layers[2].self_attn.q_proj).to(dev)
 compiled=torch.compile(module,fullgraph=True,dynamic=False);start=time.monotonic()
 with torch.no_grad():y=compiled(x['input'].to(dev)).cpu().contiguous()
 result=dict(schema='r41-vl-reason-fp8-tlm/1',domain='SystemC/TLM, not board',profile=os.environ['TNCA_PROFILE'],
             layer=2,module='self_attn.q_proj',position=24,input_shape=list(x['input'].shape),
             seconds=time.monotonic()-start,vs_cpu_reference=compare(y,x['reference']),vs_cpu_dp_emulation=compare(y,x['dp']))
 a.out.mkdir(parents=True,exist_ok=True);save_file(dict(output=y),str(a.out/'output.safetensors'));(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
