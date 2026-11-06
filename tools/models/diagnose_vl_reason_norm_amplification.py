#!/usr/bin/env python3
"""Measure FP8 V projection sensitivity to captured CPU versus TLM norm outputs."""
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
import chat

def main():
 p=argparse.ArgumentParser();p.add_argument('--norm',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','2','--new-tokens','1']);app=chat.ChatApplication(args);chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee')
 t=load_file(str(a.norm/'tensors.safetensors'));proj=app.model.model.language_model.layers[2].self_attn.v_proj
 with torch.no_grad():baseline=proj(t['reference']).detach().cpu().contiguous();candidate=proj(t['output']).detach().cpu().contiguous()
 d=(baseline.float()-candidate.float()).abs();res=dict(schema='r41-vl-reason-norm-amplification/1',domain='CPU W8A8 using CPU and TLM normalization tensors; not board',layer=2,module='self_attn.v_proj',shape=list(baseline.shape),different=int((baseline!=candidate).sum()),elements=baseline.numel(),max_abs=float(d.max()),rel_l2=float(d.norm()/baseline.float().norm().clamp_min(1e-30)))
 a.out.mkdir(parents=True,exist_ok=True);save_file(dict(reference=baseline,candidate=candidate),str(a.out/'outputs.safetensors'));(a.out/'result.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)
if __name__=='__main__':main()
