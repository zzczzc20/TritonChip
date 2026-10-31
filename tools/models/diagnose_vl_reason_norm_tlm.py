#!/usr/bin/env python3
"""Compare layer-2 input normalization on the full reason-prompt TLM input."""
import argparse,copy,json,os,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
import chat

def main():
 p=argparse.ArgumentParser();p.add_argument('--tree',required=True);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if os.environ.get('TNCA_PROFILE')!='core-vrf-sync-v5-x8-g2c4-dp':raise ValueError('DP profile required')
 dev=chat.L.use_tnca(a.tree);spec=chat.spec_of('vl');chat.L.set_threads(2);model,load=chat.L.load_model(spec,'fp8');chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee')
 rt=load_file(str(a.teacher/'tensors.safetensors'));layer=model.model.language_model.layers[2]; captured={}
 def pre(module,args,kw):
  h=kw.get('hidden_states',args[0] if args else None)
  if h is not None:captured['hidden']=h.detach().clone()
 handle=layer.register_forward_pre_hook(pre,with_kwargs=True);cache=chat.L.make_cache(model,256,torch.device('cpu'))
 try:
  with torch.no_grad():model(input_ids=rt['input_ids'],position_ids=rt['position_ids_prefill'],past_key_values=cache,use_cache=True,logits_to_keep=1)
 finally:handle.remove()
 h=captured.get('hidden')
 if h is None:raise ValueError('no layer input captured')
 with torch.no_grad():reference=layer.input_layernorm(h).detach().cpu().contiguous()
 module=copy.deepcopy(layer.input_layernorm).to(dev);fn=torch.compile(module,fullgraph=True,dynamic=False);t0=time.monotonic()
 with torch.no_grad():out,code=chat.run_with_code_capture(lambda:fn(h.to(dev)));value=out.cpu().contiguous()
 from tnca.inductor import host_fallbacks
 diff=(value.float()-reference.float()).abs();res=dict(schema='r41-vl-reason-norm-tlm/1',domain='standalone DP SystemC/TLM, not board',layer=2,shape=list(h.shape),seconds=time.monotonic()-t0,wrapper_fallbacks=host_fallbacks(code) if code else None,equal=bool(torch.equal(value,reference)),different=int((value!=reference).sum()),elements=value.numel(),max_abs=float(diff.max()),rel_l2=float(diff.norm()/reference.float().norm().clamp_min(1e-30)),load=load)
 a.out.mkdir(parents=True,exist_ok=True);save_file(dict(input=h.contiguous(),reference=reference,output=value),str(a.out/'tensors.safetensors'));(a.out/'result.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)
if __name__=='__main__':main()
