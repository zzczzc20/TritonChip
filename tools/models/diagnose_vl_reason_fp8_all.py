#!/usr/bin/env python3
"""Compare grouped DP dots and exact W8A8 for seven real layer-2 projections."""
import argparse,importlib.util,json
from pathlib import Path
import torch
import chat
import fp8_blockwise_ref as ref
import fp8_block_reference as tref
PROMPT='Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? Answer with one name.'

def cmp(x,y):
 a,b=x.float(),y.float();d=(a-b).abs();return dict(equal=bool(torch.equal(x,y)),different=int((x!=y).sum()),elements=x.numel(),max_abs=float(d.max()),rel_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--old-chat',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('legacy_reason_chat',a.old_chat);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--new-tokens','1']);app=chat.ChatApplication(args);chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee')
 inputs,text=chat.make_inputs(app.spec,app.processor,[chat.user_message(PROMPT)]);layer=app.model.model.language_model.layers[2]
 modules={f'attention.{n}':getattr(layer.self_attn,n) for n in ('q_proj','k_proj','v_proj','o_proj')}; modules.update({f'mlp.{n}':getattr(layer.mlp,n) for n in ('gate_proj','up_proj','down_proj')});capture={k:[] for k in modules};handles=[]
 def hook(name):
  def f(mod,args,out):capture[name].append((args[0].detach().clone(),out.detach().clone()))
  return f
 for name,module in modules.items():handles.append(module.register_forward_hook(hook(name)))
 try:
  with torch.no_grad():s=old.Session(app.model,app.spec,'cpu',256,8);_,calls,_=s.prefill_prompt(inputs,None)
 finally:
  for h in handles:h.remove()
 if calls[3]['first']!=24:raise RuntimeError('unexpected tail')
 tref.set_dot_mode('mma.dp'); rows={}
 for name,module in modules.items():
  x,y=capture[name][3];w=module.weight.detach().cpu();sc=module.weight_scale_inv.detach().cpu();block=list(module.block_size)
  dp=tref.block_dynamic(x,w,sc,block,torch.bfloat16,ieee=True);r=ref.w8a8_block_matmul(x,w,sc,block,dot='exact',scale_div='ieee');exact=ref.exact_w8a8(x,w,sc,block)
  rows[name]=dict(input_shape=list(x.shape),weight_shape=list(w.shape),block_size=block,dp_vs_cpu_reference=cmp(dp,r),hook_vs_reference=cmp(y,r),dp_vs_bf16_exact=cmp(dp,exact.to(torch.bfloat16)))
  print(json.dumps(dict(module=name,result=rows[name])),flush=True)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(schema='r41-vl-reason-all-projection-dp/1',domain='CPU DP mathematical rederivation; not board/TLM',prompt=PROMPT,prompt_text=text,layer=2,position=24,ieee_activation_division=True,rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
