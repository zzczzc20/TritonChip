#!/usr/bin/env python3
"""Compare DP-group and exact FP8 dot arithmetic on a real reason-prompt activation."""
import argparse,importlib.util,json,time
from pathlib import Path
import torch
import chat
import fp8_blockwise_ref as ref
import fp8_block_reference as tref
PROMPT='Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? Answer with one name.'

def cmp(x,y):
 a,b=x.float(),y.float(); d=(a-b).abs()
 return dict(equal=bool(torch.equal(x,y)),different=int((x!=y).sum()),elements=x.numel(),max_abs=float(d.max()),rel_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--old-chat',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('legacy_reason_chat',a.old_chat);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--new-tokens','1']);app=chat.ChatApplication(args)
 chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee'); inputs,text=chat.make_inputs(app.spec,app.processor,[chat.user_message(PROMPT)])
 module=app.model.model.language_model.layers[2].self_attn.q_proj; rows=[]
 def hook(mod,args,out):rows.append((args[0].detach().clone(),out.detach().clone()))
 h=module.register_forward_hook(hook)
 try:
  with torch.no_grad():s=old.Session(app.model,app.spec,'cpu',256,8);_,calls,_=s.prefill_prompt(inputs,None)
 finally:h.remove()
 x,y=rows[3];weight=module.weight.detach().cpu();scale=module.weight_scale_inv.detach().cpu();block=list(module.block_size)
 if calls[3]['first']!=24 or x.shape[1]!=1:raise RuntimeError('unexpected tail activation')
 tref.set_dot_mode('mma.dp'); start=time.monotonic(); dp=tref.block_dynamic(x,weight,scale,block,torch.bfloat16,ieee=True)
 exact_ref=ref.w8a8_block_matmul(x,weight,scale,block,dot='exact',scale_div='ieee'); exact=ref.exact_w8a8(x,weight,scale,block)
 result=dict(schema='r41-vl-reason-fp8-dot-diagnostic/1',domain='CPU mathematical emulation; not TLM or board',
             prompt=PROMPT,prompt_text=text,layer=2,module='self_attn.q_proj',position=24,input_shape=list(x.shape),
             weight_shape=list(weight.shape),block_size=block,dp_dot='ISA sec 9C.4 grouped four FP8 products',
             activation_division='IEEE in both paths, isolates grouped dot order',seconds=time.monotonic()-start,
             hook_vs_reference=cmp(y,exact_ref),dp_vs_reference=cmp(dp,exact_ref),
             dp_vs_bf16_exact=cmp(dp,exact.to(torch.bfloat16)),reference_vs_bf16_exact=cmp(exact_ref,exact.to(torch.bfloat16)))
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
