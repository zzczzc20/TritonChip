#!/usr/bin/env python3
"""Locate the first CPU-flash versus SVE prototype divergence during real prefill."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
import chat
from diagnose_vl_reason_logits import PROMPT,flash_approximation,summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--new-tokens','1']);app=chat.ChatApplication(args)
 chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee');inputs,text=chat.make_inputs(app.spec,app.processor,[chat.user_message(PROMPT)])
 n_layers=len(app.model.model.language_model.layers); original=F.scaled_dot_product_attention;rows=[];first={}; ctx=flash_approximation(sve_exp=True);ctx.__enter__(); prototype=F.scaled_dot_product_attention
 def wrapped(q,k,v,*args,**kw):
  expected=original(q,k,v,*args,**kw);got=prototype(q,k,v,*args,**kw);d=(got.float()-expected.float()).abs();i=len(rows)
  row=dict(call=i,chunk=i//n_layers,layer=i%n_layers,q_shape=list(q.shape),different=int((got!=expected).sum()),elements=got.numel(),max_abs=float(d.max()))
  rows.append(row)
  if row['different'] and not first:
   first.update(q=q.detach().cpu().contiguous(),k=k.detach().cpu().contiguous(),v=v.detach().cpu().contiguous(),expected=expected.detach().cpu().contiguous(),got=got.detach().cpu().contiguous())
   mask=kw.get('attn_mask')
   if mask is not None:first['mask']=mask.detach().cpu().contiguous()
   first['scale']=torch.tensor(kw.get('scale') or q.shape[-1]**-.5);first['is_causal']=torch.tensor(bool(kw.get('is_causal',False)))
  return got
 F.scaled_dot_product_attention=wrapped
 try:
  with torch.no_grad():s=chat.Session(app.model,app.spec,'cpu',256,8);logits,calls,_=s.prefill_prompt(inputs,None)
 finally:F.scaled_dot_product_attention=prototype;ctx.__exit__(None,None,None)
 a.out.mkdir(parents=True,exist_ok=True)
 if first:save_file(first,str(a.out/'first_mismatch.safetensors'))
 result=dict(schema='r41-vl-reason-sve-trace/1',domain='CPU diagnostic only; not board/TLM',prompt=PROMPT,prompt_text=text,
             num_layers=n_layers,calls=calls,attention_rows=rows,first_mismatch=next((r for r in rows if r['different']),None),logits=summary(logits,app.processor.tokenizer))
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(first_mismatch=result['first_mismatch'],total_calls=len(rows),logits=result['logits'])),flush=True)
if __name__=='__main__':main()
