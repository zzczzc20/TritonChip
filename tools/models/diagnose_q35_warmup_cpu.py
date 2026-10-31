#!/usr/bin/env python3
"""Check full-Qwen3.5 cache reset after a dummy decode warmup on CPU."""
import argparse,json
from pathlib import Path
import torch
import chat

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 args=chat.parser().parse_args(['--model','q35','--device','cpu','--threads','4','--bucket','64','--new-tokens','1']);app=chat.ChatApplication(args)
 chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee');inputs,text=chat.make_inputs(app.spec,app.processor,[chat.user_message('Say hi.')]);out=[]
 for warm in (False,True):
  session=chat.Session(app.model,app.spec,'cpu',64,0)
  with torch.no_grad():
   if warm:session.warmup_decode(inputs)
   logits,calls,_=session.prefill_prompt(inputs,None)
  state,meta=chat.L.cache_state(session.cache);out.append((logits.detach().clone(),state,meta,calls))
 x,y=out[0][0],out[1][0];keys=set(out[0][1])|set(out[1][1]);state={}
 for k in sorted(keys):
  if k not in out[0][1] or k not in out[1][1]:state[k]={'equal':False,'missing':True};continue
  u,v=out[0][1][k],out[1][1][k];state[k]=dict(equal=bool(torch.equal(u,v)),max_abs=float((u.float()-v.float()).abs().max()))
 result=dict(schema='r41-q35-cpu-warmup-reset/1',domain='full-checkpoint CPU diagnostic, not board or TLM',prompt='Say hi.',prompt_text=text,logits_equal=bool(torch.equal(x,y)),logits_max_abs=float((x.float()-y.float()).abs().max()),argmax=[int(x.argmax()),int(y.argmax())],states_equal=all(v.get('equal') for v in state.values()),state=state,cache_metadata=[out[0][2],out[1][2]])
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('logits_equal','logits_max_abs','argmax','states_equal')}),flush=True)
if __name__=='__main__':main()
