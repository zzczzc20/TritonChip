#!/usr/bin/env python3
"""Compare selected real VL LM-head logits on CPU and DP-profile TLM."""
import argparse,importlib.util,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
import chat
from diagnose_vl_reason_logits import PROMPT

def main():
 p=argparse.ArgumentParser();p.add_argument('--tree',required=True);p.add_argument('--old-chat',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if os.environ.get('TNCA_PROFILE')!='core-vrf-sync-v5-x8-g2c4-dp':raise ValueError('DP profile required')
 dev=chat.L.use_tnca(a.tree);sp=importlib.util.spec_from_file_location('legacy_reason_chat',a.old_chat);old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
 args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','2','--bucket','256','--new-tokens','1']);app=chat.ChatApplication(args);chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee')
 inputs,text=chat.make_inputs(app.spec,app.processor,[chat.user_message(PROMPT)]);norm=app.model.model.language_model.norm;rows=[];h=norm.register_forward_hook(lambda mod,ins,out: rows.append(out.detach().cpu().clone()));hidden={};logits={}
 try:
  for name,cls,chunk in [('whole',old.Session,0),('legacy_chunk8',old.Session,8),('grouped_chunk8',chat.Session,8)]:
   rows.clear()
   with torch.no_grad():s=cls(app.model,app.spec,'cpu',256,chunk);value,_,_=s.prefill_prompt(inputs,None)
   hidden[name]=rows[-1][:,-1,:].contiguous();logits[name]=value.detach().cpu().contiguous().reshape(-1)
 finally:h.remove()
 ids=sorted(set(int(i) for x in logits.values() for i in torch.topk(x.float(),5).indices));weight=app.model.get_output_embeddings().weight.detach().cpu()[ids].contiguous();fn=torch.compile(lambda x,w:F.linear(x,w),fullgraph=True,dynamic=False)
 from tnca.inductor import host_fallbacks
 outputs={};result={};device_weight=weight.to(dev);all_code=[]
 for name in hidden:
  start=time.monotonic()
  with torch.no_grad():y,code=chat.run_with_code_capture(lambda:fn(hidden[name].to(dev),device_weight));got=y.cpu().contiguous().reshape(-1)
  all_code.extend(code);cpu=logits[name][ids];diff=(got.float()-cpu.float()).abs();outputs[name]=got
  result[name]=dict(seconds=time.monotonic()-start,cpu_logits=[float(x) for x in cpu],tlm_logits=[float(x) for x in got],equal=bool(torch.equal(got,cpu)),max_abs=float(diff.max()),rel_l2=float(diff.norm()/cpu.float().norm().clamp_min(1e-30)),compiled_wrapper_fallbacks=host_fallbacks(all_code) if all_code else None)
  print(json.dumps(dict(event='path-complete',path=name,result=result[name])),flush=True)
 res=dict(schema='r41-vl-reason-lmhead-tlm/1',domain='selected-output standalone SystemC/TLM, CPU-captured hidden; not full TLM or board',profile=os.environ['TNCA_PROFILE'],prompt=PROMPT,prompt_text=text,token_ids=ids,token_text=[app.processor.tokenizer.decode([i]) for i in ids],paths=result)
 a.out.mkdir(parents=True,exist_ok=True);save_file(outputs,str(a.out/'outputs.safetensors'));(a.out/'result.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n');print(json.dumps(res),flush=True)
if __name__=='__main__':main()
