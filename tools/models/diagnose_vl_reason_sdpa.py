#!/usr/bin/env python3
"""Compare actual layer-2 SDPA operands for the reason prompt on CPU."""
import argparse,importlib.util,json
from pathlib import Path
import torch
import torch.nn.functional as F
import chat
PROMPT='Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? Answer with one name.'

def delta(a,b):
    x,y=a.float(),b.float(); diff=(x-y).abs()
    return dict(equal=bool(torch.equal(a,b)),max_abs=float(diff.max()),rel_l2=float((x-y).norm()/x.norm().clamp_min(1e-30)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--old-chat',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('legacy_reason_chat',a.old_chat);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--new-tokens','1']);app=chat.ChatApplication(args)
    chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee'); inputs,text=chat.make_inputs(app.spec,app.processor,[chat.user_message(PROMPT)])
    layer=app.model.model.language_model.layers[2].self_attn; traces={}; orig=F.scaled_dot_product_attention; active=[False]
    for name,cls in [('legacy_chunk8',old.Session),('grouped_chunk8',chat.Session)]:
        rows=[]
        def pre(*args): active[0]=True
        def post(*args): active[0]=False
        def sdpa(q,k,v,*args,**kw):
            out=orig(q,k,v,*args,**kw)
            if active[0]: rows.append(dict(q=q.detach().clone(),k=k.detach().clone(),v=v.detach().clone(),out=out.detach().clone(),mask=kw.get('attn_mask').detach().clone() if kw.get('attn_mask') is not None else None,is_causal=kw.get('is_causal'),scale=kw.get('scale')))
            return out
        h1=layer.register_forward_pre_hook(pre);h2=layer.register_forward_hook(post);F.scaled_dot_product_attention=sdpa
        try:
            with torch.no_grad(): s=cls(app.model,app.spec,'cpu',256,8); logits,calls,_=s.prefill_prompt(inputs,None)
        finally: F.scaled_dot_product_attention=orig;h1.remove();h2.remove();active[0]=False
        traces[name]=rows;print(json.dumps(dict(path=name,calls=len(rows),top_id=int(logits.argmax()))),flush=True)
    legacy=traces['legacy_chunk8'][3]; grouped=traces['grouped_chunk8'][3]
    comp={'query':delta(legacy['q'][:,:,0,:],grouped['q'][:,:,0,:]),'key_prefix':delta(legacy['k'][:,:,:25,:],grouped['k'][:,:,:25,:]),'value_prefix':delta(legacy['v'][:,:,:25,:],grouped['v'][:,:,:25,:]),'output':delta(legacy['out'][:,:,0,:],grouped['out'][:,:,0,:]),'old_shapes':{k:list(legacy[k].shape) for k in ('q','k','v','out')},'grouped_shapes':{k:list(grouped[k].shape) for k in ('q','k','v','out')},'old_causal':legacy['is_causal'],'grouped_causal':grouped['is_causal'],'old_scale':legacy['scale'],'grouped_scale':grouped['scale']}
    a.out.mkdir(parents=True,exist_ok=True);torch.save(traces,a.out/'captures.pt');(a.out/'result.json').write_text(json.dumps(dict(schema='r41-vl-reason-sdpa-diagnostic/1',domain='CPU',prompt=PROMPT,comparisons=comp),indent=2)+'\n');print(json.dumps(comp),flush=True)
if __name__=='__main__':main()
