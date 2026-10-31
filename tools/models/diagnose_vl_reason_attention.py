#!/usr/bin/env python3
"""Compare layer-2 CPU attention masks and rotary inputs at the reason prompt's short tail."""
import argparse,importlib.util,json,time
from pathlib import Path
import torch
import chat
PROMPT='Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? Answer with one name.'

def grab(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,(tuple,list)): return [grab(y) for y in x]
    return None

def main():
    p=argparse.ArgumentParser(); p.add_argument('--old-chat',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    spec=importlib.util.spec_from_file_location('legacy_reason_chat',a.old_chat); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--new-tokens','1'])
    app=chat.ChatApplication(args); chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee')
    inputs,text=chat.make_inputs(app.spec,app.processor,[chat.user_message(PROMPT)])
    layer=app.model.model.language_model.layers[2]; traces={}; paths={}
    for name,cls in [('legacy_chunk8',old.Session),('grouped_chunk8',chat.Session)]:
        rows=[]
        def hook(module,args,kw):
            hidden=kw.get('hidden_states',args[0] if args else None); cache=kw.get('past_key_values')
            seq=int(hidden.shape[1]); pos=sum(r['seq'] for r in rows)
            item=dict(first=pos,seq=seq,attention_mask=grab(kw.get('attention_mask')),position_embeddings=grab(kw.get('position_embeddings')))
            if cache is not None:
                item['keys_before']=grab(cache.layers[2].keys[:,:,:30]); item['values_before']=grab(cache.layers[2].values[:,:,:30])
            rows.append(item)
        handle=layer.self_attn.register_forward_pre_hook(hook,with_kwargs=True)
        try:
            with torch.no_grad():
                s=cls(app.model,app.spec,'cpu',256,8); logits,calls,_=s.prefill_prompt(inputs,None)
        finally: handle.remove()
        traces[name]=rows; paths[name]=dict(top_id=int(logits.argmax()),calls=calls)
        print(json.dumps(dict(event='path-complete',path=name,tail=[{k:v for k,v in x.items() if k in ('first','seq')} for x in rows])),flush=True)
    a.out.mkdir(parents=True,exist_ok=True); torch.save(traces,a.out/'captures.pt')
    tail_legacy=[r for r in traces['legacy_chunk8'] if r['first']>=24]; grouped=next(r for r in traces['grouped_chunk8'] if r['first']==24)
    comp=[]
    for j,row in enumerate(tail_legacy):
        info=dict(position=24+j)
        for name in ('attention_mask','position_embeddings'):
            oldv=row[name]; newv=grouped[name]
            if name=='attention_mask' and torch.is_tensor(oldv) and torch.is_tensor(newv):
                x=oldv[...,0,:]; y=newv[...,j,:]; info[name]=dict(equal=bool(torch.equal(x,y)),shape_old=list(x.shape),shape_grouped=list(y.shape),unequal=int((x!=y).sum()),old_zero=int((x==0).sum()),grouped_zero=int((y==0).sum()))
            elif name=='position_embeddings' and isinstance(oldv,list) and isinstance(newv,list):
                info[name]=[dict(equal=bool(torch.equal(x[...,0,:],y[...,j,:])),max_abs=float((x[...,0,:].float()-y[...,j,:].float()).abs().max())) for x,y in zip(oldv,newv)]
        comp.append(info)
    result=dict(schema='r41-vl-reason-attention-diagnostic/1',domain='CPU W8A8 reference; not board/TLM',prompt=PROMPT,prompt_text=text,paths=paths,comparisons=comp)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(dict(event='complete',comparisons=comp)),flush=True)
if __name__=='__main__': main()
