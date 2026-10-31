#!/usr/bin/env python3
"""Trace the real reason prompt on CPU to locate the chunk-tail divergence."""
import argparse, importlib.util, json, time
from pathlib import Path
import torch
import chat

PROMPT='Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? Answer with one name.'

def measure(a,b):
    af,bf=a.float(),b.float(); delta=(af-bf).abs(); dims=tuple(i for i in range(af.ndim) if i != 1)
    per=delta.amax(dim=dims); ix=torch.nonzero(per != 0).flatten()
    return dict(shape=list(af.shape), first_different_position=int(ix[0]) if ix.numel() else None,
                max_abs=float(delta.max()), relative_l2=float((af-bf).norm()/af.norm().clamp_min(1e-30)),
                last_position_max_abs=float(per[-1]), last_position_relative_l2=float(
                    (af[:, -1]-bf[:, -1]).norm()/af[:, -1].norm().clamp_min(1e-30)))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--old-chat',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    spec=importlib.util.spec_from_file_location('legacy_reason_chat',a.old_chat); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--new-tokens','1'])
    app=chat.ChatApplication(args); chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee')
    inputs,text=chat.make_inputs(app.spec,app.processor,[chat.user_message(PROMPT)])
    layers=app.model.model.language_model.layers
    targets={'embedding':app.model.model.language_model.embed_tokens,'layer0_input_norm':layers[0].input_layernorm,
             'layer0_attention':layers[0].self_attn,'layer0_post_attention_norm':layers[0].post_attention_layernorm,
             'layer0_mlp':layers[0].mlp,'layer0':layers[0],'layer14':layers[14],'layer27':layers[27]}
    traces={}; results={}
    for name,cls in [('legacy_chunk8',old.Session),('grouped_chunk8',chat.Session)]:
        rows={k:[] for k in targets}; handles=[]
        def hook(k):
            def save(module,args,out):
                x=out[0] if isinstance(out,(tuple,list)) else out
                if torch.is_tensor(x): rows[k].append(x.detach().cpu().clone())
            return save
        for k,m in targets.items(): handles.append(m.register_forward_hook(hook(k)))
        start=time.monotonic()
        try:
            with torch.no_grad():
                session=cls(app.model,app.spec,'cpu',256,8); logits,calls,_=session.prefill_prompt(inputs,None)
        finally:
            for h in handles: h.remove()
        traces[name]={k:torch.cat(v,dim=1) for k,v in rows.items() if v and all(t.ndim==3 for t in v)}
        results[name]=dict(seconds=time.monotonic()-start,calls=calls,top_id=int(logits.argmax()),
                           captured_shapes={k:[list(x.shape) for x in v] for k,v in rows.items()})
        print(json.dumps(dict(event='path-complete',path=name,result=results[name])),flush=True)
    comp={k:measure(traces['grouped_chunk8'][k],traces['legacy_chunk8'][k]) for k in traces['grouped_chunk8'] if k in traces['legacy_chunk8'] and traces['grouped_chunk8'][k].shape==traces['legacy_chunk8'][k].shape}
    out=dict(schema='r41-vl-reason-layer-diagnostic/1',domain='CPU W8A8 reference, not board or TLM',prompt=PROMPT,
             prompt_text=text,input_ids=inputs['input_ids'].tolist(),paths=results,comparisons=comp,versions=chat.L.versions())
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(event='complete',comparisons=comp)),flush=True)

if __name__=='__main__': main()
