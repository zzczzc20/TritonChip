#!/usr/bin/env python3
"""Compare reason-prompt prefill logits across CPU paths and a DP TLM run.

The legacy chat file is the immutable source used before grouped prefill tails.
The TLM execution uses the normal full-checkpoint model and original prompt.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file, save_file
import chat

PROMPT = 'Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? Answer with one name.'


def summary(logits, tokenizer):
    x = logits.detach().float().reshape(-1).cpu()
    vals, ids = torch.topk(x, 5)
    return dict(top5=[dict(id=int(i), text=tokenizer.decode([int(i)]), logit=float(v))
                      for v, i in zip(vals, ids)], top1_top2_gap=float(vals[0]-vals[1]))


def compare(base, got):
    x, y = base.float().reshape(-1), got.float().reshape(-1)
    return dict(rel_l2=float(torch.linalg.vector_norm(x-y) / torch.linalg.vector_norm(x).clamp_min(1e-30)),
                max_abs=float((x-y).abs().max()), argmax_equal=bool(x.argmax()==y.argmax()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode', choices=['cpu','tlm'], required=True)
    p.add_argument('--old-chat', type=Path, required=True)
    p.add_argument('--tree', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cpu-reference', type=Path)
    a=p.parse_args()
    if a.mode=='tlm':
        chat.L.use_tnca(a.tree)
    spec=importlib.util.spec_from_file_location('legacy_reason_chat',a.old_chat)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--new-tokens','1'])
    t0=time.monotonic(); app=chat.ChatApplication(args)
    inputs,text=chat.make_inputs(app.spec, app.processor, [chat.user_message(PROMPT)])
    tok=app.processor.tokenizer
    result=dict(schema='r41-vl-reason-logits-diagnostic/1',mode=a.mode,prompt=PROMPT,prompt_text=text,
                input_ids=inputs['input_ids'].tolist(),model=app.spec.name,revision=app.spec.revision,
                legacy_chat_sha256=hashlib.sha256(a.old_chat.read_bytes()).hexdigest(),
                load=app.load_report,versions=chat.L.versions(), paths={})
    tensors={}
    if a.mode=='cpu':
        chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee')
        modes=[('whole',old.Session,0),('legacy_chunk8',old.Session,8),('grouped_chunk8',chat.Session,8)]
    else:
        if not a.cpu_reference: raise ValueError('TLM mode requires --cpu-reference')
        chat.L.install_fp8_route('official',dot='exact',scale_div='ieee')
        modes=[('legacy_chunk8',old.Session,8)]
    for name, cls, chunk in modes:
        start=time.monotonic()
        session=cls(app.model,app.spec,'cpu' if a.mode=='cpu' else 'tnca',256,chunk)
        with torch.no_grad(): logits,calls,reused=session.prefill_prompt(inputs,None)
        value=logits.detach().cpu().contiguous().reshape(-1)
        tensors[name]=value; result['paths'][name]=dict(seconds=time.monotonic()-start,calls=calls,reused=reused,
                                                       logits=summary(value,tok))
        print(json.dumps(dict(event='path-complete',path=name,seconds=result['paths'][name]['seconds'],
                              logits=result['paths'][name]['logits'])),flush=True)
        del session
    if a.mode=='cpu':
        result['comparisons']={name:compare(tensors['whole'],tensors[name]) for name in ('legacy_chunk8','grouped_chunk8')}
    else:
        ref=load_file(str(a.cpu_reference))
        result['comparisons']={name:compare(ref[name],tensors['legacy_chunk8']) for name in ('whole','legacy_chunk8','grouped_chunk8')}
    a.out.mkdir(parents=True,exist_ok=True)
    save_file(tensors,str(a.out/'logits.safetensors'))
    result['elapsed_seconds']=time.monotonic()-t0
    (a.out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(event='complete',comparisons=result['comparisons'])),flush=True)

if __name__=='__main__': main()
