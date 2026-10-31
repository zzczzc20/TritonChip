#!/usr/bin/env python3
"""Compare reason-prompt prefill logits across CPU paths and a DP TLM run.

The legacy chat file is the immutable source used before grouped prefill tails.
The TLM execution uses the normal full-checkpoint model and original prompt.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import types
import torch
from safetensors.torch import load_file, save_file
import chat

PROMPT = 'Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? Answer with one name.'


@contextlib.contextmanager
def flash_approximation(sve_exp=False):
    """CPU experiment: round unnormalized attention weights before V matmul."""
    import torch.nn.functional as functional
    original=functional.scaled_dot_product_attention
    def attention(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
        if dropout_p: raise ValueError('diagnostic supports inference only')
        if enable_gqa and q.shape[-3]!=k.shape[-3]:
            repeats=q.shape[-3]//k.shape[-3]; k=k.repeat_interleave(repeats,-3); v=v.repeat_interleave(repeats,-3)
        score=torch.matmul(q.float(),k.float().transpose(-2,-1))*(scale if scale is not None else 1/math.sqrt(q.shape[-1]))
        if is_causal:
            allowed=torch.ones(q.shape[-2],k.shape[-2],dtype=torch.bool,device=q.device).tril()
            score=score.masked_fill(~allowed,float('-inf'))
        if attn_mask is not None:
            score=score.masked_fill(~attn_mask,float('-inf')) if attn_mask.dtype==torch.bool else score+attn_mask.float()
        maximum=score.amax(-1,keepdim=True); finite=torch.isfinite(maximum)
        exponent=score-torch.where(finite,maximum,torch.zeros_like(maximum))
        if sve_exp:
            # CPU-only SVE fexp_u20 rederivation; float64 expressions emulate
            # the single-rounding FMA used by the native implementation.
            lower=-float.fromhex('0x1.5d619ap+6'); shift=float.fromhex('0x1.803f8p17')
            inv=float.fromhex('0x1.715476p+0'); ln2=float.fromhex('0x1.62e43p-1')
            bounded=exponent.clamp(min=lower,max=0)
            rounded=(bounded.double()*inv+shift).float()-shift
            remainder=(bounded.double()-rounded.double()*ln2).float()
            factor=torch.exp2(rounded.double()).float()
            prob=(factor.double()+factor.double()*remainder.double()).float()
            prob=torch.where(exponent<lower,0,prob)
        else:
            prob=torch.exp(exponent)
        denominator=prob.sum(-1,keepdim=True)
        numerator=torch.matmul(prob.to(q.dtype).float(),v.float())
        return torch.where(denominator>0,numerator/denominator.clamp_min(1e-30),0).to(q.dtype)
    functional.scaled_dot_product_attention=attention
    try: yield
    finally: functional.scaled_dot_product_attention=original


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
    p.add_argument('--tlm-path', choices=['whole','legacy_chunk8','grouped_chunk8'], default='legacy_chunk8')
    p.add_argument('--cpu-sdpa', choices=['default','math','flash-approx','flash-sve-prototype'], default='default')
    p.add_argument('--cpu-fused-norm', action='store_true')
    a=p.parse_args()
    if a.mode=='tlm':
        chat.L.use_tnca(a.tree)
    spec=importlib.util.spec_from_file_location('legacy_reason_chat',a.old_chat)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--new-tokens','1'])
    t0=time.monotonic(); app=chat.ChatApplication(args)
    if a.cpu_fused_norm:
        if a.mode!='cpu': raise ValueError('fused-norm approximation is a CPU diagnostic only')
        norm_type=type(app.model.model.language_model.layers[0].input_layernorm)
        def fused_norm(module,x):
            val=x.float(); scale=torch.rsqrt(val.pow(2).mean(-1,keepdim=True)+module.variance_epsilon)
            return (module.weight.float()*(val*scale)).to(x.dtype)
        for module in app.model.modules():
            if isinstance(module,norm_type):module.forward=types.MethodType(fused_norm,module)
    inputs,text=chat.make_inputs(app.spec, app.processor, [chat.user_message(PROMPT)])
    tok=app.processor.tokenizer
    if a.mode=='tlm' and a.cpu_sdpa!='default':
        raise ValueError('CPU SDPA override is only a CPU diagnostic')
    result=dict(schema='r41-vl-reason-logits-diagnostic/1',mode=a.mode,cpu_sdpa=a.cpu_sdpa,cpu_fused_norm=a.cpu_fused_norm,prompt=PROMPT,prompt_text=text,
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
        cls=chat.Session if a.tlm_path=='grouped_chunk8' else old.Session
        modes=[(a.tlm_path,cls,0 if a.tlm_path=='whole' else 8)]
    for name, cls, chunk in modes:
        start=time.monotonic()
        session=cls(app.model,app.spec,'cpu' if a.mode=='cpu' else 'tnca',256,chunk)
        if a.cpu_sdpa=='math':
            from torch.nn.attention import sdpa_kernel, SDPBackend
            attention_context=sdpa_kernel(SDPBackend.MATH)
        elif a.cpu_sdpa=='flash-approx':
            attention_context=flash_approximation()
        elif a.cpu_sdpa=='flash-sve-prototype':
            attention_context=flash_approximation(sve_exp=True)
        else:
            attention_context=contextlib.nullcontext()
        with torch.no_grad(), attention_context: logits,calls,reused=session.prefill_prompt(inputs,None)
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
        result['comparisons']={name:compare(ref[name],tensors[a.tlm_path]) for name in ('whole','legacy_chunk8','grouped_chunk8')}
    a.out.mkdir(parents=True,exist_ok=True)
    save_file(tensors,str(a.out/'logits.safetensors'))
    result['elapsed_seconds']=time.monotonic()-t0
    (a.out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(event='complete',comparisons=result['comparisons'])),flush=True)

if __name__=='__main__': main()
