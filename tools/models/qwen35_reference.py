#!/usr/bin/env python3
"""Official all-layer Qwen3.5-2B CPU reference; no TNCA output is read.

Reference-only numerical cross-validation before prospective freeze. Records
full vocabulary logits and every persistent cache tensor at each teacher-forced
step. BF16 checkpoint states retain their actual dtype; FP32 oracle widens the
same checkpoint values. No model dimensions, weights or layer count are changed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

REVISION='15852e8c16360a2fea060d615a32b45270f8a8fc'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def tokens(length):
    return [(i*104729+17)%248000+1 for i in range(length)]

def cache_tensors(cache):
    import torch
    out={}
    for i,layer in enumerate(cache.layers):
        for name in ('keys','values','conv_states','recurrent_states','cumulative_length'):
            value=getattr(layer,name,None)
            if isinstance(value,torch.Tensor):out[f'{i}.{name}']=value
            elif isinstance(value,(tuple,list,dict)):
                entries=value.items() if isinstance(value,dict) else enumerate(value)
                for j,tensor in entries:
                    if isinstance(tensor,torch.Tensor):out[f'{i}.{name}.{j}']=tensor
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dtype',choices=['bf16','fp32'],required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.cache_utils import DynamicCache
    import transformers,importlib
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sources=json.loads(args.sources.read_text())
    for name,entry in sources['files'].items():
        module=importlib.import_module(name.removeprefix('src/').removesuffix('.py').replace('/','.'))
        assert sha(module.__file__)==entry['sha256'],name
    config=Qwen3_5TextConfig(**json.loads((args.snapshot/'config.json').read_text())['text_config'])
    config._attn_implementation='eager'
    assert config.num_hidden_layers==24 and config.vocab_size==248320
    with torch.device('meta'):model=modeling.Qwen3_5ForCausalLM(config).eval()
    inventory=json.loads(args.inventory.read_text())
    state={}
    for shard in sorted({v['shard'] for v in inventory['selected_tensors'].values()}):
        assert sha(args.snapshot/shard)==inventory['shard_lfs'][shard]['sha256']
        with safe_open(args.snapshot/shard,framework='pt',device='cpu') as f:
            for key,entry in inventory['selected_tensors'].items():
                if entry['shard']==shard:state[key.replace('model.language_model.','model.',1)]=f.get_tensor(key)
    assert len(state)==320
    state['lm_head.weight']=state['model.embed_tokens.weight']
    model.load_state_dict(state,assign=True,strict=True)
    del state
    model.tie_weights()
    # Rotary frequencies are nonpersistent buffers, absent from checkpoint state.
    # Recreate the official module on CPU after meta parameter construction.
    model.model.rotary_emb=type(model.model.rotary_emb)(config=config)
    assert not any(buffer.is_meta for buffer in model.buffers())
    if args.dtype=='fp32':model.float()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    started=time.monotonic()
    report=dict(schema='tnca-qwen35-independent-reference/1',repository='Qwen/Qwen3.5-2B',revision=REVISION,
                dtype=args.dtype,layers=24,vocabulary=248320,torch=str(torch.__version__),transformers=transformers.__version__,
                source_revision=sources['revision'],source_sha256=sources['files'],weights_verified=True,workloads={})
    with torch.inference_mode():
        for name,prefill,decode in [('Q41-core',128,16),('Q41-tail',129,4)]:
            cache=DynamicCache(config=config)
            ids=tokens(prefill+decode)
            rows=[]
            for step in range(decode+1):
                sequence=ids[:prefill] if step==0 else [ids[prefill+step-1]]
                input_ids=torch.tensor([sequence],dtype=torch.int64)
                position=0 if step==0 else prefill+step-1
                output=model(input_ids=input_ids,position_ids=torch.arange(position,position+len(sequence))[None,:],
                             past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=output.past_key_values
                tensors={'logits':output.logits.detach().contiguous()}
                tensors.update({k:t.detach().clone().contiguous() for k,t in cache_tensors(cache).items()})
                assert sum('recurrent_states' in k for k in tensors)==18
                assert sum('conv_states' in k for k in tensors)==18
                assert sum(k.endswith(('.keys','.values')) for k in tensors)==12
                assert all(t.dtype==torch.float32 for k,t in tensors.items() if 'recurrent' in k)
                path=args.out/f'{name}-{step:02}.safetensors'
                save_file(tensors,str(path))
                rows.append(dict(step=step,file=path.name,sha256=sha(path),tensors={k:dict(shape=list(t.shape),dtype=str(t.dtype)) for k,t in tensors.items()}))
                print(name,args.dtype,step,round(time.monotonic()-started,2),flush=True)
            report['workloads'][name]=dict(prefill=prefill,decode=decode,input_ids=ids,steps=rows)
            del cache,output,tensors
            gc.collect()
    report.update(wall_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
