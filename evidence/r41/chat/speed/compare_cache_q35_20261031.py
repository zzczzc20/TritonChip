import json,hashlib,sys
from pathlib import Path
from safetensors.torch import load_file
import torch
root=Path.home()/'r41_chat_cache_dp_20261031';sys.path[:0]=[str(root),str(root/'python'),str(root/'compiler/python'),str(root/'tools/asm')]
from tnca import artifact
home=Path.home();a=home/'r41_cache_q35_lru';b=home/'r41_cache_q35_sqrt'
ra=json.load(open(a/'report.json'));rb=json.load(open(b/'report.json'))
ta=json.load(open(a/'profile/launches.json'));tb=json.load(open(b/'profile/launches.json'))
la=load_file(str(a/'logits.safetensors'));lb=load_file(str(b/'logits.safetensors'))
if set(la)!=set(lb):raise ValueError('different logits keys')
logits={k:dict(elements=la[k].numel(),bit_mismatches=int((la[k].contiguous().view(torch.int16)!=lb[k].contiguous().view(torch.int16)).sum())) for k in la}
rows=[]
for label,rep,profile in [('lru',ra,ta),('sqrt',rb,tb)]:
 s=[]
 for step in rep['steps']:
  d=step['transfers_inside_step'];s.append(dict(kind=step['kind'],launches=d.get('launches'),device_cycles=d.get('device_cycles'),host_read_calls=d.get('host_read_calls'),evictions=step['code_store']['evicted']))
 p=[]
 for index in sorted({e['step'] for e in profile['launches']}):
  entries=[e for e in profile['launches'] if e['step']==index];p.append(dict(step=index,launches=len(entries),load_marks=sum(e['code_load'] for e in entries),evictions=sum(e['evicted'] for e in entries)))
 cpath=next((home/('r41_cache_q35_'+label+'_tlm')).glob('model-*/counters.json'));c=json.load(open(cpath));instructions=sum(v for k,v in c.items() if k.startswith('core_') and k.endswith('_instructions') and isinstance(v,int))
 rows.append(dict(policy=label,status=rep['status'],generated_tokens=rep['generated_tokens'],cpu_fallback=rep.get('cpu_numerical_fallback'),steps=s,profile=p,total_instructions=instructions))
left,right=ta['launches'],tb['launches'];same=len(left)==len(right);text_mismatch=[];meta_mismatch=[]
if same:
 for i,(x,y) in enumerate(zip(left,right)):
  if (x['step'],x['kernel'],x['grid'])!=(y['step'],y['kernel'],y['grid']):meta_mismatch.append(i)
  try:
   c1=artifact.load(x['artifact'])[1]['text'];c2=artifact.load(y['artifact'])[1]['text']
   if hashlib.sha256(c1).digest()!=hashlib.sha256(c2).digest():text_mismatch.append(i)
  except Exception as e:raise RuntimeError(f'could not read artifact for launch {i}: {e}') from e
output=dict(schema='r41-cache-real-model-tlm-ab/1',domain='reduced-depth four-layer Qwen3.5, uncalibrated DP SystemC/TLM, not board',
           same_launch_count=same,code_text_mismatch_indices=text_mismatch,launch_metadata_mismatch_indices=meta_mismatch,
           logits=logits,records=rows,caveat='per-launch profile cycle/instruction counters null; aggregate report cycles and whole-process instructions only, no MAC normalization; load marks are path-residency indicators')
p=home/'r41_cache_q35_compare_20261031.json';p.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
