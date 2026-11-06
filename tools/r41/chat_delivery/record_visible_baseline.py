#!/usr/bin/env python3
"""Derive a labelled lower-bound wall metric from preserved board reports."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 v=ROOT/'evidence/r41/chat/run3_priority/vl_nohost_v5_script.json'
 q=ROOT/'evidence/r41/chat/run3_priority/q35_nohost_v6_four_prompts_timeout.json'
 vl=json.loads(v.read_text())['turns'];q35=json.loads(q.read_text())['prompts'];rows=[]
 def row(model,label,tokens,stats,warm,retr):
  warm=(warm or {}).get('seconds',0.0);retr=retr or 0.0;total=warm+stats['generation_seconds']+retr
  return dict(model=model,case=label,returned_token_ids=len(tokens),includes_end_markers=True,
      warmup_seconds=warm,prefill_seconds=stats['prefill_seconds'],decode_seconds=stats['decode_seconds'],
      decode_steps=stats['decode_steps'],output_retrieval_seconds=retr,measured_wall_sum_seconds=total,
      returned_token_ids_per_measured_wall_sum=len(tokens)/total if total else None,
      non_device_fraction=stats.get('decode_estimated_non_device_wall_fraction'))
 for i,x in enumerate(vl):
  r=x['report'];rows.append(row('Qwen3-VL-2B-FP8',f'script-turn-{i+1}',r['generated_tokens'],r['statistics'],r.get('decode_warmup'),r.get('output_retrieval_seconds')))
 for x in q35:
  t=x['turns'][0];e=t['device_evidence'];rows.append(row('Qwen3.5-4B-FP8',x['id'],t['tokens'],t['statistics'],e.get('decode_warmup'),e.get('output_retrieval_seconds')))
 out=dict(schema='r41-user-visible-baseline/1',domain='recorded run-2 board',rows=rows,
  method='sum recorded first-run warmup, prefill and decode generation wall, and output retrieval; token IDs include end markers',
  limitations=['Input preparation, tokenization, model loading and gaps between recorded spans were not timed; this is not an exact prompt-to-last-token wall clock.',
               'Warmup can contain device work and is not a pure compiler timer.',
               'Decode counts include work after EOS; recorded returned IDs include end markers.'],source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (v,q)})
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps([r for r in rows if (r['model'].startswith('Qwen3-VL') and r['case']=='script-turn-1') or (r['model'].startswith('Qwen3.5') and r['case']=='fact')],indent=2))
if __name__=='__main__':main()
