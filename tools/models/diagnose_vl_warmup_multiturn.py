#!/usr/bin/env python3
"""Compare full-VL CPU multi-turn chat with and without dummy decode warmup."""
import argparse,json
from pathlib import Path
import chat

PROMPTS=('My name is Alice. Please remember it and reply with one short sentence.',
         'What is my name? Answer with one word.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--prefill-chunk','8','--new-tokens','12']);app=chat.ChatApplication(args)
 chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee');paths={}
 for warm in (False,True):
  app.session=chat.Session(app.model,app.spec,'cpu',256,8);app.reset_conversation()
  if warm:
   inputs,_=chat.make_inputs(app.spec,app.processor,[chat.user_message(PROMPTS[0])]);app.session.warmup_decode(inputs)
  turns=[]
  for prompt in PROMPTS:
   r=app.generate(prompt,keep_history=True,check=False);turns.append(dict(text=r['generated_text'],tokens=r['generated_tokens'],status=r['status']))
  paths['warmup' if warm else 'baseline']=turns
 same=all(a['tokens']==b['tokens'] and a['text']==b['text'] for a,b in zip(paths['baseline'],paths['warmup'],strict=True))
 result=dict(schema='r41-vl-cpu-warmup-multiturn/1',domain='full-checkpoint CPU diagnostic, not board/TLM',prompts=list(PROMPTS),paths=paths,equal=same)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
