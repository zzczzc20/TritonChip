#!/usr/bin/env python3
"""Compare full-VL CPU image chat with and without an earlier decode warmup."""
import argparse,json
from pathlib import Path
import chat

FIRST='What is the capital of France? Answer with one word.'
IMAGE_PROMPT='Which shapes are in this image? Answer with a short list.'

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','4','--bucket','256','--prefill-chunk','8','--new-tokens','12']);app=chat.ChatApplication(args);chat.L.install_fp8_route('reference',dot='exact',scale_div='ieee');image=chat.open_images([a.image]);paths={}
 for warm in (False,True):
  app.session=chat.Session(app.model,app.spec,'cpu',256,8);app.reset_conversation()
  if warm:
   inputs,_=chat.make_inputs(app.spec,app.processor,[chat.user_message(FIRST)]);app.session.warmup_decode(inputs)
  app.generate(FIRST,keep_history=False,check=False);r=app.generate(IMAGE_PROMPT,images=image,keep_history=False,check=False)
  paths['warmup' if warm else 'baseline']=dict(text=r['generated_text'],tokens=r['generated_tokens'],status=r['status'],image_count=r['images'])
 same=paths['baseline']['tokens']==paths['warmup']['tokens'] and paths['baseline']['text']==paths['warmup']['text']
 result=dict(schema='r41-vl-cpu-warmup-image/1',domain='full-checkpoint CPU diagnostic, not board/TLM',image=str(a.image),prompt=IMAGE_PROMPT,paths=paths,equal=same)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
