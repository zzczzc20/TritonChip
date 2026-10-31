#!/usr/bin/env python3
"""Stage registered VL reason prompt tensors for layer-local TLM diagnosis."""
import argparse,json
from pathlib import Path
from safetensors.torch import save_file
import chat
PROMPT='Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? Answer with one name.'
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 args=chat.parser().parse_args(['--model','vl','--device','cpu','--threads','2','--bucket','256','--new-tokens','1']);app=chat.ChatApplication(args)
 inputs,text=chat.make_inputs(app.spec,app.processor,[chat.user_message(PROMPT)]);pos=chat.L.prefill_position_ids(app.model,inputs)
 a.out.mkdir(parents=True,exist_ok=True);save_file(dict(input_ids=inputs['input_ids'].contiguous(),position_ids_prefill=pos.contiguous()),str(a.out/'tensors.safetensors'))
 (a.out/'manifest.json').write_text(json.dumps(dict(domain='CPU input preparation',prompt=PROMPT,prompt_text=text,revision=app.spec.revision,tokens=int(inputs['input_ids'].shape[1])),indent=2)+'\n')
if __name__=='__main__':main()
