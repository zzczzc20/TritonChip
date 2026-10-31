#!/usr/bin/env python3
"""Compare saved CPU or TLM reason-prompt logits without modifying them."""
import argparse,json
from pathlib import Path
from safetensors.torch import load_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 ref,cand=load_file(str(a.reference)),load_file(str(a.candidate));result={}
 for k in sorted(set(ref)&set(cand)):
  x,y=ref[k].float(),cand[k].float();d=(x-y).abs()
  result[k]=dict(rel_l2=float(d.norm()/x.norm().clamp_min(1e-30)),max_abs=float(d.max()),reference_argmax=int(x.argmax()),candidate_argmax=int(y.argmax()),reference_bob=float(x[32388]),candidate_bob=float(y[32388]),reference_we=float(x[1654]),candidate_we=float(y[1654]))
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
