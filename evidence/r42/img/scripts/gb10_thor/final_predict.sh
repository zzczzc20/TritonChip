#!/bin/bash
# final_predict.sh : composite arms from the measured ones and the 113-point and xl predictions
S=/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad
B=$S/img/byarm
python3 - <<PY
import glob, json, os, shutil
B='$B'
def pts(arm):
    out={}
    for f in sorted(glob.glob(f'{B}/{arm}/*.jsonl')):
        try: r=json.loads(open(f).readline())
        except Exception: continue
        out[r['point']]=f
    return out
def comp(name, layers):
    d=f'{B}/{name}'; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    seen={}
    for arm, sel in layers:                      # later layers override earlier ones
        for p,f in pts(arm).items():
            if sel(p): seen[p]=f
    for p,f in seen.items(): os.symlink(os.path.realpath(f), f'{d}/{p.replace("/","_")}.jsonl')
    print(name, len(seen), 'points')
fp32=lambda p: 'fp32' in p
fp16=lambda p: 'fp32' not in p
# VLF: LINE_HALF + MAC_F32 image: fp16 points from L (hardware equal for fp16 to LF within noise), fp32 from F, then every LF record
comp('VLF', [('L',fp16),('F',fp32),('LF',lambda p: True)])
# VLMF: the three items: LMF records; fp16 points from LM where LMF is missing, fp32 from LF + nothing else
comp('VLMF', [('LM',fp16),('LMF',lambda p: True)])
PY
cd /home/ubuntu/r42-wt/img
for a in L F LF VLF LMF VLMF LM M; do [ -d $B/$a ] || continue; echo; echo "######## $a"; python3 tools/r42/img_predict.py $S/img/ab_int2.json $B/A $a=$B/$a 2>&1 | grep -E "^===|groups|measured |family |family/2|models "; done
