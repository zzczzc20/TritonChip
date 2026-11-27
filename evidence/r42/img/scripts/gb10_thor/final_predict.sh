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
        if r.get('status')!='pass': continue          # a failed record never stands in for a measurement (it stays in the table)
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
comp('VL', [('L',lambda p: True),('LN',lambda p: True)])
comp('VF', [('F',lambda p: True)])
attn=lambda p: '/attention_' in p or p.startswith('op.attention')
# the binary32 dot is what moves the attention kernels (F: attention_prefill 0.910), the half-line rows move the GEMM-shaped ones
comp('VLF', [('L',fp16),('LN',fp16),('F',lambda p: fp32(p) or attn(p)),('LF',lambda p: True)])
# VLMF: the three items: LMF records; fp16 points from LM where LMF is missing, fp32 from LF + nothing else
comp('VLM', [('LM',lambda p: True),('LMN',lambda p: True)])
comp('VLMF', [('LM',fp16),('LMN',fp16),('LMF',lambda p: True),('LMFN',lambda p: True)])
PY
cd /home/ubuntu/r42-wt/img
for a in VL VF VLF VLM VLMF; do [ -d $B/$a ] || continue; echo; echo "######## $a"; python3 tools/r42/img_predict.py $S/img/ab_int2.json $B/A $a=$B/$a --alias resnet_block=conv2d 2>&1 | grep -E "^===|groups|measured |family |family/2|models "; done
echo; echo "######## sensitivity: the pointwise families included (the profile-name layout effect, +2 to +9 %, extended to their board points)"
python3 tools/r42/img_predict.py $S/img/ab_int2.json $B/A VLMF=$B/VLMF --alias resnet_block=conv2d --pointwise 2>&1 | grep -E "^===|measured |family |family/2|models "
echo; echo "######## xl tier (xl-cand2 board cycles as the reference)"
python3 tools/r42/img_predict_xl.py $S/img/xl_cand2.json $B/A VL=$B/VL VLF=$B/VLF VLMF=$B/VLMF 2>&1
