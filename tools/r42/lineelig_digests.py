#!/usr/bin/env python3
"""Compare the output digests (lineelig_mc.py, digest.jsonl: sha256 of every compared output tensor) of arms of an rtl-3c screen.

    lineelig_digests.py BASE_DIR ARM_DIR [ARM_DIR ...]

For every point of BASE_DIR and every arm that has it: identical / DIFFERENT, bit for bit over all compared outputs (the cold call and the timed call)."""
import json, os, sys, collections


def load(d):
    out = collections.defaultdict(list)
    p = os.path.join(d, 'digest.jsonl')
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            out[r['point']].append(r['output_sha256'])
    return out


base = load(sys.argv[1])
for arm in sys.argv[2:]:
    a = load(arm)
    print(f'{os.path.basename(arm.rstrip("/"))} against {os.path.basename(sys.argv[1].rstrip("/"))}:')
    for p in sorted(base):
        if p in a:
            same = base[p] == a[p]
            print(f'  {p:56s} {"identical" if same else "DIFFERENT"}  ({len(a[p])} compared outputs; {base[p][0][:10]} / {a[p][0][:10]})')
