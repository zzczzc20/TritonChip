#!/usr/bin/env python3
"""Build evidence/r42/kfuse_speed/xl_inputs.json: the 32 registered xl points, their planning parts and FP8 projection shapes.

    python3 tools/r42/kfuse_speed_xl_inputs.py OUT.json

Reads archbench/suites/xl.json of origin/R42-xl (the registration, sha256 recorded) and archbench/xl.py's decoder shape
(hidden 2048, MLP 6144, 16 query and 8 KV heads of 128, 28 layers). Per point: its estimate parts (bound cycles / eta, the
planning estimate of archbench/xl_estimate.py: a plan, never a result), and for the FP8 projection part the list of Hub-kernel
launches (M, N, K).
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

H, INTER, NQ, NKV, HD, VOCAB = 2048, 6144, 16, 8, 128, 151936
LAYER = [(NQ * HD, H), (NKV * HD, H), (NKV * HD, H), (H, NQ * HD), (INTER, H), (INTER, H), (H, INTER)]   # (N, K)


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'origin/R42-xl']).decode().strip()
    raw = subprocess.check_output(['git', 'show', 'origin/R42-xl:archbench/suites/xl.json'])
    d = json.loads(raw)
    out = dict(source='origin/R42-xl:archbench/suites/xl.json', commit=commit, sha256=hashlib.sha256(raw).hexdigest(),
               suite_sha256=d['sha256'], points=[])
    for p in d['points']:
        pid, family = p['id'], p['family']
        parts = [dict(part=q['part'], cycles=q['bound_cycles'] / q['eta'], bound_cycles=q['bound_cycles'], eta=q['eta'])
                 for q in p['estimate']['parts']]
        row = dict(id=pid, family=family, parts=parts, fp8_shapes=[], m_class=16)
        if family == 'fp8_linear':
            m, k, n = (int(re.search(rf'-{c}(\d+)', pid).group(1)) for c in 'mkn')
            row['fp8_shapes'] = [(m, n, k)]
            row['m_class'] = 1 if m == 1 else 16
        elif family in ('fp8_layer_decode', 'fp8_layer_prefill', 'fp8_decoder_decode', 'fp8_decoder_prefill'):
            layers = int(re.search(r'layers(\d+)', pid).group(1))
            if family.endswith('_decode'):
                m = 1
            else:
                m = int(re.search(r'-s(\d+)', pid).group(1))
            row['fp8_shapes'] = [(m, n, k) for _ in range(layers) for (n, k) in LAYER]
            row['m_class'] = 1 if m == 1 else 16
        out['points'].append(row)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(len(out['points']), 'points,', sum(1 for r in out['points'] if r['fp8_shapes']), 'with FP8 projections')


if __name__ == '__main__':
    main()
