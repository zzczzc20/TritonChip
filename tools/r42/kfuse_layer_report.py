#!/usr/bin/env python3
"""kfuse_layer_report.py -- the registered gate on the xl whole-layer points (output of kfuse_layer.py).

  python kfuse_layer_report.py DIR [--out-txt F] [--out-json F]

Per (point, attention engine): for every projection engine, over the seeds, the worst of
  ratio     combined relative L2 of the layer outputs against the point's binary64 reference, lane / `seq`
  tol       max|got - ref| / max(max|ref|, 1) over the outputs (archbench.platforms.common.compare), against the point's tolerance
and the gate: tolerance holds on every seed AND ratio <= 1.5 on every seed. `seqrev` (the contract in decreasing k) is
the control: its ratio shows what 1 looks like for this metric (the layer re-quantises activations to E4M3 between
projections, so one flipped fp16 rounding upstream flips E4M3 codes downstream).
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

GATE = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--out-txt')
    ap.add_argument('--out-json')
    a = ap.parse_args()
    recs = defaultdict(dict)                 # (point, attn, seed) -> proj -> record
    for f in sorted(glob.glob(os.path.join(a.dir, '*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            recs[(r['point'], r['attn'], r['seed'])][r['proj']] = r
    out, rows = [], []
    groups = sorted({(p, at) for p, at, s in recs})
    for point, attn in groups:
        seeds = sorted(s for p, at, s in recs if p == point and at == attn)
        base = {s: recs[(point, attn, s)].get('seq') for s in seeds}
        if any(v is None for v in base.values()):
            continue
        out.append(f'\n{point}   attention engine: {attn}   seeds {seeds}   tolerance {base[seeds[0]]["tol"]}')
        out.append('seq: rel_l2 ' + ', '.join(f'{base[s]["rel_l2"]:.3g}' for s in seeds) + '; tol-metric ' +
                   ', '.join(f'{base[s]["max_rel_err"]:.3g}' for s in seeds))
        projs = sorted({pr for s in seeds for pr in recs[(point, attn, s)] if pr != 'seq'}, key=lambda t: (t != 'seqrev', t))
        out.append('| projection engine | ratio (max over seeds) | tol-metric (max over seeds) | outputs identical to seq | gate |')
        out.append('|---|---|---|---|---|')
        for pr in projs:
            rs = [recs[(point, attn, s)].get(pr) for s in seeds]
            if any(r is None for r in rs):
                continue
            ratios = [r['rel_l2'] / base[s]['rel_l2'] if base[s]['rel_l2'] > 0 else (1.0 if r['rel_l2'] == 0 else float('inf'))
                      for r, s in zip(rs, seeds)]
            tol = max(r['max_rel_err'] for r in rs)
            ok = all(r['ok'] for r in rs)
            gate = ok and max(ratios) <= GATE
            # the error statistics of every output tensor equal to all digits: the layer outputs are the same numbers
            same = sum(1 for r, s in zip(rs, seeds) if r['per_output'] == base[s]['per_output'] and r['rel_l2'] == base[s]['rel_l2'])
            rows.append(dict(point=point, attn=attn, proj=pr, ratio_max=max(ratios), ratio_min=min(ratios), tol_max=tol,
                             tol_ok=ok, gate=gate, tolerance=rs[0]['tol'], identical_seeds=same, n_seeds=len(seeds)))
            out.append(f'| {pr} | {max(ratios):.3f} (min {min(ratios):.3f}) | {tol:.3g} | {same} of {len(seeds)} seeds | {"PASS" if gate else "FAIL"}{"" if ok else " (tolerance)"} |')
    text = '\n'.join(out) + '\n'
    if a.out_txt:
        open(a.out_txt, 'w').write(text)
    if a.out_json:
        json.dump(rows, open(a.out_json, 'w'), indent=1)
    print(text)


if __name__ == '__main__':
    main()
