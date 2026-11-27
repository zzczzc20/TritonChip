#!/usr/bin/env python3
"""R42 issue: what a measured hardware/software effect may do to the 145-point board geometric mean (DERIVED from rtl ratios; not a board number).

    issue_predict.py ORIG_COMPARISON.json XL_COMPARISON.json BASE ARM [BASE ARM ...] [--name NAME]

ORIG_COMPARISON.json / XL_COMPARISON.json: the board comparisons of the round (rows: point, family, tier or none for xl). BASE and ARM: a record file (jsonl) or a
directory of per-point jsonl records (archbench or tools/r42/img_mc.py format; time.device = device cycles of the timed call); the ratio of a point is ARM/BASE.
Several BASE ARM pairs are merged (a point of a later pair replaces an earlier one). The ratios are extended to the board points by (family, dtype): the geometric
mean of the measured points of the group; unmeasured groups keep 1. Scenarios: `measured` (only board points that were measured themselves), `family` (every board
point of a measured group at the group's ratio), `family/2` (as family, the logarithm halved for tiers medium, large, hbm and xl: the larger tiers hold shares
that a pipeline-level effect does not touch; the central scenario of tools/r42/img_predict.py). The factor is exp(-mean(log ratio)) over the 145 points (>1: fewer cycles).
"""
import glob, json, math, os, sys

BIG = ('medium', 'large', 'hbm', 'xl')


def cycles(rec):
    t = rec.get('time') or {}
    c = t.get('device')
    if not c:
        s = rec.get('device_cycles_samples') or []
        c = s[0] if s else None
    return c


def load(p):
    out = {}
    files = sorted(glob.glob(os.path.join(p, '*.jsonl'))) if os.path.isdir(p) else [p]
    for f in files:
        if f.endswith('digest.jsonl'):
            continue
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            pid = r.get('point_id') or r.get('point')
            if isinstance(r, dict) and r.get('status') == 'pass' and pid and cycles(r):
                out[pid] = cycles(r)
    return out


def dtype_of(p):
    return 'fp8' if 'fp8' in p else ('fp32' if 'fp32' in p else 'fp16')


def main():
    a = sys.argv[1:]
    name = ''
    if '--name' in a:
        i = a.index('--name'); name = a[i + 1]; del a[i:i + 2]
    orig = json.load(open(a[0]))['rows']
    xl = json.load(open(a[1]))['rows']
    rows = [(r['point'], r['family'], r['tier']) for r in orig] + [(r['point'], r['family'], 'xl') for r in xl]
    ratios = {}
    for i in range(2, len(a), 2):
        b, c = load(a[i]), load(a[i + 1])
        for p in c:
            if p in b:
                ratios[p] = c[p] / b[p]
    meta = {p: (f, t) for p, f, t in rows}
    groups = {}
    for p, r in ratios.items():
        if p in meta:
            groups.setdefault((meta[p][0], dtype_of(p)), []).append(math.log(r))
        else:
            f = p.split('/')[0].split('.')[-1]
            groups.setdefault((f, dtype_of(p)), []).append(math.log(r))   # a point that is not a board row still informs its family
    g = {k: math.exp(sum(v) / len(v)) for k, v in groups.items()}
    print(f'=== {name}: {len(ratios)} points measured, {len(g)} (family, dtype) groups')
    out = {}
    for sc in ('measured', 'family', 'family/2'):
        logs, aff = [], 0
        for p, f, t in rows:
            r = 1.0
            if sc == 'measured':
                r = ratios.get(p, 1.0)
            else:
                r = g.get((f, dtype_of(p)), 1.0)
                if sc == 'family/2' and t in BIG:
                    r = math.sqrt(r)
            aff += r != 1.0
            logs.append(math.log(r))
        fac = math.exp(-sum(logs) / len(logs))
        out[sc] = fac
        print(f'  {sc:9s} {fac:.4f}x ({(fac - 1) * 100:+.2f} %) over {aff} affected of {len(rows)} board points')
    return out


if __name__ == '__main__':
    main()
