#!/usr/bin/env python3
"""R42 issue: arm-against-base cycle table of stand-in (rtl-3c) results.
  python3 mc_table.py BASE_DIR ARM_DIR [ARM_DIR ...]      (directories of per-point .jsonl records from iss_mc.sh, digest.jsonl beside them)
Per point: device cycles of the base and of each arm (arm/base), whether the output digest of every arm record equals the base's, geometric mean."""
import glob, json, math, os, sys


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))):
        if f.endswith('digest.jsonl'):
            continue
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if isinstance(r, dict) and r.get('status') == 'pass' and 'point' in r:
                pid = r['point']
                cyc = (r.get('time') or {}).get('device')
                out[pid] = cyc
    dig = {}
    p = os.path.join(d, 'digest.jsonl')
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            dig.setdefault(r['point'], set()).add(r['output_sha256'])
    return out, dig


base, bdig = load(sys.argv[1])
arms = [(os.path.basename(a.rstrip('/')), *load(a)) for a in sys.argv[2:]]
hdr = f"{'point':<48}{'base':>9}" + ''.join(f"{n[:14]:>16}" for n, _, _ in arms)
print(hdr)
logs = [[] for _ in arms]
for p in sorted(base):
    row = f"{p:<48}{base[p]:>9}"
    for i, (n, c, d) in enumerate(arms):
        if p in c and c[p] and base[p]:
            r = c[p] / base[p]
            logs[i].append(math.log(r))
            same = (d.get(p) == bdig.get(p)) and p in d
            row += f"{c[p]:>9} {r:>5.3f}{'=' if same else '!'}"
        else:
            row += f"{'-':>16}"
    print(row)
print(f"{'geomean arm/base (n)':<48}{'':>9}" + ''.join(f"{(math.exp(sum(l) / len(l)) if l else float('nan')):>10.3f} ({len(l):>2})" for l in logs))
print("'=' output digest identical to the base's, '!' differs or missing")
