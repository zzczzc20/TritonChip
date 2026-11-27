#!/usr/bin/env python3
"""R42 issue: per-point cycle ratios between two single-core RTL record files (archbench.run --counters), reference first.
  python3 cnt_cmp.py REF.jsonl CAND.jsonl [NAME_REF NAME_CAND]
Prints per point cycles of both, ratio ref/cand (>1 : candidate faster) and the geometric mean (tiny tier: rtl-1c)."""
import json, math, sys

def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("status") == "pass" and r.get("counters"):
            d[r["point_id"]] = r["counters"]
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
na = sys.argv[3] if len(sys.argv) > 3 else "ref"
nb = sys.argv[4] if len(sys.argv) > 4 else "cand"
print(f"{'point':<48}{na:>10}{nb:>10}{'ref/cand':>10}  hazard_ref hazard_cand vrf_ref vrf_cand")
ls = []
for p in sorted(a):
    if p in b:
        x, y = a[p]["perf_cycles"], b[p]["perf_cycles"]
        ls.append(math.log(x / y))
        print(f"{p:<48}{x:>10}{y:>10}{x / y:>10.3f}  {a[p]['perf_x_hazard']:>9} {b[p]['perf_x_hazard']:>11} {a[p]['perf_x_vrf']:>7} {b[p]['perf_x_vrf']:>8}")
print(f"{'GEOMEAN ref/cand over %d points' % len(ls):<48}{'':>20}{math.exp(sum(ls) / len(ls)):>10.3f}")
