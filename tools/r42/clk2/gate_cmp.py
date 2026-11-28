#!/usr/bin/env python3
"""R42 issue: compare two `make core-run2` style logs program by program.
  python3 gate_cmp.py REF.log CAND.log [--identical]
Reports PASS counts, programs whose cycle count differs (ratio cand/ref) and the geometric mean ratio. With --identical the exit
status is 1 when any program differs or fails (the default-off identity gate)."""
import math, re, sys

pat = re.compile(r'^\s+(\S+) \(([^)]*)\)\s+(PASS|FAIL\S*)\s+cycles=(\d+)')

def load(p):
    d, fails, n = {}, [], 0
    for l in open(p):
        m = pat.match(l)
        if m:
            n += 1
            d[(m.group(1), m.group(2))] = int(m.group(4))
            if m.group(3) != 'PASS':
                fails.append((m.group(1), m.group(2)))
    ex = [l.strip() for l in open(p) if l.startswith('EXIT=')]
    return d, fails, n, ex

a, fa, na, ea = load(sys.argv[1]); b, fb, nb, eb = load(sys.argv[2])
common = sorted(set(a) & set(b))
diff = [(k, a[k], b[k]) for k in common if a[k] != b[k]]
print(f'ref: {na} runs, fail {len(fa)}, {ea};  cand: {nb} runs, fail {len(fb)}, {eb};  common {len(common)}')
for k, x, y in diff:
    print(f'  {k[0]:<28}{k[1]:<18}{x:>8}{y:>8}  {y / x:.3f}')
if common:
    g = math.exp(sum(math.log(b[k] / a[k]) for k in common) / len(common))
    print(f'differing programs {len(diff)}; geomean cand/ref cycles {g:.4f}')
if '--identical' in sys.argv:
    sys.exit(1 if diff or fa or fb or set(a) != set(b) else 0)
