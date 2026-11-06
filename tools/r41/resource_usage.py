#!/usr/bin/env python3
"""Per-kernel resource use from a compile audit (R41 G2).

  resource_usage.py REPORT.json [REPORT.json ...]

REPORT is tools/performance_harness/default_change_audit.py's report.json:
every launched kernel of every point, with the sealed manifest's resources
(whole program, all roles) and the compiler's decisions. Prints, per report,
the distribution of accumulators, barriers, descriptors and windows over all
kernels, the maximum per operator, and the role-SPMD context counts.
Nothing here is typed in; every number is read from the report.
"""
import json
import sys
from collections import Counter, defaultdict

KINDS = ('accumulators', 'barriers', 'descriptors')


def kernels(report):
    for pid, p in report['points'].items():
        if not isinstance(p, dict):
            continue
        for name, k in (p.get('kernels') or {}).items():
            yield pid, name, k


def summarize(path):
    r = json.load(open(path))
    pts = r['points']
    status = Counter(p.get('status') for p in pts.values() if isinstance(p, dict))
    print(f'## `{path}`\n')
    print(f"label `{r.get('label')}`, software tree `{r.get('software_tree')}`, "
          f"profile `{(r.get('harness_env') or {}).get('TNCA_PROFILE', r.get('profile'))}`; "
          f'points {len(pts)}, status {dict(status)}\n')
    dist = {k: Counter() for k in KINDS}
    per_op = defaultdict(lambda: Counter())
    spmd = Counter()
    worst = {k: (0, None) for k in KINDS}
    nk = 0
    for pid, name, k in kernels(r):
        nk += 1
        res = k['launch_structure']['resources']
        op = pid.split('/')[0]
        for kind in KINDS:
            v = int(res.get(kind) or 0)
            dist[kind][v] += 1
            per_op[op][kind] = max(per_op[op][kind], v)
            if v > worst[kind][0]:
                worst[kind] = (v, f'{pid} :: {name}')
        windows = len(res.get('shared_windows') or [])
        per_op[op]['shared windows'] = max(per_op[op]['shared windows'], windows)
        spmd[(k.get('decisions') or {}).get('rctx_spmd', 'none')] += 1
        per_op[op]['roles'] = max(per_op[op]['roles'],
                                  len(k['launch_structure']['entry_points']))
    print(f'{nk} launched kernels.\n')
    print('| Resource | Distribution over kernels (value: kernels) | Maximum | Where |')
    print('|---|---|---:|---|')
    for kind in KINDS:
        d = ', '.join(f'{v}: {n}' for v, n in sorted(dist[kind].items()))
        print(f'| {kind} | {d} | {worst[kind][0]} | `{worst[kind][1]}` |')
    print(f"\nrole-SPMD contexts per kernel (decision `rctx_spmd`): "
          f"{', '.join(f'{v}: {n}' for v, n in sorted(spmd.items(), key=str))}\n")
    print('| Operator | max accumulators | max barriers | max descriptors | max shared windows | max roles |')
    print('|---|---:|---:|---:|---:|---:|')
    for op in sorted(per_op):
        c = per_op[op]
        print(f"| {op} | {c['accumulators']} | {c['barriers']} | {c['descriptors']} | "
              f"{c['shared windows']} | {c['roles']} |")
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        summarize(p)
