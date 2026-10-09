#!/usr/bin/env python3
"""R39 A/B ablation over ArchBench records.

    python3 tools/r39/ablate.py A.jsonl B.jsonl [--tier tiny small] [--gate]
        [--min-speedup 1.15] [--max-regress 1.02] [--json out.json]

A is the configuration without the change (the switch off), B with it. Both
are ArchBench JSONL files (archbench.run --out), each holding ONE
configuration: a file whose records disagree on the platform key (profile,
backend, opt level, Inductor configuration, knobs, device build) is refused,
because an ablation that mixed two configurations on one side would measure
nothing in particular.

For the points measured and passing on both sides it reports, as
doc/r39_tasks_and_acceptance.md R3 asks:
  * per point: device cycles A and B, speedup A/B, eta on both sides;
  * the suite: E on both sides (balanced weights, archbench's own scoring),
    the geometric-mean speedup S_geo;
  * the stall attribution on both sides, as shares of all cycles, when the
    records carry counters: issued, every perf_x_*, and the front end;
  * every point B made slower by more than --max-regress, and every point
    that does not pass on both sides.
With --gate it exits non-zero when S_geo < --min-speedup, when any point
regressed past --max-regress (unless listed in --explained), or when any
point passes on one side only.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from archbench import report as R          # noqa: E402
from archbench import registry             # noqa: E402

STALLS = ['perf_x_hazard', 'perf_x_lsu', 'perf_x_vrf', 'perf_x_vrfwb', 'perf_x_txf',
          'perf_x_fpu', 'perf_x_mtx', 'perf_x_fence', 'perf_x_arrive', 'perf_x_rsv',
          'perf_x_wb', 'perf_x_frontend']


def side(path, tier):
    recs = R.load([path])
    if tier:
        recs = [r for r in recs if r['tier'] in tier]
    keys = sorted({R.platform_key(r) for r in recs})
    if len(keys) != 1:
        raise SystemExit(f'{path}: {len(keys)} configurations in one side ({keys}); '
                         f'an ablation side must be one configuration')
    return keys[0], {r['point_id']: r for r in recs}


def efficiency(recs, tier):
    pts = registry.points(tier=tier)
    by_family = {}
    for p in pts:
        by_family.setdefault(p.family, []).append(p.id)
    weights = R.balanced(by_family)
    ids = sorted(p.id for p in pts)
    points = [R.to_point(recs[i]) if i in recs else R.S.Point(i, '', 'not-measured')
              for i in ids]
    E, _A, _etas = R.S.efficiency(points, {i: weights[i] for i in ids if i in weights})
    return E


def attribution(recs, ids):
    tot = {}
    for i in ids:
        c = recs[i].get('counters') or {}
        for k, v in c.items():
            if isinstance(v, int):
                tot[k] = tot.get(k, 0) + v
    cyc = tot.get('perf_cycles')
    if not cyc:
        return None
    out = {'issued': tot.get('perf_issued', 0) / cyc}
    for k in STALLS:
        if k in tot:
            out[k] = tot[k] / cyc
    if 'perf_x_frontend' not in tot:
        # Older images: the front end is whatever neither issued nor stalled.
        out['unattributed'] = 1 - (tot.get('perf_issued', 0) + tot.get('perf_stall_total', 0)) / cyc
    out['cycles'] = cyc
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--tier', nargs='+')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--min-speedup', type=float, default=1.0)
    ap.add_argument('--max-regress', type=float, default=1.02)
    ap.add_argument('--explained', nargs='*', default=[],
                    help='point ids whose regression is explained in the report')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)

    ka, A = side(args.a, args.tier)
    kb, B = side(args.b, args.tier)
    print(f'A  {ka}  ({len(A)} points)  {args.a}')
    print(f'B  {kb}  ({len(B)} points)  {args.b}')
    if ka == kb:
        print('note: A and B have the same platform key -- the same configuration')

    both = sorted(i for i in set(A) & set(B)
                  if A[i]['status'] == 'pass' and B[i]['status'] == 'pass')
    def passes(side_, i):
        return (side_.get(i) or {}).get('status') == 'pass'
    one_sided = sorted(i for i in set(A) | set(B) if passes(A, i) != passes(B, i))
    rows, regress, logs = [], [], []
    for i in both:
        ta, tb = A[i]['time']['device'], B[i]['time']['device']
        sp = ta / tb
        rows.append(dict(point=i, a=ta, b=tb, speedup=sp, eta_a=A[i].get('eta'), eta_b=B[i].get('eta')))
        logs.append(math.log(sp))
        if tb > ta * args.max_regress:
            regress.append(i)
    sgeo = math.exp(sum(logs) / len(logs)) if logs else float('nan')

    print(f"\n{'point':<52s} {'A cycles':>11s} {'B cycles':>11s} {'A/B':>7s} {'eta A':>7s} {'eta B':>7s}")
    for r in rows:
        mark = '  REGRESS' if r['point'] in regress else ''
        print(f"{r['point']:<52s} {r['a']:>11d} {r['b']:>11d} {r['speedup']:>7.3f} "
              f"{r['eta_a'] or 0:>7.4f} {r['eta_b'] or 0:>7.4f}{mark}")

    tier = args.tier
    Ea, Eb = efficiency(A, tier), efficiency(B, tier)
    print(f'\nS_geo = {sgeo:.4f} over {len(both)} point(s) passing on both sides')
    print(f'E(A) = {Ea.value:.4g} ({"certified" if Ea.certified else "NOT CERTIFIED"}; '
          f'supported-only {Ea.supported_only:.4g})')
    print(f'E(B) = {Eb.value:.4g} ({"certified" if Eb.certified else "NOT CERTIFIED"}; '
          f'supported-only {Eb.supported_only:.4g})')

    at_a, at_b = attribution(A, both), attribution(B, both)
    if at_a and at_b:
        print(f"\n{'share of cycles':<20s} {'A':>8s} {'B':>8s}   (B cycles / A cycles = "
              f"{at_b['cycles'] / at_a['cycles']:.3f})")
        for k in ['issued'] + STALLS + ['unattributed']:
            if k in at_a or k in at_b:
                print(f"{k:<20s} {at_a.get(k, float('nan')):>8.3f} {at_b.get(k, float('nan')):>8.3f}")
    if regress:
        print(f'\nslower by more than {args.max_regress}x: {regress}')
    if one_sided:
        print(f'passing on one side only: {one_sided}')

    failed = []
    if args.gate:
        if not (sgeo >= args.min_speedup):
            failed.append(f'S_geo {sgeo:.4f} < {args.min_speedup}')
        unexplained = [i for i in regress if i not in args.explained]
        if unexplained:
            failed.append(f'unexplained regressions {unexplained}')
        if one_sided:
            failed.append(f'points passing on one side only {one_sided}')
        print('\nGATE ' + ('PASS' if not failed else 'FAIL: ' + '; '.join(failed)))
    if args.json:
        args.json.write_text(json.dumps(dict(
            a=dict(key=ka, path=args.a, E=Ea.value, attribution=at_a),
            b=dict(key=kb, path=args.b, E=Eb.value, attribution=at_b),
            s_geo=sgeo, points=rows, regress=regress, one_sided=one_sided,
            gate=(None if not args.gate else (not failed)), failures=failed), indent=1) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
