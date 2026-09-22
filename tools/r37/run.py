#!/usr/bin/env python3
"""Run the representative operator regression kernels on RTL.

Original TileBench source compatibility is tested by upstream_run.py.

`coverage.py` says which operators BUILD. This says which ones are
right, which is the question that matters -- a compiler that emits
plausible instructions for all forty-five and the wrong answer for one
has not supported forty-five operators.

Each case lays its inputs out in the simulated memory, executes the
compiled kernel on the iverilog core bench, reads the output region back
and compares it against a reference written out in Python from the
kernel's own text. Tolerances are the ISA's published SFU bound, not
numbers fitted to make a case pass; cases.py says which operation put
each one where it is.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SPEC = json.load(open(os.path.join(ROOT, 'spec', 'r37_tilebench.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    ap.add_argument('--cycles', type=int, default=4000000)
    ap.add_argument('--suite', choices=('baseline', 'edge', 'all'), default='all')
    ap.add_argument('--seed', type=int, default=37,
                    help='deterministic second input set for all 45 operators')
    ap.add_argument('--bench', choices=('fast', 'slow', 'ooo'), default='fast')
    args = ap.parse_args()

    import execute as E
    import cases as Cs
    import rtl_harness as H

    if not H.bench_available(args.bench):
        print(f'  r37-run                FAIL   missing bench {args.bench}')
        return 1

    known = {o['id'] for o in SPEC['operators']}
    if args.only and set(args.only) - known:
        ap.error('unknown operators: ' + ', '.join(sorted(set(args.only) - known)))
    table = {}
    if args.suite != 'edge':
        table.update(Cs.build(E.Case))
        table.update({f'{oid}/seed{args.seed}': case
                      for oid, case in Cs.build(E.Case, seed=args.seed).items()})
    if args.suite != 'baseline':
        import edge_cases
        table.update(edge_cases.build(E.Case))
    wanted = [o['id'] for o in SPEC['operators']
              if not args.only or o['id'] in args.only]

    rows, ok_n = [], 0
    for cid, case in table.items():
        oid = case.kernel
        if oid not in wanted:
            continue
        t0 = time.time()
        try:
            ok, detail = E.run(cid.replace('/', '_'), case,
                               max_cycles=args.cycles, bench=args.bench)
        except Exception as e:                            # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        took = time.time() - t0
        ok_n += bool(ok)
        rows.append({'id': oid, 'case': cid, 'ok': bool(ok), 'detail': str(detail)[:400],
                     'seconds': round(took, 1),
                     'tolerance': case.rtol, 'note': case.note})
        print(f"  {'ok ' if ok else 'NO '} {cid:40s} {took:6.1f}s  "
              f"{str(detail).splitlines()[0][:100] if detail else ''}", flush=True)

    executed = [oid for oid in wanted if any(r['id'] == oid for r in rows)]
    operators = [{'id': oid,
                  'ok': all(r['ok'] for r in rows if r['id'] == oid),
                  'cases': sum(r['id'] == oid for r in rows)} for oid in executed]
    passed = sum(r['ok'] for r in operators)
    print()
    print(f"TileBench execution: {passed}/{len(executed)} operator(s), "
          f"{ok_n}/{len(rows)} cases correct on {args.bench} RTL")
    for cat in SPEC['categories']:
        ids = {o['id'] for o in SPEC['operators'] if o['category'] == cat}
        subset = [r for r in operators if r['id'] in ids]
        print(f"    {cat:<14s} {sum(r['ok'] for r in subset)}/{len(subset)}")

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'operators': operators, 'passed': passed,
                       'total': len(executed), 'cases': rows,
                       'cases_passed': ok_n, 'cases_total': len(rows),
                       'bench': args.bench, 'suite': args.suite}, fh, indent=2)
            fh.write('\n')
    return 0 if rows and ok_n == len(rows) else 1



if __name__ == '__main__':
    sys.exit(main())
