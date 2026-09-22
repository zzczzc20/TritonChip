#!/usr/bin/env python3
"""Run every TileBench operator on the RTL and check what it computed.

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
    args = ap.parse_args()

    import execute as E
    import cases as Cs
    import rtl_harness as H

    if not H.bench_available('fast'):
        print('  r37-run                SKIP   no iverilog bench '
              '(make sim/build/tb_core.vvp)')
        return 0

    table = Cs.build(E.Case)
    wanted = [o['id'] for o in SPEC['operators']
              if not args.only or o['id'] in args.only]

    rows, ok_n = [], 0
    for oid in wanted:
        case = table[oid]
        t0 = time.time()
        try:
            ok, detail = E.run(oid.replace('/', '_'), case,
                               max_cycles=args.cycles)
        except Exception as e:                            # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        took = time.time() - t0
        ok_n += bool(ok)
        rows.append({'id': oid, 'ok': bool(ok), 'detail': str(detail)[:400],
                     'seconds': round(took, 1),
                     'tolerance': case.rtol, 'note': case.note})
        print(f"  {'ok ' if ok else 'NO '} {oid:24s} {took:6.1f}s  "
              f"{str(detail).splitlines()[0][:80] if detail else ''}")

    print()
    print(f"TileBench execution: {ok_n}/{len(wanted)} operator(s) compute "
          f"the right answer on the RTL")
    by = {}
    for o in SPEC['operators']:
        if o['id'] in {r['id'] for r in rows}:
            by.setdefault(o['category'], [0, 0])
            by[o['category']][1] += 1
            by[o['category']][0] += next(r['ok'] for r in rows
                                         if r['id'] == o['id'])
    for cat, (good, total) in by.items():
        print(f"    {cat:<14s} {good}/{total}")

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'operators': rows, 'passed': ok_n,
                       'total': len(wanted)}, fh, indent=2)
            fh.write('\n')
    return 0 if ok_n == len(wanted) else 1


if __name__ == '__main__':
    sys.exit(main())
