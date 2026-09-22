#!/usr/bin/env python3
"""Compile the representative operator regression kernels.

Original TileBench source compatibility is tested by upstream.py.

The output that matters is the blocker table: failures grouped by their
diagnostic, most common first. That is the work queue -- fixing the top
row unblocks the most operators.

A kernel that has not been written yet is NOT RUN and is counted apart
from a failure. A suite that quietly omitted the hard operators would
report a coverage number that means nothing.
"""
import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (os.path.join(ROOT, 'python'), os.path.join(ROOT, 'compiler',
                                                     'python'), HERE,
          os.path.join(ROOT, 'tools', 'r1')):
    if p not in sys.path:
        sys.path.insert(0, p)

SPEC = json.load(open(os.path.join(ROOT, 'spec', 'r37_tilebench.json')))
PROFILE = 'core-vrf-sync-v1'


def _blocker(msg):
    """The diagnostic, reduced to something countable.

    Names, shapes and register numbers vary between operators; the thing
    that does not is what the compiler refused to do. Keeping the first
    line and stripping the specifics is what makes the table a queue
    rather than a list.
    """
    line = str(msg).splitlines()[0].strip()
    # The quoted part of "no TNCA lowering for TTIR operation 'tt.foo'"
    # IS the actionable thing, so it stays. Everywhere else a quoted
    # name is an operator-specific detail and would split one blocker
    # into forty.
    keep_quotes = 'no TNCA lowering' in line or 'not lowered' in line
    if not keep_quotes:
        line = re.sub(r"'[^']*'", "'X'", line)
    line = re.sub(r'\b\d+\b', 'N', line)
    return line[:110]


def run(only=None, json_out=None, verbose=False):
    import kernels as K
    from tnca import aot
    from tnca.errors import TncaError

    rows = []
    for entry in SPEC['operators']:
        name = entry['id']
        if only and name not in only:
            continue
        spec = K.KERNELS.get(name)
        if spec is None:
            rows.append({'id': name, 'category': entry['category'],
                         'state': 'NOT WRITTEN', 'blocker': None})
            continue
        fn, sig, ce = spec
        try:
            variants = {}
            with tempfile.TemporaryDirectory(prefix='tnca-r37-') as tmp:
                for label, overrides in entry.get(
                        'signature_variants', {'default': {}}).items():
                    art = aot.compile_aot(
                        fn, signature=dict(sig, **overrides), constexprs=ce,
                        profile=PROFILE, options={'opt_level': 2},
                        output=os.path.join(tmp, f'{name}-{label}.tnca'))
                    variants[label] = len(art.asm['hex'].splitlines())
            rows.append({'id': name, 'category': entry['category'],
                         'state': 'COMPILES', 'blocker': None,
                         'words': max(variants.values()), 'variants': variants})
        except TncaError as e:
            rows.append({'id': name, 'category': entry['category'],
                         'state': 'REFUSED',
                         'blocker': _blocker(e),
                         'detail': str(e).splitlines()[0][:160]})
        except Exception as e:                             # noqa: BLE001
            rows.append({'id': name, 'category': entry['category'],
                         'state': 'ERROR',
                         'blocker': _blocker(f"{type(e).__name__}: {e}"),
                         'detail': f"{type(e).__name__}: {e}"[:160]})

    total = len(rows)
    built = [r for r in rows if r['state'] == 'COMPILES']
    written = [r for r in rows if r['state'] != 'NOT WRITTEN']
    print(f"TileBench coverage: {len(built)}/{total} operator(s) build "
          f"from Triton source")
    print(f"  {len(written)} kernel(s) written, "
          f"{total - len(written)} NOT WRITTEN (not a pass and not a "
          f"failure)")
    print()

    by_cat = {}
    for r in rows:
        by_cat.setdefault(r['category'], []).append(r)
    print("  by category")
    for cat in SPEC['categories']:
        rs = by_cat.get(cat, [])
        ok = sum(1 for r in rs if r['state'] == 'COMPILES')
        wr = sum(1 for r in rs if r['state'] != 'NOT WRITTEN')
        print(f"    {cat:<14s} {ok}/{len(rs)} built, {wr} written")

    blockers = Counter(r['blocker'] for r in rows if r['blocker'])
    if blockers:
        print()
        print("  what is in the way, most operators first")
        for b, n in blockers.most_common():
            who = [r['id'] for r in rows if r['blocker'] == b]
            print(f"    {n:>2d}  {b}")
            print(f"        {', '.join(who[:6])}"
                  + (' ...' if len(who) > 6 else ''))

    if verbose:
        print()
        for r in rows:
            mark = {'COMPILES': 'ok ', 'REFUSED': 'no ', 'ERROR': 'ERR',
                    'NOT WRITTEN': '-  '}[r['state']]
            extra = (f" {r.get('words')} words" if r['state'] == 'COMPILES'
                     else (' ' + r.get('detail', '') if r.get('detail')
                           else ''))
            print(f"    {mark} {r['id']:<24s}{extra}")

    if json_out:
        import r1lib as R
        R.write_json(json_out, {'built': len(built), 'total': total,
                                'written': len(written), 'rows': rows})
    return rows, built, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--json')
    args = ap.parse_args()
    known = {o['id'] for o in SPEC['operators']}
    if args.only and set(args.only) - known:
        ap.error('unknown operators: ' + ', '.join(sorted(set(args.only) - known)))
    _rows, built, total = run(args.only, args.json, args.verbose)
    return 0 if len(built) == total else 1


if __name__ == '__main__':
    sys.exit(main())
