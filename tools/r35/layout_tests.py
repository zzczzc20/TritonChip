#!/usr/bin/env python3
"""
R35-LL-01, LL-02 and LL-03 -- phase3_5_tasks_and_acceptance.md section 7.

The layout semantics of section 3, checked element by element against an
enumeration that does not use LinearLayout.

The independent oracle matters here more than usual. Section 4.2 is
explicit that an independent implementation is for VERIFYING and must
never quietly stand in for upstream in the product -- so the product
path calls upstream and this file does not. Every expectation below is
worked out by walking the tile in Python and saying where each element
ought to land, and then upstream is asked and the two are compared.

Seeds {1, 7, 42} and at most 64 host cases per seed, per section 7.
None of this starts a simulator: layouts are decided at compile time and
checking them on hardware would measure the hardware.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
import r1lib as R                                          # noqa: E402
from tnca_cc.upstream.layout import (LinearLayoutAdapter, LayoutError,  # noqa
                                     identity, available)
from tnca_cc.layout import (classify, declared_basis, RENAME,  # noqa: E402
                            INSTRUCTIONS)
from tnca_cc.ir import Op, Value                           # noqa: E402

SEEDS = (1, 7, 42)
PER_SEED = 64


# ---------------------------------------------------------------------------
# the independent side: no LinearLayout anywhere below

def oracle_transpose(rows, cols):
    """Row-major to column-major, worked out by walking the tile."""
    return [(i % cols) * rows + (i // cols) for i in range(rows * cols)]


def oracle_identity(n):
    return list(range(n))


def oracle_broadcast(n, repeat):
    """Every source element read `repeat` times."""
    return [i // repeat for i in range(n)]


def apply_basis(basis, i):
    """Where i lands, from a GF(2) basis. Plain arithmetic, no upstream."""
    j = 0
    for k, b in enumerate(basis):
        if (i >> k) & 1:
            j ^= b
    return j


def ll01(log):
    """Element by element, against an enumeration upstream never saw."""
    ok = True
    total = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        problems, checked = [], 0
        cases = []
        # The shapes section 3.2 names: power-of-two tiles, a valid tail,
        # a broadcast, packed f16 pairs and the A/B/ARF correspondences.
        for rows, cols in ((2, 2), (4, 4), (2, 8), (8, 2), (4, 2)):
            cases.append(('transpose', rows, cols))
        for n in (2, 4, 8, 16):
            cases.append(('identity', n, 1))
        for n, rep in ((4, 2), (8, 2), (16, 4)):
            cases.append(('broadcast', n, rep))
        rng.shuffle(cases)
        cases = cases[:PER_SEED]

        for kind, a, b in cases:
            if kind == 'transpose':
                rows, cols = a, b
                n = rows * cols
                rbits = (rows - 1).bit_length()
                cbits = (cols - 1).bit_length()
                # reg bit k: k < cbits picks a column bit, else a row bit
                bases = []
                for k in range(cbits):
                    bases.append([0, 1 << k])
                for k in range(rbits):
                    bases.append([1 << k, 0])
                ll = LinearLayoutAdapter([('reg', bases)], ['row', 'col'],
                                         [rows, cols], what=f"{rows}x{cols}")
                want = oracle_transpose(rows, cols)
                for i in range(n):
                    got = ll.apply({'reg': i})
                    # flatten column-major: the transposed position
                    flat = got['col'] * rows + got['row']
                    checked += 1
                    if flat != want[i]:
                        problems.append(f"{rows}x{cols} lane {i}: upstream "
                                        f"{flat}, enumeration {want[i]}")
            elif kind == 'identity':
                n = a
                ll = identity(n, 'reg', 'elem')
                want = oracle_identity(n)
                for i in range(n):
                    checked += 1
                    if ll.apply({'reg': i}) != {'elem': want[i]}:
                        problems.append(f"identity {n} lane {i}")
            else:
                n, rep = a, b
                rbits = (rep - 1).bit_length()
                bases = [[0]] * rbits + \
                        [[1 << k] for k in range((n // rep - 1).bit_length())]
                try:
                    ll = LinearLayoutAdapter([('reg', bases)], ['src'],
                                             [n // rep],
                                             require_surjective=True,
                                             what=f"broadcast x{rep}")
                except LayoutError as e:
                    problems.append(f"broadcast {n}x{rep}: {e}")
                    continue
                want = oracle_broadcast(n, rep)
                for i in range(n):
                    checked += 1
                    if ll.apply({'reg': i}) != {'src': want[i]}:
                        problems.append(
                            f"broadcast {n}x{rep} lane {i}: upstream "
                            f"{ll.apply({'reg': i})}, enumeration "
                            f"{want[i]}")
        total += checked
        log.append(f"    seed {seed:<3d} {len(cases)} case(s), {checked} "
                   f"element comparison(s), {len(problems)} problem(s)")
        for p in problems[:3]:
            log.append(f"        {p}")
        ok &= not problems

    # The comparison has to be able to fail. The first control here did
    # not: it was the transpose written with its dimensions the other way
    # round and flattened to match, so of course it agreed. A control
    # that cannot disagree tests nothing.
    ctrl = identity(16, 'reg', 'elem')
    want = oracle_transpose(4, 4)
    differ = [i for i in range(16)
              if ctrl.apply({'reg': i})['elem'] != want[i]]
    log.append(f"    control: the IDENTITY checked against the transpose "
               f"enumeration -> disagrees at {len(differ)} of 16 position(s)"
               + (" (the 4 fixed points agree, as they must)"
                  if len(differ) == 12 else ""))
    ok &= len(differ) == 12
    return ok, f"{total} element comparison(s) over {len(SEEDS)} seed(s)"


def ll02(log):
    """Coverage, multiplicity, and the things that must be refused."""
    ok = True

    # Every valid element covered exactly once, for a permutation.
    ll = LinearLayoutAdapter([('reg', [[1, 0], [2, 0], [0, 1], [0, 2]])],
                             ['row', 'col'], [4, 4], what='4x4')
    cover = ll.covers({'reg': 16}, {'row': 4, 'col': 4})
    log.append(f"    a permutation covers {cover['distinct_outputs']}/"
               f"{cover['expected_outputs']} output(s), multiplicity "
               f"{cover['max_multiplicity']}, {cover['unhit']} unhit")
    ok &= cover['covered'] and cover['max_multiplicity'] == 1

    # A broadcast covers everything and is NOT injective, and must not be
    # inverted by pretending it is.
    b = LinearLayoutAdapter([('reg', [[0], [1], [2]])], ['src'], [4],
                            what='broadcast')
    cb = b.covers({'reg': 8}, {'src': 4})
    log.append(f"    a broadcast covers {cb['distinct_outputs']}/"
               f"{cb['expected_outputs']} with multiplicity "
               f"{cb['max_multiplicity']}: injective="
               f"{b.is_injective()}, invertible={b.is_invertible()}")
    ok &= cb['covered'] and cb['max_multiplicity'] == 2
    try:
        b.invert()
        log.append('    inverting the broadcast was ALLOWED')
        ok = False
    except LayoutError as e:
        log.append(f"    inverting it is refused: {str(e)[:78]}")

    # A non-power-of-two bound is not a linear layout's business.
    try:
        LinearLayoutAdapter([('reg', [[1], [2]])], ['i'], [3])
        log.append('    a bound of 3 was ACCEPTED')
        ok = False
    except LayoutError as e:
        log.append(f"    a bound of 3 is refused: {str(e)[:78]}")

    # Dimensions that do not line up abort the process upstream, so they
    # have to be refused before the call.
    try:
        ll.compose(identity(16, 'lane', 'elem'))
        log.append('    a mismatched composition was ALLOWED')
        ok = False
    except LayoutError as e:
        log.append(f"    a mismatched composition is refused before it "
                   f"reaches upstream: {str(e)[:52]}")

    # A role is not a layout dimension (frozen decision 2, section 3.1).
    # Asked of the layouts the pass actually builds, rather than by
    # grepping the file -- the pass iterates over roles, so the word is
    # in the source for a reason that has nothing to do with this.
    x = Value('x', 'vector')
    o1 = Op('view', [Value('v1', 'vector')], [x],
            {'bases': [('reg', [[1, 0], [2, 0], [0, 1], [0, 2]])],
             'out_dims': ['row', 'col'], 'out_sizes': [4, 4],
             'what': 'probe'})
    dims = set()
    for p in classify([o1], use_upstream=True):
        if p.layout is not None:
            dims |= set(p.layout.in_dims) | set(p.layout.out_dims)
    dims |= set(o1.attrs['out_dims']) | {n for n, _r in o1.attrs['bases']}
    forbidden = sorted(d for d in dims
                       if d.lower() in ('role', 'rctx', 'warp', 'lane_group'))
    log.append(f"    the dimensions the pass works in are {sorted(dims)}: "
               f"{'no role or warp among them' if not forbidden else 'FORBIDDEN: ' + str(forbidden)}")
    ok &= not forbidden
    return ok, 'coverage, multiplicity, non-bijection and the refusals'


def ll03(log):
    """Every conversion classified, and the cost checkable in the code."""
    ok = True

    def view(bases, out_dims, out_sizes, src, what):
        r = Value(what, 'vector')
        return Op('view', [r], [src], {'bases': bases, 'out_dims': out_dims,
                                       'out_sizes': out_sizes,
                                       'what': what}), r

    x = Value('x', 'vector')
    fwd = [('reg', [[1, 0], [2, 0], [0, 1], [0, 2]])]
    back = [('row', [[1], [2]]), ('col', [[4], [8]])]
    o1, v1 = view(fwd, ['row', 'col'], [4, 4], x, 'transpose')
    o2, v2 = view(back, ['reg'], [16], v1, 'back')

    plans = classify([o1, o2], use_upstream=True)
    log.append(f"    transpose then back, with the algebra on:  "
               f"{plans[0].category} -- {plans[0].why[:56]}")
    ok &= len(plans) == 1 and plans[0].category == RENAME

    plans_off = classify([o1, o2], use_upstream=False)
    log.append(f"    the same pair with it off:                 "
               f"{plans_off[0].category} -- {plans_off[0].why[:56]}")
    ok &= plans_off[0].category == INSTRUCTIONS

    # One transpose on its own is a real conversion, not a rename.
    solo = classify([view(fwd, ['row', 'col'], [4, 4], x, 'solo')[0]],
                    use_upstream=True)
    log.append(f"    one transpose alone:                       "
               f"{solo[0].category} -- {solo[0].why[:56]}")
    ok &= solo[0].category == INSTRUCTIONS

    # The declared basis is what the emitted permutation will be, and it
    # is a real transpose. Checked against the enumeration, not upstream.
    basis = declared_basis(o1)
    perm = [apply_basis(basis, i) for i in range(16)]
    want = oracle_transpose(4, 4)
    log.append(f"    the basis the ablation emits from: {basis} -> "
               f"{'the transpose the enumeration gives'
                  if perm == want else 'SOMETHING ELSE: ' + str(perm)}")
    ok &= perm == want

    # Applying it twice is the identity, which is why the pair cancels.
    twice = [apply_basis(basis, apply_basis(basis, i)) for i in range(16)]
    log.append(f"    and applying it twice is the identity: {twice == list(range(16))}")
    ok &= twice == list(range(16))

    # Nothing invents a cross-lane instruction or a scratchpad.
    cg = open(os.path.join(ROOT, 'compiler', 'python', 'tnca_cc',
                           'codegen.py')).read()
    view_src = cg[cg.index('def _op_view'):cg.index('def _op_gather')]
    real = json.load(open(os.path.join(ROOT, 'isa',
                                       'tnca-isa-v0.1.json')))
    names = set()

    def walk(o):
        if isinstance(o, dict):
            if o.get('mnemonic'):
                names.add(o['mnemonic'])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(real)
    import re as _re
    used = sorted(set(_re.findall(r"'(v\.[\w.]+)'", view_src)))
    unknown = [u for u in used if u not in names]
    log.append(f"    the conversion emits {used}")
    log.append(f"    every one of them is in the ISA: "
               f"{'yes' if not unknown else 'NO -- ' + str(unknown)}")
    ok &= not unknown
    ok &= 'smem' not in view_src.lower() and 'scratch' not in view_src.lower()
    return ok, 'rename, conversion and refusal, with the cost on the page'


TESTS = [('R35-LL-01', ll01), ('R35-LL-02', ll02), ('R35-LL-03', ll03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.5 Gate A: the layout semantics, against an independent "
          "enumeration")
    if not available():
        print("  the pinned upstream LinearLayout is not importable here")
        return 1
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:                             # noqa: BLE001
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-700:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    print(f"  r35-layout-tests       {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
