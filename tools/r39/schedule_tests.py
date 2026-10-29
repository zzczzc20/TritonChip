#!/usr/bin/env python3
"""R39-01: the scheduler moves only what it may, and never makes the model's
estimate worse.

    python3 tools/r39/schedule_tests.py [--levels 0 1 2] [--profiles core-sim core-vrf-sync-v1]

Every R3.7 case is compiled with schedule=1, and every block the scheduler
reordered is checked against program order by rules written here, not by the
scheduler's own dependence graph:

1. the block is a permutation of the original, with any label still first
   and any branch or exit still last;
2. every value an operation reads was last written, before it, by the same
   operation as in program order -- read-after-write, and also the in-place
   writes (`vinsert`, `assign`) that must not pass a reader or a writer;
3. every operation with an effect keeps its order relative to every other
   one, and a memory read stays between the same two effects;
4. an operation reading a value that aliases a window register stays ahead
   of the effect that followed it;
5. the model's issue-cycle estimate of the block is no worse.

(The pass first drops loads from the argument block that nothing reads; the
reference order is program order after that drop.)

A hand-built block also checks that the scheduler does what it is for: a
dependent pair with independent work after it gets that work between them.
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'compiler' / 'python'),
                str(ROOT / 'tools' / 'r38')]

from tnca_cc import schedule as S          # noqa: E402
from tnca_cc.optimization import _values_in  # noqa: E402


def check_block(orig, new, aliased):
    """Problems with `new` as a reordering of `orig` (list of strings)."""
    probs = []
    if sorted(map(id, orig)) != sorted(map(id, new)):
        return ['not a permutation of the original block']
    if orig and orig[0].kind == 'label' and new[0] is not orig[0]:
        probs.append('the label is not first')
    if orig and orig[-1].kind in S.BLOCK_END and new[-1] is not orig[-1]:
        probs.append(f'the {orig[-1].kind} is not last')

    def writes(op):
        return [id(r) for r in op.results] + \
            [id(op.operands[i]) for i in S.MUTATES.get(op.kind, ()) if i < len(op.operands)]

    def last_writers(order):
        out, last = {}, {}
        for op in order:
            out[id(op)] = {id(v): last.get(id(v)) for v in _values_in(op)}
            for w in writes(op):
                last[w] = id(op)
        return out

    a, b = last_writers(orig), last_writers(new)
    for op in orig:
        if a[id(op)] != b[id(op)]:
            probs.append(f'{op.kind}: reads a value last written by a different operation')
    pos = {id(op): i for i, op in enumerate(new)}

    def effect(op):
        return op.kind not in S.PURE and op.kind not in S.MEMREAD and op.kind != 'comment'

    seq = [op for op in orig if op.kind != 'comment' and op.kind not in S.PURE]
    for i, x in enumerate(seq):
        for y in seq[i + 1:]:
            if (effect(x) or effect(y)) and pos[id(x)] > pos[id(y)]:
                probs.append(f'{x.kind} and {y.kind} changed order')
    for i, op in enumerate(orig):
        if op.kind in S.PURE and any(id(v) in aliased for v in _values_in(op)):
            nxt = next((y for y in orig[i + 1:] if effect(y)), None)
            if nxt is not None and pos[id(op)] > pos[id(nxt)]:
                probs.append(f'{op.kind} reading a window alias passed {nxt.kind}')
    return probs


class Recorder:
    """Wraps schedule_kernel: keeps each role's ops before and after."""

    def __init__(self):
        self.real = S.schedule_kernel
        self.problems, self.blocks, self.moved, self.worse = [], 0, 0, 0
        self.case = None

    def __call__(self, kernel, target, *, model=None, **kw):
        # The pass drops argument loads nothing reads before it schedules;
        # the reference is program order after that drop.
        before = {r.index: S.drop_unread_argument_loads(list(r.region.ops))[0]
                  for r in kernel.roles}
        model = model or S.Model.for_target(target)
        rep = self.real(kernel, target, model=model, **kw)
        for r in kernel.roles:
            aliased = {id(x) for op in before[r.index] if op.kind in S.ALIASES for x in op.results}
            ob, nb = S._blocks(before[r.index]), S._blocks(r.region.ops)
            if [len(x) for x in ob] != [len(x) for x in nb]:
                self.problems.append(f'{self.case}: role {r.index}: blocks changed shape')
                continue
            for o, n in zip(ob, nb):
                self.blocks += 1
                if any(x is not y for x, y in zip(o, n)):
                    self.moved += 1
                for p in check_block(o, n, aliased):
                    self.problems.append(f'{self.case}: role {r.index}: {p}')
                if S.estimate(n, model) > S.estimate(o, model):
                    self.worse += 1
                    self.problems.append(f'{self.case}: role {r.index}: estimate got worse')
        return rep


def hand_block():
    """`a = x*y; b = a+1; c = p+q; d = r+s` -- c and d belong between a and b."""
    from tnca_cc.ir import make_kernel, Builder
    k = make_kernel('hand')
    b = Builder(k, k.roles[0].region)
    x, y, p, q, r, s = (b.vmovi(i, dtype='f32') for i in range(6))
    a = b.vbin('mul', x, y, dtype='f32')
    b.vbin('add', a, x, dtype='f32')
    b.vbin('add', p, q, dtype='f32')
    b.vbin('add', r, s, dtype='f32')
    b.exit()
    ops = k.roles[0].region.ops
    model = S.Model('banked', __import__('json').load(open(ROOT / 'spec' / 'r39_sched_model.json'))
                    ['models']['banked'])
    before = S.estimate(ops, model)
    out = S.schedule_block(list(ops), model, set(), set())
    after = S.estimate(out, model)
    kinds = [(o.kind, o.attrs.get('op')) for o in out]
    i_mul = kinds.index(('vbin', 'mul'))
    i_dep = next(i for i, o in enumerate(out) if o.kind == 'vbin' and a in o.operands)
    between = i_dep - i_mul - 1
    return before, after, between, check_block(ops, out, set())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--levels', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--profiles', nargs='+', default=['core-sim', 'core-vrf-sync-v1'])
    ap.add_argument('--only', nargs='+')
    args = ap.parse_args()
    ok = True

    before, after, between, probs = hand_block()
    good = after < before and between >= 1 and not probs
    print(f"  {'PASS' if good else 'FAIL'}  hand block: estimate {before} -> {after}, "
          f"{between} operation(s) placed between a producer and its consumer {probs or ''}")
    ok &= good

    # Every banked-VRF target resolves to a banked latency model (R41 G4: the
    # resized profiles fell to the flip-flop default and were scheduled for a
    # different register file).
    import json as _json
    spec = _json.load(open(ROOT / 'spec' / 'r39_sched_model.json'))
    profiles = _json.load(open(ROOT / 'spec' / 'r1_profiles.json'))['profiles']
    wrong = []
    for f in sorted((ROOT / 'compiler' / 'targets').glob('*.json')):
        t = _json.load(open(f))
        impl = profiles.get(t.get('profile'), {}).get('implementation', {})
        banked = impl.get('vector_register_file') == 'banked'
        name = S.Model.model_name(spec, t['name'])
        if banked and not name.startswith('banked'):
            wrong.append(f"{t['name']} -> {name}")
    print(f"  {'PASS' if not wrong else 'FAIL'}  scheduling model per target {wrong or ''}")
    ok &= not wrong

    import harness as H
    from tnca import aot, device
    rec = Recorder()
    S.schedule_kernel = rec
    import tempfile
    cases = H.cases()
    if args.only:
        cases = {n: c for n, c in cases.items() if n in args.only}
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for prof in args.profiles:
            for level in args.levels:
                for name, case in cases.items():
                    rec.case = f'{prof} O{level} {name}'
                    fn, sig, ce = case.kernel_spec
                    try:
                        aot.compile_aot(fn, signature=sig, constexprs=ce,
                                        profile=device.base_profile(prof),
                                        options={'opt_level': level, 'schedule': 1},
                                        output=str(Path(tmp) / 'k.tnca'))
                    except Exception as e:                       # noqa: BLE001
                        failures.append(f'{rec.case}: {type(e).__name__}: {str(e)[:200]}')
    S.schedule_kernel = rec.real
    for f in failures[:20]:
        print('  FAIL  compile', f)
    for p in rec.problems[:30]:
        print('  FAIL ', p)
    n = len(cases) * len(args.levels) * len(args.profiles)
    print(f'  {n - len(failures)}/{n} compiles; {rec.blocks} blocks, {rec.moved} reordered; '
          f'{len(rec.problems)} rule violation(s); {rec.worse} worse estimate(s)')
    ok &= not failures and not rec.problems
    print(f"r39-schedule-tests     {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
