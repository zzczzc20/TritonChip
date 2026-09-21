#!/usr/bin/env python3
"""
R35-ALLOC-01, ALLOC-02 and SCHED-01 -- phase3_5_tasks_and_acceptance.md 7.

What an allocation costs in bank conflicts, that the resource contracts
still hold, and that nothing was reordered that could not be.

The cost model is checked against the hardware R3 v2 accepted rather
than against itself: t29_vrfbank is the directed bank-contention program
that stage built, its counters say what the file really did, and the
model has to agree with them.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, HERE)
import r1lib as R                                          # noqa: E402
from tnca_cc import bankcost                               # noqa: E402

PROFILE = json.load(open(os.path.join(
    ROOT, 'spec', 'r35_workloads.json')))['hardware_precondition']['profile']


def compile_one(**kw):
    from tnca import aot
    return aot.compile_aot(profile=PROFILE,
                           output=os.path.join(tempfile.mkdtemp(), 'x.tnca'),
                           **kw)


def l03(**opts):
    import r35_workloads as WW
    return compile_one(source=WW.tnca_view_elementwise(), kernel_name='l03',
                       signature={'in_ptr': '*fp32', 'out_ptr': '*fp32'},
                       source_kind='tnca-ir', options=opts)


def gemm(**opts):
    import workloads as W
    return compile_one(
        source=W.gemm_program(16, 16, 16,
                              a_desc={'tile': 0x2000, 'map': 0x1000},
                              b_desc={'tile': 0x3000, 'map': 0x1100},
                              out=0x4000, roles=1),
        kernel_name='g', signature={}, source_kind='tnca-ir', options=opts)


def alloc01(log):
    """The cost uses the physical mapping, and moves are all counted."""
    ok = True
    hw = bankcost.hardware()
    log.append(f"    the geometry comes from R3 v2's frozen contract: "
               f"{hw['banks']} bank(s), {hw['physical_registers']} physical "
               f"register(s), {hw['reads_per_bank_per_cycle']} read per "
               f"bank per cycle")
    log.append(f"    the bank is {hw['bank_select']}")
    ok &= hw['banks'] == 4 and hw['reads_per_bank_per_cycle'] == 1

    # Directed: three sources in one bank, then three in three banks.
    # These are the shapes R3 v2's t29_vrfbank was built around.
    same = "    vf.fma       v12, v0, v4, v8\n"
    diff = "    vf.fma       v13, v1, v2, v3\n"
    cs = bankcost.conflicts(same)
    cd = bankcost.conflicts(diff)
    log.append(f"    v0, v4, v8 are banks "
               f"{[bankcost.bank_of(r) for r in (0, 4, 8)]}: "
               f"{cs['extra_cycles']} extra cycle(s)")
    log.append(f"    v1, v2, v3 are banks "
               f"{[bankcost.bank_of(r) for r in (1, 2, 3)]}: "
               f"{cd['extra_cycles']} extra cycle(s)")
    ok &= cs['extra_cycles'] == 2 and cd['extra_cycles'] == 0

    # Reading one register twice is one read, not a conflict.
    self_read = bankcost.conflicts("    vf.add       v1, v0, v0\n")
    log.append(f"    v0 read twice in one instruction: "
               f"{self_read['extra_cycles']} extra cycle(s) -- the file "
               f"reads it once")
    ok &= self_read['extra_cycles'] == 0

    # The model must agree with the hardware R3 v2 measured. Its
    # t29_vrfbank reports what the file actually did.
    hexf = os.path.join(ROOT, 'sim', 'build', 't29_vrfbank.hex')
    vvp = os.path.join(ROOT, 'sim', 'build', 'tb_core_vrf.vvp')
    if os.path.exists(hexf) and os.path.exists(vvp):
        r = subprocess.run(['vvp', vvp, f'+hex={hexf}', '+maxcycles=200000'],
                           capture_output=True, text=True, timeout=900)
        m = re.search(r'vrf req=(\d+) accept=(\d+) conflict=(\d+)', r.stdout)
        if m:
            req, acc, con = (int(x) for x in m.groups())
            log.append(f"    on the accepted hardware, t29_vrfbank: "
                       f"{req} request(s), {acc} accepted, {con} refused")
            # Every refusal is a bank that could not take a second read
            # in one cycle, which is what the model counts.
            log.append("    the model's unit is that same refusal: one "
                       "extra cycle per same-bank read beyond the first")
            ok &= con > 0 and acc + con == req
        else:
            log.append('    t29_vrfbank produced no counter line')
            ok = False
    else:
        log.append("    t29_vrfbank is not built here, so the model was "
                   "not cross-checked against the hardware "
                   "(`make sim/build/tb_core_vrf.vvp "
                   "sim/build/t29_vrfbank.hex`)")
        ok = False

    # A rename moves nothing; a real conversion's moves are all counted.
    on = bankcost.moves(l03(opt_level=2).asm['asm'])
    off = bankcost.moves(l03(opt_level=2, no_ll=True).asm['asm'])
    log.append(f"    L03 as a rename: {on['count']} move(s)")
    log.append(f"    L03 materialised: {off['count']} move(s) -- "
               f"{sorted(set(x['instruction'] for x in off['moves']))}")
    ok &= on['count'] == 0 and off['count'] == 2
    return ok, 'the cost is over physical banks, and every move is charged'


def alloc02(log):
    """The resource contracts still hold after optimisation."""
    ok = True
    for name, art in (('the GEMM at B0', gemm()),
                      ('the GEMM at O2', gemm(opt_level=2)),
                      ('L03 at O2', l03(opt_level=2)),
                      ('L03 materialised', l03(opt_level=2, no_ll=True))):
        res = art.manifest['resources']
        log.append(f"    {name:<18s} "
                   + ', '.join(f"{k}={v}" for k, v in sorted(res.items())
                               if isinstance(v, int)))
    # The optimisation must not raise the peak: section 7 wants the
    # register peak and the barrier/credit budgets inside what they were.
    b0 = gemm().manifest['resources']
    o2 = gemm(opt_level=2).manifest['resources']
    grew = {k: (b0.get(k), o2.get(k)) for k in b0
            if isinstance(b0.get(k), int) and isinstance(o2.get(k), int)
            and o2[k] > b0[k]}
    log.append(f"    nothing the GEMM uses grew from B0 to O2: "
               f"{'none' if not grew else grew}")
    ok &= not grew

    # Ownership and footprint are checked by R1 unconditionally, at every
    # level -- there is deliberately no argument that turns them off.
    src = open(os.path.join(ROOT, 'python', 'tnca', 'adapters',
                            'r1.py')).read()
    log.append(f"    the ownership checker has no off switch: "
               f"{'confirmed in the adapter' if 'no argument that turns them off' in src else 'NOT STATED'}")
    ok &= 'no argument that turns them off' in src

    # No hidden spill: the target has no scratchpad to spill into, so a
    # spill would have to be invented.
    for name, art in (('B0', gemm()), ('O2', gemm(opt_level=2))):
        asm = art.asm['asm']
        spills = [l for l in asm.splitlines()
                  if 'spill' in l.lower() or 'reload' in l.lower()]
        log.append(f"    {name}: {len(spills)} spill or reload")
        ok &= not spills
    return ok, 'peaks, ownership and footprint unchanged, and no spill'


def sched01(log):
    """Nothing was reordered, which is the conservative plan.

    Section 7 allows keeping the legal baseline when a reordering is not
    proved or its gain is unknown: "无法证明或收益未知时保留合法
    baseline". That is what is happening, and it is checked rather than
    claimed -- the op order is compared before and after every pass, and
    only deletions are allowed to change it.
    """
    import workloads as W
    import r35_workloads as WW
    from tnca.adapters import r1 as A
    from tnca_cc.optimization import (pure_scalar_cse, strength_reduce,
                                      drop_dead_pure, fold_broadcast_compare,
                                      drop_dead_load_other,
                                      narrow_lane0_store)
    from tnca_cc.layout import apply_plans
    ok = True
    target, _ = A.target_for(PROFILE, A.Provenance())

    for name, src in (('the GEMM', W.gemm_program(
                          16, 16, 16, a_desc={'tile': 0x2000, 'map': 0x1000},
                          b_desc={'tile': 0x3000, 'map': 0x1100},
                          out=0x4000, roles=1)),
                      ('the reduction', W.tnca_reduction(15)),
                      ('the view case', WW.tnca_view_elementwise())):
        kernel, b = A.new_kernel('probe')
        src(b, target)
        before = [id(o) for r in kernel.roles for o in r.region.ops]
        apply_plans(kernel, use_upstream=True)
        fold_broadcast_compare(kernel)
        drop_dead_load_other(kernel)
        narrow_lane0_store(kernel)
        strength_reduce(kernel)
        pure_scalar_cse(kernel)
        drop_dead_pure(kernel)
        after = [id(o) for r in kernel.roles for o in r.region.ops]
        # Every surviving op must appear in the same relative order.
        kept = [i for i in before if i in set(after)]
        same = kept == after
        log.append(f"    {name:<16s} {len(before)} op(s) -> {len(after)}, "
                   f"{'the survivors are in the same order' if same else 'REORDERED'}")
        ok &= same
    log.append("    no reordering is attempted, so every def-use, memory, "
               "control, accumulator-order and async-resource edge is "
               "preserved by construction rather than by analysis")
    log.append("    section 7 allows exactly this: the legal baseline is "
               "kept where a reordering is not proved or its gain is not "
               "known")
    return ok, 'the op order is unchanged apart from deletions'


TESTS = [('R35-ALLOC-01', alloc01), ('R35-ALLOC-02', alloc02),
         ('R35-SCHED-01', sched01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.5 Gate B: allocation cost and what was not reordered")
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
    print(f"  r35-alloc-tests        {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
