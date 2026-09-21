#!/usr/bin/env python3
"""
R35-OPT-01, OPT-02 and OPT-03 -- phase3_5_tasks_and_acceptance.md sec 7.

That the optimisation is real, that its facts are proved rather than
assumed, and that it is safe.

OPT-03 is the one that matters most. An optimisation that is merely
fast is worth nothing if it moved a store, deleted something that could
fault, or changed a floating-point result. Everything below is checked
by running the same program both ways and comparing, rather than by
reading the passes and believing them.
"""
import argparse
import json
import os
import struct
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

PROFILE = json.load(open(os.path.join(
    ROOT, 'spec', 'r35_workloads.json')))['hardware_precondition']['profile']

# Operations that carry an effect. None of these may be removed, moved
# across one another, or duplicated by anything in optimization.py.
EFFECTFUL = ('store', 'sstore', 'load', 'gather', 'scatter', 'dma_load',
             'dma_store', 'barrier_arm', 'barrier_wait', 'barrier_arrive',
             'acc_zero', 'acc_read', 'mma', 'exit', 'trap', 'desc_snapshot')


def build_axpy(**opts):
    import workloads as W
    from tnca import aot
    return aot.compile_aot(
        W.triton_masked_axpy(), profile=PROFILE,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 16}, options=opts,
        output=os.path.join(tempfile.mkdtemp(), 'a.tnca'))


def opt01(log):
    """Upstream canonicalisation and CSE really run, and change the code."""
    from tnca_cc.upstream.passes import PureValueOptimizer
    ok = True

    base = build_axpy()
    at_o1 = build_axpy(opt_level=1)
    ttir = base.asm['ttir']
    after, diff = PureValueOptimizer().run(ttir)
    log.append(f"    the TTIR of a real Triton kernel: "
               f"{diff['ops_before']} value op(s) -> {diff['ops_after']}, "
               f"removing {diff['removed']}")
    ok &= bool(diff['ops_after'] < diff['ops_before']
               and diff['removed'])

    # The source was not written to give CSE work: the duplicates are
    # what the Triton frontend emits for an ordinary masked kernel.
    import re
    consts = re.findall(r'=\s+arith\.constant\s+(\S+)\s*:\s*(\S+)', ttir)
    from collections import Counter
    dup = [f"{v} : {t}" for (v, t), n in Counter(consts).items() if n > 1]
    log.append(f"    the duplicates it removed are the frontend's own: "
               f"{dup}")
    ok &= bool(dup)

    # And the machine code is different because of it.
    log.append(f"    the kernel is {len(base.asm['hex'].splitlines())} "
               f"word(s) at B0 and "
               f"{len(at_o1.asm['hex'].splitlines())} at O1")
    ok &= base.asm['hex'] != at_o1.asm['hex']

    # The optimised IR is dumped, so the before and after can be read.
    dumped = build_axpy(opt_level=1, dump_dir=tempfile.mkdtemp())
    log.append(f"    the stages recorded are "
               f"{sorted(dumped.asm)}, so the input and the optimised IR "
               f"are both readable")
    ok &= 'ttir' in dumped.asm and 'ttir_opt' in dumped.asm
    return ok, 'upstream runs on the product path and the code changes'


def opt02(log):
    """AxisInfo driving a lowering choice.

    This one cannot be satisfied from the pinned wheel, and the reason is
    specific rather than a shrug.
    """
    from triton._C import libtriton as L
    names = [n for n in dir(L.passes.analysis) if not n.startswith('_')]
    log.append(f"    libtriton.passes.analysis exports {names}")
    axis = [n for n in names if 'axis' in n.lower()]
    log.append(f"    AxisInfo among them: "
               f"{axis if axis else 'NO'}")
    if axis:
        return False, 'AxisInfo is exported after all; this check is stale'

    log.append("    Section 2 asks for the LOCKED VERSION's TTIR analysis "
               "to be reused, and the wheel does not expose it. Deriving "
               "the same facts in Python would be the replacement "
               "implementation section 4.2 forbids, so it is not done.")
    log.append("    The route section 4.2 describes is a separate tool "
               "built against Triton's OWN pinned LLVM "
               "(compiler/pin/toolchain.json names both: triton v3.8.0 at "
               "c01b6774, llvm at 5f07f818), speaking versioned text. "
               "tnca-opt cannot host it: it links distribution MLIR 20, "
               "and section 4.2 forbids the two LLVMs meeting in one "
               "process.")
    # A second route was tried and does not work either, recorded so
    # the next person does not spend the afternoon on it. Upstream's
    # TTIR->TTGIR conversion consults AxisInfo when it coalesces, so its
    # chosen layout should in principle reflect what the analysis proved.
    # It converts happily for this target -- threadsPerWarp 16,
    # warpsPerCTA 1, which is exactly TNCA's sixteen lanes and one role
    # -- and then `add_coalesce` leaves sizePerThread at 1 for a
    # provably contiguous access AND for a stride-7 one, at BLOCK 16 and
    # BLOCK 64 alike. The conclusion is not observable in the layout, so
    # there is no fact to consume through this door.
    log.append("    a second route was tried: upstream's TTIR->TTGIR "
               "conversion consults AxisInfo when it coalesces. It "
               "converts for this target, and then add_coalesce leaves "
               "sizePerThread at 1 for a contiguous access and for a "
               "stride-7 one alike, at BLOCK 16 and 64. The conclusion "
               "is not observable in the layout.")
    log.append("    Until the tool exists this is NOT VERIFIED. Nothing "
               "in the product currently claims an alignment or "
               "contiguity fact, so nothing is relying on an assumption "
               "either.")

    # The claim in that last line is checked, not asserted.
    src = open(os.path.join(ROOT, 'compiler', 'python', 'tnca_cc',
                            'optimization.py')).read()
    claims = [w for w in ('alignment', 'contiguous', 'contiguity',
                          'divisible', 'axisinfo') if w in src.lower()]
    log.append(f"    the optimisation passes claim no such fact: "
               f"{'none found' if not claims else 'FOUND ' + str(claims)}")
    return False, ('the pinned wheel does not export AxisInfo; the route '
                   'is named and nothing assumes its facts meanwhile')


def opt03(log):
    """Nothing effectful is touched, and floating point is unchanged."""
    import workloads as W
    import r35_workloads as WW
    from tnca import aot
    from tnca.adapters import r1 as A
    from tnca_cc.optimization import (pure_scalar_cse, strength_reduce,
                                      drop_dead_pure, fold_broadcast_compare,
                                      drop_dead_load_other,
                                      narrow_lane0_store)
    ok = True
    target, _ = A.target_for(PROFILE, A.Provenance())

    # 1. Every effectful operation survives every pass, in the same order.
    def census(kernel):
        return [o.kind for r in kernel.roles for o in r.region.ops
                if o.kind in EFFECTFUL]

    for name, src in (('the GEMM', W.gemm_program(
                          16, 16, 16, a_desc={'tile': 0x2000, 'map': 0x1000},
                          b_desc={'tile': 0x3000, 'map': 0x1100},
                          out=0x4000, roles=1)),
                      ('the reduction', W.tnca_reduction(15)),
                      ('the view case', WW.tnca_view_elementwise())):
        kernel, b = A.new_kernel('probe')
        src(b, target)
        before = census(kernel)
        fold_broadcast_compare(kernel)
        drop_dead_load_other(kernel)
        narrow_lane0_store(kernel)
        strength_reduce(kernel)
        pure_scalar_cse(kernel)
        drop_dead_pure(kernel)
        after = census(kernel)
        # narrow_lane0_store turns one store into a scalar store of the
        # same word; that is the only rewrite allowed to change this list.
        norm = [k.replace('sstore', 'store') for k in after]
        same = norm == [k.replace('sstore', 'store') for k in before]
        log.append(f"    {name:<16s} {len(before)} effectful op(s), "
                   f"{'same ops in the same order after every pass' if same else 'CHANGED: ' + str(before) + ' -> ' + str(after)}")
        ok &= same

    # 2. Floating-point special values survive. A masked AXPY carrying
    #    NaN, +/-Inf, -0.0 and a subnormal, run at B0 and at O2, compared
    #    bit for bit -- not by value, because -0.0 == 0.0 and NaN != NaN.
    from tnca import device
    from tnca.runtime import Session
    device.prepare(PROFILE, verbose=False)
    specials = [float('nan'), float('inf'), float('-inf'), -0.0,
                5e-324, 1.4e-45, 0.0, 1.0,
                -1.5, 3.4e38, -3.4e38, 1e-40, 2.5, -2.5, 7.0, -7.0]
    ones = [1.0] * 16
    bits = {}
    for tag, opts in (('B0', {}), ('O2', {'opt_level': 2})):
        art = aot.compile_aot(
            W.triton_masked_axpy(), profile=PROFILE,
            signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                       'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
            constexprs={'BLOCK': 16}, options=opts,
            output=os.path.join(tempfile.mkdtemp(), f'{tag}.tnca'))
        with Session(profile=PROFILE, deadline_cycles=50000) as s:
            mod = s.load(art.path)
            x = s.allocate(64, name='x')
            y = s.allocate(64, name='y')
            o = s.allocate(64, name='o')
            s.upload(x, b''.join(struct.pack('<f', v) for v in specials))
            s.upload(y, b''.join(struct.pack('<f', v) for v in ones))
            s.upload(o, struct.pack('<I', W.SENTINEL) * 16)
            ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                'n': 16, 'alpha': 1.0}, grid=(1, 1, 1))
            s.wait(ev)
            bits[tag] = s.download(o, 64)
    same = bits['B0'] == bits['O2']
    log.append(f"    NaN, +/-Inf, -0.0 and subnormals through a masked "
               f"AXPY: B0 and O2 agree bit for bit: {same}")
    if not same:
        for i in range(16):
            a = struct.unpack_from('<I', bits['B0'], i * 4)[0]
            c = struct.unpack_from('<I', bits['O2'], i * 4)[0]
            if a != c:
                log.append(f"        lane {i}: {a:#010x} != {c:#010x}")
    ok &= same

    # 3. No reassociation and no new contraction were switched on.
    from tnca_cc.upstream.passes import PureValueOptimizer
    pipeline = PureValueOptimizer().pipeline()
    log.append(f"    the pass pipeline is exactly {pipeline}")
    ok &= set(pipeline) <= {'canonicalize', 'cse'}
    cg = open(os.path.join(ROOT, 'compiler', 'python', 'tnca_cc',
                           'optimization.py')).read()
    bad = [w for w in ('reassoc', 'fastmath', 'fast_math', 'ffast',
                       'contract') if w in cg.lower()]
    log.append(f"    nothing enables reassociation or contraction: "
               f"{'none found' if not bad else 'FOUND ' + str(bad)}")
    ok &= not bad
    return ok, 'effects intact, floating point unchanged bit for bit'


TESTS = [('R35-OPT-01', opt01), ('R35-OPT-02', opt02), ('R35-OPT-03', opt03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.5 Gate B: the optimisation, and whether it is safe")
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
    print(f"  r35-opt-tests          {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
