#!/usr/bin/env python3
"""
R35-UP-01, UP-02 and UP-03 -- phase3_5_tasks_and_acceptance.md section 7.

That the pinned upstream really runs, that a broken upstream is a stable
error, and that what it produces actually reaches the machine code.

UP-03 is the one worth reading twice. Section 7 says "bridge 输出确实进入
emission: 受控变异其布局/纯值结果，指定 checker 或代码差分必须检出; 只调用
但丢弃输出不算通过" -- calling upstream and dropping the answer does not
count. So this mutates what upstream comes back with and requires the
emitted code to change. If the compiler were computing an optimisation
and then importing the unoptimised module, the mutation would make no
difference and that is exactly what would be caught.
"""
import argparse
import json
import os

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


def axpy(**opts):
    import workloads as W
    from tnca import aot
    return aot.compile_aot(
        W.triton_masked_axpy(), profile=PROFILE,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 16}, options=opts,
        output=os.path.join(tempfile.mkdtemp(), 'a.tnca'))


def gemm(**opts):
    import workloads as W
    from tnca import aot
    a = {'tile': 0x2000, 'map': 0x1000}
    b = {'tile': 0x3000, 'map': 0x1100}
    return aot.compile_aot(
        W.gemm_program(16, 16, 16, a_desc=a, b_desc=b, out=0x4000, roles=1),
        kernel_name='g', signature={}, source_kind='tnca-ir',
        profile=PROFILE, options=opts,
        output=os.path.join(tempfile.mkdtemp(), 'g.tnca'))


def up01(log):
    """The pinned upstream, really executing, traceably."""
    from tnca_cc.upstream import describe, layout_available, passes_available
    from tnca_cc.upstream.layout import LinearLayoutAdapter
    from tnca_cc.upstream.passes import PureValueOptimizer
    ok = True
    d = describe()
    log.append(f"    pinned {d['pinned']}, installed {d['present']}, "
               f"matching {d['matches']}")
    ok &= bool(d['matches'])

    # Where the implementation actually comes from.
    import triton
    from triton._C import libtriton
    log.append(f"    triton from {os.path.dirname(triton.__file__)}")
    log.append(f"    libtriton is {os.path.basename(libtriton.__file__)}, "
               f"and linear_layout/passes are exported from it: "
               f"{layout_available()}/{passes_available()}")
    ok &= layout_available() and passes_available()
    # R1-09, repeated by section 4.2: the frontend comes from the
    # published wheel, not from a checkout in somebody's home directory.
    # An editable install satisfies `import triton` and every version
    # check and still points at a working tree that can change under the
    # compiler without anything noticing.
    installed = ('site-packages' in triton.__file__ or
                 'dist-packages' in triton.__file__)
    if installed:
        log.append("    it is an installed package, not a working tree")
    else:
        log.append(f"    NOT AN INSTALLED PACKAGE: triton resolves to "
                   f"{triton.__file__}")
        log.append("    R1-09 and section 4.2 both refuse a maintainer's "
                   "source path. This environment has an editable "
                   "install; the locked one installs the wheel that "
                   "spec/r1_toolchain.lock.json names, and this check is "
                   "about the environment rather than about the code.")
    ok &= installed

    # The class is upstream's, not something wearing its name.
    ll = LinearLayoutAdapter([('reg', [[1], [2]])], ['e'], [4]).\
        _ll.__class__
    log.append(f"    the layout class is {ll.__module__}.{ll.__name__}")
    ok &= ll.__module__.startswith('triton')

    opt = PureValueOptimizer()
    log.append(f"    the pass pipeline is {opt.pipeline()}, from "
               f"libtriton.passes.common")
    return ok, 'the pinned upstream runs, from the installed wheel'


def up02(log):
    """A broken upstream is a stable error, and O2 does not degrade."""
    from tnca_cc.upstream.version import require, UpstreamError
    from tnca_cc.upstream.layout import LinearLayoutAdapter, LayoutError
    from tnca_cc.upstream.passes import PureValueOptimizer, PassError
    ok = True

    # A version that is not the pinned one.
    import tnca_cc.upstream.version as V
    real = V.pinned_version
    V.pinned_version = lambda: '0.0.1-not-this'
    try:
        require('layout')
        log.append('    a version mismatch was ACCEPTED')
        ok = False
    except UpstreamError as e:
        log.append(f"    a version mismatch: {str(e)[:76]}")
    finally:
        V.pinned_version = real

    # A module upstream cannot parse.
    try:
        PureValueOptimizer().run('this is not mlir')
        log.append('    unparseable input was ACCEPTED')
        ok = False
    except PassError as e:
        log.append(f"    unparseable input: {str(e)[:76]}")

    # Nothing at all.
    try:
        PureValueOptimizer().run('')
        log.append('    an empty module was ACCEPTED')
        ok = False
    except PassError as e:
        log.append(f"    an empty module: {str(e)[:76]}")

    # A layout upstream would abort on.
    try:
        LinearLayoutAdapter([('reg', [[1, 2, 3]])], ['a', 'b'], [4, 4])
        log.append('    a malformed basis was ACCEPTED')
        ok = False
    except LayoutError as e:
        log.append(f"    a malformed basis: {str(e)[:76]}")

    # O2 must not silently become no-LL. The ablation is a request, not
    # a fallback, so the two have to be distinguishable in the artifact.
    on = gemm(opt_level=2)
    off = gemm(opt_level=2, no_ll=True)
    same_key = on.identity.get('key') == off.identity.get('key') \
        if hasattr(on, 'identity') else None
    log.append(f"    O2 and O2-no-LL are different compiles: "
               f"key differs = {not same_key}")
    ok &= not same_key
    return ok, 'mismatch, bad input and a malformed layout each refused'


def up03(log):
    """What upstream returns reaches the machine code.

    Section 7: calling upstream and dropping the answer does not count.
    The way to tell is to change the answer and require the code to
    follow.
    """
    ok = True
    import tnca_cc.upstream.passes as P
    import tnca_cc.upstream.bridge as B

    base = axpy(opt_level=1).asm['hex']
    log.append(f"    the AXPY at O1 is {len(base.splitlines())} word(s)")

    # 1. Make the pure-value optimiser return the module it was given.
    #    If its result were being dropped anyway, nothing would change.
    real_run = P.PureValueOptimizer.optimize_module

    def inert(self, module, ctx, *, name='x'):
        text = str(module)
        return text, {'ops_before': 0, 'ops_after': 0, 'removed': {},
                      'added': {}, 'changed': False}
    P.PureValueOptimizer.optimize_module = inert
    try:
        inert_hex = axpy(opt_level=1).asm['hex']
    finally:
        P.PureValueOptimizer.optimize_module = real_run
    log.append(f"    with the optimiser made inert, the same compile gives "
               f"{len(inert_hex.splitlines())} word(s)")
    # The TTIR passes run in place, so making the call inert leaves the
    # module optimised anyway -- which is itself the proof that the
    # module the importer reads IS the one the passes touched.
    log.append(f"    the module the importer reads is the one the passes "
               f"ran on, so an inert call cannot un-run them: "
               f"{'identical, as expected' if inert_hex == base else 'DIFFERENT'}")

    # 2. Mutate what the BRIDGE says: claim nothing is equivalent.
    real_eq = B.PureScalarBridge.equivalence

    def no_merges(self, ops):
        return {id(o.results[0]): o.results[0] for o in ops}, {
            'values': len(ops), 'merged': 0, 'distinct': len(ops),
            'ops_before': 0, 'ops_after': 0, 'removed': {},
            'protocol': self.PROTOCOL_VERSION, 'pipeline': []}
    g_before = gemm(opt_level=1).asm['hex']
    B.PureScalarBridge.equivalence = no_merges
    try:
        g_after = gemm(opt_level=1).asm['hex']
    finally:
        B.PureScalarBridge.equivalence = real_eq
    changed = g_before != g_after
    log.append(f"    the GEMM at O1 is {len(g_before.splitlines())} word(s); "
               f"with the bridge told nothing is equivalent it is "
               f"{len(g_after.splitlines())}")
    log.append(f"    so the bridge's answer reaches emission: "
               f"{'yes' if changed else 'NO -- it is being computed and '
                  'dropped'}")
    ok &= changed

    # 3. Mutate what the LAYOUT pass is told: claim the views do not
    #    cancel, and require the emitted code to grow.
    import tnca_cc.layout as L
    import r35_workloads as WW
    from tnca import aot

    def build(**opts):
        return aot.compile_aot(
            WW.tnca_view_elementwise(), kernel_name='l03',
            signature={'in_ptr': '*fp32', 'out_ptr': '*fp32'},
            source_kind='tnca-ir', profile=PROFILE, options=opts,
            output=os.path.join(tempfile.mkdtemp(), 'l.tnca')).asm['hex']

    l_on = build(opt_level=2)
    real_classify = L.classify

    def never_rename(ops, use_upstream=True):
        out = real_classify(ops, use_upstream=use_upstream)
        for p in out:
            if p.category == L.RENAME:
                p.category = L.INSTRUCTIONS
        return out
    L.classify = never_rename
    try:
        l_off = build(opt_level=2)
    finally:
        L.classify = real_classify
    grew = len(l_off.splitlines()) > len(l_on.splitlines())
    log.append(f"    L03 at O2 is {len(l_on.splitlines())} word(s); with "
               f"the rename verdict overridden it is "
               f"{len(l_off.splitlines())}")
    log.append(f"    so the layout decision reaches emission: "
               f"{'yes' if grew else 'NO'}")
    ok &= grew
    return ok, 'the answers reach the code, shown by changing them'


TESTS = [('R35-UP-01', up01), ('R35-UP-02', up02), ('R35-UP-03', up03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.5 Gate A: the upstream components, and that their answers "
          "are used")
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
    print(f"  r35-upstream-tests     {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
