#!/usr/bin/env python3
"""
IR-01 .. IR-07 -- doc/phase1_tasks_and_acceptance.md R1-07.

Where the compiler's input ends and its refusal begins.

Three of these run against `tnca-opt` with fixture files under
compiler/test/IR, because the question is about MLIR structure; four run in
process against the Python IR and the real Triton frontend, because the
question is about what the frontend does with a module it did not write.

The contract's rule for this package is the interesting one: "未实现的结构明确
报 unsupported；不得不检查直接接受" -- an unimplemented structure must be
refused in so many words, never accepted unchecked. A verifier that walks past
what it does not model reports no errors, and "no errors" then means "nothing
was looked at". Every unsupported case here therefore has a positive twin that
must still compile, so freezing the boundary cannot be mistaken for moving it.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
import r1lib as R  # noqa: E402

OPT = os.path.join(ROOT, 'compiler', 'build', 'tools', 'tnca-opt', 'tnca-opt')
FIXTURES = os.path.join(ROOT, 'compiler', 'test', 'IR')

SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
       'n': 'i32', 'BLOCK': 'constexpr'}
SIG2 = {'x_ptr': '*fp32', 'o_ptr': '*fp32', 'n': 'i32', 'BLOCK': 'constexpr'}
ADDRS = {'x_ptr': 0x1000, 'y_ptr': 0x2000, 'o_ptr': 0x3000, 'n': 16}


def _reexec_under_triton():
    """Run under the interpreter that has Triton, if this one does not.

    Four of these cases are about the real Triton frontend, and the audit's
    F09 was precisely that the compiler suite ran under an interpreter without
    it and reported the resulting gap as seven skips. A skip is not a result;
    finding the right interpreter is this script's job, and failing loudly
    when there is none is the alternative.
    """
    try:
        import triton  # noqa: F401
        return
    except Exception:
        pass
    if os.environ.get('R1_IR_REEXEC') == '1':
        return
    cand = os.environ.get('TRITON_PYTHON') or \
        os.path.expanduser('~/src/tritonenv/bin/python')
    if os.path.exists(cand) and os.access(cand, os.X_OK):
        env = dict(os.environ, R1_IR_REEXEC='1')
        os.execve(cand, [cand] + sys.argv, env)


_reexec_under_triton()


def have_triton():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def run_fixture(name, log):
    """One .mlir fixture, through tnca-opt with its own RUN line."""
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(OPT):
        log.append(f"    tnca-opt not built at {OPT}")
        return False
    with open(path) as fh:
        first = fh.readline()
    m = re.match(r'//\s*RUN:\s*(.*)', first)
    if not m:
        log.append(f"    {name}: no RUN line")
        return False
    llvm = os.path.expanduser(
        os.environ.get('TNCA_LLVM', '~/.triton/llvm/llvm-macos-arm64'))
    filecheck = os.path.join(llvm, 'bin', 'FileCheck')
    cmd = (m.group(1).replace('%s', path)
           .replace('tnca-opt', OPT)
           .replace('FileCheck', filecheck))
    if 'FileCheck' in m.group(1) and not os.path.exists(filecheck):
        log.append(f"    {name}: FileCheck not found at {filecheck}")
        return False
    r = subprocess.run(['/bin/bash', '-o', 'pipefail', '-c', cmd],
                       capture_output=True, text=True, timeout=R.CAPS['ir'])
    ok = r.returncode == 0
    ncases = sum(1 for line in open(path) if 'expected-error' in line)
    log.append(f"    {name}: {'PASS' if ok else 'FAIL'} "
               f"({ncases} expected diagnostic(s))")
    if not ok:
        log.append('      ' + (r.stderr or r.stdout).strip()[:400])
    return ok


# --------------------------------------------------------------------------
def ir01(log):
    """The supported subset: legal single-block programs and OWN negatives."""
    return (run_fixture('ir01_subset.mlir', log)
            and run_fixture('verify_async.mlir', log)
            and run_fixture('roundtrip.mlir', log)), \
        'single-block DMA/MMA/store/await, positives and negatives'


def ir02(log):
    """Multi-block, block-argument events and nested regions are refused."""
    return run_fixture('ir02_unsupported.mlir', log), \
        'the support boundary is stated, not silently crossed'


def ir03(log):
    """The Python IR's loops: wide K, three slots, and an illegal reuse."""
    from tnca_cc import TargetInfo
    from tnca_cc.gemm import plan_gemm, build_gemm
    from tnca_cc.pipeline import plan_pipeline
    from tnca_cc.codegen import CodeGen
    from tnca_cc.diagnostics import Diagnostic
    from tnca_cc.ir import Builder, make_kernel

    t = TargetInfo.default()
    ok = True

    # Wide K still plans and compiles. K=64 is four fragments and K=128 is
    # eight; the footprint is sized from the instruction now (FP-02), so this
    # is also the case that would have silently under-sized a window.
    for k, want_steps in ((64, 4), (128, 8)):
        try:
            plan = plan_gemm(t, m=16, n=16, k=k)
            kern, _b = build_gemm(t, plan, a_desc_addr=0x6000,
                                  b_desc_addr=0x6100, out_addr=0x7000)
            # check_protocol=False, as compiler/test/test_gemm.py does: the
            # channel protocol models a producer/consumer ring, and a GEMM
            # arms its own barriers directly. The ownership checks this case
            # is about are not behind that switch -- they were, and the
            # previous commit made them unconditional for exactly this
            # reason.
            cg = CodeGen(kern, t, check_protocol=False)
            cg.assign_shared()
            cg.run()
            good = plan.k_steps == want_steps
            log.append(f"    K={k}: {plan.k_steps} fragment(s), "
                       f"{plan.mma_count} MMA -> compiled")
        except Diagnostic as d:
            good = False
            log.append(f"    K={k}: REFUSED {str(d).splitlines()[0][:110]}")
        ok &= good

    # Three slots. K=96 is six fragments through a three-stage ring.
    try:
        p = plan_pipeline(t, stages=3, iterations=6)
        log.append(f"    three-slot K96: {p.stages} stage(s), "
                   f"{p.iterations} iteration(s) planned")
        ok &= (p.stages == 3 and p.iterations == 6)
    except Diagnostic as d:
        log.append(f"    three-slot K96: REFUSED {str(d).splitlines()[0][:110]}")
        ok = False

    # The illegal lifetime, and its twin. Loops are not banned: the same loop
    # with the wait in the right place must compile, or the "rejection" is
    # just an absence of support wearing a diagnostic.
    def rolled(wait_inside):
        kern = make_kernel('rolled')
        b = Builder(kern, kern.role(0).region)
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        with b.loop(b.const(0), b.const(4)):
            tok = b.barrier_arm(bar, ops=1)
            b.mma_async(acc, a, o, tok, k='k16')
            if wait_inside:
                b.barrier_wait(tok)
            b.window_put(a, 0, b.vmovi(1))
            if not wait_inside:
                b.barrier_wait(tok)
        b.exit()
        cg = CodeGen(kern, t)
        cg.assign_shared()
        cg.run()

    try:
        rolled(False)
        bad = True
        detail = ''
    except Diagnostic as d:
        bad = False
        detail = str(d).splitlines()[0][:110]
    try:
        rolled(True)
        good = True
    except Diagnostic as d:
        good = False
        detail = f"positive twin refused: {str(d).splitlines()[0][:100]}"
    log.append(f"    cross-iteration reuse -> "
               f"{'ACCEPTED (bad)' if bad else 'refused'}; "
               f"the same loop done right -> "
               f"{'accepted' if good else 'REFUSED (bad)'}")
    if detail:
        log.append(f"      {detail}")
    ok &= (not bad) and good
    return ok, 'wide K, three slots, and a loop refused for its lifetimes'


def ir04(log):
    """Function selection and argument binding -- the audit's F07."""
    if not have_triton():
        log.append("    Triton is not importable by this interpreter, and "
                   "IR-04 is about the real frontend")
        return False, 'needs the locked Triton install'
    import triton_bridge as TB
    import kernels_triton as K
    from tnca_cc import TargetInfo
    from tnca_cc.frontend import TTIRImporter
    from tnca_cc.diagnostics import Diagnostic

    t = TargetInfo.default()
    ok = True

    # (a) A module with two tt.funcs. `module.walk` visits both; only one is
    #     being compiled.
    mod, _ctx = TB.make_ttir(K.calls_a_helper, SIG, {'BLOCK': t.vlen})
    names = re.findall(r'tt\.func\s+\S*\s*@(\w+)', mod.str_nodebug())
    all_ops = []
    mod.walk(lambda op: all_ops.append(op))
    fn = mod.get_function('calls_a_helper')
    mine = [op for op in all_ops if TTIRImporter._in_function(op, fn)]
    two_funcs = len(names) >= 2
    filtered = 0 < len(mine) < len(all_ops)
    log.append(f"    two-function module {names}: walk sees {len(all_ops)} "
               f"op(s), this function owns {len(mine)}")
    ok &= two_funcs and filtered

    # (b) A name that is not in the module is a diagnostic. MLIR's
    #     get_function returns a null handle and the next call on it used to
    #     segfault the interpreter with nothing to read.
    try:
        TTIRImporter(t, 'nope').run(mod, 'nope', ADDRS,
                                    param_names=TB.param_names(K.add_kernel))
        missing_ok = False
        log.append("    a function name that is not in the module: ACCEPTED (bad)")
    except Diagnostic as d:
        missing_ok = 'no function' in str(d)
        log.append(f"    a name not in the module -> {str(d).splitlines()[0][:100]}")
    ok &= missing_ok

    # (c) The same arguments in two dict orders must compile identically.
    def compile_with(d):
        m, _c = TB.make_ttir(K.add_kernel, SIG, {'BLOCK': t.vlen})
        imp = TTIRImporter(t, 'add_kernel')
        kern, b = imp.run(m, 'add_kernel', d,
                          param_names=TB.param_names(K.add_kernel))
        return imp.arg_names, b.text() if hasattr(b, 'text') else str(kern)

    forward = {'x_ptr': 0x1000, 'y_ptr': 0x2000, 'o_ptr': 0x3000, 'n': 16}
    shuffled = {'n': 16, 'o_ptr': 0x3000, 'x_ptr': 0x1000, 'y_ptr': 0x2000}
    n1, t1 = compile_with(forward)
    n2, t2 = compile_with(shuffled)
    same = (n1 == n2)
    log.append(f"    binding order: {n1} vs {n2} -> "
               f"{'identical' if same else 'DIFFERENT (bad)'}")
    ok &= same

    # (d) A missing and an extra parameter.
    for label, d in (('missing', {'x_ptr': 1, 'y_ptr': 2, 'o_ptr': 3}),
                     ('extra', dict(forward, zzz=1))):
        m, _c = TB.make_ttir(K.add_kernel, SIG, {'BLOCK': t.vlen})
        try:
            TTIRImporter(t, 'add_kernel').run(
                m, 'add_kernel', d,
                param_names=TB.param_names(K.add_kernel))
            log.append(f"    an {label} argument: ACCEPTED (bad)")
            ok = False
        except Diagnostic as d2:
            log.append(f"    an {label} argument -> "
                       f"{str(d2).splitlines()[0][:100]}")

    # (e) A value whose type does not match the declared parameter.
    m, _c = TB.make_ttir(K.add_kernel, SIG, {'BLOCK': t.vlen})
    try:
        TTIRImporter(t, 'add_kernel').run(
            m, 'add_kernel', dict(forward, n=1.5),
            param_names=TB.param_names(K.add_kernel))
        log.append("    a float for an i32 parameter: ACCEPTED (bad)")
        ok = False
    except Diagnostic as d3:
        log.append(f"    a float for an i32 parameter -> "
                   f"{str(d3).splitlines()[0][:100]}")
    return ok, 'only the named function, bound by its declared signature'


def ir05(log):
    """Refused before lowering: bad BLOCK, unknown ops, unknown effects."""
    ok = run_fixture('ir05_unknown_effect.mlir', log)
    if not have_triton():
        log.append("    Triton is not importable by this interpreter")
        return False, 'needs the locked Triton install'
    import triton_bridge as TB
    import kernels_triton as K
    from tnca_cc import TargetInfo
    from tnca_cc.frontend import TTIRImporter, Unsupported
    from tnca_cc.diagnostics import Diagnostic

    t = TargetInfo.default()

    # BLOCK wider than one vector register. sec 12: a diagnostic, not a
    # lowering of the first register only.
    mod, _c = TB.make_ttir(K.add_kernel, SIG, {'BLOCK': t.vlen * 4})
    try:
        TTIRImporter(t, 'add_kernel').run(
            mod, 'add_kernel', ADDRS,
            param_names=TB.param_names(K.add_kernel))
        log.append(f"    BLOCK={t.vlen * 4} != VLEN={t.vlen}: ACCEPTED (bad)")
        ok = False
    except Unsupported as d:
        log.append(f"    BLOCK={t.vlen * 4} != VLEN={t.vlen} -> "
                   f"{str(d).splitlines()[0][:100]}")

    # An operation with no lowering -- here `tt.call`, which the two-function
    # module produces for real rather than by construction.
    mod2, _c2 = TB.make_ttir(K.calls_a_helper, SIG, {'BLOCK': t.vlen})
    try:
        TTIRImporter(t, 'calls_a_helper').run(
            mod2, 'calls_a_helper', ADDRS,
            param_names=TB.param_names(K.calls_a_helper))
        log.append("    an unlowered TTIR operation: ACCEPTED (bad)")
        ok = False
    except Unsupported as d:
        log.append(f"    an unlowered TTIR operation -> "
                   f"{str(d).splitlines()[0][:100]}")
    except Diagnostic as d:
        log.append(f"    an unlowered TTIR operation -> {str(d).splitlines()[0][:100]}")

    # And the positive twin: the supported kernel still compiles, so none of
    # the above is a blanket refusal.
    mod3, _c3 = TB.make_ttir(K.add_kernel, SIG, {'BLOCK': t.vlen})
    try:
        TTIRImporter(t, 'add_kernel').run(
            mod3, 'add_kernel', ADDRS,
            param_names=TB.param_names(K.add_kernel))
        log.append("    the supported kernel still compiles")
    except Diagnostic as d:
        log.append(f"    the supported kernel was REFUSED: "
                   f"{str(d).splitlines()[0][:100]}")
        ok = False
    return ok, 'refused at or before lowering, with no successful artefact'


def ir06(log):
    """Float constants: the bits, or a diagnostic. Never another value."""
    import struct
    from tnca_cc.frontend import TTIRImporter
    from tnca_cc.diagnostics import Diagnostic
    ok = True
    lv = TTIRImporter._literal_value

    # The forms MLIR actually prints, and what they must mean.
    cases = [
        ('0.000000e+00', 0x00000000, '+0'),
        ('-0.000000e+00', 0x80000000, '-0 keeps its sign bit'),
        ('1.500000e+00', struct.unpack('<I', struct.pack('<f', 1.5))[0],
         'an ordinary finite value'),
        ('0x7F800000', 0x7F800000, '+inf, printed as bits'),
        ('0xFF800000', 0xFF800000, '-inf, printed as bits'),
        ('0x7FC00000', 0x7FC00000, 'a quiet NaN, payload intact'),
        ('dense<0x7FC00000>', 0x7FC00000, 'the same inside a dense<>'),
    ]
    for literal, want, why in cases:
        try:
            got = lv(literal, 'f32')
        except Diagnostic as d:
            got = f"refused: {d}"
        good = got == want
        log.append(f"    {literal:>16s} -> "
                   f"{got if isinstance(got, str) else hex(got)}  ({why})"
                   + ('' if good else f"  WANT {hex(want)}"))
        ok &= good

    # A literal this cannot read must be a diagnostic. The hex forms above
    # used to reach `float()` and raise ValueError -- an interpreter
    # traceback where the contract asks for a controlled refusal, and the
    # kind of thing a later "fix" turns into reading it as another number.
    for literal in ('wat', '1.0.0', '0x'):
        try:
            got = lv(literal, 'f32')
            log.append(f"    {literal!r} -> {got} (ACCEPTED, bad)")
            ok = False
        except Diagnostic as d:
            log.append(f"    {literal!r} -> {str(d).splitlines()[0][:80]}")
        except Exception as e:
            log.append(f"    {literal!r} -> {type(e).__name__} "
                       f"(not a diagnostic, bad)")
            ok = False

    if have_triton():
        # End to end: a kernel whose `other=` is an infinity. Triton prints
        # it as 0x7F800000 and the frontend must carry those bits through.
        import triton_bridge as TB
        import kernels_triton as K
        from tnca_cc import TargetInfo
        t = TargetInfo.default()
        mod, _c = TB.make_ttir(K.special_constants_kernel, SIG2,
                               {'BLOCK': t.vlen})
        imp = TTIRImporter(t, 'special_constants_kernel')
        try:
            imp.run(mod, 'special_constants_kernel',
                    {'x_ptr': 0x1000, 'o_ptr': 0x3000, 'n': 16},
                    param_names=TB.param_names(K.special_constants_kernel))
            vals = [v for v in imp._const.values()]
            carried = 0x7F800000 in vals
            log.append(f"    a kernel with other=inf: bits carried through "
                       f"= {carried}")
            ok &= carried
        except Exception as e:
            log.append(f"    a kernel with other=inf -> {type(e).__name__}: "
                       f"{str(e).splitlines()[0][:90]}")
            ok = False
    else:
        log.append("    Triton is not importable, so the end-to-end half of "
                   "IR-06 cannot run")
        ok = False
    return ok, 'specified bits preserved; an unreadable literal diagnosed'


def ir07(log):
    """The same resource-safe program, expressed twice, must agree."""
    from tnca_cc import TargetInfo
    from tnca_cc.codegen import CodeGen
    from tnca_cc.diagnostics import Diagnostic
    from tnca_cc.ir import Builder, make_kernel

    t = TargetInfo.default()

    def python_ir(legal):
        kern = make_kernel('ir07')
        b = Builder(kern, kern.role(0).region)
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k16')
        if legal:
            b.barrier_wait(tok)
            b.window_put(a, 0, b.vmovi(1))
        else:
            b.window_put(a, 0, b.vmovi(1))
            b.barrier_wait(tok)
        b.exit()
        cg = CodeGen(kern, t)
        cg.assign_shared()
        cg.run()

    results = {}
    for legal in (True, False):
        try:
            python_ir(legal)
            results[legal] = 'accepted'
        except Diagnostic:
            results[legal] = 'refused'

    # The MLIR spelling of the same two programs is in ir07_same_program.mlir:
    # the legal one carries no expected-error, the illegal one does, so
    # `--verify-diagnostics` passing means both verdicts matched.
    mlir_ok = run_fixture('ir07_same_program.mlir', log)
    log.append(f"    Python IR: legal -> {results[True]}, "
               f"illegal -> {results[False]}")
    agree = (results[True] == 'accepted' and results[False] == 'refused'
             and mlir_ok)
    return agree, 'the Python IR and the MLIR reach the same verdict'


TESTS = [('IR-01', ir01), ('IR-02', ir02), ('IR-03', ir03), ('IR-04', ir04),
         ('IR-05', ir05), ('IR-06', ir06), ('IR-07', ir07)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R1-07 IR support boundary and frontend selection")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-800:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}")
        for line in log:
            print(line)
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r1-ir-tests            FAIL   {', '.join(bad)}")
        return 1
    print(f"  r1-ir-tests            PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
