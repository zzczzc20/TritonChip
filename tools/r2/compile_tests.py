#!/usr/bin/env python3
"""
CMP-01 .. CMP-05 -- doc/phase2_tasks_and_acceptance.md R2-03.

The product compile entry point, through its public interface. Nothing here
reaches into the compiler's internals: if `tnca.aot.compile_aot` cannot do it,
it does not count.

CMP-01 runs the three kernels the contract names -- add, mul-add and an AXPY
with a dynamic f32 alpha -- on the RTL and compares against a reference
computed in Python. CMP-03 is the one worth reading twice: it feeds R1's own
negative cases in through the product path with debug and optimisation options
ON, because an option that turns a safety check off is a safety check that is
off.
"""
import argparse
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
import r1lib as R                                        # noqa: E402

import devsim                                            # noqa: E402
from tnca import aot                                     # noqa: E402
from tnca.errors import (CompileError, SignatureError,    # noqa: E402
                         TncaError, UnsupportedError)

SIG3 = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
        'n': 'i32', 'BLOCK': 'constexpr'}
SIG_AXPY = dict(SIG3)
SIG_AXPY['alpha'] = 'fp32'
SIG_AXPY = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
            'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'}

N = 16
_TMP = None


def tmp():
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix='r2-cmp-')
    return _TMP


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def _compile(fn, sig, name):
    return aot.compile_aot(fn, signature=sig, constexprs={'BLOCK': N},
                           profile='core-sim',
                           output=os.path.join(tmp(), f"{name}.tnca"))


def _execute(res, args, xs, ys, addrs, log):
    xb, yb, ob = addrs
    m = devsim.Memory()
    m.put_f32(xb, xs)
    m.put_f32(yb, ys)
    m.put(ob, b'\xef\xbe\xad\xde' * len(xs))
    full = dict(args)
    full.update({'x_ptr': xb, 'y_ptr': yb, 'o_ptr': ob})
    run = devsim.run_artifact(res.path, full, memory=m, arg_base=0x8000,
                              dump_at=ob, dump_words=len(xs) // 2)
    if not run.ok:
        log.append('      ' + run.stdout.strip().splitlines()[-1][:110])
    return run


def cmp01(log):
    """add, mul-add and dynamic AXPY, against an independent reference."""
    if not have_triton():
        log.append("    Triton is not importable, and CMP-01 is about the "
                   "real frontend")
        return False, 'needs the locked Triton'
    import kernels_triton as K

    xs = [float(i) for i in range(N)]
    ys = [float(100 + i) for i in range(N)]
    ok = True

    cases = [
        ('add_kernel', K.add_kernel, SIG3, {'n': N},
         [a + b for a, b in zip(xs, ys)]),
        ('mul_add_kernel', K.mul_add_kernel, SIG3, {'n': N},
         [a * b + a for a, b in zip(xs, ys)]),
        ('axpy_kernel', K.axpy_kernel, SIG_AXPY, {'n': N, 'alpha': 2.5},
         [a * 2.5 + b for a, b in zip(xs, ys)]),
    ]
    for name, fn, sig, args, want in cases:
        try:
            res = _compile(fn, sig, name)
        except TncaError as e:
            log.append(f"    {name:<16s} compile FAILED: {e.message[:70]}")
            ok = False
            continue
        run = _execute(res, args, xs, ys, (0x9000, 0xA000, 0xB000), log)
        got = run.f32(N)
        same = got == want
        log.append(f"    {name:<16s} {res.manifest['code']['segments'][0]['length']:>4d} "
                   f"bytes, ran={run.ok}, matches the reference={same}")
        if not same:
            log.append(f"      got  {got[:4]}")
            log.append(f"      want {want[:4]}")
        ok &= run.ok and same

    # The claim the phase exists for: the SAME artifact, different numbers.
    res = _compile(K.axpy_kernel, SIG_AXPY, 'axpy_kernel')
    code_hash = res.manifest['code']['segments'][0]['sha256']
    for alpha, addrs in ((2.5, (0x9000, 0xA000, 0xB000)),
                         (-0.5, (0xD000, 0xE000, 0xF000)),
                         (0.0, (0x9000, 0xA000, 0xB000))):
        run = _execute(res, {'n': N, 'alpha': alpha}, xs, ys, addrs, log)
        want = [a * alpha + b for a, b in zip(xs, ys)]
        same = run.f32(N) == want
        log.append(f"    axpy alpha={alpha:<5} at {addrs[0]:#x} -> "
                   f"ran={run.ok} matches={same}")
        ok &= run.ok and same
    after = aot.inspect_artifact(res.path)['segments'][0]['sha256']
    log.append(f"    the code hash before and after three launches: "
               f"{'unchanged' if after == code_hash else 'CHANGED (bad)'}")
    ok &= (after == code_hash)
    return ok, 'three real Triton kernels, and one artifact with three alphas'


def cmp02(log):
    """Malformed inputs, each refused in its own way."""
    if not have_triton():
        return False, 'needs the locked Triton'
    import kernels_triton as K
    ok = True

    def refused(label, fn, expect=TncaError, **kw):
        nonlocal ok
        try:
            aot.compile_aot(output=os.path.join(tmp(), 'x.tnca'), **kw)
            log.append(f"    {label:<38s} -> ACCEPTED (bad)")
            ok = False
        except expect as e:
            log.append(f"    {label:<38s} -> [{e.code}] {e.message[:44]}")
        except Exception as e:
            log.append(f"    {label:<38s} -> {type(e).__name__} (wrong kind)")
            ok = False

    refused('a kernel name not in the module', None,
            source=K.add_kernel, kernel_name='not_there', signature=SIG3,
            constexprs={'BLOCK': N}, profile='core-sim')
    refused('an unknown compile option', None,
            source=K.add_kernel, signature=SIG3, constexprs={'BLOCK': N},
            profile='core-sim', options={'opt_levle': 2})
    refused('a parameter missing from the signature', None,
            source=K.add_kernel,
            signature={k: v for k, v in SIG3.items() if k != 'n'},
            constexprs={'BLOCK': N}, profile='core-sim')
    refused('a scalar type the ABI does not carry', None,
            source=K.axpy_kernel,
            signature=dict(SIG_AXPY, alpha='fp64'),
            constexprs={'BLOCK': N}, profile='core-sim')
    refused('an unknown profile', None,
            source=K.add_kernel, signature=SIG3, constexprs={'BLOCK': N},
            profile='core-9x9')
    refused('an unknown source kind', None,
            source=K.add_kernel, signature=SIG3, constexprs={'BLOCK': N},
            profile='core-sim', source_kind='ptx')

    # Different dict order, same artifact. The signature decides the layout,
    # not the order a caller happened to build the mapping in.
    a = aot.compile_aot(K.add_kernel, signature=SIG3, constexprs={'BLOCK': N},
                        profile='core-sim',
                        output=os.path.join(tmp(), 'order-a.tnca'))
    shuffled = {k: SIG3[k] for k in reversed(list(SIG3))}
    b = aot.compile_aot(K.add_kernel, signature=shuffled,
                        constexprs={'BLOCK': N}, profile='core-sim',
                        output=os.path.join(tmp(), 'order-b.tnca'))
    same_layout = (a.manifest['parameters']['signature']
                   == b.manifest['parameters']['signature'])
    same_code = (a.manifest['code']['segments'][0]['sha256']
                 == b.manifest['code']['segments'][0]['sha256'])
    log.append(f"    a signature dict in reverse order      -> same layout="
               f"{same_layout}, same code={same_code}")
    ok &= same_layout and same_code
    return ok, 'each malformed input refused in its own way, by code'


def cmp03(log):
    """R1's negatives, through the product path, with the options on.

    The point is the options. A compiler that refuses a program at -O0 and
    emits it at -O2 has a safety check for the default case and none for the
    case people ship.
    """
    ok = True

    def unsafe(b, target):
        """The write-into-an-outstanding-fragment case: R1's OWN-01."""
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k16')
        b.window_put(a, 0, b.vmovi(7))      # inside the outstanding read set
        b.barrier_wait(tok)
        b.exit()

    def leaked(b, target):
        """R1's OWN-06: a command issued and never awaited."""
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k16')
        b.exit()

    def oversized(b, target):
        """R1's FP-02: a K64 fragment in an 8-register window."""
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k64', dtype='f16')
        b.barrier_wait(tok)
        b.exit()

    for label, program in (('a write into an outstanding fragment', unsafe),
                           ('a command that is never awaited', leaked),
                           ('a K64 fragment in an 8-register window',
                            oversized)):
        for opts in ({}, {'opt_level': 2, 'debug': True},
                     {'check_protocol': False}):
            try:
                aot.compile_aot(program, kernel_name='neg', signature={},
                                profile='core-sim', source_kind='tnca-ir',
                                options=opts,
                                output=os.path.join(tmp(), 'neg.tnca'))
                log.append(f"    {label} with {opts or 'defaults'}: "
                           f"ACCEPTED (bad)")
                ok = False
            except TncaError as e:
                pass
        log.append(f"    {label:<38s} -> refused with defaults, with "
                   f"-O2 --debug, and with check_protocol off")
    return ok, "R1's negatives refused through the product path, at every option"


def cmp04(log):
    """A failed compile leaves nothing, and damages nothing."""
    if not have_triton():
        return False, 'needs the locked Triton'
    import kernels_triton as K
    path = os.path.join(tmp(), 'survivor.tnca')
    good = aot.compile_aot(K.add_kernel, signature=SIG3,
                           constexprs={'BLOCK': N}, profile='core-sim',
                           output=path)
    before = open(path, 'rb').read()
    log.append(f"    a good artifact at {os.path.basename(path)}: "
               f"{len(before)} bytes")

    def bad(b, target):
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k16')
        b.window_put(a, 0, b.vmovi(1))
        b.barrier_wait(tok)
        b.exit()

    try:
        aot.compile_aot(bad, kernel_name='bad', signature={},
                        profile='core-sim', source_kind='tnca-ir',
                        output=path)
        log.append("    a failing compile wrote to that path (bad)")
        return False, 'a failed compile overwrote a good artifact'
    except TncaError as e:
        log.append(f"    a compile that fails                   -> "
                   f"[{e.code}] {e.message[:44]}")

    after = open(path, 'rb').read()
    same = after == before
    log.append(f"    the artifact that was already there    -> "
               f"{'unchanged' if same else 'DAMAGED (bad)'}")
    leftovers = [f for f in os.listdir(tmp()) if f.startswith('.tnca-')]
    log.append(f"    partial files a loader could find      -> "
               f"{leftovers or 'none'}")
    return same and not leftovers, 'nothing written, nothing damaged'


def cmp05(log):
    """The TNCA IR input produces the same format, labelled as itself."""
    def gemm_like(b, target):
        """A two-window MMA: R1's dedicated path, not Triton."""
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k16')
        b.barrier_wait(tok)
        v = b.acc_read(acc, b.const(0))
        b.exit()

    res = aot.compile_aot(gemm_like, kernel_name='gemm_like', signature={},
                          profile='core-sim', source_kind='tnca-ir',
                          output=os.path.join(tmp(), 'gemm.tnca'))
    kind = res.manifest['identity']['source_kind']
    log.append(f"    a TNCA IR program compiles: "
               f"{res.manifest['code']['segments'][0]['length']} bytes, "
               f"source_kind={kind!r}")
    log.append("    and the manifest says `tnca-ir`, not `triton` -- the "
               "contract forbids writing this up as Triton support")
    info = aot.inspect_artifact(res.path)
    same_format = info['manifest']['identity']['format_version'] == \
        res.manifest['identity']['format_version']
    log.append(f"    the same container format as a Triton artifact: "
               f"{same_format}")
    return kind == 'tnca-ir' and same_format, \
        'the second input, in the same format, labelled as itself'


TESTS = [('CMP-01', cmp01), ('CMP-02', cmp02), ('CMP-03', cmp03),
         ('CMP-04', cmp04), ('CMP-05', cmp05)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R2-03 compile driver, through the product entry point")
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
            log.append('    ' + traceback.format_exc()[-700:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-compile-tests       FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-compile-tests       PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
