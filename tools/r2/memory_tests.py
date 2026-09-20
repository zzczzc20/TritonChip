#!/usr/bin/env python3
"""
MEM-03, MEM-05 and MEM-06 -- doc/phase2_tasks_and_acceptance.md R2-10.

The host memory transport, and the line the contract draws around it:

    "不得用 testbench 层级引用读取/写入 cache tag/data/RF 数组来实现产品数据
     接口。testbench 内部观察仍可用于验证断言，不能作为 runtime 结果来源。"

MEM-05 is that line, checked two ways. Statically: nothing under `python/tnca`
names a hierarchical path into the device -- no `dut.`, no `u_mem.mem`, no
register file array -- so the product could not read one if it wanted to.
Dynamically: a download is made to fail by breaking the TRANSPORT, and the
result is an error rather than a number, which it could not be if the value
were coming from somewhere else.

MEM-01, MEM-02 and MEM-04 are NOT RUN and say why. They are about a value
living in a dirty cache line, which the single-core profile does not have:
its memory model is a flat array with no cache at all. They need the
four-tile device with the NUCA slices, which is R2-10's RTL half.
"""
import argparse
import os
import re
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402

from tnca import aot                                     # noqa: E402
from tnca import transport as T                          # noqa: E402
from tnca.runtime import Session                         # noqa: E402
from tnca.errors import BufferError, TncaError           # noqa: E402

N = 16
SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
       'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'}

# Things a product data path must never name. Each is a way of reaching
# INSIDE the device model instead of asking it.
FORBIDDEN = [
    (r'\bdut\.', 'a hierarchical reference into the DUT'),
    (r'u_mem\s*\.\s*mem', "the memory model's array"),
    (r'\bmem\[', 'a memory array subscript'),
    (r'vector_rf|tnca_vector_rf', 'the vector register file'),
    (r'cache_tag|tag_array|data_array', 'cache tag or data arrays'),
    (r'\$readmem|\$dumpvars|\bforce\s+\w+(?:\.\w+)+\s*=',
     'simulator-level access'),
]


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def session():
    d = tempfile.mkdtemp(prefix='r2-mem-')
    return Session(workdir=os.path.join(d, 'dev'), deadline_cycles=200000,
                   host_timeout=180), d


def artifact(d):
    import kernels_triton as K
    return aot.compile_aot(K.axpy_kernel, signature=SIG,
                           constexprs={'BLOCK': N}, profile='core-sim',
                           output=os.path.join(d, 'axpy.tnca')).path


def f32s(v):
    return b''.join(struct.pack('<f', x) for x in v)


def mem03(log):
    """A -> B -> C entirely in device memory, across batches and relaunches."""
    if not have_triton():
        return False, 'needs the locked Triton'
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
    s, d = session()
    ok = True
    try:
        mod = s.load(artifact(d))
        a, b, c, zero = (s.allocate(N * 4, name=n) for n in 'abcz')
        src = [float(i) for i in range(N)]
        s.upload(a, f32s(src))
        s.upload(zero, f32s([0.0] * N))

        # B = 2*A, C = 3*B. Nothing is downloaded until the end, so B only
        # ever exists on the device.
        e1 = s.submit(mod, {'x_ptr': a, 'y_ptr': zero, 'o_ptr': b,
                            'n': N, 'alpha': 2.0})
        e2 = s.submit(mod, {'x_ptr': b, 'y_ptr': zero, 'o_ptr': c,
                            'n': N, 'alpha': 3.0}, depends_on=[e1])
        s.wait(e2)
        got = list(struct.unpack(f'<{N}f', s.download(c, N * 4)))
        want = [6.0 * v for v in src]
        log.append(f"    A->B->C with no host read of B: {e1.state}/"
                   f"{e2.state}, correct={got == want}")
        ok &= got == want
        log.append(f"    downloads performed before the last one: 0 "
                   f"(B never left the device)")

        # And again, with the chain re-run on the same session so the
        # intermediate is overwritten in place rather than reallocated.
        for round_ in range(3):
            alpha = 1.0 + round_
            e1 = s.submit(mod, {'x_ptr': a, 'y_ptr': zero, 'o_ptr': b,
                                'n': N, 'alpha': alpha})
            e2 = s.submit(mod, {'x_ptr': b, 'y_ptr': zero, 'o_ptr': c,
                                'n': N, 'alpha': 2.0}, depends_on=[e1])
            s.wait(e2)
            got = list(struct.unpack(f'<{N}f', s.download(c, N * 4)))
            want = [2.0 * alpha * v for v in src]
            if got != want:
                log.append(f"        round {round_} alpha={alpha}: WRONG")
                ok = False
        log.append(f"    three more rounds on the same buffers: "
                   f"{'all correct' if ok else 'a round was wrong'}")
    finally:
        s.poisoned = True
        s.close()
    return ok, 'the middle result never leaves the device'


def _code_only(path):
    """The file with comments and string literals blanked, lines preserved."""
    import io
    import tokenize
    src = open(path, 'rb').read()
    lines = src.decode('utf-8').splitlines()
    out = [list(line) for line in lines]
    try:
        toks = list(tokenize.tokenize(io.BytesIO(src).readline))
    except tokenize.TokenError:
        return '\n'.join(lines)
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (r1, c1), (r2, c2) = tok.start, tok.end
        for row in range(r1, r2 + 1):
            if row - 1 >= len(out):
                break
            lo = c1 if row == r1 else 0
            hi = c2 if row == r2 else len(out[row - 1])
            for col in range(lo, min(hi, len(out[row - 1]))):
                out[row - 1][col] = ' '
    return '\n'.join(''.join(row) for row in out)


def mem05(log):
    """The product path cannot reach inside the device, statically or not."""
    ok = True
    product = os.path.join(ROOT, 'python', 'tnca')
    scanned = 0
    hits = []
    for base, dirs, files in os.walk(product):
        dirs[:] = [x for x in dirs if x not in ('__pycache__', 'data')]
        for f in sorted(files):
            if not f.endswith('.py'):
                continue
            scanned += 1
            path = os.path.join(base, f)
            # Comments and docstrings talk ABOUT these things -- this very
            # file does -- and prose is not access. Only code counts, so
            # comments and string literals are blanked out first, keeping
            # line numbers so a real hit can still be pointed at.
            code = _code_only(path)
            for pattern, what in FORBIDDEN:
                for m in re.finditer(pattern, code):
                    line = code[:m.start()].count('\n') + 1
                    hits.append(f"{f}:{line} {what} ({m.group(0)!r})")
    log.append(f"    scanned {scanned} product module(s) for hierarchical "
               f"device access")
    if hits:
        for h in hits[:6]:
            log.append(f"        {h}")
        ok = False
    else:
        log.append(f"    none found: the product asks the device, it does "
                   f"not reach into it")

    # Dynamically: break the transport and the download becomes an error,
    # which it could not if the value were coming from anywhere else.
    if have_triton():
        sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
        s, d = session()
        try:
            mod = s.load(artifact(d))
            x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
            s.upload(x, f32s([2.0] * N))
            s.upload(y, f32s([1.0] * N))
            ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                'n': N, 'alpha': 1.0})
            s.wait(ev)
            first = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
            log.append(f"    a normal download returns {first[0]} through "
                       f"the transport")
            # Kill the device: if download read an array, it would still work.
            s.dev.proc.kill()
            s.dev.proc.wait(timeout=10)
            try:
                s.download(o, N * 4)
                log.append("    a download with the device dead SUCCEEDED "
                           "(bad: the value came from somewhere else)")
                ok = False
            except TncaError as e:
                log.append(f"    with the device killed, download -> "
                           f"[{e.code}]: the data path is the transport and "
                           f"nothing else")
        finally:
            s.poisoned = True
            s.close()
    else:
        log.append("    (the dynamic half needs a session)")
        ok = False
    return ok, 'no hierarchical access, and the transport is the only path'


def mem06(log):
    """Every host request is recorded; out of range is refused; canaries hold."""
    if not have_triton():
        return False, 'needs the locked Triton'
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
    s, d = session()
    ok = True
    try:
        mod = s.load(artifact(d))
        guard_lo = s.allocate(64, name='canary-lo')
        x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
        guard_hi = s.allocate(64, name='canary-hi')
        canary = b'\xA5' * 64
        s.upload(guard_lo, canary)
        s.upload(guard_hi, canary)
        s.upload(x, f32s([3.0] * N))
        s.upload(y, f32s([4.0] * N))

        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                            'n': N, 'alpha': 1.0})
        s.wait(ev)
        out = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
        ok &= out == [7.0] * N

        # Every request recorded, with what it was.
        history = s.dev.history
        fields = {'seq', 'op', 'a0', 'a1', 'status'}
        complete = all(fields <= set(h) for h in history)
        ops = sorted({h['op'] for h in history})
        log.append(f"    {len(history)} transport request(s) recorded, "
                   f"every one with address, length, direction and status: "
                   f"{complete}")
        log.append(f"    directions used: "
                   f"{', '.join(T.OP_NAME.get(o, str(o)) for o in ops)}")
        ok &= complete

        # Out of range, in both directions, refused without touching memory.
        for label, call in (
                ('upload past the buffer',
                 lambda: s.upload(o, b'\x00' * (N * 4 + 8))),
                ('download past the buffer',
                 lambda: s.download(o, N * 4 + 8)),
                ('upload at a negative offset',
                 lambda: s.upload(o, b'\x00' * 4, offset=-4))):
            try:
                call()
                log.append(f"    {label:<30s} -> ACCEPTED (bad)")
                ok = False
            except BufferError as e:
                log.append(f"    {label:<30s} -> [{e.code}]")

        lo = s.download(guard_lo, 64)
        hi = s.download(guard_hi, 64)
        log.append(f"    the canaries either side: "
                   f"{'intact' if lo == canary and hi == canary else 'CLOBBERED (bad)'}")
        ok &= lo == canary and hi == canary

        # A failed request does not report success.
        bad = s.dev.request(T.OP_UPLOAD, 0xFFFFFFF0, 1)
        log.append(f"    an upload beyond device memory -> "
                   f"{T.STATUS_NAME.get(bad.status)} (not OK)")
        ok &= not bad.ok
    finally:
        s.poisoned = True
        s.close()
    return ok, 'recorded, bounded, and a failure that says so'


TESTS = [('MEM-03', mem03), ('MEM-05', mem05), ('MEM-06', mem06)]

NOT_YET = {
    'MEM-01': 'a value living in a dirty cache line needs the four-tile '
              'profile: the single-core memory model is a flat array with '
              'no cache at all (R2-10 RTL half)',
    'MEM-02': 'a host upload overwriting a cached line needs the same',
    'MEM-04': 'delaying the last store to prove SUCCEEDED is not early '
              'needs an injection hook in the memory model',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R2-10 host memory transport and completion")
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
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    for cid, why in sorted(NOT_YET.items()):
        if args.only and cid not in args.only:
            continue
        results.append(dict(id=cid, passed=False, detail=f"NOT RUN: {why}"))
        print(f"  {cid}  NOT RUN  {why}", flush=True)
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-memory-tests        FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-memory-tests        PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
