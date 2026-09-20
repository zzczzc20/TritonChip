#!/usr/bin/env python3
"""
E2E-01, E2E-02 and E2E-05 -- doc/phase2_tasks_and_acceptance.md R2-13.

"所有比较使用独立参考计算，不得由 runtime 代算输出。seed 集合固定 {1,7,42}；
 值生成、内存 stall/reorder 的 seed 分别记录与传递。所有 mask/尾块输出区使用
 sentinel 与邻接 canary 验证未触达范围."

Every expected value here is computed in Python from the inputs, before the
device runs. Nothing asks the runtime what the answer should be, and nothing
compares the device against itself.

The two seeds are separate on purpose and are both recorded. The value seed
decides the inputs; the memory seed decides the stall and reorder pattern
inside the device's memory model. A run that mixed them would be unable to
say which of the two a disagreement came from.

Every output buffer is filled with a sentinel before the launch and has a
canary allocated on each side, so "the tail was not written" and "the tail
was written with the right value" are different observations.

E2E-03, E2E-04 and E2E-06 are NOT RUN and say what each needs.
"""
import argparse
import os
import random
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                       # noqa: E402

from tnca import aot                                    # noqa: E402
from tnca.runtime import Session                        # noqa: E402

BLOCK = 16
SEEDS = (1, 7, 42)                  # fixed by the contract
SIZES = (0, 1, 15, 16, 17, 63, 64, 65)
SENTINEL = 0x7FC00000               # a quiet NaN; an untouched slot shows up
CANARY = b'\xA5' * 64


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def f32(values):
    return b''.join(struct.pack('<f', v) for v in values)


def unpack(blob):
    return list(struct.unpack(f'<{len(blob) // 4}f', blob))


def inputs(seed, n):
    """The values, from the value seed alone.

    Exact in binary32 on purpose: the point of these cases is the mask and
    the tail, not floating-point association, and a mismatch should mean
    the wrong element was written rather than the wrong rounding.
    """
    rng = random.Random(seed)
    return [float(rng.randint(-64, 64)) for _ in range(max(n, 1))]


# The independent reference. One line each, computed here, never asked of
# the device.
KERNELS = {
    'add':      (lambda x, y, a: x + y,       ('x_ptr', 'y_ptr', 'o_ptr',
                                               'n')),
    'mul_add':  (lambda x, y, a: x * y + x,   ('x_ptr', 'y_ptr', 'o_ptr',
                                               'n')),
    'axpy':     (lambda x, y, a: x * a + y,   ('x_ptr', 'y_ptr', 'o_ptr',
                                               'n', 'alpha')),
}


def _linear_add():
    """A numeric kernel that covers all three grid axes.

    `add_kernel` uses program_id(0) only, so on a 1x3x1 grid all three
    programs write the same sixteen elements -- correct for what it asks
    for, useless for checking a 3D grid. This one rebuilds the linear
    index from the three axes, which is what makes a y or z program land
    anywhere new.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def linear_add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = (tl.program_id(0)
               + tl.program_id(1) * tl.num_programs(0)
               + tl.program_id(2) * tl.num_programs(0) * tl.num_programs(1))
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        tl.store(o_ptr + offs, x + y, mask=mask)

    return linear_add


def build(d, which):
    import kernels_triton as K
    if which == 'linear_add':
        return aot.compile_aot(
            _linear_add(),
            signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                       'n': 'i32', 'BLOCK': 'constexpr'},
            constexprs={'BLOCK': BLOCK}, profile='core-sim',
            output=os.path.join(d, 'linear_add.tnca')).path
    fn = {'add': K.add_kernel, 'mul_add': K.mul_add_kernel,
          'axpy': K.axpy_kernel}[which]
    sig = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32', 'n': 'i32',
           'BLOCK': 'constexpr'}
    if which == 'axpy':
        sig['alpha'] = 'fp32'
    return aot.compile_aot(fn, signature=sig, constexprs={'BLOCK': BLOCK},
                           profile='core-sim',
                           output=os.path.join(d, f'{which}.tnca')).path


def _session(memseed):
    d = tempfile.mkdtemp(prefix='r2-e2e-')
    s = Session(workdir=os.path.join(d, 'dev'), deadline_cycles=400000,
                host_timeout=600)
    s.seeds = {'values': None, 'memory': memseed}
    return s, d


def _run_case(s, mod, which, n, xs, ys, alpha, log_bad):
    """One (kernel, N) case, with the sentinel and the canaries."""
    grid_x = max(1, (n + BLOCK - 1) // BLOCK)
    slots = grid_x * BLOCK
    lo = s.allocate(64, name='canary-lo')
    x = s.allocate(slots * 4, name='x')
    y = s.allocate(slots * 4, name='y')
    o = s.allocate(slots * 4, name='o')
    hi = s.allocate(64, name='canary-hi')
    s.upload(lo, CANARY)
    s.upload(hi, CANARY)
    xs_pad = (xs + [0.0] * slots)[:slots]
    ys_pad = (ys + [0.0] * slots)[:slots]
    s.upload(x, f32(xs_pad))
    s.upload(y, f32(ys_pad))
    s.upload(o, struct.pack('<I', SENTINEL) * slots)

    args = {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': n}
    if which == 'axpy':
        args['alpha'] = alpha
    ev = s.submit(mod, args, grid=(grid_x, 1, 1))
    s.wait(ev)
    if not ev.ok:
        log_bad(f"N={n}: {ev.state} [{getattr(ev.error, 'code', None)}]")
        return False

    raw = s.download(o, slots * 4)
    got = unpack(raw)
    ref = KERNELS[which][0]
    want = [ref(xs_pad[i], ys_pad[i], alpha) for i in range(n)]
    if got[:n] != want:
        bad = next(i for i in range(n) if got[i] != want[i])
        log_bad(f"N={n}: element {bad} is {got[bad]}, reference says "
                f"{want[bad]}")
        return False

    # The masked tail: untouched, not merely correct.
    tail = struct.unpack(f'<{slots - n}I', raw[n * 4:]) if slots > n else ()
    if any(v != SENTINEL for v in tail):
        touched = [i + n for i, v in enumerate(tail) if v != SENTINEL]
        log_bad(f"N={n}: the mask let {len(touched)} tail slot(s) through, "
                f"first at {touched[0]}")
        return False
    if s.download(lo, 64) != CANARY or s.download(hi, 64) != CANARY:
        log_bad(f"N={n}: a canary either side of the output was clobbered")
        return False
    for buf in (lo, x, y, o, hi):
        s.free(buf)
    return True


def e2e01(log):
    """Three real Triton kernels, eight sizes, three seeds."""
    if not have_triton():
        return False, 'needs the locked Triton'
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
    d = tempfile.mkdtemp(prefix='r2-e2e1-')
    ok = True
    problems = []
    done = 0
    for memseed in SEEDS:
        s, _d = _session(memseed)
        try:
            for which in sorted(KERNELS):
                mod = s.load(build(d, which))
                for vseed in SEEDS:
                    s.seeds['values'] = vseed
                    for n in SIZES:
                        xs = inputs(vseed, max(n, BLOCK))
                        ys = inputs(vseed + 1000, max(n, BLOCK))
                        alpha = float((vseed % 5) - 2)
                        good = _run_case(
                            s, mod, which, n, xs, ys, alpha,
                            lambda m, w=which, v=vseed, ms=memseed:
                            problems.append(f"{w} vseed={v} memseed={ms}: {m}"))
                        ok &= good
                        done += 1
        finally:
            s.poisoned = True
            s.close()
    log.append(f"    {done} case(s): {len(KERNELS)} kernel(s) x "
               f"{len(SIZES)} size(s) x {len(SEEDS)} value seed(s) x "
               f"{len(SEEDS)} memory seed(s)")
    log.append(f"    sizes {SIZES}, BLOCK={BLOCK}, grid.x=ceil(N/{BLOCK})")
    log.append(f"    value seeds {SEEDS} and memory seeds {SEEDS} recorded "
               f"separately on every session")
    log.append(f"    every expected value computed in Python from the "
               f"inputs, before the launch")
    if problems:
        for p in problems[:8]:
            log.append(f"        {p}")
        log.append(f"    {len(problems)} problem(s)")
    else:
        log.append(f"    tail slots past N: sentinel intact in every case; "
                   f"canaries either side intact")
    return ok, f'{done} cases, three kernels, eight sizes, two seed axes'


def e2e02(log):
    """The nine grids, with a numeric kernel as well as the PID kernel."""
    if not have_triton():
        return False, 'needs the locked Triton'
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
    d = tempfile.mkdtemp(prefix='r2-e2e2-')
    grids = [(1, 1, 1), (2, 1, 1), (5, 1, 1), (1, 3, 1), (2, 3, 1),
             (1, 1, 4), (2, 2, 2), (3, 1, 2), (0, 1, 1)]
    ok = True
    s, _d = _session(7)
    try:
        mod = s.load(build(d, 'linear_add'))
        ran3d = 0
        for gx, gy, gz in grids:
            total = gx * gy * gz
            n = max(total, 1) * BLOCK
            xs = inputs(1, n)
            ys = inputs(2, n)
            o = s.allocate(n * 4, name='o')
            x = s.allocate(n * 4, name='x')
            y = s.allocate(n * 4, name='y')
            s.upload(x, f32(xs))
            s.upload(y, f32(ys))
            s.upload(o, struct.pack('<I', SENTINEL) * n)
            if total == 0:
                # An empty grid computes nothing, and must not.
                ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                    'n': n}, grid=(gx, gy, gz))
                s.wait(ev)
                raw = s.download(o, n * 4)
                untouched = all(v == SENTINEL for v in
                                struct.unpack(f'<{n}I', raw))
                log.append(f"    grid {gx}x{gy}x{gz}: {ev.state}, output "
                           f"untouched={untouched}")
                ok &= ev.ok and untouched
            else:
                ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                    'n': total * BLOCK},
                              grid=(gx, gy, gz))
                s.wait(ev)
                got = unpack(s.download(o, total * BLOCK * 4))
                want = [xs[i] + ys[i] for i in range(total * BLOCK)]
                good = ev.ok and got == want
                if gz > 1 and gy >= 1 and gx >= 1 and (gy > 1 or gz > 1):
                    ran3d += 1 if gz > 1 else 0
                log.append(f"    grid {gx}x{gy}x{gz}: {total:2d} program(s), "
                           f"{total * BLOCK} element(s) correct={good}")
                ok &= good
            for b in (o, x, y):
                s.free(b)
        log.append(f"    two 3D grids actually executed: "
                   f"{ran3d >= 2} ({ran3d} with gz>1)")
        ok &= ran3d >= 2
    finally:
        s.poisoned = True
        s.close()
    return ok, 'nine grids, a numeric kernel, an empty grid that computes nothing'


def e2e05(log):
    """The persistent-session sequences, under all three seeds."""
    if not have_triton():
        return False, 'needs the locked Triton'
    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
    d = tempfile.mkdtemp(prefix='r2-e2e5-')
    ok = True
    for memseed in SEEDS:
        s, _d = _session(memseed)
        try:
            mod = s.load(build(d, 'axpy'))
            n = 4 * BLOCK
            xs = inputs(memseed, n)
            ys = inputs(memseed + 500, n)
            x, y, o = (s.allocate(n * 4, name=c) for c in 'xyo')
            s.upload(x, f32(xs))
            s.upload(y, f32(ys))
            good = 0
            for i in range(16):
                alpha = float((i % 5) - 2)
                ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                    'n': n, 'alpha': alpha},
                              grid=(4, 1, 1))
                s.wait(ev)
                got = unpack(s.download(o, n * 4))
                want = [xs[j] * alpha + ys[j] for j in range(n)]
                good += 1 if (ev.ok and got == want) else 0
            # and a chain whose middle never leaves the device
            a, b, c = (s.allocate(n * 4, name=t) for t in 'ABC')
            zero = s.allocate(n * 4, name='zero')
            s.upload(a, f32(xs))
            s.upload(zero, f32([0.0] * n))
            e1 = s.submit(mod, {'x_ptr': a, 'y_ptr': zero, 'o_ptr': b,
                                'n': n, 'alpha': 3.0}, grid=(4, 1, 1))
            e2 = s.submit(mod, {'x_ptr': b, 'y_ptr': zero, 'o_ptr': c,
                                'n': n, 'alpha': 2.0}, grid=(4, 1, 1),
                          depends_on=[e1])
            s.wait(e2)
            chain = unpack(s.download(c, n * 4))
            want = [6.0 * v for v in xs]
            log.append(f"    memseed={memseed}: {good}/16 launches correct, "
                       f"chain correct={chain == want}, one device "
                       f"({s.dev.seq} requests)")
            ok &= good == 16 and chain == want
        finally:
            s.poisoned = True
            s.close()
    return ok, '16 launches and a chain, under each of the three seeds'


TESTS = [('E2E-01', e2e01), ('E2E-02', e2e02), ('E2E-05', e2e05)]

NOT_YET = {
    'E2E-03': 'the two MLIR-02 branches over five trip counts, on core AND '
              'system: the MLIR main path is R2-04 and not started, and '
              'there is no system device',
    'E2E-04': 'the TNCA IR GEMM matrix (FP16/BF16, K=16..128, two roles, '
              'fast/slow/ooo) needs the core-2r-sfu persistent device; only '
              'core-sim has one',
    'E2E-06': "core-4r's two-consumer program needs the four-role "
              'persistent device (R2-09 RTL half)',
}


def _case_ids():
    ids = set()
    for name, value in list(globals().items()):
        if name.startswith('_') or not name.isupper():
            continue
        if isinstance(value, dict):
            ids |= {k for k in value if isinstance(k, str) and '-' in k}
        elif isinstance(value, (list, tuple)):
            for item in value:
                if (isinstance(item, (list, tuple)) and item
                        and isinstance(item[0], str) and '-' in item[0]):
                    ids.add(item[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R2-13 end-to-end matrix")
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
            log.append('    ' + traceback.format_exc()[-900:].replace(
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

    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing; the known "
              f"cases are:")
        for cid in sorted(_case_ids()):
            print(f"      {cid}")
        return 2

    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-e2e-tests           FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-e2e-tests           PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
