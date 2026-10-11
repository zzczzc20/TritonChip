#!/usr/bin/env python3
"""
Tensor descriptors at ranks 1-5 through the real Triton front end and the
TNCA plugin, launched on the RTL (R38; ISA sec 9D-9E).

Every case is an ordinary `@triton.jit` kernel using the API Triton defines
for NVIDIA's TMA -- `tl.make_tensor_descriptor`, a host
`TensorDescriptor` argument, `desc.load/store/atomic_*`, `desc.gather/
scatter` -- launched as `kernel[grid](...)` on CPU tensors, and checked
element for element against PyTorch computing the same thing from the
definition: box element i of a load at offsets c is tensor[c + i] when every
index is in bounds, else 0 (or NaN under padding_option="nan"); a store,
atomic or scatter touches exactly the in-bounds elements.

    python tools/r38/tdesc_tests.py                 # every case, in parallel
    python tools/r38/tdesc_tests.py --only rank3_dma
    python tools/r38/tdesc_tests.py --profile core-vrf-sync-v1

Each case also records which path the compiler chose (the transfer engine
or the LSU), read from the kernel's assembly, so the engine's rank-3..5 path
is known to have been exercised rather than assumed.
"""
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# The cases. Each returns (ok, detail, info); runs in its own process.
# ---------------------------------------------------------------------------

def _setup(profile):
    os.environ.setdefault('TNCA_TRITON', '1')
    os.environ['TNCA_PROFILE'] = profile
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'compiler' / 'python')]
    import torch
    import triton  # noqa: F401  (before the driver)
    return torch


def _box_ref(torch, t, offs, block, pad_nan=False):
    """The box a load at `offs` returns, from the definition."""
    out = torch.full(block, float('nan') if pad_nan else 0, dtype=t.dtype)
    import itertools
    for idx in itertools.product(*[range(b) for b in block]):
        g = [o + i for o, i in zip(offs, idx)]
        if all(0 <= x < n for x, n in zip(g, t.shape)):
            out[idx] = t[tuple(g)]
    return out


def _same(torch, got, want):
    if got.shape != want.shape:
        return False, f'shape {tuple(got.shape)} != {tuple(want.shape)}'
    g, w = got.float(), want.float()
    nan_ok = torch.equal(torch.isnan(g), torch.isnan(w))
    fin = ~torch.isnan(w)
    exact = torch.equal(g[fin], w[fin])
    if nan_ok and exact:
        return True, 'bit-exact'
    bad = int((g[fin] != w[fin]).sum()) + int((torch.isnan(g) != torch.isnan(w)).sum())
    return False, f'{bad} elements differ'


def _asm(_kernel_fn=None):
    """The assembly of every kernel this case compiled, from the per-case
    TNCA_KERNEL_DUMP directory (run_case sets it)."""
    root = Path(os.environ.get('TNCA_DUMP_DIR', ''))
    if not root.is_dir():
        return ''
    return '\n'.join(p.read_text() for p in sorted(root.rglob('*.s')))


def _paths(asm):
    return {'t.ld.async': asm.count('t.ld.async'), 't.st.async': asm.count('t.st.async'),
            'v.ld.gather': asm.count('v.ld.gather'), 'v.st.scatter': asm.count('v.st.scatter')}


def case_rank1_copy(torch, tl, triton):
    """Rank 1, runtime shape, negative and past-the-end offsets, f32 and f16."""
    @triton.jit
    def k(x, y, n, off, B: tl.constexpr):
        dx = tl.make_tensor_descriptor(x, shape=[n], strides=[1], block_shape=[B])
        dy = tl.make_tensor_descriptor(y, shape=[n], strides=[1], block_shape=[B])
        # Upcast before arithmetic, as Inductor does: raw FP16 arithmetic in a
        # hand-written kernel is a known gap of the vector lowering, separate
        # from descriptors (doc/compiler_overview.md sec 14).
        dy.store([off], (dx.load([off]).to(tl.float32) * 2).to(x.dtype.element_ty))
    res = []
    for dt, n, off in ((torch.float32, 37, -5), (torch.float16, 40, 30), (torch.float32, 16, 3)):
        x = torch.randn(n, dtype=dt)
        y = torch.full((n,), 7, dtype=dt)
        k[(1,)](x, y, n, off, 16)
        want = torch.full((n,), 7, dtype=dt)
        lo, hi = max(off, 0), min(off + 16, n)
        want[lo:hi] = x[lo:hi] * 2
        res.append(_same(torch, y, want))
    return all(r[0] for r in res), '; '.join(r[1] for r in res), _paths(_asm(k))


def case_rank2_nan_pad(torch, tl, triton):
    """Rank 2, runtime strides, padding_option="nan", a box straddling two edges."""
    @triton.jit
    def k(x, out, M, N, sm, r0, c0, BM: tl.constexpr, BN: tl.constexpr):
        d = tl.make_tensor_descriptor(x, shape=[M, N], strides=[sm, 1],
                                      block_shape=[BM, BN], padding_option="nan")
        v = d.load([r0, c0])
        rm = tl.arange(0, BM)[:, None]
        cn = tl.arange(0, BN)[None, :]
        tl.store(out + rm * BN + cn, v)
    x = torch.randn(10, 24, dtype=torch.float32)[:, :13]        # a strided view
    out = torch.zeros(4, 16, dtype=torch.float32)
    k[(1,)](x, out, 10, 13, x.stride(0), 7, -3, 4, 16)
    return (*_same(torch, out, _box_ref(torch, x, (7, -3), (4, 16), pad_nan=True)), _paths(_asm(k)))


def case_rank3_dma(torch, tl, triton):
    """Rank 3, f32, a [2, 4, 16] box: loaded and stored by the transfer engine."""
    @triton.jit
    def k(x, y, A, Bd, C, sa, sb, a0, b0, c0, BA: tl.constexpr, BB: tl.constexpr, BC: tl.constexpr):
        dx = tl.make_tensor_descriptor(x, shape=[A, Bd, C], strides=[sa, sb, 1],
                                       block_shape=[BA, BB, BC])
        dy = tl.make_tensor_descriptor(y, shape=[A, Bd, C], strides=[sa, sb, 1],
                                       block_shape=[BA, BB, BC])
        dy.store([a0, b0, c0], dx.load([a0, b0, c0]))
    x = torch.randn(3, 5, 32, dtype=torch.float32)
    y = torch.full_like(x, -1.0)
    k[(1,)](x, y, 3, 5, 32, x.stride(0), x.stride(1), 2, 3, 9, 2, 4, 16)
    want = torch.full_like(x, -1.0)
    want[2:3, 3:5, 9:25] = x[2:3, 3:5, 9:25]
    ok, detail = _same(torch, y, want)
    return ok, detail, _paths(_asm(k))


def case_rank3_small_trans(torch, tl, triton):
    """Rank 3, f32, a [2, 2, 4] box (one register): gathered, permuted, stored."""
    @triton.jit
    def k(x, out, A, Bd, C, sa, sb, a0, b0, c0):
        d = tl.make_tensor_descriptor(x, shape=[A, Bd, C], strides=[sa, sb, 1],
                                      block_shape=[2, 2, 4])
        v = d.load([a0, b0, c0])                # [2, 2, 4]
        t = tl.permute(v, (2, 0, 1))            # [4, 2, 2]
        tl.store(out + tl.arange(0, 16), tl.reshape(t, [16]))
    x = torch.randn(3, 3, 8, dtype=torch.float32)
    out = torch.zeros(16, dtype=torch.float32)
    k[(1,)](x, out, 3, 3, 8, x.stride(0), x.stride(1), 1, 2, 6)
    want = _box_ref(torch, x, (1, 2, 6), (2, 2, 4)).permute(2, 0, 1).reshape(-1)
    return (*_same(torch, out, want), _paths(_asm(k)))


def case_rank4_rows(torch, tl, triton):
    """Rank 4, f16, a [2, 2, 2, 16] box: row-by-row loads, arithmetic, store."""
    @triton.jit
    def k(x, y, s0, s1, s2, c0, c1, c2, c3):
        d = tl.make_tensor_descriptor(x, shape=[3, 3, 4, 20], strides=[s0, s1, s2, 1],
                                      block_shape=[2, 2, 2, 16])
        e = tl.make_tensor_descriptor(y, shape=[3, 3, 4, 20], strides=[s0, s1, s2, 1],
                                      block_shape=[2, 2, 2, 16])
        v = d.load([c0, c1, c2, c3]).to(tl.float32)
        e.store([c0, c1, c2, c3], (v + v).to(tl.float16))
    x = torch.randn(3, 3, 4, 20, dtype=torch.float16)
    y = torch.zeros_like(x)
    k[(1,)](x, y, *x.stride()[:3], 2, -1, 3, 10)
    want = torch.zeros_like(x)
    sl = (slice(2, 3), slice(0, 1), slice(3, 4), slice(10, 20))
    want[sl] = x[sl] + x[sl]
    return (*_same(torch, y, want), _paths(_asm(k)))


def case_rank5_fill(torch, tl, triton):
    """Rank 5, f32, a box half outside on two axes; zero fill checked."""
    @triton.jit
    def k(x, out, s0, s1, s2, s3, c0, c1, c2, c3, c4):
        d = tl.make_tensor_descriptor(x, shape=[2, 2, 2, 3, 12], strides=[s0, s1, s2, s3, 1],
                                      block_shape=[1, 2, 1, 2, 16])
        v = d.load([c0, c1, c2, c3, c4])        # [1, 2, 1, 2, 16]
        w = tl.reshape(v, [4, 16])
        r = tl.arange(0, 4)[:, None] * 16 + tl.arange(0, 16)[None, :]
        tl.store(out + r, w)
    x = torch.randn(2, 2, 2, 3, 12, dtype=torch.float32)
    out = torch.full((64,), -9.0, dtype=torch.float32)
    k[(1,)](x, out, *x.stride()[:4], 1, 1, 0, 2, -4)
    want = _box_ref(torch, x, (1, 1, 0, 2, -4), (1, 2, 1, 2, 16)).reshape(-1)
    return (*_same(torch, out, want), _paths(_asm(k)))


def case_host_descriptor(torch, tl, triton):
    """A host TensorDescriptor (rank 3, NaN padding) as a kernel argument."""
    from tnca.triton_backend import TensorDescriptor

    @triton.jit
    def k(dx, out, a0, b0, c0):
        v = dx.load([a0, b0, c0])               # [1, 2, 16]
        w = tl.reshape(v, [2, 16])
        r = tl.arange(0, 2)[:, None] * 16 + tl.arange(0, 16)[None, :]
        tl.store(out + r, w)
    x = torch.randn(2, 3, 20, dtype=torch.float32)
    out = torch.zeros(32, dtype=torch.float32)
    k[(1,)](TensorDescriptor(x, list(x.shape), list(x.stride()), [1, 2, 16], padding='nan'),
            out, 1, 2, 9)
    want = _box_ref(torch, x, (1, 2, 9), (1, 2, 16), pad_nan=True).reshape(-1)
    return (*_same(torch, out, want), _paths(_asm(k)))


def case_atomic_add(torch, tl, triton):
    """desc.atomic_add on f32 and desc.atomic_max on i32, partly out of bounds."""
    @triton.jit
    def k(y, z, M, N, r0, c0):
        dy = tl.make_tensor_descriptor(y, shape=[M, N], strides=[N, 1], block_shape=[2, 16])
        dz = tl.make_tensor_descriptor(z, shape=[M, N], strides=[N, 1], block_shape=[2, 16])
        one = tl.full([2, 16], 1.5, tl.float32)
        dy.atomic_add([r0, c0], one)
        dz.atomic_max([r0, c0], tl.full([2, 16], 5, tl.int32))
    y = torch.zeros(3, 12, dtype=torch.float32)
    z = torch.arange(36, dtype=torch.int32).reshape(3, 12)
    k[(1,)](y, z, 3, 12, 2, -4)
    wy = torch.zeros(3, 12)
    wy[2:3, 0:12] += 1.5
    wz = torch.arange(36, dtype=torch.int32).reshape(3, 12)
    wz[2:3, 0:12] = torch.maximum(wz[2:3, 0:12], torch.tensor(5, dtype=torch.int32))
    a, b = _same(torch, y, wy), _same(torch, z, wz)
    return a[0] and b[0], f'add {a[1]}; max {b[1]}', _paths(_asm(k))


def case_gather_scatter(torch, tl, triton):
    """desc.gather / desc.scatter: rows by index, one past the end."""
    @triton.jit
    def k(x, y, M, N, rows, y0):
        dx = tl.make_tensor_descriptor(x, shape=[M, N], strides=[N, 1], block_shape=[1, 16])
        dy = tl.make_tensor_descriptor(y, shape=[M, N], strides=[N, 1], block_shape=[1, 16])
        idx = tl.load(rows + tl.arange(0, 8))
        v = dx.gather(idx, y0)
        dy.scatter(v * 3, idx, y0)
    x = torch.randn(6, 20, dtype=torch.float32)
    y = torch.zeros_like(x)
    rows = torch.tensor([0, 5, 2, 6, 3, 1, 4, 5], dtype=torch.int32)
    k[(1,)](x, y, 6, 20, rows, 8)
    want = torch.zeros_like(x)
    for r in rows.tolist():
        if 0 <= r < 6:
            want[r, 8:20] = x[r, 8:20] * 3
    return (*_same(torch, y, want), _paths(_asm(k)))


def case_rank3_dot(torch, tl, triton):
    """A [1, 16, 16] f16 rank-3 box per operand, reshaped into tl.dot."""
    @triton.jit
    def k(a, b, c, sa, sb):
        da = tl.make_tensor_descriptor(a, shape=[2, 16, 16], strides=[sa, 16, 1],
                                       block_shape=[1, 16, 16])
        db = tl.make_tensor_descriptor(b, shape=[2, 16, 16], strides=[sb, 16, 1],
                                       block_shape=[1, 16, 16])
        x = tl.reshape(da.load([1, 0, 0]), [16, 16])
        y = tl.reshape(db.load([1, 0, 0]), [16, 16])
        acc = tl.dot(x, y)
        r = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        tl.store(c + r, acc)
    a = torch.randn(2, 16, 16, dtype=torch.float16)
    b = torch.randn(2, 16, 16, dtype=torch.float16)
    c = torch.zeros(256, dtype=torch.float32)
    k[(1,)](a, b, c, a.stride(0), b.stride(0))
    want = (a[1].float() @ b[1].float()).reshape(-1)
    err = float((c - want).abs().max() / want.abs().max())
    return err < 1e-5, f'max rel err {err:.2e}', _paths(_asm(k))


def case_rank1_relabel_in_loop(torch, tl, triton):
    """Rank 1 loaded in a loop and relabelled [16] -> [16, 1] -> [1, 16] before
    use, the way Inductor reads an RMSNorm weight (R39-00: llama_decode)."""
    @triton.jit
    def k(w, x, out, N: tl.constexpr):
        dw = tl.make_tensor_descriptor(w, shape=[N], strides=[1], block_shape=[16])
        acc = tl.zeros([1, 16], tl.float32)
        for off in range(0, N, 16):
            idx = off + tl.arange(0, 16)[None, :]
            a = tl.trans(tl.expand_dims(dw.load([off]), 1)).to(tl.float32)
            t = a + tl.load(x + idx).to(tl.float32)
            tl.store(out + idx, t)
            acc += t * t
        tl.store(out + N + tl.arange(0, 16)[None, :], acc)
    w = torch.randn(64, dtype=torch.float16)
    x = torch.randn(64, dtype=torch.float16)
    out = torch.zeros(80, dtype=torch.float32)
    k[(1,)](w, x, out, 64)
    t = w.float() + x.float()
    want = torch.cat([t, (t * t).reshape(4, 16).sum(0)])
    ok = bool(torch.allclose(out, want, rtol=1e-3, atol=1e-3))
    return ok, f'max abs error {float((out - want).abs().max()):.3g}', _paths(_asm(k))


def case_rank1_store_relabel(torch, tl, triton):
    """A [1, 16] row relabelled [16, 1] -> [16] and stored through a rank-1
    descriptor, the way Inductor writes an RMSNorm output (R39-00)."""
    @triton.jit
    def k(x, out, N: tl.constexpr):
        do = tl.make_tensor_descriptor(out, shape=[N], strides=[1], block_shape=[16])
        for off in range(0, N, 16):
            t = tl.load(x + off + tl.arange(0, 16)[None, :]).to(tl.float32) * 3
            v = tl.reshape(tl.trans(t), [16])
            do.store([off], v.to(tl.float16))
    x = torch.randn(64, dtype=torch.float16)
    out = torch.zeros(64, dtype=torch.float16)
    k[(1,)](x, out, 64)
    want = (x.float() * 3).half()
    ok = bool(torch.equal(out, want))
    return ok, f'max abs error {float((out.float() - want.float()).abs().max()):.3g}', _paths(_asm(k))


CASES = {n[5:]: f for n, f in sorted(globals().items()) if n.startswith('case_')}


def run_case(name, profile):
    import tempfile
    os.environ['TNCA_KERNEL_DUMP'] = '1'
    os.environ['TNCA_DUMP_DIR'] = tempfile.mkdtemp(prefix=f'tdesc-{name}-')
    torch = _setup(profile)
    import triton
    import triton.language as tl
    t0 = time.monotonic()
    try:
        ok, detail, info = CASES[name](torch, tl, triton)
    except Exception as e:                               # noqa: BLE001
        import traceback
        ok, detail, info = False, f'{type(e).__name__}: {str(e)[:600]}', {
            'trace': traceback.format_exc()[-2000:]}
    return dict(case=name, ok=bool(ok), detail=detail, info=info,
                seconds=round(time.monotonic() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--profile', default='core-sim')
    ap.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--json', type=Path, default=ROOT / 'sim' / 'build' / 'r38' / 'tdesc_tests.json')
    ap.add_argument('--child', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(run_case(args.child, args.profile)))
        return 0
    names = args.only or list(CASES)
    py = os.environ.get('TORCH_PYTHON', sys.executable)

    # A GPU next to TNCA would make two Triton drivers active; the children
    # see none (as archbench.run does).
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TNCA_TRITON='1')

    def child(n):
        r = subprocess.run([py, __file__, '--child', n, '--profile', args.profile],
                           capture_output=True, text=True, timeout=7200, env=env)
        line = (r.stdout.strip().splitlines() or [''])[-1]
        try:
            return json.loads(line)
        except ValueError:
            return dict(case=n, ok=False, detail=f'no result (rc {r.returncode}): '
                        f'{(r.stderr or r.stdout)[-800:]}', info={})

    rows = []
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for res in ex.map(child, names):
            rows.append(res)
            print(f"{'PASS' if res['ok'] else 'FAIL'} {res['case']:22} {res['detail'][:110]}"
                  f"  {res.get('info') if res['ok'] else ''}  [{res.get('seconds', '?')}s]",
                  flush=True)
            if not res['ok'] and res.get('info', {}).get('trace'):
                print('     ' + res['info']['trace'].strip().replace('\n', '\n     ')[-1500:])
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(dict(profile=args.profile, rows=rows), indent=1) + '\n')
    n_ok = sum(r['ok'] for r in rows)
    print(f'tensor descriptors [{args.profile}]: {n_ok}/{len(rows)} pass')
    return 0 if n_ok == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
