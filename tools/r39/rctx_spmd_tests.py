#!/usr/bin/env python3
"""R39-05 Mode B: every Role Context runs programs of one grid row.

    python3 tools/r39/rctx_spmd_tests.py [--profile core-vrf-sync-v2] [--only NAME ...]

Each case is a `@triton.jit` kernel launched through the TNCA plugin on CPU
tensors twice -- TNCA_RCTX_SPMD=0, then 1 -- and checked element for element
against PyTorch both times. With the switch on, a kernel the pass accepts
must say so in its manifest (role_mask has a bit per context) and one it
refuses must not; the cycles of both runs are reported.

The grids are chosen to break the loop the pass builds: more programs than
contexts and not a multiple of them, one program (one context started), a
second grid axis, an inner loop whose counter starts from a constant (the
pass must not compute that constant once for every program), and trip
counts that differ between programs. Three kernels hold per-core resources
-- a tl.dot (accumulator), tensor descriptors (TensorMaps in scratch, a
descriptor, a window and a barrier) and a grid of dot tiles: they run on
several contexts only where the target can slice the resources (RSLICE,
rctx_slice), each context in its own slice and its own scratch, and must be
refused anywhere else.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def kernels(tl, triton):
    @triton.jit
    def add(x, y, out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(out + offs, tl.load(x + offs, mask=m) + tl.load(y + offs, mask=m), mask=m)

    @triton.jit
    def row_sum(x, out, ncols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for c in range(0, ncols, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            acc += tl.load(x + row * ncols + offs, mask=offs < ncols, other=0.0)
        tl.store(out + row, tl.sum(acc, axis=0))

    @triton.jit
    def grid2d(x, out, ncols, BLOCK: tl.constexpr):
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = tl.program_id(1)
        m = col < ncols
        v = tl.load(x + row * ncols + col, mask=m)
        tl.store(out + row * ncols + col, v * 2.0 + row, mask=m)

    @triton.jit
    def ragged(out, BLOCK: tl.constexpr):
        # program p sums 0..p: a different trip count for every program
        p = tl.program_id(0)
        acc = 0
        for i in range(p + 1):
            acc += i
        tl.store(out + p, acc)

    @triton.jit
    def row_cumsum(x, out, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        tl.store(out + row * BLOCK + offs, tl.cumsum(tl.load(x + row * BLOCK + offs), axis=0))

    @triton.jit
    def row_softmax(x, out, ncols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        m = offs < ncols
        v = tl.load(x + row * ncols + offs, mask=m, other=-float('inf'))
        e = tl.exp(v - tl.max(v, axis=0))
        tl.store(out + row * ncols + offs, e / tl.sum(e, axis=0), mask=m)

    @triton.jit
    def dot(a, b, c, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        A = tl.load(a + r[:, None] * BLOCK + r[None, :])
        B = tl.load(b + r[:, None] * BLOCK + r[None, :])
        tl.store(c + r[:, None] * BLOCK + r[None, :], tl.dot(A, B))

    @triton.jit
    def tdesc_rows(x, y, M, N, BN: tl.constexpr):
        # a [2, BN] box per program through tensor descriptors (TensorMaps
        # built in scratch, the transfer engine, a barrier and a window)
        row = tl.program_id(0) * 2
        dx = tl.make_tensor_descriptor(x, shape=[M, N], strides=[N, 1], block_shape=[2, BN])
        dy = tl.make_tensor_descriptor(y, shape=[M, N], strides=[N, 1], block_shape=[2, BN])
        dy.store([row, 0], (dx.load([row, 0]).to(tl.float32) * 2 + 1).to(tl.float16))

    @triton.jit
    def dot_grid(a, b, c, K: tl.constexpr, BLOCK: tl.constexpr):
        # C[m, n] tile = A[m, :] @ B[:, n]; n on axis 0, m on axis 1
        n0 = tl.program_id(0) * BLOCK
        m0 = tl.program_id(1) * BLOCK
        r = tl.arange(0, BLOCK)
        k = tl.arange(0, K)
        A = tl.load(a + (m0 + r)[:, None] * K + k[None, :])
        B = tl.load(b + k[:, None] * (3 * BLOCK) + (n0 + r)[None, :])
        tl.store(c + (m0 + r)[:, None] * (3 * BLOCK) + (n0 + r)[None, :], tl.dot(A, B))

    return dict(add=add, row_sum=row_sum, grid2d=grid2d, ragged=ragged,
                row_cumsum=row_cumsum, row_softmax=row_softmax, dot=dot,
                tdesc_rows=tdesc_rows, dot_grid=dot_grid)


def cases(torch, k):
    g = torch.Generator().manual_seed(0)

    def rnd(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=g).to(dtype)

    def add_case(n):
        def run():
            x, y = rnd(n), rnd(n)
            out = torch.zeros(n)
            ck = k['add'][((n + 15) // 16,)](x, y, out, n, BLOCK=16)
            return ck, out, x + y
        return run

    def row_sum():
        x = rnd(5, 40)
        out = torch.zeros(5)
        ck = k['row_sum'][(5,)](x, out, 40, BLOCK=16)
        return ck, out, x.sum(1)

    def grid2d():
        x = rnd(2, 40)
        out = torch.zeros(2, 40)
        ck = k['grid2d'][(3, 2)](x, out, 40, BLOCK=16)
        return ck, out, x * 2 + torch.arange(2).float()[:, None]

    def ragged():
        out = torch.zeros(7, dtype=torch.int32)
        ck = k['ragged'][(7,)](out, BLOCK=16)
        return ck, out, torch.tensor([p * (p + 1) // 2 for p in range(7)], dtype=torch.int32)

    def row_cumsum():
        x = rnd(5, 16)
        out = torch.zeros(5, 16)
        ck = k['row_cumsum'][(5,)](x, out, BLOCK=16)
        return ck, out, x.cumsum(1)

    def row_softmax():
        x = rnd(3, 13)
        out = torch.zeros(3, 13)
        ck = k['row_softmax'][(3,)](x, out, 13, BLOCK=16)
        return ck, out, torch.softmax(x, 1)

    def dot():
        a, b = rnd(16, 16, dtype=torch.float16), rnd(16, 16, dtype=torch.float16)
        c = torch.zeros(16, 16)
        ck = k['dot'][(1,)](a, b, c, BLOCK=16)
        return ck, c, a.float() @ b.float()

    def tdesc_rows():
        x = rnd(10, 16, dtype=torch.float16)
        y = torch.zeros(10, 16, dtype=torch.float16)
        ck = k['tdesc_rows'][(5,)](x, y, 10, 16, BN=16)
        return ck, y.float(), (x.float() * 2 + 1).half().float()

    def dot_grid():
        a, b = rnd(32, 16, dtype=torch.float16), rnd(16, 48, dtype=torch.float16)
        c = torch.zeros(32, 48)
        ck = k['dot_grid'][(3, 2)](a, b, c, K=16, BLOCK=16)
        return ck, c, a.float() @ b.float()

    # name: (run, the pass must accept it: True, False, or 'slice' -- only
    # where the target has rctx_slice)
    return {
        'add_odd_grid': (add_case(16 * 5 + 3), True),
        'add_one_program': (add_case(10), True),
        'row_sum_inner_loop': (row_sum, True),
        'grid_2d': (grid2d, True),
        'ragged_trip_counts': (ragged, True),
        'row_cumsum': (row_cumsum, True),
        'row_softmax_masked': (row_softmax, True),
        'dot_one_program': (dot, False),     # never reads program_id(0)
        'tdesc_rows': (tdesc_rows, 'slice'),
        'dot_grid': (dot_grid, 'slice'),
    }


def run_case(name, profile):
    os.environ.setdefault('TNCA_TRITON', '1')
    os.environ['TNCA_PROFILE'] = profile
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'compiler' / 'python')]
    import torch
    import triton
    import triton.language as tl
    from tnca import artifact
    from tnca.triton_backend import driver as D
    run, accept = cases(torch, kernels(tl, triton))[name]
    if accept == 'slice':
        from tnca import device, profiles
        from tnca_cc.target import TargetInfo
        tfile = profiles.profile(device.base_profile(profile))['target_file']
        target = TargetInfo.load(tfile[:-5] if tfile.endswith('.json') else tfile)
        accept = bool(target.capabilities.get('rctx_slice'))
    ck, got, want = run()
    with tempfile.NamedTemporaryFile(suffix='.tnca') as f:
        f.write(ck.asm['tnca'])
        f.flush()
        man, _ = artifact.read_header(f.name)
    tol = 2e-2 if name in ('dot_one_program', 'dot_grid') else 1e-5
    ok = torch.allclose(got.to(want.dtype), want, rtol=tol, atol=tol)
    err = (got.to(want.dtype) - want).abs().max().item() if want.is_floating_point() \
        else int((got != want).sum())
    return dict(ok=bool(ok), err=err, role_mask=man['execution']['role_mask'],
                cycles=D.TNCALauncher.last.last_cycles, accept=accept)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default='core-vrf-sync-v2')
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--child')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(run_case(args.child, args.profile)))
        return 0
    names = ['add_odd_grid', 'add_one_program', 'row_sum_inner_loop', 'grid_2d',
             'ragged_trip_counts', 'row_cumsum', 'row_softmax_masked', 'dot_one_program',
             'tdesc_rows', 'dot_grid']
    if args.only:
        names = [n for n in names if n in args.only]
    rows, good = [], True
    contexts = None
    for n in names:
        res = {}
        for v in ('0', '1'):
            env = dict(os.environ, TNCA_RCTX_SPMD=v,
                       TRITON_CACHE_DIR=tempfile.mkdtemp(prefix='tnca-spmd-'))
            p = subprocess.run([sys.executable, __file__, '--child', n, '--profile', args.profile],
                               capture_output=True, text=True, env=env)
            try:
                res[v] = json.loads(p.stdout.strip().splitlines()[-1])
            except Exception:                                   # noqa: BLE001
                res[v] = dict(ok=False, err=(p.stderr or p.stdout)[-400:], role_mask=None,
                              cycles=None, accept=None)
        off, on = res['0'], res['1']
        if on.get('role_mask') and on['role_mask'] > 1:
            contexts = on['role_mask'].bit_length()
        accept = on.get('accept')
        manifest_ok = off.get('role_mask') == 1 and (
            (on.get('role_mask') or 0) > 1 if accept else on.get('role_mask') == 1)
        ok = off['ok'] and on['ok'] and manifest_ok
        good &= ok
        sp = (off['cycles'] / on['cycles']) if off.get('cycles') and on.get('cycles') else None
        print(f"  {'PASS' if ok else 'FAIL'}  {n:<22s} off {off.get('cycles')!s:>7} cyc  "
              f"on {on.get('cycles')!s:>7} cyc  {('%.3fx' % sp) if sp else '':>7}  "
              f"role_mask {off.get('role_mask')}->{on.get('role_mask')}"
              + ('' if off['ok'] and on['ok'] else f"  err off {off['err']} on {on['err']}"))
        rows.append(dict(name=n, off=off, on=on, ok=ok))
    if args.json:
        args.json.write_text(json.dumps(dict(profile=args.profile, contexts=contexts, rows=rows),
                                        indent=1) + '\n')
    print(f"r39-rctx-spmd-tests [{args.profile}]  {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
