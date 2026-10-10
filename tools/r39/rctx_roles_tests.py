#!/usr/bin/env python3
"""R39-12 Mode A: a GEMM loop as a producer and a consumer on a Role Context pair.

    python3 tools/r39/rctx_roles_tests.py [--profile core-vrf-sync-v5] [--only NAME ...]

Each case is a `@triton.jit` GEMM with a K loop, launched through the TNCA
plugin on CPU tensors twice -- TNCA_RCTX_ROLES=0, then 2 (the split forced
past ring_estimate, so the mechanism is tested on every loop; rctx_spmd on in
both) -- and checked against PyTorch both times. With the switch on, a target
with `rctx_pair` must compile every case as two roles per program (the
manifest's program_to_role has two roles per program) and one without it must
not; the cycles of both runs are reported.

The shapes are chosen to break the ring: one K step (the ring never wraps),
an odd number (the loop leaves after stage 0 of a trip), several programs per
pair (the epochs carry from one program to the next), more programs than
pairs and not a multiple of them, a second grid axis, B read as W transposed
(the transposed transfer, Inductor's linear layer) and a bias in the
epilogue (Inductor's addmm).
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
    def gemm(a, b, c, N, K: tl.constexpr, BLOCK: tl.constexpr):
        # C[m, n] = sum over K of A[m, k] B[k, n], one 16x16 tile per program
        n0 = tl.program_id(0) * BLOCK
        m0 = tl.program_id(1) * BLOCK
        r = tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        for k in range(0, K, BLOCK):
            A = tl.load(a + (m0 + r)[:, None] * K + (k + r)[None, :])
            B = tl.load(b + (k + r)[:, None] * N + (n0 + r)[None, :])
            acc += tl.dot(A, B)
        tl.store(c + (m0 + r)[:, None] * N + (n0 + r)[None, :], acc)

    @triton.jit
    def linear(x, w, bias, out, N, K: tl.constexpr, BLOCK: tl.constexpr):
        # out = x @ w.T + bias: B[k, n] = w[n, k], contiguous down its columns
        n0 = tl.program_id(0) * BLOCK
        m0 = tl.program_id(1) * BLOCK
        r = tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        for k in range(0, K, BLOCK):
            A = tl.load(x + (m0 + r)[:, None] * K + (k + r)[None, :])
            B = tl.load(w + (n0 + r)[None, :] * K + (k + r)[:, None])
            acc += tl.dot(A, B)
        acc += tl.load(bias + n0 + r)[None, :]
        tl.store(out + (m0 + r)[:, None] * N + (n0 + r)[None, :], acc)

    @triton.jit
    def gemm_ptrs(a, b, c, N, K: tl.constexpr, BLOCK: tl.constexpr):
        # the pointers carried round the loop, as Inductor's bmm template
        # does: the importer spills the pointer tiles to scratch in the loop
        n0 = tl.program_id(0) * BLOCK
        m0 = tl.program_id(1) * BLOCK
        r = tl.arange(0, BLOCK)
        pa = a + (m0 + r)[:, None] * K + r[None, :]
        pb = b + r[:, None] * N + (n0 + r)[None, :]
        acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        for k in range(0, K, BLOCK):
            acc += tl.dot(tl.load(pa), tl.load(pb))
            pa += BLOCK
            pb += BLOCK * N
        tl.store(c + (m0 + r)[:, None] * N + (n0 + r)[None, :], acc)

    return dict(gemm=gemm, linear=linear, gemm_ptrs=gemm_ptrs)


def cases(torch, k):
    g = torch.Generator().manual_seed(0)

    def rnd(*shape):
        return torch.randn(*shape, generator=g).to(torch.float16)

    def gemm(m, n, kk):
        def run():
            a, b = rnd(m, kk), rnd(kk, n)
            c = torch.zeros(m, n)
            ck = k['gemm'][(n // 16, m // 16)](a, b, c, n, K=kk, BLOCK=16)
            return ck, c, a.float() @ b.float()
        return run

    def gemm_ptrs(m, n, kk):
        def run():
            a, b = rnd(m, kk), rnd(kk, n)
            c = torch.zeros(m, n)
            ck = k['gemm_ptrs'][(n // 16, m // 16)](a, b, c, n, K=kk, BLOCK=16)
            return ck, c, a.float() @ b.float()
        return run

    def linear(m, n, kk):
        def run():
            x, w, bias = rnd(m, kk), rnd(n, kk), rnd(n).float()
            out = torch.zeros(m, n)
            ck = k['linear'][(n // 16, m // 16)](x, w, bias, out, n, K=kk, BLOCK=16)
            return ck, out, x.float() @ w.float().T + bias
        return run

    # name: (run, specialised where the target has rctx_pair)
    return {
        # One K step: Triton folds the loop away, so there is nothing to
        # overlap; the kernel must be refused, and still be right.
        'gemm_one_step': (gemm(16, 16, 16), False),
        'gemm_odd_steps': (gemm(16, 16, 48), True),
        'gemm_grid_3x2': (gemm(32, 48, 64), True),
        'gemm_five_programs': (gemm(16, 80, 32), True),
        'linear_one_program': (linear(16, 16, 64), True),
        'linear_four_programs': (linear(16, 64, 64), True),
        'linear_odd_steps_grid': (linear(32, 48, 48), True),
        # The loop writes memory (its spilled pointer tiles): refused.
        'gemm_carried_pointers': (gemm_ptrs(32, 32, 48), False),
    }


def run_case(name, profile):
    os.environ.setdefault('TNCA_TRITON', '1')
    os.environ['TNCA_PROFILE'] = profile
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'compiler' / 'python')]
    import torch
    import triton
    import triton.language as tl
    from tnca import artifact, device, profiles
    from tnca.triton_backend import driver as D
    from tnca_cc.target import TargetInfo
    tfile = profiles.profile(device.base_profile(profile))['target_file']
    target = TargetInfo.load(tfile[:-5] if tfile.endswith('.json') else tfile)
    run, want_pair = cases(torch, kernels(tl, triton))[name]
    accept = bool(target.capabilities.get('rctx_pair')) and want_pair
    ck, got, want = run()
    with tempfile.NamedTemporaryFile(suffix='.tnca') as f:
        f.write(ck.asm['tnca'])
        f.flush()
        man, _ = artifact.read_header(f.name)
    ok = torch.allclose(got, want, rtol=2e-2, atol=2e-2)
    p2r = man['execution']['program_to_role']
    return dict(ok=bool(ok), err=(got - want).abs().max().item(),
                roles=len(p2r.get('0', [])), programs=len(p2r),
                cycles=D.TNCALauncher.last.last_cycles, accept=accept,
                why=man['optional_metadata']['provenance'].get('rctx_roles_refused'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default='core-vrf-sync-v5')
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--child')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(run_case(args.child, args.profile)))
        return 0
    names = ['gemm_one_step', 'gemm_odd_steps', 'gemm_grid_3x2', 'gemm_five_programs',
             'linear_one_program', 'linear_four_programs', 'linear_odd_steps_grid',
             'gemm_carried_pointers']
    if args.only:
        names = [n for n in names if n in args.only]
    rows, good = [], True
    for n in names:
        res = {}
        for v in ('0', '2'):
            env = dict(os.environ, TNCA_RCTX_ROLES=v,
                       TRITON_CACHE_DIR=tempfile.mkdtemp(prefix='tnca-roles-'))
            p = subprocess.run([sys.executable, __file__, '--child', n, '--profile', args.profile],
                               capture_output=True, text=True, env=env)
            try:
                res[v] = json.loads(p.stdout.strip().splitlines()[-1])
            except Exception:                                   # noqa: BLE001
                res[v] = dict(ok=False, err=(p.stderr or p.stdout)[-400:], roles=None,
                              cycles=None, accept=None)
        off, on = res['0'], res['2']
        manifest_ok = off.get('roles') == 1 and on.get('roles') == (2 if on.get('accept') else 1)
        ok = off['ok'] and on['ok'] and manifest_ok
        good &= ok
        sp = (off['cycles'] / on['cycles']) if off.get('cycles') and on.get('cycles') else None
        print(f"  {'PASS' if ok else 'FAIL'}  {n:<24s} off {off.get('cycles')!s:>7} cyc  "
              f"on {on.get('cycles')!s:>7} cyc  {('%.3fx' % sp) if sp else '':>7}  "
              f"roles {off.get('roles')}->{on.get('roles')}"
              + ('' if ok else f"  err off {off.get('err')} on {on.get('err')} {on.get('why') or ''}"))
        rows.append(dict(name=n, ok=ok, off=off, on=on, speedup=sp))
    print(f"r39-rctx-roles-tests [{args.profile}]  {'PASS' if good else 'FAIL'}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(profile=args.profile, ok=good, cases=rows), indent=2))
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
