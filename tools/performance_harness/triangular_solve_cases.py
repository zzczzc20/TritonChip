#!/usr/bin/env python3
"""`torch.linalg.solve_triangular` on `tnca`, through the standard path.

    $TORCH_PYTHON tools/performance_harness/triangular_solve_cases.py --out DIR \
        [--quick] [--case NAME ...]

Every graph is `torch.compile(backend="inductor", fullgraph=True)` of plain
Torch code on `tnca` tensors: Inductor, the TNCA decomposition of
`aten.linalg_solve_triangular` (tnca.linalg_decomp: blocked substitution
with inverted diagonal blocks, as matmuls and pointwise kernels), Triton,
the TNCA backend, sealed artifacts and the Session, executing on
TNCA_BACKEND (default `systemc-tlm-grid`, the SystemC/TLM model;
`sim-file` is the single-core RTL simulator). Nothing here names a model,
a layer or a checkpoint.

Per case: the generated wrapper must hold no host computation
(tnca.inductor.host_fallbacks) -- in particular no ATen
`linalg_solve_triangular` call -- every output must stay on the device,
and a second (steady) call must return the cold call's bits.

Numerics. The decomposition is not substitution's operation order, so it
is not bitwise equal to LAPACK; it is judged against an independent
float64 reference (torch CPU in float64 on the same binary32 inputs).
Per matrix of the batch:

    err    = ||X - X64||_inf / ||X64||_inf          (matrix infinity norm)
    kappa  = ||T||_inf * ||T^-1||_inf               (T: the triangle the
             flags select, unit diagonal if unitriangular; float64)
    bound  = n * u * kappa,  u = 2**-24

A case passes when err <= bound for every matrix: the first-order
normwise forward-error bound of a backward-stable triangular solve
(Higham, Accuracy and Stability of Numerical Algorithms, sec 8.1-8.2:
|dT| <= gamma_n |T| gives err <~ gamma_n kappa). The same error of torch's
own binary32 CPU solve (LAPACK trsm) is reported beside it, and the ratio
between the two.

Development evidence: model results are uncalibrated model predictions
and RTL results are simulations; neither is a board record.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
                str(ROOT/'tools/asm'), str(ROOT/'tools/performance_harness/systemc')]

U = 2.0 ** -24


# ------------------------------------------------------------ inputs --------
def triangular(batch, n, *, upper, scale, diag, seed):
    """A random binary32 matrix whose selected triangle has off-diagonal
    entries `scale`*N(0,1)/sqrt(n) and diagonal magnitudes in `diag`
    (random signs). The other triangle is filled with junk the solve must
    ignore."""
    import torch
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(*batch, n, n, generator=g) * (scale / max(n, 1) ** 0.5)
    lo, hi = diag
    d = (torch.rand(*batch, n, generator=g) * (hi - lo) + lo)
    d = d * torch.where(torch.rand(*batch, n, generator=g) < 0.5, -1.0, 1.0)
    A = A.triu(1) if upper else A.tril(-1)
    junk = torch.randn(*batch, n, n, generator=g) * 1e3
    A = A + (junk.tril(-1) if upper else junk.triu(1))
    return (A + torch.diag_embed(d)).float()


def gram_unit_lower(batch, n, d, seed):
    """A unit-lower system with a decayed Gram-matrix strict part:
    T[i, j] = beta_i <k_i, k_j> exp(c_i - c_j) for j < i, with unit-norm k,
    beta in (0, 1) and a non-increasing log-decay c -- the system a chunked
    linear-attention update solves, written as generic math. The diagonal
    holds junk: unitriangular solves never read it."""
    import torch
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(*batch, n, d, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    beta = torch.sigmoid(torch.randn(*batch, n, generator=g))
    c = (-torch.rand(*batch, n, generator=g) * 0.1).cumsum(-1)
    T = (beta[..., :, None] * (k @ k.mT)) * torch.exp(c[..., :, None] - c[..., None, :])
    T = T.tril(-1) + torch.diag_embed(torch.full((*batch, n), 7.0))
    return T.float()


def rhs(shape, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).float()


def effective(A, upper, unitriangular):
    import torch
    T = A.triu() if upper else A.tril()
    if unitriangular:
        T = T - torch.diag_embed(torch.diagonal(T, dim1=-2, dim2=-1)) + torch.eye(A.shape[-1], dtype=A.dtype)
    return T


def inf_norm(M):
    return M.abs().sum(-1).amax(-1)


# ------------------------------------------------------------ cases ---------
# name, builder(quick) -> (A, B, flags, graph); graph(A, B) is the compiled fn.
def _cases(quick):
    import torch
    sb = (1, 2, 2) if quick else (1, 16, 2)
    cases = []

    def solve(flags):
        def fn(A, B):
            return torch.linalg.solve_triangular(A, B, **flags)
        return fn

    def add(name, A, B, flags, fn=None):
        cases.append((name, A, B, flags, fn or solve(flags)))

    # The batched 64x64, 128-column unit-lower case, both right-hand sides
    # of one system solved in one graph.
    A = gram_unit_lower(sb, 64, 128, seed=1)
    B1, B2 = rhs((*sb, 64, 128), 2), rhs((*sb, 64, 128), 3)
    flags = dict(upper=False, unitriangular=True)

    def two(A, B):
        return (torch.linalg.solve_triangular(A, B[0], **flags),
                torch.linalg.solve_triangular(A, B[1], **flags))
    cases.append((f'gram unit-lower {"x".join(map(str, sb))} 64x64 @ 64x128, two RHS', A,
                  torch.stack([B1, B2]), flags, two))
    add('lower non-unit 3x64x64 @ 64x32', triangular((3,), 64, upper=False, scale=1.0, diag=(1, 2), seed=4),
        rhs((3, 64, 32), 5), dict(upper=False))
    add('upper non-unit 37x37 @ 37x5 (odd n)', triangular((), 37, upper=True, scale=1.0, diag=(1, 2), seed=6),
        rhs((37, 5), 7), dict(upper=True))
    add('upper unit 2x100x100 @ 100x16', triangular((2,), 100, upper=True, scale=1.0, diag=(1, 2), seed=8),
        rhs((2, 100, 16), 9), dict(upper=True, unitriangular=True))
    add('lower right-side X A = B, 48x48, B 20x48', triangular((), 48, upper=False, scale=1.0, diag=(1, 2), seed=10),
        rhs((20, 48), 11), dict(upper=False, left=False))
    add('upper unit right-side, 2x20x20, B 2x9x20', triangular((2,), 20, upper=True, scale=1.0, diag=(1, 2), seed=12),
        rhs((2, 9, 20), 13), dict(upper=True, left=False, unitriangular=True))
    add('broadcast batch A 2x1x24x24, B 1x3x24x6', triangular((2, 1), 24, upper=False, scale=1.0, diag=(1, 2), seed=14),
        rhs((1, 3, 24, 6), 15), dict(upper=False))
    add('n=1', triangular((4,), 1, upper=False, scale=1.0, diag=(1, 2), seed=16), rhs((4, 1, 3), 17),
        dict(upper=False))
    add('n=5 lower non-unit @ 5x5', triangular((), 5, upper=False, scale=1.0, diag=(0.5, 2), seed=18),
        rhs((5, 5), 19), dict(upper=False))
    # Moderately ill-conditioned: large off-diagonal entries against small
    # diagonals make kappa_inf grow to ~1e4-1e6.
    add('ill-conditioned lower 32x32 @ 32x8', triangular((2,), 32, upper=False, scale=4.0, diag=(0.5, 1.0), seed=20),
        rhs((2, 32, 8), 21), dict(upper=False))
    add('ill-conditioned upper unit 40x40 @ 40x8', triangular((2,), 40, upper=True, scale=6.0, diag=(1, 1), seed=22),
        rhs((2, 40, 8), 23), dict(upper=True, unitriangular=True))
    # Composed with other device arithmetic in one graph.
    fl = dict(upper=False)

    def composed(A, B):
        X = torch.linalg.solve_triangular(A * 2.0, B + 1.0, **fl)
        return (X @ B.mT).relu()
    add('composed: solve(2A, B+1) @ B^T, relu', triangular((2,), 40, upper=False, scale=1.0, diag=(1, 2), seed=24),
        rhs((2, 40, 12), 25), fl, composed)
    if quick:
        keep = ('upper non-unit 37x37', 'n=5', 'gram unit-lower', 'lower right-side')
        cases = [c for c in cases if c[0].startswith(keep)]
    return cases


def reference(name, A, B, flags, fn):
    """float64 on the CPU (the reference) and binary32 on the CPU (torch's
    own LAPACK, for comparison), through the same Torch function."""
    import torch
    return fn(A.double(), B.double()), fn(A, B)


def judge(A, B, flags, got, ref64, cpu32, *, solve_only):
    """Per-matrix forward error against float64, with the n*u*kappa bound."""
    import torch
    n = A.shape[-1]
    outs = got if isinstance(got, tuple) else (got,)
    r64 = ref64 if isinstance(ref64, tuple) else (ref64,)
    c32 = cpu32 if isinstance(cpu32, tuple) else (cpu32,)
    T = effective(A.double(), flags.get('upper', False), flags.get('unitriangular', False))
    kappa = inf_norm(T) * inf_norm(torch.linalg.inv(T))
    rows = []
    ok = True
    for g, r, c in zip(outs, r64, c32):
        g64 = g.double()
        if not flags.get('left', True) and solve_only:
            # X A = B: the error of X^T against A^T's conditioning.
            g64, r, c = g64.mT, r.mT, c.mT
            kap = inf_norm(T.mT) * inf_norm(torch.linalg.inv(T).mT)
        else:
            kap = kappa
        denom = inf_norm(r).clamp_min(1e-300)
        err = inf_norm(g64 - r) / denom
        err_cpu = inf_norm(c.double() - r) / denom
        finite = bool(torch.isfinite(g64).all())
        if solve_only:
            batch = torch.broadcast_shapes(err.shape, kap.shape)
            bound = n * U * kap.expand(batch)
            err = err.expand(batch)
            err_cpu = err_cpu.expand(batch)
            passed = finite and bool((err <= bound).all())
            rows.append(dict(max_err=float(err.max()), max_err_cpu_fp32=float(err_cpu.max()),
                             max_ratio_to_cpu_fp32=float((err / err_cpu.clamp_min(1e-30)).max()),
                             kappa_inf_max=float(kap.max()), kappa_inf_min=float(kap.min()),
                             bound_min=float(bound.min()), worst_err_over_bound=float((err / bound).max()),
                             finite=finite, passed=passed))
        else:
            # A composed graph: no clean conditioning bound; the device error
            # must stay within 16x torch CPU binary32's (both against float64).
            passed = finite and bool((err <= 16 * err_cpu.clamp_min(n * U)).all())
            rows.append(dict(max_err=float(err.max()), max_err_cpu_fp32=float(err_cpu.max()),
                             max_ratio_to_cpu_fp32=float((err / err_cpu.clamp_min(1e-30)).max()),
                             criterion='err <= 16 * max(err_cpu_fp32, n*u)', finite=finite, passed=passed))
        ok = ok and passed
    return ok, rows


# ------------------------------------------------------------ the run -------
def compile_run(out, name, fn, args):
    import torch
    from torch._inductor.utils import run_and_get_code
    from tnca.inductor import host_fallbacks, triton_kernels
    from tnca.triton_backend.driver import launch_log
    torch._dynamo.reset()
    dargs = [a.to('tnca') for a in args]
    compiled = torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)
    t0 = time.monotonic()
    cold, code = run_and_get_code(compiled, *dargs)
    compile_s = time.monotonic() - t0
    t0 = time.monotonic()
    with launch_log(isolated=True) as launches:
        steady = compiled(*dargs)
    run_s = time.monotonic() - t0
    outs = steady if isinstance(steady, tuple) else (steady,)
    colds = cold if isinstance(cold, tuple) else (cold,)
    host = tuple(o.cpu() for o in outs)
    src = '\n\n'.join(code)
    # Executable lines only: Inductor's comments name the source node.
    executable = '\n'.join(line for line in src.splitlines() if not line.lstrip().startswith('#'))
    record = dict(kernels=triton_kernels(code), host_fallbacks=host_fallbacks(code),
                  aten_solve_in_wrapper=bool(re.search(r'aten\.linalg_solve_triangular|linalg\.solve_triangular',
                                                       executable)),
                  launches_per_call=len(launches),
                  launch_kernels=[e['kernel'] for e in launches],
                  device_cycles=sum(e.get('device_cycles') or 0 for e in launches),
                  outputs_on_device=all(o.device.type == 'tnca' for o in outs),
                  cold_equals_steady=all(torch.equal(c.cpu().view(torch.int32), h.view(torch.int32))
                                         for c, h in zip(colds, host)),
                  compile_seconds=round(compile_s, 3), steady_seconds=round(run_s, 3))
    (out/'code').mkdir(exist_ok=True)
    (out/'code'/(re.sub(r'[^A-Za-z0-9_.-]+', '_', name) + '.py')).write_text(src)
    return (host if isinstance(steady, tuple) else host[0]), record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--quick', action='store_true', help='small batches and a case subset (RTL simulator)')
    ap.add_argument('--case', action='append', help='run only cases whose name starts with this')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1',
                      TRITON_CACHE_DIR=str(out/'triton-cache'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
    os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
    backend = os.environ.setdefault('TNCA_BACKEND', 'systemc-tlm-grid')
    import torch
    import triton
    import tnca
    if not Path(tnca.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit('tnca imported outside the requested source tree')
    if backend.startswith('systemc-tlm'):
        import tlm_backend
        tlm_backend.register()
    import tnca.torch_device as device
    from archbench.provenance import software_tree
    device.register()
    torch._dynamo.config.recompile_limit = 1 << 20
    torch._dynamo.config.accumulated_recompile_limit = 1 << 20
    report = dict(schema='tnca-triangular-solve-cases/1', status='running', quick=args.quick, backend=backend,
                  execution='torch.compile(backend=inductor, fullgraph=True) on tnca tensors, Session on ' + backend,
                  evidence=('model-prediction-uncalibrated' if backend.startswith('systemc') else 'rtl-simulation'),
                  numerics=('per-matrix err = ||X-X64||_inf/||X64||_inf against torch CPU float64 on the same '
                            'binary32 inputs; pass when err <= n*u*kappa_inf(T), u=2^-24 (normwise forward-error '
                            'bound of a backward-stable triangular solve); torch CPU binary32 error reported beside it'),
                  software_tree=software_tree(ROOT, refresh=True),
                  toolchain=dict(torch=str(torch.__version__), triton=str(triton.__version__)),
                  knobs={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')}, rows=[])

    def save():
        report['summary'] = {s: sum(r['status'] == s for r in report['rows']) for s in ('pass', 'fail', 'error')}
        (out/'report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    save()
    started = time.monotonic()
    for name, A, B, flags, fn in _cases(args.quick):
        if args.case and not any(name.startswith(c) for c in args.case):
            continue
        row = dict(case=name, flags=flags, A_shape=list(A.shape), B_shape=list(B.shape))
        try:
            ref64, cpu32 = reference(name, A, B, flags, fn)
            got, record = compile_run(out, name, fn, (A, B))
            solve_only = not name.startswith('composed')
            ok, numerics = judge(A, B, flags, got, ref64, cpu32, solve_only=solve_only)
            structural = (not record['host_fallbacks'] and not record['aten_solve_in_wrapper']
                          and record['outputs_on_device'] and record['cold_equals_steady'])
            row.update(record=record, numerics=numerics, structural_ok=structural,
                       status='pass' if ok and structural else 'fail')
            n0 = numerics[0]
            row['summary'] = (f"err {n0['max_err']:.3g} (cpu fp32 {n0['max_err_cpu_fp32']:.3g}), "
                              f"{record['launches_per_call']} launches, {len(record['kernels'])} kernels, "
                              f"fallbacks {len(record['host_fallbacks'])}")
        except Exception as e:                                   # noqa: BLE001
            row.update(status='error', error=f'{type(e).__name__}: {str(e)[:2000]}',
                       traceback=traceback.format_exc()[-6000:])
            row['summary'] = row['error'][:200]
        report['rows'].append(row)
        print(f"{row['status'].upper():6s} {name:55s} {row['summary']}", flush=True)
        save()
    report.update(status='complete', wall_seconds=round(time.monotonic() - started, 1))
    save()
    print(json.dumps(report['summary']))
    return 0 if not (report['summary']['fail'] or report['summary']['error']) else 1


if __name__ == '__main__':
    sys.exit(main())
