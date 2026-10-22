#!/usr/bin/env python3
"""Generic operators the standard path lacked, on `tnca`: sin/cos, log1p and
softplus, int64 -> float, long scans, skinny GEMMs.

    $TORCH_PYTHON tools/performance_harness/std_ops_cases.py --out DIR \
        [--group trig|log1p|int64|scan|gemm ...] [--quick]

Every graph is `torch.compile(backend="inductor", fullgraph=True)` of plain
Torch code on `tnca` tensors, executed on TNCA_BACKEND (default
`systemc-tlm-grid`; `sim-file` is the single-core RTL simulator). The CPU
computes every reference; nothing names a model, a layer or a checkpoint.

groups
  trig    torch.sin / torch.cos of binary32 over every exponent (each
          binade's ends and middle, both signs, zeros, subnormals, inf, NaN,
          random patterns): bit for bit against the compiler's own sequence
          executed on numpy (transcendental_numeric.Emu), and within one
          binary32 ulp of float64; a rotary table from int64 positions
          (positions.float() * inv_freq, cos and sin, cast to bf16) against
          torch CPU and float64
  log1p   torch.log1p the same way; F.softplus (threshold 20, Inductor's
          log1p(exp(x))) on binary32 and bf16 against float64, with the SFU
          exp's 2**-22 bound in the tolerance
  int64   int64 / uint-range values converted to binary32, bit for bit
          against torch CPU's conversion (one rounding)
  scan    torch.cumsum over 64..4096 elements, last and inner dims, binary32
          and int64, against float64 / exact int64, and no split scan in the
          generated code
  gemm    skinny F.linear (M <= 3, K >= 64*max(M, N)): no decompose-K in the
          generated code, and bit for bit against a per-K binary32 FMA in
          ascending K (ISA sec 9C.1)
  rope    RMS norm then rotary embedding of one decode token, the angle
          table built in the graph (int64 positions x inv_freq as a K = 1
          bmm, cat((f, f))) and fused with the norm's reduction: int64 ->
          float, sin and cos together. Against float64 from the binary32
          angles, normwise, and torch CPU binary32
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm'),
                str(HERE/'systemc'), str(HERE)]

from fp8_cases import Run, bits, compare, per_k_fma  # noqa: E402

GROUPS = ('trig', 'log1p', 'int64', 'scan', 'gemm', 'rope')


def _torch():
    import torch
    return torch


def f32_full_range(quick):
    """binary32 patterns over every exponent: ends and middle of every binade,
    both signs, plus random patterns (all quiet, so NaN in = NaN out)."""
    import numpy as np
    from transcendental_numeric import structured
    v = structured()
    if quick:
        rng = np.random.default_rng(5)
        keep = np.zeros(v.size, dtype=bool)
        keep[rng.choice(v.size, 2048, replace=False)] = True
        e = (v >> 23) & 0xFF
        keep |= (e == 0) | (e == 0xFF) | ((v & 0x7FFFFF) == 0)
        v = v[keep]
    return v


def ulp_report(got, x, fn64):
    """Max |got - fn64(x)| in binary32 ulps at the reference (NaN must match)."""
    import numpy as np
    from transcendental_numeric import ulp32
    y = got.numpy().astype(np.float64)
    with np.errstate(all='ignore'):
        ref = fn64(x.numpy().astype(np.float64))
    yn, rn = np.isnan(y), np.isnan(ref)
    ok = ~(yn | rn)
    with np.errstate(all='ignore'):
        both_inf = np.isinf(y[ok]) & np.isinf(ref[ok]) & (np.sign(y[ok]) == np.sign(ref[ok]))
        err = np.where(both_inf, 0.0, np.abs(y[ok] - ref[ok]) / ulp32(ref[ok]))
    err = np.nan_to_num(err, nan=np.inf)
    return dict(nan_mismatch=int(np.count_nonzero(yn != rn)), max_ulp=float(err.max()) if err.size else 0.0,
                over_1_ulp=int(np.count_nonzero(err > 1.0)))


def bitwise_vs_emulation(got, x_bits, fname):
    import numpy as np
    from transcendental_numeric import evaluate
    want = evaluate(fname, x_bits)
    g = got.contiguous().view(_torch().int32).numpy().view(np.uint32)
    gn = np.isnan(g.view(np.float32))
    wn = np.isnan(want.view(np.float32))
    diff = (g != want) & ~(gn & wn)
    idx = np.flatnonzero(diff)[:5]
    return dict(elements=int(g.size), mismatches=int(np.count_nonzero(diff)),
                nan_encoding_only=int(np.count_nonzero((g != want) & gn & wn)),
                examples=[dict(input=f'0x{int(x_bits[i]):08x}', got=f'0x{int(g[i]):08x}',
                               emulated=f'0x{int(want[i]):08x}') for i in idx])


def _fail(run, group, name, e):
    run.add(group, name, 'error', summary=f'{type(e).__name__}: {str(e)[:300]}',
            traceback=traceback.format_exc()[-4000:])


# ------------------------------------------------------------ groups -------
def group_trig(run):
    torch = _torch()
    import numpy as np
    xb = f32_full_range(run.quick)
    x = torch.from_numpy(xb.view(np.int32).copy()).view(torch.float32)
    for name, fn, f64 in (('sin', torch.sin, np.sin), ('cos', torch.cos, np.cos)):
        case = f'{name} binary32 full range ({x.numel()} values)'
        try:
            (y,), rec = run.compile_run(f'trig/{name}', lambda t, f=fn: (f(t),), (x,), dump=False)
            emu = bitwise_vs_emulation(y, xb, name)
            acc = ulp_report(y, x, f64)
            ok = (emu['mismatches'] == 0 and acc['nan_mismatch'] == 0 and acc['over_1_ulp'] == 0
                  and not rec['host_fallbacks'])
            run.add('trig', case, 'pass' if ok else 'fail', emulation=emu, accuracy=acc, graph=rec,
                    summary=f"emulation mismatches {emu['mismatches']}, max {acc['max_ulp']:.3f} ulp vs float64")
        except Exception as e:                                   # noqa: BLE001
            _fail(run, 'trig', case, e)
    # A rotary table: int64 positions, binary32 frequencies, cos and sin,
    # scaled, cast to bf16 -- the standard formulation.
    g = torch.Generator().manual_seed(3)
    for start, L, dim in ((0, 64, 32), (100000, 48, 64), (262000, 16, 64)) if not run.quick else ((0, 16, 32),):
        case = f'rotary table int64 positions [{start}, {start + L}) dim {dim}'
        try:
            inv_freq = 1.0 / (1e7 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            pos = torch.arange(start, start + L, dtype=torch.int64)[None]
            scale = 1.0

            def rope(p, f):
                freqs = p[..., None].float() * f
                emb = torch.cat((freqs, freqs), dim=-1)
                return (emb.cos() * scale).to(torch.bfloat16), (emb.sin() * scale).to(torch.bfloat16)
            (c, s), rec = run.compile_run(f'trig/rope{start}', rope, (pos, inv_freq), dump=False)
            ce, se = rope(pos, inv_freq)
            freqs64 = (pos[..., None].double() * inv_freq.double())
            emb64 = torch.cat((freqs64, freqs64), dim=-1)
            # the binary32 argument the graph computes, then cos/sin exactly
            arg32 = (pos[..., None].float() * inv_freq)
            arg = torch.cat((arg32, arg32), dim=-1).double()
            cmp = dict(cos_vs_cpu=compare(c, ce), sin_vs_cpu=compare(s, se),
                       cos_vs_exact_of_f32_argument=compare(c, arg.cos().to(torch.bfloat16)),
                       sin_vs_exact_of_f32_argument=compare(s, arg.sin().to(torch.bfloat16)),
                       max_abs_vs_float64_angle=dict(cos=float((c.double() - emb64.cos()).abs().max()),
                                                     sin=float((s.double() - emb64.sin()).abs().max())))
            # bf16 of a <= 1 ulp binary32 value: equal to bf16(exact) except
            # where the exact value is within 1 binary32 ulp of a bf16 tie.
            near = 0
            for t in (arg.cos(), arg.sin()):
                b = t.to(torch.bfloat16).double()
                up = torch.nextafter(b.to(torch.bfloat16), torch.tensor(2.0, dtype=torch.bfloat16)).double()
                dn = torch.nextafter(b.to(torch.bfloat16), torch.tensor(-2.0, dtype=torch.bfloat16)).double()
                mid = torch.minimum((t - (b + up) / 2).abs(), (t - (b + dn) / 2).abs())
                near += int((mid <= t.abs() * 2.0 ** -23).sum())
            mism = cmp['cos_vs_exact_of_f32_argument']['value_mismatches'] + \
                cmp['sin_vs_exact_of_f32_argument']['value_mismatches']
            ok = mism <= near and not rec['host_fallbacks']
            run.add('trig', case, 'pass' if ok else 'fail', comparisons=cmp, graph=rec,
                    within_one_f32_ulp_of_bf16_tie=near,
                    summary=f'bf16 mismatches vs exact(f32 arg) {mism} (allowed: {near} near-tie), '
                            f"vs CPU {cmp['cos_vs_cpu']['value_mismatches']}+{cmp['sin_vs_cpu']['value_mismatches']}")
        except Exception as e:                                   # noqa: BLE001
            _fail(run, 'trig', case, e)


def group_log1p(run):
    torch = _torch()
    import numpy as np
    F = torch.nn.functional
    xb = f32_full_range(run.quick)
    x = torch.from_numpy(xb.view(np.int32).copy()).view(torch.float32)
    case = f'log1p binary32 full range ({x.numel()} values)'
    try:
        (y,), rec = run.compile_run('log1p/log1p', lambda t: (torch.log1p(t),), (x,), dump=False)
        emu = bitwise_vs_emulation(y, xb, 'log1p')
        acc = ulp_report(y, x, np.log1p)
        ok = emu['mismatches'] == 0 and acc['nan_mismatch'] == 0 and acc['over_1_ulp'] == 0 and \
            not rec['host_fallbacks']
        run.add('log1p', case, 'pass' if ok else 'fail', emulation=emu, accuracy=acc, graph=rec,
                summary=f"emulation mismatches {emu['mismatches']}, max {acc['max_ulp']:.3f} ulp vs float64")
    except Exception as e:                                       # noqa: BLE001
        _fail(run, 'log1p', case, e)
    g = torch.Generator().manual_seed(11)
    n = 4096 if run.quick else 1 << 16
    xs = torch.cat([torch.linspace(-120, 40, n // 2), torch.randn(n // 2, generator=g) * 8,
                    torch.tensor([0.0, -0.0, 20.0, 20.000002, 19.999998, -1e-30, 1e-30, float('inf'),
                                  -float('inf'), float('nan')])])
    # exp on its own (the SFU exp2 behind it, ISA sec 8.8): what softplus
    # inherits, measured, not assumed.
    exp_dev = None
    case = f'exp binary32 ({xs.numel()} values), the input of softplus'
    try:
        (exp_dev,), rec = run.compile_run('log1p/exp', lambda t: (torch.exp(t),), (xs.float(),), dump=False)
        e64 = torch.exp(xs.double())
        normal = torch.isfinite(e64) & (e64 >= 2.0 ** -126) & (e64 <= float(torch.finfo(torch.float32).max))
        rel = ((exp_dev.double() - e64).abs() / e64)[normal]
        worst = int(torch.argmax(rel))
        run.add('log1p', case, 'pass' if not rec['host_fallbacks'] else 'fail', graph=rec,
                max_relative_error_normal_results=float(rel.max()),
                worst_input=float(xs.float()[normal][worst]),
                over_2_pow_minus_22=int((rel > 2.0 ** -22).sum()),
                summary=f'measured only: max rel err {float(rel.max()):.3e} on normal results '
                        f'(worst x = {float(xs.float()[normal][worst]):.6g}); '
                        f'{int((rel > 2.0 ** -22).sum())} above 2**-22')
    except Exception as e:                                       # noqa: BLE001
        _fail(run, 'log1p', case, e)
    for dtype in (torch.float32, torch.bfloat16):
        case = f'softplus {dtype} ({xs.numel()} values)'
        try:
            xin = xs.to(dtype)
            (y,), rec = run.compile_run(f'log1p/softplus-{dtype}', lambda t: (F.softplus(t),), (xin,), dump=False)
            x64 = xin.double()
            # The log1p part, isolated: Inductor computes
            # where(x > 20, x, log1p(exp(x))) in binary32. Fed the device's
            # own exp (same lowering, run alone above), the exact log1p,
            # rounded to dtype, must agree within one ulp of dtype.
            if exp_dev is None:
                raise RuntimeError('no device exp to compose with')
            xf = xin.float()
            (e_same,), _ = run.compile_run(f'log1p/exp-{dtype}', lambda t: (torch.exp(t),), (xf,), dump=False)
            composed = torch.where(xf.double() > 20, xf.double(), torch.log1p(e_same.double())).to(dtype)
            cmp = compare(y, composed)
            ulps = cmp.get('max_ulp_diff', 0)
            nan_ok = bool((torch.isnan(y.double()) == torch.isnan(composed.double())).all())
            # End to end against float64, for the record (exp's error
            # dominates where softplus(x) ~ exp(x)).
            ref = torch.where(x64 > 20, x64, torch.log1p(torch.exp(x64)))
            fin = torch.isfinite(ref) & torch.isfinite(y.double()) & (ref.abs() >= 2.0 ** -126)
            rel = ((y.double() - ref).abs() / ref.abs())[fin]
            ok = ulps <= 1 and nan_ok and not rec['host_fallbacks']
            run.add('log1p', case, 'pass' if ok else 'fail', graph=rec, versus_log1p_of_device_exp=cmp,
                    max_ulp_vs_log1p_of_device_exp=ulps, nan_ok=nan_ok,
                    end_to_end_max_relative_error_normal=float(rel.max()),
                    end_to_end_worst_input=float(xin.float()[fin][int(torch.argmax(rel))]),
                    flushed_subnormal_results=int(((y.double() == 0) & (ref != 0) & (ref.abs() < 2.0 ** -126)).sum()),
                    versus_eager_cpu=compare(y, F.softplus(xin)),
                    summary=f'<= {ulps} ulp vs exact log1p of the device exp; end to end '
                            f'{float(rel.max()):.3e} rel (exp-bound)')
        except Exception as e:                                   # noqa: BLE001
            _fail(run, 'log1p', case, e)


def group_int64(run):
    torch = _torch()
    g = torch.Generator().manual_seed(17)
    n = 2048 if run.quick else 1 << 16
    length = torch.randint(0, 64, (n,), generator=g)
    raw = torch.randint(-2 ** 63, 2 ** 63 - 1, (n,), generator=g, dtype=torch.int64)
    vals = torch.where(length < 63, raw % (torch.ones_like(raw) << length.clamp(max=62)), raw)
    vals = vals * torch.where(torch.rand(n, generator=g) < 0.5, -1, 1)
    edges = []
    for p in (24, 25, 31, 32, 33, 53, 62):
        edges += [(1 << p) + d for d in (-1, 0, 1, 2, 3)] + [(1 << p) + (1 << max(p - 24, 0))]
    edges += [0, 1, -1, -(2 ** 63), 2 ** 63 - 1]
    vals = torch.cat([vals, torch.tensor(edges, dtype=torch.int64), -torch.tensor(edges[:-2], dtype=torch.int64)])
    for name, fn in (('to float32', lambda t: (t.to(torch.float32),)),
                     ('to float32 * scale (positions * inv_freq form)', lambda t: (t.float() * 0.5,))):
        case = f'int64 {name} ({vals.numel()} values)'
        try:
            (y,), rec = run.compile_run(f'int64/{name}', fn, (vals,), dump=False)
            ref = fn(vals)[0]
            cmp = compare(y, ref)
            ok = cmp['equal'] and not rec['host_fallbacks']
            run.add('int64', case, 'pass' if ok else 'fail', comparison=cmp, graph=rec,
                    summary=f"{cmp['elements']} elements, {cmp['bit_mismatches']} bit mismatches vs torch CPU")
        except Exception as e:                                   # noqa: BLE001
            _fail(run, 'int64', case, e)


def group_scan(run):
    torch = _torch()
    g = torch.Generator().manual_seed(29)
    shapes = ((2, 3, 4, 64), (1, 2, 2, 256), (3, 4096)) if not run.quick else ((1, 2, 2, 64), (2, 300))
    for shape in shapes:
        for dim in sorted({len(shape) - 1, max(len(shape) - 2, 0)}):
            case = f'cumsum float32 {shape} dim {dim}'
            try:
                x = torch.randn(*shape, generator=g)
                (y,), rec = run.compile_run(f'scan/{shape}/{dim}', lambda t, d=dim: (torch.cumsum(t, d),),
                                            (x,), dump=False)
                ref = torch.cumsum(x.double(), dim)
                # Any summation order of n terms: |err| <= (n-1) u sum|x_i|.
                bound = (shape[dim] - 1) * 2.0 ** -24 * torch.cumsum(x.double().abs(), dim)
                err = (y.double() - ref).abs()
                split = any('split_scan' in k or 'triton_spl' in k for k in rec['kernels'])
                ok = bool((err <= bound + 1e-30).all()) and not split and not rec['host_fallbacks']
                run.add('scan', case, 'pass' if ok else 'fail', graph=rec, max_abs_err=float(err.max()),
                        max_err_over_bound=float((err / (bound + 1e-30)).max()), split_scan=split,
                        versus_cpu=compare(y, torch.cumsum(x, dim)),
                        summary=f'max err/bound {float((err / (bound + 1e-30)).max()):.3f}, split scan {split}')
            except Exception as e:                               # noqa: BLE001
                _fail(run, 'scan', case, e)
    for n in (64, 300):
        case = f'cumsum int64 over {n} (positions from a mask)'
        try:
            mask = (torch.rand(2, n, generator=g) > 0.2).to(torch.int64)
            (y,), rec = run.compile_run(f'scan/int64-{n}', lambda m: (m.cumsum(-1) - 1,), (mask,), dump=False)
            cmp = compare(y, mask.cumsum(-1) - 1)
            run.add('scan', case, 'pass' if cmp['equal'] and not rec['host_fallbacks'] else 'fail',
                    comparison=cmp, graph=rec, summary=f"{cmp['bit_mismatches']} mismatches (exact)")
        except Exception as e:                                   # noqa: BLE001
            if '64-bit tensor operation scf.for' in str(e):
                # A 64-bit scan longer than one register block is a looped
                # scan carrying an int64 across iterations, which the 64-bit
                # lowering (tnca_cc.wide) does not carry: refused at compile
                # time with that diagnostic. Not one of the gaps fixed here.
                run.add('scan', case, 'refused', summary='known gap: 64-bit looped scan '
                        '(64-bit tensor operation scf.for), refused at compile time')
            else:
                _fail(run, 'scan', case, e)


def group_gemm(run):
    torch = _torch()
    F = torch.nn.functional
    g = torch.Generator().manual_seed(31)
    shapes = ((1, 2048, 16), (3, 2048, 16), (1, 4096, 32), (2, 1024, 8)) if not run.quick else ((1, 1024, 16),)
    for dtype in (torch.bfloat16, torch.float32):
        for M, K, N in shapes:
            case = f'skinny linear {dtype} {M}x{K}x{N}'
            try:
                x = torch.randn(M, K, generator=g).to(dtype)
                w = (torch.randn(N, K, generator=g) * 0.05).to(dtype)
                (y,), rec = run.compile_run(f'gemm/{dtype}/{M}x{K}x{N}', lambda a, b: (F.linear(a, b),), (x, w),
                                            dump=False)
                import re
                code = (run.out/'code'/(re.sub(r'[^A-Za-z0-9_.-]+', '_', f'gemm/{dtype}/{M}x{K}x{N}') + '.py')).read_text()
                decomposed = 'decompose_k' in code or any('decompose_k' in k for k in rec['kernels'])
                if dtype == torch.bfloat16:
                    ref = per_k_fma(x, w.t()).to(torch.bfloat16)
                    cmp = compare(y, ref)
                    ok = cmp['equal']
                else:
                    # There is no binary32 MMA: a binary32 tl.dot is vector
                    # FMAs, per-K in ascending K from +0.0 within one K block
                    # (frontend._dot_f32), and the mm template adds each
                    # block's dot to its accumulator (`acc += tl.dot(a, b)`).
                    # Reference: that order, each fma rounded once (exact
                    # emulation, transcendental_numeric.fma32).
                    import numpy as np
                    from transcendental_numeric import fma32
                    bk = int(os.environ.get('TNCA_GEMM_TILE', '16,16,16').split(',')[2])
                    xa, wa = x.numpy(), w.t().contiguous().numpy()
                    acc = np.zeros((M, N), dtype=np.float32)
                    for k0 in range(0, K, bk):
                        part = np.zeros((M, N), dtype=np.float32)
                        for k in range(k0, min(k0 + bk, K)):
                            part = fma32(np.broadcast_to(xa[:, k:k+1], (M, N)), np.broadcast_to(wa[k:k+1, :], (M, N)),
                                         part)
                        acc = (acc + part).astype(np.float32)
                    cmp = compare(y, torch.from_numpy(acc))
                    ref64 = x.double() @ w.double().t()
                    cmp['max_abs_vs_float64'] = float((y.double() - ref64).abs().max())
                    cmp['reference'] = f'binary32 fma per K within {bk}-deep K blocks, blocks summed in order'
                    ok = cmp['value_mismatches'] == 0
                ok = ok and not decomposed and not rec['host_fallbacks']
                run.add('gemm', case, 'pass' if ok else 'fail', comparison=cmp, graph=rec, decompose_k=decomposed,
                        summary=f"{cmp['bit_mismatches']} bit mismatches vs per-K FMA (ascending K); "
                                f'decompose-K {decomposed}')
            except Exception as e:                               # noqa: BLE001
                _fail(run, 'gemm', case, e)


def group_rope(run):
    """RMS norm + rotary embedding of a decode token (any RoPE model's q/k path).

    Inductor fuses the norm's reduction over D with the rotary epilogue, which
    reads the angle table at `r % (D/2)` (the cat of two copies). Pass:
    compiles, no host fallback, and the normwise error against float64 (from
    the same binary32 angles) is within 64 binary32 ulps (2**-18: rsqrt on
    the SFU, sin/cos <= 1 ulp, a mean over D, two products and a sum).
    Not a reproducer of the Qwen3-VL decode descriptor failure (that kernel
    also selects mrope sections); this fused form compiles with or without
    tnca.descriptor_broadcast.
    """
    torch = _torch()
    g = torch.Generator().manual_seed(37)
    shapes = ((1, 16, 128), (2, 4, 64)) if not run.quick else ((1, 2, 64),)

    def angles(pos, inv_freq):
        # inv_freq @ positions as a (K = 1) batched matmul, as HF rotary
        # embeddings compute it: one binary32 product per element.
        B = pos.shape[0]
        f = inv_freq[None, :, None].expand(B, -1, 1) @ pos[:, None, :].float()
        return f.transpose(1, 2)                                          # [B, 1, D/2]

    def fn(x, w, freqs):
        emb = torch.cat((freqs, freqs), dim=-1)[:, None]                  # [B, 1, 1, D]
        xf = x.to(freqs.dtype)
        xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        half = x.shape[-1] // 2
        rot = torch.cat((-xn[..., half:], xn[..., :half]), dim=-1)
        return (xn * emb.cos() + rot * emb.sin(),)

    for B, H, D in shapes:
        case = f'rms norm + rotary, decode, B={B} H={H} D={D}'
        try:
            x = torch.randn(B, H, 1, D, generator=g).to(torch.bfloat16)
            w = 1 + 0.1 * torch.randn(D, generator=g)
            pos = torch.randint(0, 32768, (B, 1), generator=g, dtype=torch.int64)
            inv_freq = 1.0 / (1e6 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
            (y,), rec = run.compile_run(f'rope/{B}x{H}x{D}', lambda a, b, c, d: fn(a, b, angles(c, d)),
                                        (x, w, pos, inv_freq), dump=False)
            # Reference: the binary32 angles (exact product, one rounding --
            # what any binary32 rotary computes), the rest in float64.
            f32 = angles(pos, inv_freq)
            ref = fn(x, w.double(), f32.double())[0]
            cpu = fn(x, w, f32)[0]
            den = ref.abs().max().item()
            err = (y.double() - ref).abs().max().item() / den
            err_cpu = (cpu.double() - ref).abs().max().item() / den
            ok = err <= 2.0 ** -18 and not rec['host_fallbacks']
            run.add('rope', case, 'pass' if ok else 'fail', graph=rec, err=err, err_cpu=err_cpu,
                    summary=f'normwise err vs float64 {err:.3g} (bound 2**-18 = {2.0 ** -18:.3g}; '
                            f'torch CPU fp32 {err_cpu:.3g}), host fallbacks {len(rec["host_fallbacks"])}')
        except Exception as e:                                   # noqa: BLE001
            _fail(run, 'rope', case, e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--group', action='append', choices=GROUPS)
    ap.add_argument('--quick', action='store_true', help='reduced inputs (for the RTL simulator)')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1',
                      TRITON_CACHE_DIR=str(out/'triton-cache'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
    os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
    # A launch over ~10**6 elements runs longer than the driver's default
    # 5e6-cycle hang guard; the guard is not part of what is tested.
    os.environ.setdefault('TNCA_DEADLINE', '200000000')
    backend = os.environ.setdefault('TNCA_BACKEND', 'systemc-tlm-grid')
    run = Run(out, args.quick, backend)
    import torch
    import triton
    import tnca
    if not Path(tnca.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit('tnca imported outside the requested source tree')
    if backend.startswith('systemc-tlm'):
        import tlm_backend
        run.models = tlm_backend.register()
    import tnca.torch_device as device
    from archbench.provenance import software_tree
    device.register()
    torch._dynamo.config.recompile_limit = 1 << 20
    torch._dynamo.config.accumulated_recompile_limit = 1 << 20
    run.report = dict(schema='tnca-std-ops-cases/1', status='running', quick=args.quick, backend=backend,
                      execution='torch.compile(backend=inductor, fullgraph=True) on tnca tensors, Session on '
                                + backend, evidence=('model-prediction-uncalibrated' if backend.startswith('systemc')
                                                     else 'rtl-simulation'),
                      software_tree=software_tree(ROOT, refresh=True),
                      toolchain=dict(torch=str(torch.__version__), triton=str(triton.__version__)),
                      knobs={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')})
    run.save()
    started = time.monotonic()
    for group in args.group or GROUPS:
        globals()[f'group_{group}'](run)
    run.report.update(status='complete', wall_seconds=round(time.monotonic() - started, 1))
    run.save()
    print(json.dumps(run.report['summary']))
    return 0 if not (run.report['summary']['fail'] or run.report['summary']['error']) else 1


if __name__ == '__main__':
    sys.exit(main())
