"""Blockwise-scaled `_scaled_mm` / `_scaled_mm_v2` on `tnca` through Inductor.

    fp8_blockwise_cases.py --out OUT [--quick] [--only SUBSTR]

Each case compiles `F.scaled_mm` (aten._scaled_mm_v2, recipes BlockWise1x128 /
BlockWise128x128) or `torch._scaled_mm` (v1, blockwise scale shapes) with
`torch.compile(backend=inductor, fullgraph=True)` on tnca tensors, and
compares the output bit for bit with `blockwise_reference`: plain torch in
the order the device executes. CPU eager refuses blockwise scales, so there
is no torch CPU result to compare with.

  per 128-deep K block kb, ascending:
      partial = binary32 sum of the exact FP8 products, one rounding per
                product-add, ascending K, from +0.0 (per_k_fma)
      acc = acc + (partial * sa[row, kb]) * sb[kb, col]   (each op rounded once)
  out = (acc [+ bias]) rounded once to the output dtype

The scales are laid out as the op's convention says (v2: the CUDA kernels'
shapes and strides; v1: the meta function's shapes), plus one case with
plain contiguous scales; `SA[m, kb]` and `SB[kb, n]` below are the logical
scale of each output element's row and column in K block kb.
"""
import argparse
import json
import os
import re
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools/performance_harness')]
import fp8_cases as fc                      # noqa: E402

TILE = 128


def cdiv(a, b):
    return -(-a // b)


def rand_scales(g, *shape):
    """Scales as an amax/448 would give: not powers of two, so the
    multiplies round."""
    import torch
    return (torch.rand(*shape, generator=g) * 0.05 + 0.002).float()


def blockwise_reference(a, b, SA, SB, bias, out_dtype):
    """a (M, K), b (K, N) FP8; SA (M, KB), SB (KB, N) the logical scales."""
    import torch
    M, K = a.shape
    N = b.shape[1]
    acc = torch.zeros(M, N, dtype=torch.float32)
    for kb in range(cdiv(K, TILE)):
        ks = slice(kb * TILE, min(K, (kb + 1) * TILE))
        partial = fc.per_k_fma(a[:, ks], b[ks, :])
        acc = acc + (partial * SA[:, kb:kb+1]) * SB[kb:kb+1, :]
    if bias is not None:
        acc = acc + bias.float()
    return acc.to(out_dtype)


def make_case(api, pair, M, N, K, out_dtype, fmt_a, fmt_b, g, *, contiguous=False, bias=False):
    """Operands, scales in the op's layout, the logical SA/SB, and the call."""
    import torch
    import torch.nn.functional as F
    from torch.nn.functional import ScalingType as ST
    a, _ = fc.fp8_operands(M, N, K, fmt_a, g)
    _, b = fc.fp8_operands(M, N, K, fmt_b, g)           # b (K, N) column-major
    KB, MB, NB, L4 = cdiv(K, TILE), cdiv(M, TILE), cdiv(N, TILE), cdiv(cdiv(K, TILE), 4) * 4
    ta, tb = pair
    # Logical per-(row-block, kb) values.
    va = rand_scales(g, M if ta == 1 else MB, KB)
    vb = rand_scales(g, N if tb == 1 else NB, KB)
    SA = va.repeat_interleave(ta, 0)[:M]                # (M, KB)
    SB = vb.repeat_interleave(tb, 0)[:N].t().contiguous()   # (KB, N)
    pad = lambda v: torch.cat([v, rand_scales(g, v.shape[0], L4 - KB)], 1)   # (rows, L4)
    if api == 'v2':
        rec = {1: ST.BlockWise1x128, TILE: ST.BlockWise128x128}
        def lay(v, t):
            if contiguous:
                return v.contiguous() if t == 1 else v.t().contiguous()   # (rows, KB) / (KB, blocks)
            # CUDA (ATen ScaledBlas.cpp): 1x128 (rows, KB) stride (1, rows);
            # 128x128 (round_up(KB, 4), blocks) stride (1, round_up(KB, 4)).
            return v.t().contiguous().t() if t == 1 else pad(v).t()
        sa, sb = lay(va, ta), lay(vb, tb)
        o = out_dtype
        if bias:
            fn = lambda a, b, sa, sb, c: F.scaled_mm(a, b, sa, rec[ta], sb, rec[tb], bias=c, output_dtype=o)
        else:
            fn = lambda a, b, sa, sb: F.scaled_mm(a, b, sa, rec[ta], sb, rec[tb], output_dtype=o)
    else:
        # v1, the meta function's shapes: scale_a (rows, KB), scale_b (KB, cols).
        sa = va.t().contiguous().t() if ta == 1 else va
        sb = vb.t().contiguous()
        o = out_dtype
        if bias:
            fn = lambda a, b, sa, sb, c: torch._scaled_mm(a, b, sa, sb, bias=c, out_dtype=o)
        else:
            fn = lambda a, b, sa, sb: torch._scaled_mm(a, b, sa, sb, out_dtype=o)
    args = [a, b, sa, sb]
    c = None
    if bias:
        c = (torch.randn(N, generator=g) * 0.5).to(out_dtype)
        args.append(c)
    ref = blockwise_reference(a, b, SA, SB, c, out_dtype)
    return fn, tuple(args), ref


def cases(quick):
    import torch
    E4, E5, BF, F32 = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float32
    P1, P2, P3 = (1, TILE), (1, 1), (TILE, 1)       # (1x128,128x128), (1x128,1x128), (128x128,1x128)
    name = {P1: '1x128*128x128', P2: '1x128*1x128', P3: '128x128*1x128'}
    rows = [
        ('v2', P1, 1, 256, 256, BF), ('v2', P1, 7, 200, 328, F32), ('v2', P1, 129, 136, 200, BF),
        ('v2', P2, 1, 136, 328, F32), ('v2', P2, 7, 256, 256, BF), ('v2', P2, 129, 200, 200, BF),
        ('v2', P3, 1, 200, 256, BF), ('v2', P3, 7, 136, 328, F32), ('v2', P3, 129, 256, 200, BF),
        ('v1', P1, 7, 200, 328, BF), ('v1', P2, 129, 136, 256, F32), ('v1', P3, 129, 200, 328, BF),
    ]
    out = [(f'{api} {name[p]} {M}x{N}x{K} -> {str(o)[6:]}', dict(api=api, pair=p, M=M, N=N, K=K,
            out_dtype=o, fmt_a=E4, fmt_b=E4)) for api, p, M, N, K, o in rows]
    out += [
        ('v2 1x128*128x128 7x256x256 +bias -> bfloat16',
         dict(api='v2', pair=P1, M=7, N=256, K=256, out_dtype=BF, fmt_a=E4, fmt_b=E4, bias=True)),
        ('v2 1x128*1x128 e4m3*e5m2 7x136x200 -> bfloat16',
         dict(api='v2', pair=P2, M=7, N=136, K=200, out_dtype=BF, fmt_a=E4, fmt_b=E5)),
        ('v2 1x128*128x128 contiguous scales 7x136x200 -> float32',
         dict(api='v2', pair=P1, M=7, N=136, K=200, out_dtype=F32, fmt_a=E4, fmt_b=E4, contiguous=True)),
        ('v1 1x128*1x128 7x136x328 +bias -> bfloat16',
         dict(api='v1', pair=P2, M=7, N=136, K=328, out_dtype=BF, fmt_a=E4, fmt_b=E4, bias=True)),
    ]
    if quick:
        out = [('v2 1x128*128x128 1x136x200 -> bfloat16',
                dict(api='v2', pair=P1, M=1, N=136, K=200, out_dtype=BF, fmt_a=E4, fmt_b=E4)),
               ('v1 128x128*1x128 7x16x144 -> float32',
                dict(api='v1', pair=P3, M=7, N=16, K=144, out_dtype=F32, fmt_a=E4, fmt_b=E4))]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--quick', action='store_true', help='two small cases (for the RTL simulator)')
    ap.add_argument('--only', action='append', help='run the cases whose name contains this')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1',
                      TRITON_CACHE_DIR=str(out/'triton-cache'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
    os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5-x8')
    backend = os.environ.setdefault('TNCA_BACKEND', 'systemc-tlm-grid')
    run = fc.Run(out, args.quick, backend)
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
    run.report = dict(schema='tnca-fp8-blockwise-cases/1', status='running', quick=args.quick, backend=backend,
                      reference='blockwise_reference (plain torch, device order); CPU eager refuses blockwise',
                      evidence=('model-prediction-uncalibrated' if backend.startswith('systemc')
                                else 'rtl-simulation'),
                      software_tree=software_tree(ROOT, refresh=True),
                      toolchain=dict(torch=str(torch.__version__), triton=str(triton.__version__)),
                      knobs={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')})
    run.save()
    started = time.monotonic()
    for name, kw in cases(args.quick):
        if args.only and not any(s in name for s in args.only):
            continue
        g = torch.Generator().manual_seed(len(run.rows) + 1)
        try:
            kw = dict(kw)
            fn, cargs, ref = make_case(kw.pop('api'), kw.pop('pair'), kw.pop('M'), kw.pop('N'), kw.pop('K'),
                                       kw.pop('out_dtype'), kw.pop('fmt_a'), kw.pop('fmt_b'), g, **kw)
            (y,), rec = run.compile_run(f'blockwise/{name}', fn, cargs, dump=True)
            cmp = fc.compare(y, ref)
            code = (out/'code'/(re.sub(r'[^A-Za-z0-9_.-]+', '_', f'blockwise/{name}') + '.py')).read_text()
            rec['template'] = 'acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]' in code
            ok = (fc.judge(cmp) == 'pass' and not rec['host_fallbacks'] and rec['template']
                  and any('e4m3' in k or 'e5m2' in k or 'f16' in k for k in rec.get('mma', {})))
            run.add('blockwise', name, 'pass' if ok else 'fail', summary=fc.summary_of(cmp),
                    comparison=cmp, scales=[list(t.shape) + list(t.stride()) for t in cargs[2:4]], **rec)
        except Exception as e:                       # noqa: BLE001
            run.add('blockwise', name, 'error', summary=f'{type(e).__name__}: {str(e)[:400]}',
                    traceback=traceback.format_exc()[-4000:])
    run.report.update(status='complete', wall_seconds=round(time.monotonic() - started, 1))
    run.save()
    print(json.dumps(run.report['summary']))
    return 0 if not (run.report['summary']['fail'] or run.report['summary']['error']) else 1


if __name__ == '__main__':
    sys.exit(main())
