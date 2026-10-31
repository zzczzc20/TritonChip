#!/usr/bin/env python3
"""The Hub FP8 kernels (kernels-community/finegrained-fp8), unchanged, on tnca.

    ~/flux-venv/bin/python tools/performance_harness/fp8_kernels_cases.py --out DIR \
        [--group v4|v1|compile|transformers ...] [--quick] [--v1-path DIR]

The kernels are loaded exactly as their users load them: `kernels.get_kernel(
"kernels-community/finegrained-fp8", version=4)` is what transformers 5.20.2's
FP8Linear calls (integrations/hub_kernels.py); the v1 build (the repository's
`main` branch, act_quant.py / matmul.py) is loaded from a local copy with
`kernels.get_local_kernel`. Their Triton source is compiled by the TNCA Triton
backend and runs on TNCA_BACKEND (default systemc-tlm-grid, the SystemC/TLM
model). Nothing here names a model or a layer.

Every output is compared bit for bit with fp8_block_reference: a plain-torch
re-derivation of each kernel in the order TNCA executes it (per K block, one
binary32 rounding per product, ascending K), with IEEE division -- the
semantics of the kernels' `/`.

groups
  v4            fp8_act_quant, matmul_2d block-dynamic / block-static / tensor
  v1            fp8_act_quant, w8a8_block_fp8_matmul, w8a8_tensor_fp8_matmul
  compile       the same ops inside torch.compile(fullgraph=True)
  transformers  FP8Linear forward, eager and compiled; the quantizer's check

Development evidence: model results are uncalibrated model predictions.
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
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'),
                str(ROOT / 'tools/asm'), str(ROOT / 'tools/performance_harness/systemc'),
                str(ROOT / 'tools/performance_harness')]
GROUPS = ('v4', 'v1', 'compile', 'transformers')


class Run:
    def __init__(self, out):
        self.out = out
        self.rows = []
        self.report = {}

    def save(self):
        self.report['cases'] = self.rows
        self.report['summary'] = {s: sum(r['status'] == s for r in self.rows)
                                  for s in ('pass', 'fail', 'error')}
        (self.out / 'report.json').write_text(json.dumps(self.report, indent=1, default=str))

    def case(self, group, name, fn):
        torch = _torch()
        from tnca.triton_backend.driver import launch_log
        row = dict(group=group, case=name)
        t = time.monotonic()
        try:
            with launch_log() as log:
                got, want, extra = fn()
            row['launches'] = [dict(kernel=e['kernel'], grid=list(e['grid']), device_cycles=e['device_cycles'])
                               for e in log if not e.get('skipped')]
            cmp = [compare(g, w) for g, w in zip(got, want)]
            row['compare'] = cmp
            row.update(extra or {})
            row['status'] = 'pass' if all(c['bit_mismatches'] == 0 for c in cmp) else 'fail'
        except Exception as e:                         # noqa: BLE001
            row['status'] = 'error'
            row['error'] = f'{type(e).__name__}: {str(e)[:600]}'
            row['traceback'] = traceback.format_exc()[-3000:]
        row['wall_seconds'] = round(time.monotonic() - t, 1)
        self.rows.append(row)
        self.save()
        brief = '; '.join(f"{c['elements']} el, {c['bit_mismatches']} mismatches" for c in row.get('compare', []))
        print(f"[{row['status']}] {group} {name}: {brief or row.get('error', '')[:200]}", flush=True)
        torch.manual_seed(0)


def _torch():
    import torch
    return torch


def compare(got, want):
    torch = _torch()
    got = got.detach().cpu()
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[got.element_size()]
    g, w = got.contiguous().view(width).flatten(), want.contiguous().view(width).flatten()
    bad = (g != w)
    out = dict(elements=want.numel(), bit_mismatches=int(bad.sum()), dtype=str(want.dtype))
    if bad.any():
        i = int(bad.nonzero()[0])
        out['first'] = dict(index=i, got=float(got.flatten()[i].float()), want=float(want.flatten()[i].float()))
    return out


def inputs(M, N, K, dt):
    torch = _torch()
    torch.manual_seed(M * 7919 + N * 31 + K)
    A = torch.randn(M, K, dtype=dt) * (1 + torch.rand(M, 1)).to(dt)
    W = (torch.randn(N, K) * 0.05).to(torch.float8_e4m3fn)
    Ws = torch.rand((N + 127) // 128, (K + 127) // 128) * 0.01 + 0.001
    return A, W, Ws


def dev(t):
    return t.to('tnca')


# ------------------------------------------------------------------ groups ----
def block_shapes(quick):
    tails = [(1, 256, 256), (7, 136, 144), (16, 256, 256), (33, 256, 384), (129, 200, 328)]
    real = [(7, 6144, 2048), (7, 2048, 6144), (16, 9216, 2560), (129, 2048, 2048)]
    return tails if quick else tails + real


def group_v4(run, k, quick):
    import fp8_block_reference as ref
    torch = _torch()
    for M, K in ([(4, 256), (7, 384), (129, 256)] + ([] if quick else [(16, 2048), (3, 6144)])):
        def f(M=M, K=K):
            A, _, _ = inputs(M, 128, K, torch.bfloat16)
            y, s = k.fp8_act_quant(dev(A), 128)
            ry, rs = ref.act_quant(A, 128, ieee=True)
            return (y, s), (ry, rs), None
        run.case('v4', f'fp8_act_quant {M}x{K} bf16, block 128', f)
    for M, N, K in block_shapes(quick):
        def f(M=M, N=N, K=K):
            A, W, Ws = inputs(M, N, K, torch.bfloat16)
            C = k.matmul_2d(dev(A), dev(W), dev(Ws), [128, 128], torch.bfloat16)
            return (C,), (ref.block_dynamic(A, W, Ws, [128, 128], torch.bfloat16, ieee=True),), None
        run.case('v4', f'matmul_2d block-dynamic {M}x{N}x{K} bf16', f)
    for M, N, K in [(7, 136, 144), (129, 200, 328)] + ([] if quick else [(16, 2048, 2048)]):
        def f(M=M, N=N, K=K):
            A, W, Ws = inputs(M, N, K, torch.bfloat16)
            As = torch.tensor(0.0173)
            C = k.matmul_2d(dev(A), dev(W), dev(Ws), [128, 128], torch.bfloat16, activation_scale=dev(As))
            return (C,), (ref.block_static(A, W, Ws, As, [128, 128], torch.bfloat16, ieee=True),), None
        run.case('v4', f'matmul_2d block-static {M}x{N}x{K} bf16', f)
    for M, N, K in [(7, 256, 256), (129, 200, 328)] + ([] if quick else [(16, 512, 2048)]):
        def f(M=M, N=N, K=K):
            A, W, _ = inputs(M, N, K, torch.bfloat16)
            Bs = torch.tensor(0.0117)
            C = k.matmul_2d(dev(A), dev(W), dev(Bs), None, torch.bfloat16)
            return (C,), (ref.tensor_dynamic(A, W, Bs, torch.bfloat16, ieee=True),), None
        run.case('v4', f'matmul_2d tensor-dynamic {M}x{N}x{K} bf16', f)


def group_v1(run, k1, quick):
    import fp8_block_reference as ref
    torch = _torch()
    for M, K in [(7, 384), (129, 256)]:
        def f(M=M, K=K):
            A, _, _ = inputs(M, 128, K, torch.bfloat16)
            y, s = k1.fp8_act_quant(dev(A), 128)
            ry, rs = ref.act_quant(A, 128, ieee=True, clamp=False)
            return (y, s), (ry, rs), None
        run.case('v1', f'fp8_act_quant {M}x{K} bf16, block 128', f)
    for M, N, K in [(7, 136, 256), (33, 256, 384), (129, 200, 384)] + ([] if quick else [(7, 6144, 2048)]):
        def f(M=M, N=N, K=K):
            A, W, Ws = inputs(M, N, K, torch.bfloat16)
            qA, As = ref.act_quant(A, 128, ieee=True, clamp=False)
            C = k1.w8a8_block_fp8_matmul(dev(qA), dev(W), dev(As), dev(Ws), [128, 128], torch.bfloat16)
            return (C,), (ref.block_prequantised(qA, W, As, Ws, [128, 128], torch.bfloat16),), None
        run.case('v1', f'w8a8_block_fp8_matmul {M}x{N}x{K} bf16', f)
    for M, N, K in [(7, 256, 256), (129, 200, 384)]:
        def f(M=M, N=N, K=K):
            A, W, _ = inputs(M, N, K, torch.bfloat16)
            qA, As = ref.act_quant(A, K, ieee=True, clamp=False)
            Bs = torch.tensor(0.0117)
            C = k1.w8a8_tensor_fp8_matmul(dev(qA), dev(W), dev(As.reshape(M)), dev(Bs), torch.bfloat16)
            return (C,), (ref.tensor_scaled(qA, W, As, Bs, torch.bfloat16),), None
        run.case('v1', f'w8a8_tensor_fp8_matmul {M}x{N}x{K} bf16', f)


def group_compile(run, k, quick):
    import fp8_block_reference as ref
    torch = _torch()
    torch._dynamo.config.recompile_limit = 64
    for M, N, K in [(7, 256, 256), (33, 136, 384)]:
        def f(M=M, N=N, K=K):
            A, W, Ws = inputs(M, N, K, torch.bfloat16)
            fn = torch.compile(lambda a, w, s: k.matmul_2d(a, w, s, [128, 128], torch.bfloat16) * 2.0,
                               backend='inductor', fullgraph=True)
            C = fn(dev(A), dev(W), dev(Ws))
            return (C,), (ref.block_dynamic(A, W, Ws, [128, 128], torch.bfloat16, ieee=True) * 2.0,), None
        run.case('compile', f'block-dynamic in a compiled graph {M}x{N}x{K}', f)

    def f():
        A, W, _ = inputs(7, 256, 256, torch.bfloat16)
        Bs = torch.tensor(0.0117)
        fn = torch.compile(lambda a, w, s: k.matmul_2d(a, w, s, None, torch.bfloat16),
                           backend='inductor', fullgraph=True)
        return (fn(dev(A), dev(W), dev(Bs)),), (ref.tensor_dynamic(A, W, Bs, torch.bfloat16, ieee=True),), None
    run.case('compile', 'tensor-dynamic (act_quant + GEMM) in a compiled graph 7x256x256', f)

    def g():
        A, _, _ = inputs(7, 128, 384, torch.bfloat16)
        fn = torch.compile(lambda a: k.fp8_act_quant(a, 128), backend='inductor', fullgraph=True)
        y, s = fn(dev(A))
        return (y, s), ref.act_quant(A, 128, ieee=True), None
    run.case('compile', 'fp8_act_quant in a compiled graph 7x384', g)


def group_transformers(run, quick):
    import fp8_block_reference as ref
    torch = _torch()
    import transformers
    from transformers import FineGrainedFP8Config
    from transformers.integrations import deepgemm
    from transformers.integrations.finegrained_fp8 import replace_with_fp8_linear
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
    q = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(), pre_quantized=True)
    q.validate_environment(device_map={'': 'tnca'}, weights_only=True)
    run.report['transformers'] = dict(
        version=transformers.__version__, cuda_available=torch.cuda.is_available(),
        deepgemm_loadable=deepgemm.is_deepgemm_loadable(),
        quantizer_dequantize_after_validate_environment=q.quantization_config.dequantize)
    for M, N, K, compiled in [(7, 320, 384, False), (7, 320, 384, True), (16, 2048, 2048, False)]:
        def f(M=M, N=N, K=K, compiled=compiled):
            cfg = FineGrainedFP8Config(weight_block_size=(128, 128))
            model = replace_with_fp8_linear(torch.nn.Sequential(torch.nn.Linear(K, N, bias=False)),
                                            quantization_config=cfg, pre_quantized=True)
            lin = model[0]
            _, W, _ = inputs(1, N, K, torch.bfloat16)
            Ws = torch.rand(*lin.weight_scale_inv.shape) * 0.01 + 0.001
            lin.weight = torch.nn.Parameter(W, requires_grad=False)
            lin.weight_scale_inv = torch.nn.Parameter(Ws, requires_grad=False)
            model = model.to('tnca')
            x = torch.randn(1, M, K, dtype=torch.bfloat16)
            call = torch.compile(model, backend='inductor', fullgraph=True) if compiled else model
            with torch.no_grad():
                y = call(dev(x))
            return (y,), (ref.block_dynamic(x, W, Ws, [128, 128], torch.bfloat16, ieee=True),), \
                dict(module=type(lin).__name__)
        run.case('transformers', f"FP8Linear {'compiled' if compiled else 'eager'} x(1,{M},{K}) -> {N}", f)


# ------------------------------------------------------------ code facts ----
def code_facts(dump):
    """FP8 MMA and the per-block scaling in each compiled Hub kernel's asm."""
    facts = {}
    for s in sorted(Path(dump).glob('*/*.s')):
        text = s.read_text()
        mma = re.findall(r'm\.mma\.async\.(k\d+)\.(\w+)', text)
        facts[s.parent.name] = dict(
            instructions=sum(1 for l in text.splitlines() if l.startswith('    ') and not l.strip().startswith('//')),
            mma=sorted({f'{k}.{d}' for k, d in mma}), mma_static=len(mma),
            row_split='row_split' in text or 'rows [r0' in text,
            swapped_dot='B^T A^T' in text,
            acc_reads=text.count('a.vread'), vf_mul=text.count('vf.mul'),
            rcp=text.count('v.rcp.approx'), cvt_e4m3=text.count('.e4m3'))
    return facts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--group', action='append', choices=GROUPS)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--v1-path', type=Path, help='a local copy of the repository at main (with build/torch-cuda)')
    ap.add_argument('--reference-dot-mode', choices=('auto', 'mma', 'mma.dp'), default='auto',
                    help="tl.dot reduction the reference follows: 'auto' reads the profile's MATRIX_DP "
                         "capability; the others exist only as negative controls")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1', TNCA_TRITON='1',
                      TRITON_CACHE_DIR=str(out / 'triton-cache'), TORCHINDUCTOR_CACHE_DIR=str(out / 'inductor-cache'),
                      TNCA_KERNEL_DUMP='1', TNCA_DUMP_DIR=str(out / 'dump'))
    os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
    # The kernels' `/` is IEEE division. Kernels the extended lowering
    # compiles divide that way always; the rest (act_quant) need the option,
    # since the default quotient is faithful (mathlib: <= 1 ulp).
    os.environ.setdefault('TNCA_IEEE_DIV', '1')
    os.environ.setdefault('TNCA_DEADLINE', '40000000000')
    os.environ.setdefault('TNCA_HOST_TIMEOUT', '40000')
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
    device.register()
    import fp8_block_reference as _ref
    dot_mode = args.reference_dot_mode if args.reference_dot_mode != 'auto' else \
        _ref.dot_mode_for_profile(os.environ['TNCA_PROFILE'])
    _ref.set_dot_mode(dot_mode)
    run = Run(out)
    import kernels
    run.report = dict(schema='tnca-fp8-hub-kernels/1', backend=backend, quick=args.quick, reference_dot_mode=dot_mode,
                      toolchain=dict(torch=torch.__version__, triton=triton.__version__,
                                     kernels=kernels.__version__),
                      knobs={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')})
    groups = args.group or GROUPS
    k = kernels.get_kernel('kernels-community/finegrained-fp8', version=4)
    run.report['v4_kernel'] = dict(module=k.__name__, path=str(Path(k.__file__).parent))
    for group in groups:
        if group == 'v4':
            group_v4(run, k, args.quick)
        elif group == 'v1':
            if not args.v1_path:
                raise SystemExit('--v1-path is needed for the v1 group')
            k1 = kernels.get_local_kernel(args.v1_path, trust_remote_code=True)
            run.report['v1_kernel'] = dict(module=k1.__name__, path=str(args.v1_path))
            group_v1(run, k1, args.quick)
        elif group == 'compile':
            group_compile(run, k, args.quick)
        else:
            group_transformers(run, args.quick)
    run.report['code'] = code_facts(out / 'dump')
    run.report['status'] = 'complete'
    run.save()
    print(json.dumps(run.report['summary']))
    return 0 if not (run.report['summary']['fail'] or run.report['summary']['error']) else 1


if __name__ == '__main__':
    sys.exit(main())
