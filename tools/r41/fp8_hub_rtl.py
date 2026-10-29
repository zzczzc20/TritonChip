#!/usr/bin/env python3
"""The Hub FP8 block-scaled matmul through the standard PyTorch path on the board-top RTL simulation.

    ~/flux-venv/bin/python tools/r41/fp8_hub_rtl.py --out DIR [--image-tcl IMAGE.tcl] \
        [--shape M,N,K ...] [--profile PROFILE] [--memory-bytes N]

R41 G2, run-2 gate: the compiled `matmul_2d` block-dynamic kernel of
kernels-community/finegrained-fp8 (version 4, unchanged: activation quantisation
and the block-scaled FP8 GEMM, `w8a8_block_dynamic_fp8_matmul`) is compiled by
the TNCA Triton backend for the dp profile and executed, as launched by
`torch.compile` / eager PyTorch on `tnca` tensors, on the `tnca_multicore_board_regs`
bench elaborated at the parameters of an image .tcl (default: G4's run-2 t1 bench
image, MAC_DP 1, MAC_CUTS 29, FMA_IMPL 1, MTX_VRF_REG 1, MTX_ABUF 1, four contexts,
TENSOR_LINE 1, hybrid LSU). Every output is compared bit for bit with
`fp8_block_reference` (per K block, exact group sums through tools/model/fdp.py's
semantics, one RNE per group, ascending K). RTL simulation: not a board result.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm'),
                str(ROOT / 'tools/performance_harness/systemc'), str(ROOT / 'tools/performance_harness'),
                str(ROOT / 'tools/r40')]
from image_geometry_tests import BENCH, DERIVED, LINK, image_parameters, rtl_defaults   # noqa: E402


def bench_arguments(image_tcl, memory_bytes, profile):
    image = image_parameters(image_tcl)
    defaults = rtl_defaults()
    bench, link = {}, {}
    for name, value in sorted(image.items()):
        if name in BENCH:
            bench[BENCH[name]] = value
        elif name in LINK:
            link[LINK[name]] = value
        elif name in DERIVED:
            assert DERIVED[name](image) == value, name
        elif name in defaults:
            assert defaults[name] == value, f'{name}={value} differs from the RTL default {defaults[name]}'
        else:
            raise SystemExit(f'image parameter {name} is unknown to the bench and the RTL top')
    args = dict(core_count=3, memory_ports=link['memory_ports'], board_top=True,
                link_stages=link['link_stages'], noc_link_stages=link['noc_link_stages'],
                dispatch_stages=link['dispatch_stages'], home_stages=link['home_stages'],
                code_stages=link['code_stages'], maintenance=link['maintenance'],
                memory_controls=bool(link['memory_controls']), memory_bytes=memory_bytes,
                rtl_params=bench, hbm_dma=True, host_timeout=3600, image_profile=profile)
    return image, args


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--image-tcl', default=str(ROOT / 'evidence/r41/hbm_bw/g4/images/image_run2_t1_bench.tcl'))
    ap.add_argument('--profile', default='core-vrf-sync-v5-x8-g2c4-dp')
    ap.add_argument('--shape', action='append', default=[], metavar='M,N,K')
    ap.add_argument('--memory-bytes', type=int, default=1 << 22)
    ap.add_argument('--reference-dot-mode', choices=('auto', 'mma', 'mma.dp'), default='auto')
    a = ap.parse_args()
    shapes = [tuple(int(x) for x in s.split(',')) for s in a.shape] or [(1, 256, 256), (16, 256, 256)]
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    image, backend_args = bench_arguments(a.image_tcl, a.memory_bytes, a.profile)
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1', TNCA_TRITON='1',
                      TRITON_CACHE_DIR=str(out / 'triton-cache'), TORCHINDUCTOR_CACHE_DIR=str(out / 'inductor-cache'),
                      TNCA_KERNEL_DUMP='1', TNCA_DUMP_DIR=str(out / 'dump'), TNCA_IEEE_DIV='1',
                      TNCA_PROFILE=a.profile, TNCA_BACKEND='fpga-multicore-sim',
                      TNCA_BACKEND_ARGS=json.dumps(backend_args),
                      TNCA_DEADLINE='40000000000', TNCA_HOST_TIMEOUT='40000')
    import torch
    import tnca
    import tnca.torch_device as device
    device.register()
    import fp8_block_reference as ref
    import fp8_kernels_cases as fk
    import kernels
    mode = a.reference_dot_mode if a.reference_dot_mode != 'auto' else ref.dot_mode_for_profile(a.profile)
    ref.set_dot_mode(mode)
    run = fk.Run(out)
    run.report = dict(schema='tnca-fp8-hub-rtl/1', backend='fpga-multicore-sim (board-top RTL)', profile=a.profile,
                      reference_dot_mode=mode, image_tcl=str(a.image_tcl), image=image, backend_args=backend_args,
                      toolchain=dict(torch=torch.__version__), evidence='RTL simulation, not a board result')
    k = kernels.get_kernel('kernels-community/finegrained-fp8', version=4)
    for M, N, K in shapes:
        def f(M=M, N=N, K=K):
            A, W, Ws = fk.inputs(M, N, K, torch.bfloat16)
            C = k.matmul_2d(fk.dev(A), fk.dev(W), fk.dev(Ws), [128, 128], torch.bfloat16)
            return (C,), (ref.block_dynamic(A, W, Ws, [128, 128], torch.bfloat16, ieee=True),), None
        run.case('hub-rtl', f'matmul_2d block-dynamic {M}x{N}x{K} bf16', f)
    run.report['code'] = fk.code_facts(out / 'dump')
    run.report['status'] = 'complete'
    run.save()
    print(json.dumps(run.report['summary']))
    return 0 if not (run.report['summary']['fail'] or run.report['summary']['error']) else 1


if __name__ == '__main__':
    sys.exit(main())
