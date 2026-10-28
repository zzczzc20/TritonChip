#!/usr/bin/env python3
"""Compile-only run of the Hub FP8 kernels for one profile (R41 G2 sizing).

    TNCA_PROFILE=<profile> HF_HOME=... ~/flux-venv/bin/python -I \
        tools/r41/fp8_hub_compile_only.py --tree ROOT --dump DIR

The Hub FP8 kernels (kernels-community/finegrained-fp8, version 4: the build
transformers' FP8Linear loads for Qwen's FP8 checkpoints) are called exactly as
tools/performance_harness/fp8_kernels_cases.py calls them, on the layer shapes
it uses (the real Qwen projections and the tail shapes). As in
tools/performance_harness/default_change_audit.py, tensors live on the `tnca`
device over a host-memory stand-in session, every launch is skipped
(TNCA_LAUNCH=skip) and every device session is refused: the run produces the
compiled programs (TNCA_KERNEL_DUMP, under --dump) and nothing else. Results
are not computed, so nothing here is a numeric or performance claim.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ENV = dict(TNCA_LAUNCH='skip', TNCA_OPT_LEVEL='2', TNCA_TRITON='1', TNCA_KERNEL_DUMP='1',
           TNCA_BACKEND='fpga-u280-multicore', TNCA_TORCH_DEVICE='native', CUDA_VISIBLE_DEVICES='',
           TNCA_MAX_BLOCK='16', TNCA_GEMM_TILE='16,16,16', TNCA_REDUCTION_ROWS='1',
           TNCA_BANK_ALLOC='0', TNCA_WINDOW_REUSE='0', TNCA_SDPA='decomposed',
           TNCA_PIPELINE='1', TNCA_TUNE='0')
# fp8_kernels_cases.block_shapes with K a multiple of the 128 quantisation block
# (the kernels' act_quant asserts it): tails, then the real projections.
# The largest projections are left out: with every launch skipped they add no
# program (the kernels are already compiled) and hours of host-side tensor setup.
SHAPES = [(1, 256, 256), (16, 256, 256), (33, 256, 384), (7, 6144, 2048)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tree', type=Path, required=True)
    ap.add_argument('--dump', type=Path, required=True)
    a = ap.parse_args()
    root = a.tree.resolve()
    os.environ.update(ENV, TNCA_DUMP_DIR=str(a.dump.resolve()))
    a.dump.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root), str(root/'python'), str(root/'compiler/python'), str(root/'tools/asm'),
                    str(root/'tools/r40'), str(root/'tools/performance_harness')]
    import torch
    import triton  # noqa: F401  (before tnca: the backend plugin is discovered by triton's import)
    import tnca
    import tnca.inductor  # noqa: F401
    import tnca.torch_device as tdev
    from tnca.triton_backend import driver
    from v3_compile_preflight import HostSession
    import kernels
    if not Path(tnca.__file__).resolve().is_relative_to(root):
        raise SystemExit(f'tnca imported from {tnca.__file__}, outside {root}')

    def forbidden(*args, **kwargs):
        raise RuntimeError('compile-only run forbids every device session')
    driver._Sessions.get = classmethod(forbidden)
    stub = HostSession(os.environ['TNCA_PROFILE'])
    tdev.MEMORY.session = lambda index=0: stub
    tdev.register()
    k = kernels.get_kernel('kernels-community/finegrained-fp8', version=4)
    out = dict(profile=os.environ['TNCA_PROFILE'], cases=[])
    for M, N, K in SHAPES:
        torch.manual_seed(M * 7919 + N * 31 + K)
        A = torch.randn(M, K, dtype=torch.bfloat16)
        W = (torch.randn(N, K) * 0.05).to(torch.float8_e4m3fn)
        Ws = torch.rand((N + 127) // 128, (K + 127) // 128) * 0.01 + 0.001
        with driver.launch_log() as log:
            k.fp8_act_quant(A.to('tnca'), 128)
            k.matmul_2d(A.to('tnca'), W.to('tnca'), Ws.to('tnca'), [128, 128], torch.bfloat16)
            k.matmul_2d(A.to('tnca'), W.to('tnca'), Ws.to('tnca'), [128, 128], torch.bfloat16,
                        activation_scale=torch.tensor(0.0173).to('tnca'))
            k.matmul_2d(A.to('tnca'), W.to('tnca'), torch.tensor(0.0117).to('tnca'), None, torch.bfloat16)
        out['cases'].append(dict(shape=[M, N, K], launches=[dict(kernel=e['kernel'], grid=list(e['grid']),
                                                                  skipped=e.get('skipped')) for e in log]))
        print(f'compiled {M}x{N}x{K}: {len(log)} launches', flush=True)
    (a.dump/'compile_only.json').write_text(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
