"""op.matmul/tiny/fp16-m16-k32-n16

A point of the benchmark itself (op matmul, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at dc80a07ca012; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.matmul/tiny/fp16-m16-k32-n16'
TOL = 0.02
DEMAND = bound.Demand(macs=8192, fp_ops=0.0, sfu_ops=0.0, read_bytes=2048, write_bytes=512)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class MatMul(torch.nn.Module):

    def forward(self, a, b):
        return a @ b


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, m=16, k=32, n=16, dtype='fp16'):
    return (MatMul(), (randn(gen(seed), (m, k), dtype), randn(gen(seed + 1), (k, n), dtype)))


def make():
    return _point_make(0)
