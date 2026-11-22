"""op.transpose/membound/fp32-r4096-c4096

A stand-in for op.transpose/hbm/fp32-r8192-c8192: the same operator at a smaller size, for the performance models; like a point of the benchmark (op transpose, fp32), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.transpose/membound/fp32-r4096-c4096'
TOL = 0.0
DEMAND = bound.Demand(macs=0.0, fp_ops=0.0, sfu_ops=0.0, read_bytes=67108864, write_bytes=67108864)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class Transpose(torch.nn.Module):

    def forward(self, x):
        return x.t().contiguous()


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, tr=4096):
    return (Transpose(), (randn(gen(seed), (tr, tr), 'fp32'),))


def make():
    return _point_make(0)
