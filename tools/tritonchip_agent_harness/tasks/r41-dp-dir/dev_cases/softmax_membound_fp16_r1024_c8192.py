"""op.softmax/membound/fp16-r1024-c8192

A stand-in for op.softmax/hbm/fp16-r4096-c8192: the same operator at a smaller size, for the performance models; like a point of the benchmark (op softmax, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.softmax/membound/fp16-r1024-c8192'
TOL = 0.02
DEMAND = bound.Demand(macs=0.0, fp_ops=25165824, sfu_ops=8388608, read_bytes=16777216, write_bytes=16777216)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class Softmax(torch.nn.Module):

    def forward(self, x):
        return torch.softmax(x, dim=-1)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, r=1024, c=8192):
    return (Softmax(), (randn(gen(seed), (r, c), 'fp16', 2.0),))


def make():
    return _point_make(0)
