"""op.softmax/tiny/fp32-r3-c37

A point of the benchmark itself (op softmax, fp32), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at dc80a07ca012; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.softmax/tiny/fp32-r3-c37'
TOL = 0.0002
DEMAND = bound.Demand(macs=0.0, fp_ops=333, sfu_ops=111, read_bytes=444, write_bytes=444)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class Softmax(torch.nn.Module):

    def forward(self, x):
        return torch.softmax(x, dim=-1)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed):
    return (Softmax(), (randn(gen(seed), (3, 37), 'fp32', 2.0),))


def make():
    return _point_make(0)
