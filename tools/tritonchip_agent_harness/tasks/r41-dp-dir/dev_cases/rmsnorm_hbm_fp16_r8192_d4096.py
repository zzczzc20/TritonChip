"""op.rmsnorm/hbm/fp16-r8192-d4096

A point of the benchmark itself (op rmsnorm, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import torch.nn.functional as F
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.rmsnorm/hbm/fp16-r8192-d4096'
TOL = 0.02
DEMAND = bound.Demand(macs=0.0, fp_ops=100663296, sfu_ops=8192, read_bytes=67117056, write_bytes=67108864)

ND = 4096
NR = 8192
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class RMSNorm(torch.nn.Module):

    def __init__(self, d, dtype, g):
        super().__init__()
        self.weight = torch.nn.Parameter(1 + randn(g, (d,), dtype, 0.1))
        self.eps = 1e-05

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def gen(seed):
    return torch.Generator().manual_seed(seed)


def _point_make(seed):
    g = gen(seed)
    return (RMSNorm(ND, 'fp16', g), (randn(g, (NR, ND), 'fp16'),))


def make():
    return _point_make(0)
