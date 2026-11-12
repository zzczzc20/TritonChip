"""op.layernorm/small/fp16-r64-d256

A point of the benchmark itself (op layernorm, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at dc80a07ca012; regenerated, not edited.
"""
import torch
import torch.nn.functional as F
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.layernorm/small/fp16-r64-d256'
TOL = 0.02
DEMAND = bound.Demand(macs=0.0, fp_ops=65536, sfu_ops=64, read_bytes=33792, write_bytes=32768)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class LayerNorm(torch.nn.Module):

    def __init__(self, d, dtype, g):
        super().__init__()
        self.weight = torch.nn.Parameter(1 + randn(g, (d,), dtype, 0.1))
        self.bias = torch.nn.Parameter(randn(g, (d,), dtype, 0.1))

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, 1e-05)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def gen(seed):
    return torch.Generator().manual_seed(seed)


def _point_make(seed, r=64, d=256):
    g = gen(seed)
    return (LayerNorm(d, 'fp16', g), (randn(g, (r, d), 'fp16'),))


def make():
    return _point_make(0)
