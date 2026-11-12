"""op.argmax/tiny/fp16-r2-v256

A point of the benchmark itself (op argmax, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at dc80a07ca012; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.argmax/tiny/fp16-r2-v256'
TOL = 0.0
DEMAND = bound.Demand(macs=0.0, fp_ops=512, sfu_ops=0.0, read_bytes=1024, write_bytes=16)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class Argmax(torch.nn.Module):

    def forward(self, x):
        return torch.argmax(x, dim=-1)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, r=2, v=256):
    return (Argmax(), (randn(gen(seed), (r, v), 'fp16', 3.0),))


def make():
    return _point_make(0)
