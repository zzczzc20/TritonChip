"""op.add/tiny/fp32-n256

A point of the benchmark itself (op add, fp32), given to you as the evaluation runs it: the module
and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the demand the
benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to vary a
dimension.

Generated from the benchmark at a3e73d08f865; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.add/tiny/fp32-n256'
TOL = 0.0002
DEMAND = bound.Demand(macs=0.0, fp_ops=256, sfu_ops=0, read_bytes=2048, write_bytes=1024)

dtype = 'fp32'
n_inputs = 2
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))

module = Add


def _point_make(seed, n=256):
    g = gen(seed)
    return (module(), tuple((randn(g, (n,), dtype) for _ in range(n_inputs))))


def make():
    return _point_make(0)
