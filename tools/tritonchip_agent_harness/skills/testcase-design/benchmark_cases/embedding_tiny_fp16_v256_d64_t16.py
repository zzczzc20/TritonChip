"""op.embedding/tiny/fp16-v256-d64-t16

A point of the benchmark itself (op embedding, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at a3e73d08f865; regenerated, not edited.
"""
import torch
import torch.nn.functional as F
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.embedding/tiny/fp16-v256-d64-t16'
TOL = 0.0
DEMAND = bound.Demand(macs=0.0, fp_ops=0.0, sfu_ops=0.0, read_bytes=2176, write_bytes=2048)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class Embedding(torch.nn.Module):
    def __init__(self, v, d, dtype, g):
        super().__init__()
        self.weight = torch.nn.Parameter(randn(g, (v, d), dtype))

    def forward(self, ids):
        return F.embedding(ids, self.weight)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def gen(seed):
    return torch.Generator().manual_seed(seed)


def _point_make(seed, v=256, d=64, t=16):
    g = gen(seed)
    ids = torch.randperm(v, generator=g)[:t]
    return (Embedding(v, d, 'fp16', g), (ids,))


def make():
    return _point_make(0)
