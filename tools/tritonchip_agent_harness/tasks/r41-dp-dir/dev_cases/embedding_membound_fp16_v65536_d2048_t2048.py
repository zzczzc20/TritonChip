"""op.embedding/membound/fp16-v65536-d2048-t2048

A stand-in for op.embedding/hbm/fp16-v65536-d2048-t8192: the same operator at a smaller size, for the performance models; like a point of the benchmark (op embedding, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import torch.nn.functional as F
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.embedding/membound/fp16-v65536-d2048-t2048'
TOL = 0.0
DEMAND = bound.Demand(macs=0.0, fp_ops=0.0, sfu_ops=0.0, read_bytes=8404992, write_bytes=8388608)

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


def _point_make(seed, ev=65536, ed=2048, et=2048):
    g = gen(seed)
    ids = torch.randperm(ev, generator=g)[:et]
    return (Embedding(ev, ed, 'fp16', g), (ids,))


def make():
    return _point_make(0)
