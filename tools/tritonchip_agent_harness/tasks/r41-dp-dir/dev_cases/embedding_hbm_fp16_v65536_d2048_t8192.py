"""op.embedding/hbm/fp16-v65536-d2048-t8192

A point of the benchmark itself (op embedding, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import torch.nn.functional as F
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.embedding/hbm/fp16-v65536-d2048-t8192'
TOL = 0.0
DEMAND = bound.Demand(macs=0.0, fp_ops=0.0, sfu_ops=0.0, read_bytes=33619968, write_bytes=33554432)

ED = 2048
ET = 8192
EV = 65536
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


def _point_make(seed):
    g = gen(seed)
    ids = torch.randperm(EV, generator=g)[:ET]
    return (Embedding(EV, ED, 'fp16', g), (ids,))


def make():
    return _point_make(0)
