"""op.wfp8_linear/tiny/bf16-m16-k64-n64-we4m3

A point of the benchmark itself (op wfp8_linear, bf16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at dc80a07ca012; regenerated, not edited.
"""
import torch
import torch.nn.functional as F
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.wfp8_linear/tiny/bf16-m16-k64-n64-we4m3'
TOL = 0.06
DEMAND = bound.Demand(macs=65536, fp_ops=5120, sfu_ops=0.0, read_bytes=6276, write_bytes=2048)

W8A8_SCALES = (0.0213, 0.0157)
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class WeightFP8Linear(torch.nn.Module):

    def forward(self, x, w, ws, bias):
        return F.linear(x, w.to(x.dtype) * ws, bias)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, m=16, k=64, n=64):
    g = gen(seed)
    x = randn(g, (m, k), 'bf16', 2.0)
    w = (torch.randn(n, k, generator=g) * torch.pow(2.0, torch.randint(-2, 3, (n, k), generator=g).float())).to(torch.float8_e4m3fn)
    bias = randn(g, (n,), 'bf16')
    return (WeightFP8Linear(), (x, w, torch.tensor(W8A8_SCALES[0]), bias))


def make():
    return _point_make(0)
