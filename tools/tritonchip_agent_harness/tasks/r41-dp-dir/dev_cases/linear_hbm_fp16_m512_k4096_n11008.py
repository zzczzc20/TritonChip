"""op.linear/hbm/fp16-m512-k4096-n11008

A point of the benchmark itself (op linear, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.linear/hbm/fp16-m512-k4096-n11008'
TOL = 0.02
DEMAND = bound.Demand(macs=23085449216, fp_ops=5636096, sfu_ops=0.0, read_bytes=94393856, write_bytes=11272192)

LK = 4096
LM = 512
LN = 11008
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


def _linear(k, n, dtype, g, bias):
    lin = torch.nn.Linear(k, n, bias=bias)
    with torch.no_grad():
        lin.weight.copy_(randn(g, (n, k), 'fp32', k ** (-0.5)))
        if bias:
            lin.bias.copy_(randn(g, (n,), 'fp32', 0.1))
    return lin.to(getattr(torch, TORCH_DTYPE[dtype]))


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def gen(seed):
    return torch.Generator().manual_seed(seed)


def _point_make(seed):
    g = gen(seed)
    return (_linear(LK, LN, 'fp16', g, True), (randn(g, (LM, LK), 'fp16'),))


def make():
    return _point_make(0)
