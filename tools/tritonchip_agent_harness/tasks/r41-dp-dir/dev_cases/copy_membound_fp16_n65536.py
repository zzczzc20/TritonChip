"""op.copy/membound/fp16-n65536

A calibration case, not scored (CALIBRATION.json says what it reaches); like a point of the benchmark (op copy, fp16), given to you as the evaluation runs it: the module
and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the demand the
benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to vary a
dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
from calib_points import Copy
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.copy/membound/fp16-n65536'
TOL = 0.02
DEMAND = bound.Demand(macs=0.0, fp_ops=0.0, sfu_ops=0.0, read_bytes=131072, write_bytes=131072)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, n=65536):
    return (Copy(), (randn(gen(seed), (n,), 'fp16'),))


def make():
    return _point_make(0)
