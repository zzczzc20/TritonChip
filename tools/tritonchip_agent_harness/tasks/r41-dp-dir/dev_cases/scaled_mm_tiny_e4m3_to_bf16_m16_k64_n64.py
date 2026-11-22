"""op.scaled_mm/tiny/e4m3-to-bf16-m16-k64-n64

A point of the benchmark itself (op scaled_mm, e4m3 -> bf16), given to you as the evaluation runs
it: the module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is
the demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy
it to vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.scaled_mm/tiny/e4m3-to-bf16-m16-k64-n64'
TOL = 0.06
DEMAND = bound.Demand(macs=65536, fp_ops=1024, sfu_ops=0.0, read_bytes=5128, write_bytes=2048)

FP8_SCALES = (0.0371, 1.83)
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class ScaledMM(torch.nn.Module):

    def forward(self, a, bt, sa, sb):
        return torch._scaled_mm(a, bt.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randfp8(g, shape, dtype='e4m3'):
    mag = torch.pow(2.0, torch.randint(-3, 4, shape, generator=g).float())
    return (torch.randn(shape, generator=g) * mag).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, m=16, k=64, n=64):
    g = gen(seed)
    a, bt = (randfp8(g, (m, k)), randfp8(g, (n, k)))
    return (ScaledMM(), (a, bt, torch.tensor(FP8_SCALES[0]), torch.tensor(FP8_SCALES[1])))


def make():
    return _point_make(0)
