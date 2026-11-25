"""A projection of many rows at a medium size: the weights are reused by every row.

The dimension it isolates is reuse at a size between the small cases and a model's. Each weight is read
once and used by every row, so the matrix unit can bind rather than memory -- if the program keeps the
operands close; the weights (under a MiB) are larger than one home cache and smaller than the three
together. `projection_bias.py` is the same computation at a size that fits one cache.

About two million device cycles on the base: the CA Model runs it in well under a minute; the RTL
simulator would take a day.
"""
import torch

ROWS, K, N = 128, 768, 512


def make():
    g = torch.Generator().manual_seed(0)
    lin = torch.nn.Linear(K, N, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(N, K, generator=g) * K ** -0.5)
        lin.bias.copy_(torch.randn(N, generator=g) * 0.1)
    return lin.half(), (torch.randn(ROWS, K, generator=g).half(),)
