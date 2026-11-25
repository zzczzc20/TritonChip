"""One row against a weight matrix far larger than every cache: the decode-time projection at a model's size.

The dimension it isolates is size itself. The weights are streamed once from memory and never reused, so the
memory outlets bind; at this size nothing of them fits in the home caches, unlike the small cases, where a
working set that fits can hide what the memory path costs. Compare it with `one_row.py` and
`long_stream_row.py`: the same work per byte, three sizes, three regimes.

About four million device cycles on the base: too slow for the RTL simulator; the CA Model runs it in about
a minute.
"""
import torch

K = 1792
N = 3072


def make():
    g = torch.Generator().manual_seed(0)
    lin = torch.nn.Linear(K, N, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(N, K, generator=g) * K ** -0.5)
    return lin.half(), (torch.randn(1, K, generator=g).half(),)
