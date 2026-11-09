"""One query token against a long cached context: the attention half of a decode step at a model's size.

The dimension it isolates is the context length past the caches: the stored keys and values (several MiB)
are streamed once per step, so, as in `stream_weights_large.py`, the memory outlets bind and nothing of the
stream stays in a home cache. `cached_context.py` is the same computation at a length that fits.

About seven million device cycles on the base: too slow for the RTL simulator; the CA Model runs it in about
a minute.
"""
import torch
import torch.nn.functional as F

HEADS, LENGTH, HEAD_DIM = 8, 2048, 64


class Decode(torch.nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


def make():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, HEADS, 1, HEAD_DIM, generator=g).half()
    k = torch.randn(1, HEADS, LENGTH, HEAD_DIM, generator=g).half()
    v = torch.randn(1, HEADS, LENGTH, HEAD_DIM, generator=g).half()
    return Decode(), (q, k, v)
