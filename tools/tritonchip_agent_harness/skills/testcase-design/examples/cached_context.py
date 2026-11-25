"""One query against a cached context: a decode step reads a context it
computed earlier (keys and values stored per head) and produces one output
row. The stream is the cache, not the weights, and it grows with the context;
the arithmetic per byte is tiny, and the reduction over the context ends in
a softmax, so the time goes to moving the cache and to the one-lane parts of
the machine rather than to the matrix unit."""
import torch


class OneQuery(torch.nn.Module):
    def forward(self, q, k, v):
        # q: (heads, 1, d); k, v: (heads, context, d). One row of attention per head.
        scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(weights, v)


def make():
    g = torch.Generator().manual_seed(0)
    heads, context, d = 2, 48, 24
    q = torch.randn(heads, 1, d, generator=g).half()
    k = torch.randn(heads, context, d, generator=g).half()
    v = torch.randn(heads, context, d, generator=g).half()
    return OneQuery(), (q, k, v)
