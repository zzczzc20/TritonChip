"""A batch of independent products."""
import torch


class Batched(torch.nn.Module):
    def forward(self, a, b):
        return torch.bmm(a, b)


def make():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, 16, 32, generator=g).half()
    b = torch.randn(3, 32, 16, generator=g).half()
    return Batched(), (a, b)
