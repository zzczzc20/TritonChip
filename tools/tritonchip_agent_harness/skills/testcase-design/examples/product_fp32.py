"""The same product on the FP32 path (FP32 tolerance, exact vector FMAs)."""
import torch


class Product(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


def make():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(32, 48, generator=g)
    b = torch.randn(48, 32, generator=g)
    return Product(), (a, b)
