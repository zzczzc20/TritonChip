"""A plain product whose dimensions are not multiples of the vector length."""
import torch


class Product(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


def make():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(24, 40, generator=g).half()
    b = torch.randn(40, 48, generator=g).half()
    return Product(), (a, b)
