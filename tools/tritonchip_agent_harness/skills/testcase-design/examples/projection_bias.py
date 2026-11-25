"""A projection with a bias: the product plus an epilogue."""
import torch


def make():
    g = torch.Generator().manual_seed(0)
    lin = torch.nn.Linear(80, 48, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(48, 80, generator=g) * 80 ** -0.5)
        lin.bias.copy_(torch.randn(48, generator=g) * 0.1)
    x = torch.randn(16, 80, generator=g).half()
    return lin.half(), (x,)
