"""One row against a weight matrix: every weight is read once and used once,
so the memory port binds rather than the multipliers (a decode-time projection)."""
import torch


def make():
    g = torch.Generator().manual_seed(0)
    lin = torch.nn.Linear(96, 80, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(80, 96, generator=g) * 96 ** -0.5)
    x = torch.randn(1, 96, generator=g).half()
    return lin.half(), (x,)
