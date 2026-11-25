"""Two products with a nonlinearity between them: a program of several
launches, judged at the model tolerance."""
import torch
import torch.nn.functional as F

MODEL = True


class TwoLayers(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.up = torch.nn.Linear(48, 96, bias=True)
        self.down = torch.nn.Linear(96, 48, bias=True)
        with torch.no_grad():
            for lin in (self.up, self.down):
                lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * lin.in_features ** -0.5)
                lin.bias.copy_(torch.randn(lin.bias.shape, generator=g) * 0.1)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate='tanh'))


def make():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 48, generator=g).half()
    return TwoLayers(g).half(), (x,)
