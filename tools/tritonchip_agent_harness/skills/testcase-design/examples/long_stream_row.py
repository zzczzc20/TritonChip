"""One row against a weight matrix large enough that the weights do not fit in
what a core keeps in flight: the launch streams the whole matrix through the
memory path once, so the time is set by how many requests the machine keeps
outstanding against the memory's round trip, not by the multipliers and not
by the port's width (a decode-time projection at a realistic size). Compare
with one_row.py, whose matrix is small enough to hide that."""
import torch


def make():
    g = torch.Generator().manual_seed(0)
    lin = torch.nn.Linear(192, 320, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(320, 192, generator=g) * 192 ** -0.5)
    x = torch.randn(1, 192, generator=g).half()
    return lin.half(), (x,)
