"""Extra ArchBench points for the packed-MMA (m.mma.dp) what-if study.

Loaded by run_graphs.py --points-module; registers standard-Torch points
outside the frozen suites (no suite hash covers them). Plain torch, no
model-specific code:

  op.w8a8_linear/hbm/bf16-m{M}-k{K}-n{N}
      static W8A8 linear: x (bf16) quantised with a static tensorwise scale
      to float8_e4m3fn, then torch._scaled_mm(q, W.t(), scale_a, scale_b,
      out_dtype=bf16) -- e4m3 x e4m3, binary32 accumulation, no bias. W is an
      (N, K) e4m3 weight (column-major B for _scaled_mm, as a Linear stores it).
  op.bf16_linear/hbm/bf16-m{M}-k{K}-n{N}
      torch.nn.Linear(K, N, bias=False) in bf16 at the same shapes.

Shapes: decode M=1 and prefill M=128 with K x N = 2048x6144, 2560x9216 and
9216x2560 (Qwen3-VL-2B / Qwen3.5-4B projections).

The ArchBench reference upcasts the module and inputs to binary32
(Module.float()), which also widens an FP8 weight buffer to binary32
exactly; the forward re-narrows it with .to(float8_e4m3fn), exact for
values that came from FP8 and the identity on the device, where the
buffer is already e4m3. The reference therefore runs the same graph with
binary32 activations: CPU _scaled_mm (scale-first emulation) with a
binary32 output, against the device's bf16 output.
"""
import torch

from archbench.registry import Demand, Point, register

SHAPES = ((2048, 6144), (2560, 9216), (9216, 2560))
MS = (1, 128)
E4M3 = torch.float8_e4m3fn


def _gen(seed):
    return torch.Generator().manual_seed(seed)


class W8A8Linear(torch.nn.Module):
    def __init__(self, k, n, g):
        super().__init__()
        w = torch.randn(n, k, generator=g) * k ** -0.5
        wscale = float(w.abs().max()) / 448.0
        self.register_buffer('weight', (w / wscale).clamp(-448.0, 448.0).to(E4M3))
        self.register_buffer('weight_scale', torch.tensor(wscale, dtype=torch.float32))
        # Static activation scale for randn inputs (|x| < ~6): a calibrated
        # constant, not derived from the call's data.
        self.register_buffer('input_scale', torch.tensor(6.0 / 448.0, dtype=torch.float32))

    def forward(self, x):
        q = (x / self.input_scale).clamp(-448.0, 448.0).to(E4M3)
        return torch._scaled_mm(q, self.weight.to(E4M3).t(), scale_a=self.input_scale,
                                scale_b=self.weight_scale, out_dtype=x.dtype)


def _bf16_linear(k, n, g):
    lin = torch.nn.Linear(k, n, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(n, k, generator=g) * k ** -0.5)
    return lin.to(torch.bfloat16)


def _x(g, m, k):
    return torch.randn(m, k, generator=g).to(torch.bfloat16)


def register_all():
    for m in MS:
        for k, n in SHAPES:
            params = dict(m=m, k=k, n=n)
            tag = '-'.join(f'{a}{b}' for a, b in params.items())

            def mk_fp8(seed, m=m, k=k, n=n):
                g = _gen(seed)
                return W8A8Linear(k, n, g), (_x(g, m, k),)

            def mk_bf16(seed, m=m, k=k, n=n):
                g = _gen(seed)
                return _bf16_linear(k, n, g), (_x(g, m, k),)

            register(Point(f'op.w8a8_linear/hbm/bf16-{tag}', 'op', 'w8a8_linear', 'hbm', 'bf16', params, mk_fp8,
                           Demand(macs=m * k * n, fp_ops=2 * m * k + m * n, read_bytes=m * k * 2 + k * n,
                                  write_bytes=m * n * 2),
                           tol=2e-2, note='static W8A8 via torch._scaled_mm (e4m3 x e4m3, fp32 accumulate), bf16 out'))
            register(Point(f'op.bf16_linear/hbm/bf16-{tag}', 'op', 'bf16_linear', 'hbm', 'bf16', params, mk_bf16,
                           Demand(macs=m * k * n, read_bytes=(m * k + k * n) * 2, write_bytes=m * n * 2),
                           note='bf16 nn.Linear without bias'))


register_all()
