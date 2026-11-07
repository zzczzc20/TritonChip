"""op.w8a8_block_linear/small/e4m3-to-bf16-m1-k128-n384

A point of the benchmark itself (op w8a8_block_linear, e4m3 -> bf16), given to you as the evaluation
runs it: the module and inputs make() returns are the point's own (seed 0), judged at TOL, and
DEMAND is the demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any
case; copy it to vary a dimension.

Generated from the benchmark at a3e73d08f865; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.w8a8_block_linear/small/e4m3-to-bf16-m1-k128-n384'
TOL = 0.06
DEMAND = bound.Demand(macs=49152, fp_ops=1280, sfu_ops=0.0, read_bytes=49420, write_bytes=768)

TILE_FP8 = 128
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class W8A8BlockLinear(torch.nn.Module):
    """Dynamic W8A8 with block scales: the activation quantised per 1x128
    group (amax / 448), the E4M3 weight (n, k) with fp32 scales (k/128,
    n/128), a blockwise _scaled_mm, bf16 out. CPU eager refuses blockwise
    scales, so on the CPU (the reference) the same block form runs in fp32,
    K blocks ascending."""
    def forward(self, x, w, ws):
        m, k = x.shape
        g = x.float().reshape(m, k // TILE_FP8, TILE_FP8)
        sa = g.abs().amax(-1).clamp(min=1e-12) / 448.0
        q = (g / sa[..., None]).clamp(-448.0, 448.0).reshape(m, k).to(torch.float8_e4m3fn)
        if q.device.type == 'cpu':
            out = torch.zeros(m, w.shape[0])
            for kb in range(k // TILE_FP8):
                s = slice(kb * TILE_FP8, (kb + 1) * TILE_FP8)
                part = q[:, s].float() @ w[:, s].float().t()
                out = out + part * sa[:, kb:kb + 1] * ws[kb].repeat_interleave(TILE_FP8)[None, :]
            return out.to(torch.bfloat16)
        return torch._scaled_mm(q, w.t(), scale_a=sa, scale_b=ws, out_dtype=torch.bfloat16)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, m=1, k=128, n=384):
    g = gen(seed)
    x = randn(g, (m, k), 'bf16', 2.0)
    w = (torch.randn(n, k, generator=g) * torch.pow(2.0, torch.randint(-2, 3, (n, k), generator=g).float())).to(torch.float8_e4m3fn)
    ws = torch.rand(k // TILE_FP8, n // TILE_FP8, generator=g) * 0.02 + 0.005
    return (W8A8BlockLinear(), (x, w, ws))


def make():
    return _point_make(0)
