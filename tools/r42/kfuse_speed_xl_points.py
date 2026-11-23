"""Proxy points for the kfuse-speed study: the xl tier's Hub FP8 kernels at reduced shapes (domain: model prediction).

Loaded with `kfuse_speed_tlm_run.py --points-module` from a tree that has archbench/xl.py (origin/R42-xl). It registers
`op.fp8_linear` points of the same kernel and the same construction as the xl tier's (`archbench.xl`), at shapes small
enough for the SystemC/TLM model. They are measurement proxies of the model study only: they are not xl-tier points, are
not registered in any suite and carry no result outside evidence/r42/kfuse_speed/.
"""
import archbench.xl as X

SHAPES = []          # (m, n, k)


def _add(m, n, k):
    def make(seed, device='cpu', m=m, n=n, k=k):
        g = X.gen(seed)
        lin = X.FP8Linear(n, k, g, device)
        return lin, (X._randn(g, (m, k), 1.0, X.FP16, device),)
    mem = X.memory(m * k * 2 + X._weights(n, k), m * n * 2)
    X.xl('op', 'fp8_linear', 'fp8', dict(m=m, k=k, n=n), make, X.fp8_gemm_demand(m, n, k), tol=4e-3,
         tol_basis=X.TOL_GEMM, mem=mem, fp8=True, reference=X.xl_reference, note='kfuse-speed proxy')
    SHAPES.append((m, n, k))


for _m, _n, _k in ((1, 384, 512), (1, 384, 1024), (16, 384, 512), (16, 384, 1024), (16, 384, 2048), (48, 384, 1024),
                   (16, 256, 256), (1, 256, 256), (16, 1152, 1024), (1, 1152, 1024), (48, 1536, 512), (1, 1536, 512), (48, 1536, 1024), (1, 1536, 1024)):
    try:
        _add(_m, _n, _k)
    except Exception:      # an id of the registered xl tier (never a duplicate here, kept for safety)
        pass


# ---- the existing alternative stream (scenario S1-ref): quantise once, then the blockwise `torch._scaled_mm` -----------
# python/tnca/scaled_mm_blockwise.py lowers a blockwise-scaled torch._scaled_mm on tnca to one Triton template whose K loop
# is acc += tl.dot(a[:, kb], b[kb, :]) * a_s * b_s with 2-D tile loads (R41, evidence/r41/fp8/blockwise). Together with the
# Hub activation-quantisation kernel (one launch per projection input instead of one per N block) this is the standard-path
# program that BALANCE.md sec 6.1 projects ("tile loads, quant hoisted"). Used here only as a measurement of a real stream.
import torch   # noqa: E402


class QuantThenScaledMM(torch.nn.Module):
    """x (M, K) fp16 -> Hub fp8_act_quant -> torch._scaled_mm with the 1x128 activation and 128x128 weight scales."""

    def __init__(self, n, k, g, device='cpu'):
        super().__init__()
        w, s = X._fp8_weight(g, n, k, device)          # (N, K) E4M3, (NB, KB) fp32
        self.register_buffer('weight', w)
        self.register_buffer('scale', s)
        if device != 'meta':
            X._bind_kernels()

    def forward(self, x):
        if X._MODE[0] == 'ref':
            xq, a_s = X.ActQuant()(x)                   # float64 codes and scales of the reference quantiser
            m, k = xq.shape
            kb = k // X.BLOCK
            n = self.weight.shape[0]
            out = torch.zeros(m, n, dtype=torch.float64)
            for b in range(kb):
                ks = slice(b * X.BLOCK, (b + 1) * X.BLOCK)
                part = xq[:, ks].double() @ self.weight[:, ks].double().t()
                sb = self.scale[:, b].double().repeat_interleave(X.BLOCK)[:n]
                out = out + (part * a_s[:, b:b + 1].double()) * sb[None, :]
            return out.to(torch.float16).to(torch.float64)
        q, a_s = X.ActQuant()(x)
        # b (K, N) column-major as torch._scaled_mm requires; scale_b (KB, NB) for the 128x128 blocks
        return torch._scaled_mm(q, self.weight.t(), a_s, self.scale.t().contiguous(), out_dtype=torch.float16)


def _add_s1(m, n, k):
    def make(seed, device='cpu', m=m, n=n, k=k):
        g = X.gen(seed)
        return QuantThenScaledMM(n, k, g, device), (X._randn(g, (m, k), 1.0, X.FP16, device),)
    mem = X.memory(m * k * 2 + X._weights(n, k) + m * k, m * n * 2)
    X.xl('op', 'fp8_linear_s1ref', 'fp8', dict(m=m, k=k, n=n), make, X.fp8_gemm_demand(m, n, k), tol=4e-3,
         tol_basis=X.TOL_GEMM, mem=mem, fp8=False, reference=X.xl_reference, note='kfuse-speed proxy: quantise + _scaled_mm')


for _m, _n, _k in ((1, 384, 512), (1, 384, 1024), (16, 384, 512), (16, 384, 1024), (16, 384, 2048), (48, 384, 1024),
                   (16, 256, 256), (1, 256, 256), (16, 1152, 1024), (1, 1152, 1024), (48, 1536, 512), (1, 1536, 512), (48, 1536, 1024), (1, 1536, 1024)):
    try:
        _add_s1(_m, _n, _k)
    except Exception:
        pass
