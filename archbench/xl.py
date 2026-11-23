"""The `xl` tier (R42): AI-practical points beyond `large`, FP8 block-wise matrix multiplication first.

Registered before any xl result exists (doc/engineering_discipline.md 2.3). The
author asked for (2026-11-23) "a suite bigger than `large`, with a higher share
of FP8 matrix multiplication and of really AI-practical kernels, [that] does
not blow the device memory". Rules, fixed here:

Shapes. The decoder is the 2B-class one the FP8 work of R41 validated: the text
decoder of Qwen3-VL-2B-Instruct-FP8, hidden 2048, MLP 6144, 16 query heads, 8
KV heads, head dim 128, 28 layers, vocabulary 151936, tied embedding and head
(tools/r41/balance_model.py MODELS["vl"], from the archived plan.json of
evidence/r41/fp8/qwen: K/N pairs 2048/6144 x 2048/1024/6144, and a KV state of
29 360 128 B = 28 layers x 2 x 8 x 256 x 128 x 2 B). Nothing is invented: a
shape is a projection of that decoder, or its sequence/batch is a prefill
(512..2048 tokens) or decode (1 and 16 tokens, 2k..8k cache) the same decoder
serves.

FP8. The matrix points call the Hub kernels `kernels-community/finegrained-fp8`
(v4, kernel commit fe2bb66) unchanged (archbench/fp8_hub.py pins their source by
sha256): `matmul_2d` with 128x128 block scales (E4M3 weights, fp32 scales,
activations quantised dynamically per row and 128-wide K block inside the
kernel, fp32 accumulation, fp16 output) and `fp8_act_quant`. Everything else is
fp16 through the standard path (torch.compile, Inductor, Triton, tnca, default
knobs). No compiler or runtime code recognises these points.

Memory. A launch sees 4 GiB of device memory. Every point declares its live
device bytes -- arguments (inputs, weights, scales, caches, tables), outputs and
scratch (every intermediate of the largest layer alive at once, an upper bound:
no buffer reuse, no fusion saving, softmax written out in binary32) -- and
`memory()` refuses to register a point above 3 GiB. archbench.selftest checks
the declared argument and output bytes against the tensors of the point.

Reference and tolerance. A point's reference is the same module evaluated in
binary64 with exact block-wise FP8 arithmetic (xl_reference). Its tolerance is
derived from rounding floors and is written into the point (`extra.tolerance`)
and PROTOCOL.md 4.2 before any device output exists.

Demand (PROTOCOL.md section 4): counted from the semantic operation. FP8 MACs
are `macs_fp8` (a subset of `macs`). A block-wise W8A8 product needs, per
activation element, a max-abs fold and a scale multiplication; per output element
and K block one fused multiply-add applying the product of the two block scales;
per (row, K block, column block) one multiply forming that product; and one
reciprocal per (row, K block). Undercounting is safe, so nothing else is counted.
"""
import contextlib
import math

import torch
import torch.nn.functional as F

from . import fp8_hub
from .ops import Argmax, Rope, SwiGLU, gen, rope_tables, rotate_half
from .registry import Demand, Point, register

# ---- the decoder (tools/r41/balance_model.py MODELS["vl"]) -----------------
H, INTER, NQ, NKV, HD, VOCAB, LAYERS = 2048, 6144, 16, 8, 128, 151936, 28
BLOCK = 128                      # weight scale block and activation quantisation block
EPS = 1e-6
GiB = 1 << 30
CAP_BYTES = 3 * GiB              # live device bytes of one point (the board exposes 4 GiB to a launch)
FP16 = torch.float16
E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0

_MODE = ['device']               # device: the Hub kernels; emu: CPU fp16 + binary32 twin; ref: binary64 reference


@contextlib.contextmanager
def mode(m):
    old = _MODE[0]
    _MODE[0] = m
    try:
        yield
    finally:
        _MODE[0] = old


def _dt():
    return torch.float64 if _MODE[0] == 'ref' else FP16


def _r(x):
    """A stage boundary: the device stores fp16 here, the binary64 reference rounds to it."""
    return x.to(FP16).to(torch.float64) if x.dtype == torch.float64 else x


def _cast_in(x):
    return x.to(_dt()) if x.is_floating_point() else x


# ---- tensors, built identically on every platform ---------------------------
def _randn(g, shape, std, dtype, device='cpu'):
    """randn * std in `dtype`, drawn in row stripes so the binary32 temporary stays small."""
    if device == 'meta':
        return torch.empty(shape, dtype=dtype, device='meta')
    out = torch.empty(shape, dtype=dtype)
    row = max(1, math.prod(shape[1:]))
    per = max(1, (1 << 24) // row)
    for r0 in range(0, shape[0], per):
        n = min(per, shape[0] - r0)
        out[r0:r0 + n] = (torch.randn((n,) + tuple(shape[1:]), generator=g) * std).to(dtype)
    return out


def _fp8_weight(g, n, k, device='cpu'):
    """(N, K) E4M3 weights with (ceil(N/128), K/128) fp32 block scales, as a block-wise FP8
    checkpoint stores them: scale = amax / 448 of the 128 x 128 block of N(0, 1/K) values."""
    nb, kb = -(-n // BLOCK), k // BLOCK
    assert k % BLOCK == 0
    if device == 'meta':
        return (torch.empty((n, k), dtype=E4M3, device='meta'),
                torch.empty((nb, kb), dtype=torch.float32, device='meta'))
    w = torch.empty((n, k), dtype=E4M3)
    s = torch.empty((nb, kb), dtype=torch.float32)
    std = k ** -0.5
    for i in range(nb):
        r0, r1 = i * BLOCK, min(n, (i + 1) * BLOCK)
        blk = (torch.randn((r1 - r0, k), generator=g) * std).reshape(r1 - r0, kb, BLOCK)
        sc = blk.abs().amax(dim=(0, 2)) / E4M3_MAX
        w[r0:r1] = (blk / sc[None, :, None]).to(E4M3).reshape(r1 - r0, k)
        s[i] = sc
    return w, s


def _ids(g, shape, high, device='cpu'):
    if device == 'meta':
        return torch.empty(shape, dtype=torch.int64, device='meta')
    return torch.randint(0, high, shape, generator=g)


def _ones(shape, device='cpu'):
    return torch.ones(shape, dtype=FP16) if device != 'meta' else torch.empty(shape, dtype=FP16, device='meta')


def _norm_weight(g, d, device='cpu'):
    return 1 + _randn(g, (d,), 0.1, FP16, device)


def _rope(s, offset, device='cpu'):
    if device == 'meta':
        return (torch.empty((s, HD), dtype=FP16, device='meta'), torch.empty((s, HD), dtype=FP16, device='meta'))
    return rope_tables(s, HD, 'fp16', offset=offset)


# ---- block-wise FP8 arithmetic ----------------------------------------------
def _quantise(x2):
    """x2 (M, K): the per-row, per-128-wide-K-block dynamic E4M3 quantisation the Hub kernels perform:
    scale = amax / 448 (binary32 division), codes = RNE(E4M3)(x / max(scale, 1e-12))."""
    m, k = x2.shape
    xf = x2.reshape(m, k // BLOCK, BLOCK)
    a_s = xf.abs().amax(dim=-1) / E4M3_MAX
    q = (xf / a_s.clamp(min=1e-12).unsqueeze(-1)).to(E4M3)
    return q, a_s


def _block_matmul_cpu(x, w, s, exact):
    """C = quantise(x) @ w^T with 128x128 block scales, on the CPU. Per K block the E4M3 products are
    summed (exactly in binary64 when `exact`: every E4M3 value is a multiple of 2^-9, so a 128-term
    dot product needs 43 bits; in binary32 otherwise) and scaled by the row scale and the column-block
    scale, then accumulated in ascending K."""
    lead, k = x.shape[:-1], x.shape[-1]
    n = w.shape[0]
    x2 = x.reshape(-1, k).to(FP16).float()           # the device sees fp16 here (exact in binary32)
    q, a_s = _quantise(x2)
    kb = k // BLOCK
    dt = torch.float64 if exact else torch.float32
    sb = s.repeat_interleave(BLOCK, dim=0)[:n]         # (N, KB) the scale of every column's block
    wq = w.reshape(n, kb, BLOCK)
    acc = torch.zeros((x2.shape[0], n), dtype=dt, device=x.device)
    for j in range(kb):
        part = q[:, j, :].to(dt) @ wq[:, j, :].to(dt).t()
        acc = acc + part * a_s[:, j].to(dt).unsqueeze(1) * sb[:, j].to(dt).unsqueeze(0)
    out = acc.reshape(*lead, n)
    return out.to(FP16).to(torch.float64) if exact else out.to(FP16)


class _Hub:
    """The Hub kernel entry points, bound when a module is built on a real device: Dynamo traces the
    call, so the loader (files, environment, hashes) must have run before."""
    matmul_2d = None
    fp8_act_quant = None


def _bind_kernels():
    try:
        k = fp8_hub.kernels()
    except RuntimeError as e:                              # no Hub source on this host
        return str(e)
    _Hub.matmul_2d, _Hub.fp8_act_quant = k.matmul_2d, k.fp8_act_quant
    return None


def linear_fp8(x, w, s):
    """x (..., K) fp16 -> (..., N) fp16 through the block-wise W8A8 kernel."""
    m = _MODE[0]
    if m == 'device':
        if _Hub.matmul_2d is None:
            raise RuntimeError('the Hub FP8 kernels are not loaded (xl modules bind them when built)')
        return _Hub.matmul_2d(x, w, s, [BLOCK, BLOCK], x.dtype)
    return _block_matmul_cpu(x, w, s, exact=(m == 'ref'))


class FP8Linear(torch.nn.Module):
    def __init__(self, n, k, g, device='cpu'):
        super().__init__()
        w, s = _fp8_weight(g, n, k, device)
        self.register_buffer('weight', w)
        self.register_buffer('scale', s)
        if device != 'meta':
            _bind_kernels()

    def forward(self, x):
        return linear_fp8(_cast_in(x), self.weight, self.scale)


class ActQuant(torch.nn.Module):
    """fp8_act_quant: x (M, K) -> (E4M3 codes (M, K), fp32 block scales (M, K/128))."""

    def __init__(self, device='cpu'):
        super().__init__()
        if device != 'meta':
            _bind_kernels()

    def forward(self, x):
        if _MODE[0] == 'device':
            if _Hub.fp8_act_quant is None:
                raise RuntimeError('the Hub FP8 kernels are not loaded (xl modules bind them when built)')
            return _Hub.fp8_act_quant(x, BLOCK)
        m, k = x.shape
        q, a_s = _quantise(x.to(FP16).float())
        q = q.reshape(m, k)
        return (q.to(torch.float64), a_s.to(torch.float64)) if _MODE[0] == 'ref' else (q, a_s)


# ---- fp16 building blocks ---------------------------------------------------
class Dense(torch.nn.Module):
    """Linear without bias, weights N(0, 1/K) in fp16 (the head, a baseline projection)."""

    def __init__(self, n, k, g, device='cpu'):
        super().__init__()
        self.weight = torch.nn.Parameter(_randn(g, (n, k), k ** -0.5, FP16, device), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


class Embed(torch.nn.Module):
    def __init__(self, v, d, g, device='cpu'):
        super().__init__()
        self.weight = torch.nn.Parameter(_randn(g, (v, d), 1.0, FP16, device), requires_grad=False)

    def forward(self, ids):
        return F.embedding(ids, self.weight)


class GQA(torch.nn.Module):
    """Grouped-query attention over given q, k, v: the KV heads are repeated to the query heads the way
    transformers' repeat_kv does (an expand and a reshape), then scaled_dot_product_attention."""

    def __init__(self, causal):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        b, nq, _, hd = q.shape
        nkv, kv = k.shape[1], k.shape[2]
        rep = nq // nkv
        k = k[:, :, None].expand(b, nkv, rep, kv, hd).reshape(b, nq, kv, hd)
        v = v[:, :, None].expand(b, nkv, rep, kv, hd).reshape(b, nq, kv, hd)
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)


class RMSNormResidual(torch.nn.Module):
    """h = x + y; return (h, rmsnorm(h)): the pair every decoder layer boundary produces."""

    def __init__(self, d, g, device='cpu'):
        super().__init__()
        self.weight = torch.nn.Parameter(_norm_weight(g, d, device), requires_grad=False)
        self.d = d

    def forward(self, x, y):
        h = x + y
        return h, F.rms_norm(h, (self.d,), self.weight, EPS)


# ---- the FP8 decoder --------------------------------------------------------
class Fp8Layer(torch.nn.Module):
    """One LLaMA/Qwen-style layer with block-wise FP8 projections (q, k, v, o, gate, up, down), RMSNorm,
    RoPE, grouped-query attention and SwiGLU. `past_k/past_v` None: prefill (causal); given: one decode
    token against that cache. Returns (hidden, new k, new v)."""

    def __init__(self, g, device='cpu'):
        super().__init__()
        self.attn_norm = torch.nn.Parameter(_norm_weight(g, H, device), requires_grad=False)
        self.q = FP8Linear(NQ * HD, H, g, device)
        self.k = FP8Linear(NKV * HD, H, g, device)
        self.v = FP8Linear(NKV * HD, H, g, device)
        self.o = FP8Linear(H, NQ * HD, g, device)
        self.mlp_norm = torch.nn.Parameter(_norm_weight(g, H, device), requires_grad=False)
        self.gate = FP8Linear(INTER, H, g, device)
        self.up = FP8Linear(INTER, H, g, device)
        self.down = FP8Linear(H, INTER, g, device)

    def forward(self, x, cos, sin, past_k=None, past_v=None):
        dt = x.dtype
        b, s, _ = x.shape
        h = _r(F.rms_norm(x, (H,), self.attn_norm.to(dt), EPS))
        q = _r(self.q(h)).view(b, s, NQ, HD).transpose(1, 2)
        k = _r(self.k(h)).view(b, s, NKV, HD).transpose(1, 2)
        v = _r(self.v(h)).view(b, s, NKV, HD).transpose(1, 2)
        cos, sin = cos.to(dt), sin.to(dt)
        q = _r(q * cos + rotate_half(q) * sin)
        k = _r(k * cos + rotate_half(k) * sin)
        new_k, new_v = k, v
        if past_k is not None:
            k = torch.cat((past_k.to(dt), k), dim=2)
            v = torch.cat((past_v.to(dt), v), dim=2)
        rep = NQ // NKV
        kv = k.shape[2]
        k = k[:, :, None].expand(b, NKV, rep, kv, HD).reshape(b, NQ, kv, HD)
        v = v[:, :, None].expand(b, NKV, rep, kv, HD).reshape(b, NQ, kv, HD)
        o = _r(F.scaled_dot_product_attention(q, k, v, is_causal=past_k is None))
        o = o.transpose(1, 2).reshape(b, s, NQ * HD)
        x = _r(x + _r(self.o(o)))
        y = _r(F.rms_norm(x, (H,), self.mlp_norm.to(dt), EPS))
        act = _r(F.silu(_r(self.gate(y))) * _r(self.up(y)))
        x = _r(x + _r(self.down(act)))
        return x, new_k, new_v


class Fp8LayerPrefill(torch.nn.Module):
    """hidden (1, S, H) -> (hidden, k (1, 8, S, 128), v)."""

    def __init__(self, g, s, device='cpu'):
        super().__init__()
        self.layer = Fp8Layer(g, device)
        cos, sin = _rope(s, 0, device)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x):
        return self.layer(_cast_in(x), self.cos, self.sin)


class Fp8LayerDecode(torch.nn.Module):
    """hidden (1, 1, H) at position P, cache (1, 8, P, 128) x2 -> (hidden, new k (1, 8, 1, 128), new v)."""

    def __init__(self, g, p, device='cpu'):
        super().__init__()
        self.layer = Fp8Layer(g, device)
        cos, sin = _rope(1, p, device)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x, past_k, past_v):
        return self.layer(_cast_in(x), self.cos, self.sin, past_k, past_v)


def _head(h, tok):
    """h (1, 1, H) @ tok^T -> (1, 1, V); binary64 reference in row chunks (the table is 622 MB)."""
    if h.dtype != torch.float64:
        return F.linear(h, tok)
    out = torch.empty((1, 1, tok.shape[0]), dtype=torch.float64, device=h.device)
    for r0 in range(0, tok.shape[0], 1 << 14):
        out[0, 0, r0:r0 + (1 << 14)] = tok[r0:r0 + (1 << 14)].to(torch.float64) @ h[0, 0]
    return _r(out)


class Fp8DecoderPrefill(torch.nn.Module):
    """tokens (1, S) -> logits of the last position (1, 1, V); tied embedding and head."""

    def __init__(self, g, layers, s, device='cpu'):
        super().__init__()
        self.tok = torch.nn.Parameter(_randn(g, (VOCAB, H), 0.02, FP16, device), requires_grad=False)
        self.layers = torch.nn.ModuleList(Fp8Layer(g, device) for _ in range(layers))
        self.norm = torch.nn.Parameter(_ones((H,), device), requires_grad=False)
        cos, sin = _rope(s, 0, device)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, tokens):
        x = _cast_in(F.embedding(tokens, self.tok))
        for layer in self.layers:
            x, _, _ = layer(x, self.cos, self.sin)
        h = _r(F.rms_norm(x[:, -1:, :], (H,), self.norm.to(x.dtype), EPS))
        return _head(h, self.tok)


class Fp8DecoderDecode(torch.nn.Module):
    """token (1, 1) at position P, caches (L, 1, 8, P, 128) x2 -> (logits (1, 1, V), new k (L, 1, 8, 1, 128), new v)."""

    def __init__(self, g, layers, p, device='cpu'):
        super().__init__()
        self.tok = torch.nn.Parameter(_randn(g, (VOCAB, H), 0.02, FP16, device), requires_grad=False)
        self.layers = torch.nn.ModuleList(Fp8Layer(g, device) for _ in range(layers))
        self.norm = torch.nn.Parameter(_ones((H,), device), requires_grad=False)
        cos, sin = _rope(1, p, device)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, token, past_k, past_v):
        x = _cast_in(F.embedding(token, self.tok))
        nks, nvs = [], []
        for i, layer in enumerate(self.layers):
            x, nk, nv = layer(x, self.cos, self.sin, past_k[i], past_v[i])
            nks.append(nk)
            nvs.append(nv)
        h = _r(F.rms_norm(x, (H,), self.norm.to(x.dtype), EPS))
        return _head(h, self.tok), torch.stack(nks), torch.stack(nvs)


def xl_reference(module, inputs):
    """The point's reference: the same module in binary64, exact block-wise FP8 arithmetic, fp16 at stage
    boundaries (the precision of the tensors the device stores between its kernels)."""
    from .platforms.common import flatten
    with mode('ref'), torch.no_grad():
        return flatten(module(*inputs))


# ---- demand (semantic, PROTOCOL.md section 4) --------------------------------
def _cdiv(a, b):
    return -(-a // b)


def fp8_gemm_demand(m, n, k):
    kb, nb = k // BLOCK, _cdiv(n, BLOCK)
    return Demand(macs=m * n * k, macs_fp8=m * n * k,
                  fp_ops=2 * m * k + m * n * kb + m * nb * kb, sfu_ops=m * kb,
                  read_bytes=m * k * 2 + n * k + nb * kb * 4, write_bytes=m * n * 2)


def act_quant_demand(m, k):
    return Demand(fp_ops=2 * m * k, sfu_ops=m * (k // BLOCK),
                  read_bytes=m * k * 2, write_bytes=m * k + m * (k // BLOCK) * 4)


def _weights(n, k):
    return n * k + _cdiv(n, BLOCK) * (k // BLOCK) * 4          # E4M3 weights + fp32 block scales


LAYER_LINEARS = ((NQ * HD, H), (NKV * HD, H), (NKV * HD, H), (H, NQ * HD), (INTER, H), (INTER, H), (H, INTER))


def layer_weight_bytes():
    return sum(_weights(n, k) for n, k in LAYER_LINEARS) + 2 * H * 2      # + the two RMSNorm weights, fp16


def layer_components(t, past, *, boundary):
    """One layer over `t` tokens, split by what executes it (the split is also how the device time of a
    composite point is estimated, archbench/xl_estimate.py): the seven block-wise FP8 projections, the
    attention, and the element-wise kernels (norms, residuals, RoPE, SwiGLU). Prefill: `past` None, causal,
    t(t+1)/2 scores per head. Decode: t = 1 against `past` cached entries. `boundary`: the hidden state enters
    and leaves through memory (a standalone layer); otherwise it is an intermediate of a bigger point."""
    work = t * (t + 1) // 2 if past is None else t * (past + 1)
    macs_fp8 = sum(t * n * k for n, k in LAYER_LINEARS)
    quantised = (H, NQ * HD, H, INTER)              # x1 (q, k, v), attention output, y2 (gate, up), swiglu output (down)
    fp_ops = sum(2 * t * k for k in quantised)
    fp_ops += sum(t * n * (k // BLOCK) + t * _cdiv(n, BLOCK) * (k // BLOCK) for n, k in LAYER_LINEARS)
    fp8 = Demand(macs=macs_fp8, macs_fp8=macs_fp8, fp_ops=fp_ops, sfu_ops=sum(t * (k // BLOCK) for k in quantised),
                 read_bytes=sum(_weights(n, k) for n, k in LAYER_LINEARS))
    attn = Demand(macs=2 * NQ * HD * work, fp_ops=3 * NQ * work, sfu_ops=NQ * work,
                  read_bytes=0 if past is None else 2 * NKV * past * HD * 2)
    elem = Demand(fp_ops=2 * 3 * t * H + 2 * t * H + 3 * t * (NQ + NKV) * HD + 3 * t * INTER,
                  sfu_ops=2 * t + t * INTER,
                  read_bytes=2 * H * 2 + 2 * t * HD * 2 + (t * H * 2 if boundary else 0),      # norms, cos and sin, input
                  write_bytes=2 * t * NKV * HD * 2 + (t * H * 2 if boundary else 0))        # new k and v, output
    return dict(fp8_projections=fp8, attention=attn, elementwise=elem)


def layer_demand(t, past, *, boundary):
    c = layer_components(t, past, boundary=boundary)
    return c['fp8_projections'] + c['attention'] + c['elementwise']


def decoder_components(layers, t, past, *, prefill):
    """`layers` layers plus the tied-table embedding gather, the final norm and the head (last position only for
    prefill). Hidden states between stages are not counted; the table is read once (the head reads all of it)."""
    c = layer_components(t, None if prefill else past, boundary=False)

    def times(d, n):
        return Demand(d.macs * n, d.fp_ops * n, d.sfu_ops * n, d.read_bytes * n, d.write_bytes * n, d.macs_fp8 * n)
    elem = times(c['elementwise'], layers)
    elem = Demand(elem.macs, elem.fp_ops + 3 * H, elem.sfu_ops + 1,
                  elem.read_bytes - (layers - 1) * 2 * t * HD * 2 + t * 8 + H * 2,    # one cos/sin table, shared; ids; norm
                  0 if prefill else layers * 2 * NKV * HD * 2)                         # the new k and v of every layer (decode)
    head = Demand(macs=VOCAB * H, read_bytes=VOCAB * H * 2, write_bytes=VOCAB * 2)
    return dict(fp8_projections=times(c['fp8_projections'], layers), attention=times(c['attention'], layers),
                elementwise=elem, head=head)


def decoder_demand(layers, t, past, *, prefill):
    d = Demand()
    for part in decoder_components(layers, t, past, prefill=prefill).values():
        d = d + part
    return d


# ---- live device bytes -------------------------------------------------------
def layer_scratch(t, kvlen, decode):
    """Every intermediate of one layer alive at once (an upper bound): the normalised input, q/k/v and their
    RoPE'd copies, the concatenated cache (decode) and the KV heads repeated to the query heads, the scores in
    fp16 plus the binary32 softmax written out (12 bytes per score), the attention output and its transposed
    copy, the projections' outputs, both residual states and the MLP's three tensors."""
    b = 2
    n = t * H * b                                            # x1
    n += t * (NQ + 2 * NKV) * HD * b                         # q, k, v
    n += t * (NQ + NKV) * HD * b                             # roped q, k
    n += 2 * NKV * kvlen * HD * b if decode else 0           # cat(past, new)
    n += 2 * NQ * kvlen * HD * b                             # repeated k, v
    n += NQ * t * kvlen * 12                                 # scores fp16 + softmax fp32 + probs
    n += 2 * t * NQ * HD * b                                 # o and its transposed copy
    n += 4 * t * H * b                                       # o_proj out, x2, y2, down out
    n += 3 * t * INTER * b                                   # gate, up, act
    n += t * H * b                                           # layer output
    return n


def memory(args, out, scratch=0):
    live = args + out + scratch
    if live > CAP_BYTES:
        raise ValueError(f'live device bytes {live} exceed the {CAP_BYTES} cap')
    return dict(args_bytes=args, out_bytes=out, scratch_bytes=scratch, live_bytes=live, cap_bytes=CAP_BYTES)


# ---- registration ------------------------------------------------------------
def _fmt(params):
    return '-'.join(f'{k}{v}' for k, v in params.items())


FP8_NOTE = ('E4M3 weights with 128x128 fp32 block scales; activations fp16, quantised per row and 128-wide K block '
            'inside the kernel; fp32 accumulation; fp16 output')
KERNELS = dict(fp8_hub.IDENTITY)


COMPONENTS = {}       # point id -> {part: Demand}: how a composite point's device time is estimated (xl_estimate.py)


def xl(kind, family, dtype, params, make, demand, *, tol, tol_basis, mem, track=('llm',), note='',
       fp8=False, reference=None, components=None):
    pid = f'{kind}.{family}/xl/{dtype}-{_fmt(params)}'
    if components is not None:
        total = Demand()
        for part in components.values():
            total = total + part
        assert total == demand, f'{pid}: the components do not add up to the demand'
        COMPONENTS[pid] = components
    extra = dict(memory=mem, tolerance_basis=tol_basis)
    if fp8:
        extra.update(kernels=KERNELS, fp8=FP8_NOTE)
    return register(Point(pid, kind, family, 'xl', dtype, dict(params), make, demand, track=tuple(track), tol=tol,
                          note=note, reference=reference, extra=extra))


# Tolerance rules (written before any device output; PROTOCOL.md 4.2 and the registration section of
# doc/r42_plan.md derive them). `max|got - ref| / max(max|ref|, 1)`.
TOL_F16 = ('registry fp16 convention, 2e-2: 40x the fp16 output rounding (2^-11 of the value), as for every other '
           'fp16 point; the reference is the same module in binary32')
TOL_GEMM = ('4e-3 = 8x the fp16 output rounding floor (2^-11 relative to the value): the block-wise product is '
            'exact up to binary32 accumulation (<= 1e-5), the activation codes are reproduced exactly '
            '(IEEE binary32 division in matmul_2d), the reference is the binary64 block-wise product')
TOL_ACTQ = ('8e-2 = the largest one-code deviation of an E4M3 value relative to the largest code in its block: one ulp '
            'in the top binade is 32/448 = 7.1e-2; fp8_act_quant divides with the default (faithful, <= 1 ulp) '
            'quotient, which flips about 0.5% of codes by one ulp (evidence/r41/fp8/kernels/README.md), and no '
            'two-ulp error is tolerated at the top')


def _gemm_points():
    # (M, N, K, role): prefill GEMMs of one layer's projections at 512 and 2048 tokens, decode at 1 and 16 tokens,
    # and the head against the full vocabulary (a 311 MB weight)
    for m, n, k in ((2048, 4096, 2048), (2048, 2048, 2048), (2048, 12288, 2048), (2048, 2048, 6144),
                    (512, 12288, 2048), (512, 2048, 6144),
                    (1, 4096, 2048), (1, 12288, 2048), (1, 2048, 6144), (1, VOCAB, 2048),
                    (16, 12288, 2048), (16, 2048, 6144), (16, VOCAB, 2048)):
        def make(seed, device='cpu', m=m, n=n, k=k):
            g = gen(seed)
            lin = FP8Linear(n, k, g, device)
            return lin, (_randn(g, (m, k), 1.0, FP16, device),)
        mem = memory(m * k * 2 + _weights(n, k), m * n * 2)
        xl('op', 'fp8_linear', 'fp8', dict(m=m, k=k, n=n), make, fp8_gemm_demand(m, n, k), tol=4e-3,
           tol_basis=TOL_GEMM, mem=mem, fp8=True, reference=xl_reference,
           note=f'{"prefill" if m >= 512 else "decode"} projection, weight {n * k / 2**20:.0f} MiB of E4M3')


_gemm_points()


def _act_quant_points():
    for m, k in ((2048, 2048), (2048, 6144)):
        def make(seed, device='cpu', m=m, k=k):
            return ActQuant(device), (_randn(gen(seed), (m, k), 1.0, FP16, device),)
        xl('op', 'fp8_act_quant', 'fp8', dict(m=m, k=k), make, act_quant_demand(m, k), tol=8e-2, tol_basis=TOL_ACTQ,
           mem=memory(m * k * 2, m * k + m * (k // BLOCK) * 4), fp8=True, reference=xl_reference,
           note='per-row, 128-wide block E4M3 activation quantisation of a prefill activation')


_act_quant_points()


def _attention_points():
    for s in (1024, 2048):
        def make(seed, device='cpu', s=s):
            g = gen(seed)
            return GQA(True), (_randn(g, (1, NQ, s, HD), 1.0, FP16, device), _randn(g, (1, NKV, s, HD), 1.0, FP16, device),
                               _randn(g, (1, NKV, s, HD), 1.0, FP16, device))
        tri = s * (s + 1) // 2
        d = Demand(macs=2 * NQ * HD * tri, fp_ops=3 * NQ * tri, sfu_ops=NQ * tri,
                   read_bytes=(NQ + 2 * NKV) * s * HD * 2, write_bytes=NQ * s * HD * 2)
        mem = memory((NQ + 2 * NKV) * s * HD * 2, NQ * s * HD * 2,
                     2 * NQ * s * HD * 2 + NQ * s * s * 12)
        xl('op', 'attention_prefill', 'fp16', dict(b=1, h=NQ, kv=NKV, s=s, d=HD), make, d, tol=2e-2, tol_basis=TOL_F16,
           mem=mem, note='causal grouped-query attention: only the s(s+1)/2 unmasked scores are work')
    for l in (2048, 4096, 8192):
        def make(seed, device='cpu', l=l):
            g = gen(seed)
            return GQA(False), (_randn(g, (1, NQ, 1, HD), 1.0, FP16, device), _randn(g, (1, NKV, l, HD), 1.0, FP16, device),
                                _randn(g, (1, NKV, l, HD), 1.0, FP16, device))
        d = Demand(macs=2 * NQ * HD * l, fp_ops=3 * NQ * l, sfu_ops=NQ * l,
                   read_bytes=(NQ * HD + 2 * NKV * l * HD) * 2, write_bytes=NQ * HD * 2)
        mem = memory((NQ * HD + 2 * NKV * l * HD) * 2, NQ * HD * 2, 2 * NQ * l * HD * 2 + NQ * l * 12)
        xl('op', 'attention_decode', 'fp16', dict(b=1, h=NQ, kv=NKV, l=l, d=HD), make, d, tol=2e-2, tol_basis=TOL_F16,
           mem=mem, note='one query token against an l-entry KV cache of 8 KV heads')


_attention_points()


def _fp16_points():
    r = 2048

    def mk_rmsres(seed, device='cpu'):
        g = gen(seed)
        m = RMSNormResidual(H, g, device)
        return m, (_randn(g, (r, H), 1.0, FP16, device), _randn(g, (r, H), 1.0, FP16, device))
    n = r * H
    xl('op', 'rmsnorm_residual', 'fp16', dict(r=r, d=H), mk_rmsres,
       Demand(fp_ops=4 * n, sfu_ops=r, read_bytes=(2 * n + H) * 2, write_bytes=2 * n * 2), tol=2e-2, tol_basis=TOL_F16,
       mem=memory((2 * n + H) * 2, 2 * n * 2, n * 2),
       note='the residual add and the following RMSNorm of a layer boundary, both results written')

    def mk_swiglu(seed, device='cpu'):
        g = gen(seed)
        return SwiGLU(), (_randn(g, (r, INTER), 1.0, FP16, device), _randn(g, (r, INTER), 1.0, FP16, device))
    n = r * INTER
    xl('op', 'swiglu', 'fp16', dict(r=r, i=INTER), mk_swiglu,
       Demand(fp_ops=3 * n, sfu_ops=n, read_bytes=2 * n * 2, write_bytes=n * 2), tol=2e-2, tol_basis=TOL_F16,
       mem=memory(2 * n * 2, n * 2, n * 2), note='the MLP activation of a 2048-token prefill')

    def mk_rope(seed, device='cpu'):
        cos, sin = _rope(2048, 0, device)
        return Rope(), (_randn(gen(seed), (1, NQ, 2048, HD), 1.0, FP16, device), cos, sin)
    n = NQ * 2048 * HD
    xl('op', 'rope', 'fp16', dict(b=1, h=NQ, s=2048, d=HD), mk_rope,
       Demand(fp_ops=3 * n, read_bytes=(n + 2 * 2048 * HD) * 2, write_bytes=n * 2), tol=2e-2, tol_basis=TOL_F16,
       mem=memory((n + 2 * 2048 * HD) * 2, n * 2, 2 * n * 2), note='query heads of a 2048-token prefill')

    def mk_gemv(seed, device='cpu'):
        g = gen(seed)
        return Dense(VOCAB, H, g, device), (_randn(g, (1, H), 1.0, FP16, device),)
    xl('op', 'gemv', 'fp16', dict(k=H, n=VOCAB), mk_gemv,
       Demand(macs=H * VOCAB, read_bytes=(H + H * VOCAB) * 2, write_bytes=VOCAB * 2), tol=2e-2, tol_basis=TOL_F16,
       mem=memory((H + H * VOCAB) * 2, VOCAB * 2), note='the fp16 tied head: a 594 MiB weight against one token')

    def mk_linear(seed, device='cpu'):
        g = gen(seed)
        return Dense(INTER, H, g, device), (_randn(g, (2048, H), 1.0, FP16, device),)
    xl('op', 'linear', 'fp16', dict(m=2048, k=H, n=INTER, bias=0), mk_linear,
       Demand(macs=2048 * H * INTER, read_bytes=(2048 * H + H * INTER) * 2, write_bytes=2048 * INTER * 2), tol=2e-2,
       tol_basis=TOL_F16, mem=memory((2048 * H + H * INTER) * 2, 2048 * INTER * 2),
       note='the fp16 counterpart of the FP8 gate projection at 2048 tokens')

    def mk_emb(seed, device='cpu'):
        g = gen(seed)
        m = Embed(VOCAB, H, g, device)
        ids = (torch.randperm(VOCAB, generator=g)[:2048] if device != 'meta'
               else torch.empty((2048,), dtype=torch.int64, device='meta'))
        return m, (ids,)
    xl('op', 'embedding', 'fp16', dict(v=VOCAB, d=H, t=2048), mk_emb,
       Demand(read_bytes=2048 * 8 + 2048 * H * 2, write_bytes=2048 * H * 2), tol=0.0, tol_basis='exact: a gather',
       mem=memory(VOCAB * H * 2 + 2048 * 8, 2048 * H * 2), note='2048 rows of the 594 MiB tied table')

    def mk_argmax(seed, device='cpu'):
        return Argmax(), (_randn(gen(seed), (16, VOCAB), 3.0, FP16, device),)
    xl('op', 'argmax', 'fp16', dict(r=16, v=VOCAB), mk_argmax,
       Demand(fp_ops=16 * VOCAB, read_bytes=16 * VOCAB * 2, write_bytes=16 * 8), tol=0.0,
       tol_basis='exact: greedy sampling (the data of seed 0 has no tie at the maximum, checked in the selftest)',
       mem=memory(16 * VOCAB * 2, 16 * 8), note='greedy sampling of 16 sequences over the vocabulary')


_fp16_points()


def _model_points():
    # a complete layer, prefill at 512 tokens
    s = 512

    def mk(seed, device='cpu', s=s):
        g = gen(seed)
        return Fp8LayerPrefill(g, s, device), (_randn(g, (1, s, H), 1.0, FP16, device),)
    mem = memory(layer_weight_bytes() + s * H * 2 + 2 * s * HD * 2, s * H * 2 + 2 * NKV * s * HD * 2,
                 layer_scratch(s, s, False))
    xl('model', 'fp8_layer_prefill', 'fp8', dict(b=1, s=s, layers=1), mk, layer_demand(s, None, boundary=True),
       tol=TOL_MODEL[0], tol_basis=TOL_MODEL[1], mem=mem, fp8=True, reference=xl_reference,
       components=layer_components(s, None, boundary=True),
       note='one of 28 layers: 7 block-wise FP8 projections, RMSNorm, RoPE, GQA, SwiGLU')
    # a complete layer, decode against 2048 cached positions
    p = 2048

    def mk(seed, device='cpu', p=p):
        g = gen(seed)
        m = Fp8LayerDecode(g, p, device)
        return m, (_randn(g, (1, 1, H), 1.0, FP16, device), _randn(g, (1, NKV, p, HD), 1.0, FP16, device),
                   _randn(g, (1, NKV, p, HD), 1.0, FP16, device))
    mem = memory(layer_weight_bytes() + H * 2 + 2 * HD * 2 + 2 * NKV * p * HD * 2, H * 2 + 2 * NKV * HD * 2,
                 layer_scratch(1, p + 1, True))
    xl('model', 'fp8_layer_decode', 'fp8', dict(b=1, p=p, layers=1), mk, layer_demand(1, p, boundary=True),
       tol=TOL_MODEL[0], tol_basis=TOL_MODEL[1], mem=mem, fp8=True, reference=xl_reference,
       components=layer_components(1, p, boundary=True),
       note='one token through one layer against a 2048-entry cache')
    # the whole decoder, prefill at 512 tokens; and a quarter-depth slice at 2048 tokens
    for layers, s in ((LAYERS, 512), (4, 2048)):
        def mk(seed, device='cpu', layers=layers, s=s):
            g = gen(seed)
            return Fp8DecoderPrefill(g, layers, s, device), (_ids(g, (1, s), VOCAB, device),)
        wb = VOCAB * H * 2 + layers * layer_weight_bytes() + H * 2
        mem = memory(wb + 2 * s * HD * 2 + s * 8, VOCAB * 2, layer_scratch(s, s, False) + s * H * 2 + H * 2)
        xl('model', 'fp8_decoder_prefill', 'fp8', dict(b=1, s=s, layers=layers), mk,
           decoder_demand(layers, s, None, prefill=True), tol=TOL_MODEL[0], tol_basis=TOL_MODEL[1], mem=mem,
           fp8=True, reference=xl_reference, components=decoder_components(layers, s, None, prefill=True),
           note=('the whole 28-layer text decoder' if layers == LAYERS else
                 f'{layers} of 28 layers (shrunk from the real depth: the {s}-token score tensors and compile time)')
                + ', tied embedding and head, logits of the last position')
    # the whole decoder, one decode step against 2048 cached positions
    p, layers = 2048, LAYERS

    def mk(seed, device='cpu', p=p, layers=layers):
        g = gen(seed)
        m = Fp8DecoderDecode(g, layers, p, device)
        shape = (layers, 1, NKV, p, HD)
        return m, (_ids(g, (1, 1), VOCAB, device), _randn(g, shape, 1.0, FP16, device),
                   _randn(g, shape, 1.0, FP16, device))
    wb = VOCAB * H * 2 + layers * layer_weight_bytes() + H * 2
    mem = memory(wb + 2 * HD * 2 + 8 + 2 * layers * NKV * p * HD * 2, VOCAB * 2 + 2 * layers * NKV * HD * 2,
                 layer_scratch(1, p + 1, True) + H * 2)
    xl('model', 'fp8_decoder_decode', 'fp8', dict(b=1, p=p, layers=layers), mk,
       decoder_demand(layers, 1, p, prefill=False), tol=TOL_MODEL[0], tol_basis=TOL_MODEL[1], mem=mem, fp8=True,
       reference=xl_reference, components=decoder_components(layers, 1, p, prefill=False), note='the whole 28-layer text decoder, one token, tied embedding and head')


# Layer and model tolerance: set by tools/r42/xl_tolerance.py from the CPU rounding floor (see PROTOCOL.md 4.2);
# the value is written here, never changed after a device output exists.
TOL_MODEL = (0.1, 'placeholder until xl_tolerance.py has run')
_model_points()
