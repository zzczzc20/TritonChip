"""End-to-end model points.

Self-contained module definitions (no `transformers` dependency, no weight
download): random weights from the point's seed, because what ArchBench
measures -- time and agreement with the reference -- does not depend on the
weights being trained, and a fixed seed makes every platform compute the same
numbers. The architectures follow the published designs:

  llama     LLaMA-2/3 decoder: RMSNorm, grouped-query attention with RoPE,
            SwiGLU MLP, untied LM head. `large` is the Llama-3.2-1B shape.
            Two entry points, because they stress a chip differently:
              prefill  S prompt tokens, logits for the last one only
              decode   one token against a P-entry KV cache, returning the
                       logits and the new K/V entries (the caller appends)
  gpt2      GPT-2 block: LayerNorm, fused QKV with bias, causal attention,
            GELU(tanh) MLP with bias
  mlp       Linear -> GELU(tanh) -> Linear
  resnet    ResNet basic block (general track): conv3x3-BN-ReLU-conv3x3-BN,
            residual, ReLU; BN folded to its eval affine form

Demands are analytic (counted from the configuration, not from a run) and
include only what the result needs: weights read once, the embedding rows
actually gathered, the KV cache read once, outputs written once.
"""
from dataclasses import dataclass, asdict
import math

import torch
import torch.nn.functional as F

from .ops import gen, randn, rope_tables, rotate_half
from .registry import Demand, Point, TORCH_DTYPE, register

I64 = 8


def _dt(dtype):
    return getattr(torch, TORCH_DTYPE[dtype])


def _lin(k, n, g, dtype, bias=False, std=0.02):
    m = torch.nn.Linear(k, n, bias=bias)
    with torch.no_grad():
        m.weight.copy_(randn(g, (n, k), 'fp32', std))
        if bias:
            m.bias.copy_(randn(g, (n,), 'fp32', std))
    return m.to(_dt(dtype))


# ---------------------------------------------------------------------------
# LLaMA
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LlamaCfg:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn: int
    vocab: int
    eps: float = 1e-5

    @property
    def hd(self):
        return self.dim // self.n_heads


class LlamaAttention(torch.nn.Module):
    def __init__(self, c, g, dtype):
        super().__init__()
        self.c = c
        self.wq = _lin(c.dim, c.n_heads * c.hd, g, dtype)
        self.wk = _lin(c.dim, c.n_kv_heads * c.hd, g, dtype)
        self.wv = _lin(c.dim, c.n_kv_heads * c.hd, g, dtype)
        self.wo = _lin(c.n_heads * c.hd, c.dim, g, dtype)

    def forward(self, x, cos, sin, past_k=None, past_v=None):
        c = self.c
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, c.n_heads, c.hd).transpose(1, 2)
        k = self.wk(x).view(b, s, c.n_kv_heads, c.hd).transpose(1, 2)
        v = self.wv(x).view(b, s, c.n_kv_heads, c.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        new_k, new_v = k, v
        if past_k is not None:
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        rep = c.n_heads // c.n_kv_heads
        if rep > 1:                              # grouped-query attention, as repeat_kv does
            k = k[:, :, None].expand(b, c.n_kv_heads, rep, k.shape[2], c.hd).reshape(b, c.n_heads, -1, c.hd)
            v = v[:, :, None].expand(b, c.n_kv_heads, rep, v.shape[2], c.hd).reshape(b, c.n_heads, -1, c.hd)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=past_k is None)
        o = o.transpose(1, 2).reshape(b, s, c.n_heads * c.hd)
        return self.wo(o), new_k, new_v


class LlamaBlock(torch.nn.Module):
    def __init__(self, c, g, dtype):
        super().__init__()
        self.c = c
        self.attn_norm = torch.nn.Parameter(torch.ones(c.dim, dtype=_dt(dtype)))
        self.attn = LlamaAttention(c, g, dtype)
        self.mlp_norm = torch.nn.Parameter(torch.ones(c.dim, dtype=_dt(dtype)))
        self.w1 = _lin(c.dim, c.ffn, g, dtype)
        self.w3 = _lin(c.dim, c.ffn, g, dtype)
        self.w2 = _lin(c.ffn, c.dim, g, dtype)

    def forward(self, x, cos, sin, past_k=None, past_v=None):
        h, nk, nv = self.attn(F.rms_norm(x, (self.c.dim,), self.attn_norm, self.c.eps),
                              cos, sin, past_k, past_v)
        x = x + h
        y = F.rms_norm(x, (self.c.dim,), self.mlp_norm, self.c.eps)
        x = x + self.w2(F.silu(self.w1(y)) * self.w3(y))
        return x, nk, nv


class Llama(torch.nn.Module):
    def __init__(self, c, g, dtype):
        super().__init__()
        self.c = c
        self.tok = torch.nn.Parameter(randn(g, (c.vocab, c.dim), dtype, 0.02))
        self.layers = torch.nn.ModuleList(LlamaBlock(c, g, dtype) for _ in range(c.n_layers))
        self.norm = torch.nn.Parameter(torch.ones(c.dim, dtype=_dt(dtype)))
        self.out = _lin(c.dim, c.vocab, g, dtype)
        self.dtype = dtype

    def _head(self, x):
        return self.out(F.rms_norm(x, (self.c.dim,), self.norm, self.c.eps))


class LlamaPrefill(Llama):
    """tokens [B, S] -> logits of the last position [B, 1, V]."""

    def __init__(self, c, g, dtype, s):
        super().__init__(c, g, dtype)
        cos, sin = rope_tables(s, c.hd, dtype)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, tokens):
        x = F.embedding(tokens, self.tok)
        for layer in self.layers:
            x, _, _ = layer(x, self.cos, self.sin)
        return self._head(x[:, -1:, :])


class LlamaDecode(Llama):
    """token [B, 1] at position P, cache [L, B, Hkv, P, hd] x2 ->
    (logits [B, 1, V], new k [L, B, Hkv, 1, hd], new v)."""

    def __init__(self, c, g, dtype, p):
        super().__init__(c, g, dtype)
        cos, sin = rope_tables(1, c.hd, dtype, offset=p)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, token, past_k, past_v):
        x = F.embedding(token, self.tok)
        nks, nvs = [], []
        for i, layer in enumerate(self.layers):
            x, nk, nv = layer(x, self.cos, self.sin, past_k[i], past_v[i])
            nks.append(nk)
            nvs.append(nv)
        return self._head(x), torch.stack(nks), torch.stack(nvs)


def _llama_layer_macs(c, tokens):
    proj = c.dim * c.n_heads * c.hd * 2 + c.dim * c.n_kv_heads * c.hd * 2
    return tokens * (proj + 3 * c.dim * c.ffn)


def _llama_weight_bytes(c, b):
    per_layer = (c.dim * c.n_heads * c.hd * 2 + c.dim * c.n_kv_heads * c.hd * 2
                 + 3 * c.dim * c.ffn + 2 * c.dim)
    return b * (c.n_layers * per_layer + c.dim + c.dim * c.vocab)   # final norm + head; table excluded


def llama_prefill_demand(c, bsz, s, b=2):
    tokens = bsz * s
    tri = s * (s + 1) // 2
    attn = 2 * bsz * c.n_heads * c.hd * tri
    return Demand(macs=c.n_layers * (_llama_layer_macs(c, tokens) + attn) + bsz * c.dim * c.vocab,
                  sfu_ops=c.n_layers * bsz * c.n_heads * tri,
                  read_bytes=_llama_weight_bytes(c, b) + tokens * I64 + tokens * c.dim * b,
                  write_bytes=bsz * c.vocab * b)


def llama_decode_demand(c, bsz, p, b=2):
    attn = 2 * bsz * c.n_heads * c.hd * (p + 1)
    cache = 2 * c.n_layers * bsz * c.n_kv_heads * p * c.hd * b
    new = 2 * c.n_layers * bsz * c.n_kv_heads * c.hd * b
    return Demand(macs=c.n_layers * (_llama_layer_macs(c, bsz) + attn) + bsz * c.dim * c.vocab,
                  sfu_ops=c.n_layers * bsz * c.n_heads * (p + 1),
                  read_bytes=_llama_weight_bytes(c, b) + bsz * I64 + bsz * c.dim * b + cache,
                  write_bytes=bsz * c.vocab * b + new)


LLAMA = {
    'tiny': (LlamaCfg(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn=128, vocab=256), 1, 16, 16),
    'small': (LlamaCfg(dim=128, n_layers=4, n_heads=4, n_kv_heads=2, ffn=352, vocab=512), 1, 64, 128),
    # R39-13: as large as every kernel's arguments allow in the device heap:
    # the attention's (6 heads of 64 x 64 scores in and out, with q, k and v)
    # the down projection's 512 x 192 weight and the head's 640 x 192 one.
    'medium': (LlamaCfg(dim=192, n_layers=4, n_heads=6, n_kv_heads=2, ffn=512, vocab=640), 1, 64, 256),
    # Llama-3.2-1B's shape (its embeddings are tied; here the head is a
    # separate matrix, which changes no FLOP and only the weight bytes).
    'large': (LlamaCfg(dim=2048, n_layers=16, n_heads=32, n_kv_heads=8, ffn=8192, vocab=128256), 1, 512, 512),
}


def model(family, tier, dtype, params, make, demand, *, track=('llm',), tol=None, note=''):
    tag = '-'.join(f'{k}{v}' for k, v in params.items() if not isinstance(v, dict))
    pid = f'model.{family}/{tier}/{dtype}-{tag}'
    return register(Point(pid, 'model', family, tier, dtype, dict(params), make, demand,
                          track=tuple(track), tol=tol, note=note))


MODEL_TOL = {'fp16': 5e-2}

for tier, (c, bsz, s, p) in LLAMA.items():
    cfg = asdict(c)

    def mk_prefill(seed, c=c, bsz=bsz, s=s):
        g = gen(seed)
        m = LlamaPrefill(c, g, 'fp16', s).eval()
        return m, (torch.randint(0, c.vocab, (bsz, s), generator=g),)

    def mk_decode(seed, c=c, bsz=bsz, p=p):
        g = gen(seed)
        m = LlamaDecode(c, g, 'fp16', p).eval()
        shape = (c.n_layers, bsz, c.n_kv_heads, p, c.hd)
        return m, (torch.randint(0, c.vocab, (bsz, 1), generator=g),
                   randn(g, shape, 'fp16'), randn(g, shape, 'fp16'))
    model('llama_prefill', tier, 'fp16', dict(b=bsz, s=s, cfg=cfg), mk_prefill,
          llama_prefill_demand(c, bsz, s), tol=MODEL_TOL['fp16'])
    model('llama_decode', tier, 'fp16', dict(b=bsz, p=p, cfg=cfg), mk_decode,
          llama_decode_demand(c, bsz, p), tol=MODEL_TOL['fp16'])


# ---------------------------------------------------------------------------
# GPT-2 block, MLP
# ---------------------------------------------------------------------------

class GPT2Block(torch.nn.Module):
    def __init__(self, dim, heads, g, dtype):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.ln1 = torch.nn.LayerNorm(dim).to(_dt(dtype))
        self.qkv = _lin(dim, 3 * dim, g, dtype, bias=True)
        self.proj = _lin(dim, dim, g, dtype, bias=True)
        self.ln2 = torch.nn.LayerNorm(dim).to(_dt(dtype))
        self.fc = _lin(dim, 4 * dim, g, dtype, bias=True)
        self.fc2 = _lin(4 * dim, dim, g, dtype, bias=True)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (t.view(b, s, self.heads, d // self.heads).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, s, d)
        x = x + self.proj(o)
        return x + self.fc2(F.gelu(self.fc(self.ln2(x)), approximate='tanh'))


GPT2 = dict(tiny=(64, 4, 1, 16), small=(128, 4, 1, 64), medium=(160, 5, 1, 64), large=(1024, 16, 1, 512))
for tier, (d, h, bsz, s) in GPT2.items():
    t = bsz * s
    tri = s * (s + 1) // 2
    wbytes = 2 * (d * 3 * d + 3 * d + d * d + d + 2 * d * 4 * d + 4 * d + d + 4 * d)

    def mk(seed, d=d, h=h, bsz=bsz, s=s):
        g = gen(seed)
        return GPT2Block(d, h, g, 'fp16').eval(), (randn(g, (bsz, s, d), 'fp16'),)
    model('gpt2_block', tier, 'fp16', dict(d=d, h=h, b=bsz, s=s), mk,
          Demand(macs=t * (3 * d * d + d * d + 8 * d * d) + 2 * bsz * d * tri,
                 sfu_ops=bsz * h * tri + t * 4 * d,
                 read_bytes=wbytes + t * d * 2, write_bytes=t * d * 2),
          track=('llm', 'general'), tol=MODEL_TOL['fp16'])


class MLP(torch.nn.Module):
    def __init__(self, d, g, dtype):
        super().__init__()
        self.fc = _lin(d, 4 * d, g, dtype, bias=True, std=d ** -0.5)
        self.fc2 = _lin(4 * d, d, g, dtype, bias=True, std=(4 * d) ** -0.5)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc(x), approximate='tanh'))


for tier, (t, d) in dict(tiny=(16, 64), small=(64, 128), medium=(128, 128), large=(512, 2048)).items():
    def mk(seed, t=t, d=d):
        g = gen(seed)
        return MLP(d, g, 'fp16').eval(), (randn(g, (t, d), 'fp16'),)
    model('mlp', tier, 'fp16', dict(t=t, d=d), mk,
          Demand(macs=8 * t * d * d, sfu_ops=4 * t * d,
                 read_bytes=2 * (8 * d * d + 5 * d) + t * d * 2, write_bytes=t * d * 2),
          track=('llm', 'general'), tol=MODEL_TOL['fp16'])


# ---------------------------------------------------------------------------
# ResNet basic block (general track)
# ---------------------------------------------------------------------------

class ResNetBlock(torch.nn.Module):
    def __init__(self, ch, g):
        super().__init__()
        self.c1 = torch.nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.c2 = torch.nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = torch.nn.BatchNorm2d(ch)
        self.b2 = torch.nn.BatchNorm2d(ch)
        with torch.no_grad():
            for conv in (self.c1, self.c2):
                conv.weight.copy_(randn(g, conv.weight.shape, 'fp32', (9 * ch) ** -0.5))
            for bn in (self.b1, self.b2):
                bn.running_mean.copy_(randn(g, (ch,), 'fp32', 0.1))
                bn.running_var.copy_(1 + randn(g, (ch,), 'fp32', 0.1).abs())
                bn.weight.copy_(1 + randn(g, (ch,), 'fp32', 0.1))
                bn.bias.copy_(randn(g, (ch,), 'fp32', 0.1))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(y)) + x)


for tier, (nb, ch, hw) in dict(tiny=(1, 8, 8), small=(1, 16, 16), medium=(1, 24, 16), large=(8, 64, 56)).items():
    taps = (3 * hw - 2) ** 2

    def mk(seed, nb=nb, ch=ch, hw=hw):
        g = gen(seed)
        return ResNetBlock(ch, g).eval(), (randn(g, (nb, ch, hw, hw), 'fp32'),)
    model('resnet_block', tier, 'fp32', dict(n=nb, c=ch, hw=hw), mk,
          Demand(macs=2 * nb * ch * ch * taps,
                 read_bytes=4 * (2 * ch * ch * 9 + 8 * ch + nb * ch * hw * hw),
                 write_bytes=4 * nb * ch * hw * hw),
          track=('general',), tol=1e-3)
