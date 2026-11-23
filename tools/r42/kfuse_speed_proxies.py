"""Reduced-grid proxies of the large and hbm GEMM-shaped points of the 113 (kfuse-speed study, domain: model prediction).

Loaded with `kfuse_speed_tlm_run.py --points-module`. A proxy keeps everything that defines a kernel and its per-tile cost (the
reduction depth K, head dimension, hidden and FFN sizes, dtype, epilogue) and reduces only what adds independent output
tiles (rows, batch, heads, layers, vocabulary) so that the SystemC/TLM model finishes in minutes. They are measurement proxies
of the model study: not registered in any suite, no result outside evidence/r42/kfuse_speed/. Each proxy id carries the
point it stands for in `note`.
"""
import torch

import archbench.ops as O
import archbench.models as M
from archbench.registry import Demand

PROXY_OF = {}


def _reg(pid_hint, stands_for):
    PROXY_OF[pid_hint] = stands_for


def mm(m, k, n, stands_for):
    O.op('matmul', 'large', 'fp16', dict(m=m, k=k, n=n),
         lambda seed, m=m, k=k, n=n: (O.MatMul(), (O.randn(O.gen(seed), (m, k), 'fp16'),
                                                    O.randn(O.gen(seed + 1), (k, n), 'fp16'))),
         Demand(macs=m * k * n, read_bytes=(m * k + k * n) * 2, write_bytes=m * n * 2), note='proxy of ' + stands_for)
    _reg(f'op.matmul/large/fp16-m{m}-k{k}-n{n}', stands_for)


def lin(m, k, n, stands_for):
    def mk(seed, m=m, k=k, n=n):
        g = O.gen(seed)
        return O._linear(k, n, 'fp16', g, True), (O.randn(g, (m, k), 'fp16'),)
    O.op('linear', 'large', 'fp16', dict(m=m, k=k, n=n), mk,
         Demand(macs=m * k * n, fp_ops=m * n, read_bytes=(m * k + k * n + n) * 2, write_bytes=m * n * 2),
         note='proxy of ' + stands_for)
    _reg(f'op.linear/large/fp16-m{m}-k{k}-n{n}', stands_for)


def gemv(k, n, stands_for):
    def mk(seed, k=k, n=n):
        g = O.gen(seed)
        return O._linear(k, n, 'fp16', g, False), (O.randn(g, (1, k), 'fp16'),)
    O.op('gemv', 'large', 'fp16', dict(k=k, n=n), mk,
         Demand(macs=k * n, read_bytes=(k + k * n) * 2, write_bytes=n * 2), note='proxy of ' + stands_for)
    _reg(f'op.gemv/large/fp16-k{k}-n{n}', stands_for)


def bmm(bt, m, k, n, stands_for):
    O.op('bmm', 'large', 'fp16', dict(b=bt, m=m, k=k, n=n),
         lambda seed, bt=bt, m=m, k=k, n=n: (O.BMM(), (O.randn(O.gen(seed), (bt, m, k), 'fp16'),
                                                       O.randn(O.gen(seed + 1), (bt, k, n), 'fp16'))),
         Demand(macs=bt * m * k * n, read_bytes=bt * (m * k + k * n) * 2, write_bytes=bt * m * n * 2),
         note='proxy of ' + stands_for)
    _reg(f'op.bmm/large/fp16-b{bt}-m{m}-k{k}-n{n}', stands_for)


def mlp(t, d, stands_for):
    def mk(seed, t=t, d=d):
        g = O.gen(seed)
        return M.MLP(d, g, 'fp16').eval(), (O.randn(g, (t, d), 'fp16'),)
    M.model('mlp', 'large', 'fp16', dict(t=t, d=d), mk,
            Demand(macs=8 * t * d * d, sfu_ops=4 * t * d, read_bytes=2 * (8 * d * d + 5 * d) + t * d * 2,
                   write_bytes=t * d * 2), track=('llm', 'general'), tol=M.MODEL_TOL['fp16'], note='proxy of ' + stands_for)
    _reg(f'model.mlp/large/fp16-t{t}-d{d}', stands_for)


def gpt2(d, h, s, stands_for):
    def mk(seed, d=d, h=h, s=s):
        g = O.gen(seed)
        return M.GPT2Block(d, h, g, 'fp16').eval(), (O.randn(g, (1, s, d), 'fp16'),)
    t = s
    tri = s * (s + 1) // 2
    M.model('gpt2_block', 'large', 'fp16', dict(d=d, h=h, b=1, s=s), mk,
            Demand(macs=t * 12 * d * d + 2 * d * tri, sfu_ops=h * tri + t * 4 * d,
                   read_bytes=2 * 12 * d * d, write_bytes=t * d * 2), track=('llm', 'general'), tol=M.MODEL_TOL['fp16'],
            note='proxy of ' + stands_for)
    _reg(f'model.gpt2_block/large/fp16-d{d}-h{h}-b1-s{s}', stands_for)


def llama(kind, cfg, s_or_p, stands_for):
    c = M.LlamaCfg(**cfg)
    if kind == 'prefill':
        def mk(seed, c=c, s=s_or_p):
            g = O.gen(seed)
            return M.LlamaPrefill(c, g, 'fp16', s).eval(), (torch.randint(0, c.vocab, (1, s), generator=g),)
        M.model('llama_prefill', 'large', 'fp16', dict(b=1, s=s_or_p, cfg=cfg), mk, M.llama_prefill_demand(c, 1, s_or_p),
                tol=M.MODEL_TOL['fp16'], note='proxy of ' + stands_for)
        _reg(f'model.llama_prefill/large/fp16-b1-s{s_or_p}', stands_for)
    else:
        def mk(seed, c=c, p=s_or_p):
            g = O.gen(seed)
            m = M.LlamaDecode(c, g, 'fp16', p).eval()
            shape = (c.n_layers, 1, c.n_kv_heads, p, c.hd)
            return m, (torch.randint(0, c.vocab, (1, 1), generator=g), O.randn(g, shape, 'fp16'), O.randn(g, shape, 'fp16'))
        M.model('llama_decode', 'large', 'fp16', dict(b=1, p=s_or_p, cfg=cfg), mk, M.llama_decode_demand(c, 1, s_or_p),
                tol=M.MODEL_TOL['fp16'], note='proxy of ' + stands_for)
        _reg(f'model.llama_decode/large/fp16-b1-p{s_or_p}', stands_for)


# large tier: rows reduced, K and N kept (the weight matrix keeps its size and its reuse pattern); gemv, bmm and the
# attention points of the large tier are run directly, not as proxies
mm(48, 2048, 2048, 'op.matmul/large/fp16-m2048-k2048-n2048')
mm(48, 4096, 4096, 'op.matmul/hbm/fp16-m4096-k4096-n4096')
lin(48, 2048, 8192, 'op.linear/large/fp16-m512-k2048-n8192')
lin(48, 4096, 11008, 'op.linear/hbm/fp16-m512-k4096-n11008')
mlp(48, 2048, 'model.mlp/large/fp16-t512-d2048')
gpt2(1024, 16, 64, 'model.gpt2_block/large/fp16-d1024-h16-b1-s512')
# llama: same dimensions, one layer, a short sequence and a small vocabulary
llama('prefill', dict(dim=2048, n_layers=1, n_heads=32, n_kv_heads=8, ffn=8192, vocab=2048), 64,
      'model.llama_prefill/large/fp16-b1-s512')
llama('prefill', dict(dim=1024, n_layers=1, n_heads=16, n_kv_heads=4, ffn=2816, vocab=2048), 48,
      'model.llama_prefill/hbm/fp16-b1-s512')
llama('decode', dict(dim=2048, n_layers=1, n_heads=32, n_kv_heads=8, ffn=8192, vocab=2048), 480,
      'model.llama_decode/large/fp16-b1-p512')
llama('decode', dict(dim=1024, n_layers=1, n_heads=16, n_kv_heads=4, ffn=2816, vocab=2048), 496,
      'model.llama_decode/hbm/fp16-b1-p512')
