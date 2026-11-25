"""Stand-ins for the development points too long for the CA model (decided 2026-11-23).

The R41 development set is the 43 board points, visible to the agent. Eight of them take more than about an
hour each on the CA model (45k device cycles per second of host time, measured 2026-11-13 on the base): the
five largest (4-20 h each) and three hbm points of 1-1.7 h. Each gets a stand-in here: the same operator, module
and dtype at a smaller size whose working set is still far beyond the R41 machine's 96 KiB of home cache, so its
memory behaviour stays in the same regime. A stand-in estimates its point's speedup over the base, never its
cycle count; how well it does is part of the tool's error card.

The first five are the stand-ins of the cache-headroom study (2026-11-13, `~/tritonchip-cache-headroom/cfg/
proxies.py` on Myserver); the last three are new. They register as ArchBench points of tier `membound`, so the
case exporter (skill_cases.py --points-module) and the model runner (run_graphs.py --points-module) take them
like any point. `STANDS_IN_FOR` maps each stand-in to the development point it stands in for.
"""
from dataclasses import asdict

import torch
from archbench.models import (LlamaCfg, LlamaDecode, LlamaPrefill, MODEL_TOL, llama_decode_demand,
                              llama_prefill_demand, model)
from archbench.ops import CausalAttention, Embedding, MatMul, Softmax, Transpose, _linear, gen, op, randn
from archbench.registry import Demand

STANDS_IN_FOR = {}


def _stand_in(point_id: str, for_point: str) -> None:
    STANDS_IN_FOR[point_id] = for_point


# matmul m4096 k4096 n4096 -> m1024 k1024 n1024
m, k, n = 1024, 1024, 1024
op("matmul", "membound", "fp16", dict(m=m, k=k, n=n),
   lambda seed, m=m, k=k, n=n: (MatMul(), (randn(gen(seed), (m, k), "fp16"), randn(gen(seed + 1), (k, n), "fp16"))),
   Demand(macs=m * k * n, read_bytes=(m * k + k * n) * 2, write_bytes=m * n * 2))
_stand_in("op.matmul/membound/fp16-m1024-k1024-n1024", "op.matmul/hbm/fp16-m4096-k4096-n4096")

# linear m512 k4096 n11008 -> m128 k4096 n2752
m, k, n = 128, 4096, 2752


def _mk_linear(seed, m=m, k=k, n=n):
    g = gen(seed)
    return _linear(k, n, "fp16", g, True), (randn(g, (m, k), "fp16"),)


op("linear", "membound", "fp16", dict(m=m, k=k, n=n), _mk_linear,
   Demand(macs=m * k * n, fp_ops=m * n, read_bytes=(m * k + k * n + n) * 2, write_bytes=m * n * 2))
_stand_in("op.linear/membound/fp16-m128-k4096-n2752", "op.linear/hbm/fp16-m512-k4096-n11008")

# attention_prefill h32 s512 d64 -> h4
b, h, s, d = 1, 4, 512, 64
tri = s * (s + 1) // 2
op("attention_prefill", "membound", "fp16", dict(b=b, h=h, s=s, d=d),
   lambda seed, b=b, h=h, s=s, d=d: (CausalAttention(), tuple(randn(gen(seed + i), (b, h, s, d), "fp16")
                                                               for i in range(3))),
   Demand(macs=2 * b * h * d * tri, fp_ops=3 * b * h * tri, sfu_ops=b * h * tri,
          read_bytes=3 * b * h * s * d * 2, write_bytes=b * h * s * d * 2))
_stand_in("op.attention_prefill/membound/fp16-b1-h4-s512-d64", "op.attention_prefill/large/fp16-b1-h32-s512-d64")

# llama_decode large (Llama-3.2-1B shape, p512) -> one layer, vocab 2048
cd = LlamaCfg(dim=2048, n_layers=1, n_heads=32, n_kv_heads=8, ffn=8192, vocab=2048)


def _mk_decode(seed, c=cd, bsz=1, p=512):
    g = gen(seed)
    mdl = LlamaDecode(c, g, "fp16", p).eval()
    shape = (c.n_layers, bsz, c.n_kv_heads, p, c.hd)
    return mdl, (torch.randint(0, c.vocab, (bsz, 1), generator=g), randn(g, shape, "fp16"), randn(g, shape, "fp16"))


model("llama_decode", "membound", "fp16", dict(b=1, p=512, cfg=asdict(cd)), _mk_decode,
      llama_decode_demand(cd, 1, 512), tol=MODEL_TOL["fp16"])
_stand_in("model.llama_decode/membound/fp16-b1-p512", "model.llama_decode/large/fp16-b1-p512")

# llama_prefill hbm (dim 1024, s512) -> one layer, vocab 2048, s128
cp = LlamaCfg(dim=1024, n_layers=1, n_heads=16, n_kv_heads=4, ffn=2816, vocab=2048)


def _mk_prefill(seed, c=cp, bsz=1, s=128):
    g = gen(seed)
    mdl = LlamaPrefill(c, g, "fp16", s).eval()
    return mdl, (torch.randint(0, c.vocab, (bsz, s), generator=g),)


model("llama_prefill", "membound", "fp16", dict(b=1, s=128, cfg=asdict(cp)), _mk_prefill,
      llama_prefill_demand(cp, 1, 128), tol=MODEL_TOL["fp16"])
_stand_in("model.llama_prefill/membound/fp16-b1-s128", "model.llama_prefill/hbm/fp16-b1-s512")

# transpose fp32 8192x8192 (512 MiB moved) -> 4096x4096 (128 MiB moved)
TR = 4096
op("transpose", "membound", "fp32", dict(r=TR, c=TR),
   lambda seed, tr=TR: (Transpose(), (randn(gen(seed), (tr, tr), "fp32"),)),
   Demand(read_bytes=TR * TR * 4, write_bytes=TR * TR * 4), tol=0.0)
_stand_in("op.transpose/membound/fp32-r4096-c4096", "op.transpose/hbm/fp32-r8192-c8192")

# embedding v65536 d2048 t8192 -> the same 256 MiB table, 2048 random rows (8 MiB gathered, 8 MiB written)
EV, ED, ET = 65536, 2048, 2048


def _mk_emb(seed, ev=EV, ed=ED, et=ET):
    g = gen(seed)
    ids = torch.randperm(ev, generator=g)[:et]
    return Embedding(ev, ed, "fp16", g), (ids,)


op("embedding", "membound", "fp16", dict(v=EV, d=ED, t=ET), _mk_emb,
   Demand(read_bytes=ET * 8 + ET * ED * 2, write_bytes=ET * ED * 2), tol=0.0)
_stand_in("op.embedding/membound/fp16-v65536-d2048-t2048", "op.embedding/hbm/fp16-v65536-d2048-t8192")

# softmax fp16 r4096 c8192 -> r1024 c8192 (the row length kept: it sets the reduction)
R, C = 1024, 8192
op("softmax", "membound", "fp16", dict(r=R, c=C),
   lambda seed, r=R, c=C: (Softmax(), (randn(gen(seed), (r, c), "fp16", 2.0),)),
   Demand(fp_ops=3 * R * C, sfu_ops=R * C, read_bytes=R * C * 2, write_bytes=R * C * 2))
_stand_in("op.softmax/membound/fp16-r1024-c8192", "op.softmax/hbm/fp16-r4096-c8192")
