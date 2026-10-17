"""The HBM tier (R40-00): working sets that need external memory.

doc/r40_tasks_and_acceptance.md R40-00: the HBM suite is frozen before any
R40 optimisation is measured. It keeps every existing point (tiny, small,
medium and the original large, unshrunk) and adds points whose working set
-- the bytes a call actually touches, not the size of a tensor it indexes --
is at least four times the largest NUCA the architecture defines: 512 KiB a
slice, 8 MiB in all (doc/architecture_overview.md sec 8), so 32 MiB. That
holds for any slice count R40 builds on the U280, and leaves room for
R40-06 to grow the slices without the suite fitting in cache. The points
cover streaming, GEMM, transpose and gather, several of them past the 64 MiB
R40-04 asks a standard graph to reach. The multi-step decoder is a workload
of archbench.generate (spec/r40/suites/hbm.json, `workloads`), not a point:
a point is one call.

Every shape is a property of the workload, not of the board: none is chosen
to fit a heap, a channel or an SLR. Until a TNCA image has external memory
these record `capacity-exceeded`, exactly as `large` does.
"""
import torch

from .ops import B, Add, SiLU, Softmax, RMSNorm, MatMul, Transpose, Embedding, \
    _linear, gen, op, randn
from .registry import Demand

MiB = 1 << 20


class Copy(torch.nn.Module):
    """A tensor copied into a new one: streaming read and write, nothing else."""

    def forward(self, x):
        return x.clone()


# ---- streaming ------------------------------------------------------------
# `large` already streams 2**24 elements (96 MiB for add); these are four
# times that, so no point here repeats one of `large`.
for family, module, n_in, fp, sfu, n in (('add', Add, 2, 1, 0, 1 << 26),
                                         ('silu', SiLU, 1, 2, 1, 1 << 26),
                                         ('copy', Copy, 1, 0, 0, 1 << 26)):
    def mk(seed, module=module, n=n, n_in=n_in):
        g = gen(seed)
        return module(), tuple(randn(g, (n,), 'fp16') for _ in range(n_in))
    op(family, 'hbm', 'fp16', dict(n=n), mk,
       Demand(fp_ops=fp * n, sfu_ops=sfu * n, read_bytes=n_in * n * 2, write_bytes=n * 2),
       note=f'streaming, {(n_in + 1) * n * 2 // MiB} MiB moved')

R, C = 4096, 8192
op('softmax', 'hbm', 'fp16', dict(r=R, c=C),
   lambda seed: (Softmax(), (randn(gen(seed), (R, C), 'fp16', 2.0),)),
   Demand(fp_ops=3 * R * C, sfu_ops=R * C, read_bytes=R * C * 2, write_bytes=R * C * 2),
   note='rows streamed through a reduction, 128 MiB moved')

NR, ND = 8192, 4096


def _mk_rms(seed):
    g = gen(seed)
    return RMSNorm(ND, 'fp16', g), (randn(g, (NR, ND), 'fp16'),)


op('rmsnorm', 'hbm', 'fp16', dict(r=NR, d=ND), _mk_rms,
   Demand(fp_ops=3 * NR * ND, sfu_ops=NR, read_bytes=(NR * ND + ND) * 2, write_bytes=NR * ND * 2),
   note='128 MiB moved')

# ---- GEMM -----------------------------------------------------------------
M = K = N = 4096
op('matmul', 'hbm', 'fp16', dict(m=M, k=K, n=N),
   lambda seed: (MatMul(), (randn(gen(seed), (M, K), 'fp16'), randn(gen(seed + 1), (K, N), 'fp16'))),
   Demand(macs=M * K * N, read_bytes=(M * K + K * N) * 2, write_bytes=M * N * 2),
   track=('llm', 'general'), note='96 MiB of operands')

LM, LK, LN = 512, 4096, 11008


def _mk_linear(seed):
    g = gen(seed)
    return _linear(LK, LN, 'fp16', g, True), (randn(g, (LM, LK), 'fp16'),)


op('linear', 'hbm', 'fp16', dict(m=LM, k=LK, n=LN), _mk_linear,
   Demand(macs=LM * LK * LN, fp_ops=LM * LN, read_bytes=(LM * LK + LK * LN + LN) * 2,
          write_bytes=LM * LN * 2),
   note='an 86 MiB weight')


def _mk_gemv(seed):
    g = gen(seed)
    return _linear(LK, LN, 'fp16', g, False), (randn(g, (1, LK), 'fp16'),)


op('gemv', 'hbm', 'fp16', dict(k=LK, n=LN), _mk_gemv,
   Demand(macs=LK * LN, read_bytes=(LK + LK * LN) * 2, write_bytes=LN * 2),
   note='decode-time projection against an 86 MiB weight: bandwidth bound')

# ---- transpose and gather -------------------------------------------------
TR = 8192
op('transpose', 'hbm', 'fp32', dict(r=TR, c=TR),
   lambda seed: (Transpose(), (randn(gen(seed), (TR, TR), 'fp32'),)),
   Demand(read_bytes=TR * TR * 4, write_bytes=TR * TR * 4), tol=0.0, track=('general',),
   note='512 MiB moved')

EV, ED, ET = 65536, 2048, 8192


def _mk_emb(seed):
    g = gen(seed)
    ids = torch.randperm(EV, generator=g)[:ET]
    return Embedding(EV, ED, 'fp16', g), (ids,)


op('embedding', 'hbm', 'fp16', dict(v=EV, d=ED, t=ET), _mk_emb,
   Demand(read_bytes=ET * 8 + ET * ED * 2, write_bytes=ET * ED * 2), tol=0.0,
   note='8192 random rows of a 256 MiB table: 32 MiB gathered, 32 MiB written')
