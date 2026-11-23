#!/usr/bin/env python3
"""kfuse_study.py -- accuracy of the K-fused lane on the real GEMM shapes and input distributions (R42-kfuse).

One job = one (format, point, seed): build the tensors, compute the binary64 reference, the sequential contract
(one RNE binary32 FMA per k) and every lane configuration (variant x rounding x G x W), and write one JSON line
per candidate to --out. Aggregation and the gate are in kfuse_report.py.

  python kfuse_study.py --list
  python kfuse_study.py --job fp16/mm-k2048 --seed 0 --out F.jsonl [--quick]

PYTHONPATH must contain the tree that provides `archbench` (origin/R42-xl: ops.py generators, xl.py FP8 helpers).

POINTS (format/name): the tensors are drawn from the registered generators at a REDUCED output size (M x N), at the
point's real K.  Input distributions are the points' own:
  fp16/mm-kK     op.matmul: A = randn(M,K), B = randn(K,N) in fp16                        (archbench/ops.py MatMul)
  fp16/lin-kK    op.linear / op.gemv: x = randn(M,K), W = randn(N,K) * K^-1/2 rounded to fp16 (ops.py _linear)
  fp16/pv-lL     the P.V product of attention: P = softmax(q.k^T / sqrt(64)) in fp16 (q, k ~ N(0,1)), V ~ N(0,1), K = L
  fp16/mlp-fc2-k8192, gpt2-fc2-k4096, llama-w2-k8192   the second projection of model.mlp / model.gpt2_block /
                 model.llama_* (large tier shapes) on the true activations: GELU(tanh) or SwiGLU of the first
                 projection(s), computed in binary64 with fp16 rounding at the stage boundaries; weights as models.py
  bf16/...       the fp16 points' tensors rounded to bfloat16 (no BF16 GEMM point exists in the suite)
  e4m3/xl-kK     the Hub finegrained-fp8 block-wise W8A8 matmul of the `xl` fp8_linear points: E4M3 weights with
                 128x128 fp32 block scales (xl._fp8_weight), activations randn(M,K) fp16 quantised per row and
                 128-wide K block (xl._quantise); the kernel's tl.dot over one K block starts at a zero accumulator,
                 is scaled by the row and block scales in binary32 and added to the running binary32 sum
                 (xl._block_matmul_cpu is the CPU twin); the lane replaces the per-block dot, the rest is unchanged.
  e5m2/xl-kK     the same pipeline with E5M2 codes (scale = amax / 57344): a FORMAT study, no suite point uses E5M2.

Candidates: `seq` (the contract), `seqrev` (control: the same contract in decreasing k) and the lane configurations.

Metrics per candidate (output = the point's output dtype: fp16, or bf16 for the bf16 points; reference = the binary64
result rounded once to that dtype, as the xl points define their reference):
  out_rel_l2   ||out - ref|| / ||ref||        (the registered gate's metric)
  out_exp_rel_l2  the noise-free EXPECTATION of out_rel_l2 over the position of the exact value in the output grid
               (expected_out_err2): the measured out_rel_l2 is one realisation with a few hundred flips (FP16) or a
               few tens (BF16); this has no sampling noise. Added after the first 37 jobs had run (K >= 4096 jobs
               carry it only where re-run, see kfuse_report.py).
  out_tol      max|out - ref| / max(max|ref|, 1)   (the frozen-tolerance criterion of PROTOCOL.md sec 2)
  out_flips    number of outputs that differ from the reference
  acc_rel_l2   the same relative L2 on the binary32 accumulator BEFORE the output rounding (diagnostic)
  acc_bias     mean(acc - exact) / rms(exact)   (signed, diagnostic)
  blk_rel_l2   (block-wise FP8 only) relative L2 error of the per-K-block dot BEFORE the scales: the lane's own error,
               not masked by the scale products and the binary32 block accumulation that every candidate shares
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kfuse_model as km  # noqa: E402

GS = (2, 4, 8, 16)
WS = (0, 3, 8, 24)                # the registered menu
WS_EXPLORE = (-8, -16)            # beyond the menu (windows narrower than a binary32 significand), only where the
                                  # accumulator is not in the window ('acc_after', 'sum_fma'); reported as exploratory


_ONLY = {}                        # set by --only-ws: {'ws': [...], 'variants': [...]}
_FILTER = []                      # set by --cfg-filter: a regular expression the lane tags must match


def lane_configs(quick=False):
    import re
    for c in _lane_configs(quick):
        if not _FILTER or re.search(_FILTER[0], c.tag()):
            yield c


def _lane_configs(quick=False):
    if _ONLY:
        for variant, rnd, G, W in itertools.product(_ONLY['variants'], km.ROUNDINGS, GS, _ONLY['ws']):
            yield km.Cfg(G, W, rnd, variant)
        return
    if quick:
        yield from (km.Cfg(4, 8, 'trunc', 'acc_in'), km.Cfg(16, 0, 'trunc', 'acc_in'))
        return
    for variant, rnd, G, W in itertools.product(km.VARIANTS, km.ROUNDINGS, GS, WS):
        yield km.Cfg(G, W, rnd, variant)
    for variant, rnd, G, W in itertools.product(('acc_after', 'sum_fma'), km.ROUNDINGS, GS, WS_EXPLORE):
        yield km.Cfg(G, W, rnd, variant)


# ---- points --------------------------------------------------------------------------------------

def _plain(fmt, kind, K, M, N, out_dtype, tol, **kw):
    return dict(fmt=fmt, kind=kind, K=K, M=M, N=N, out=out_dtype, tol=tol, **kw)


POINTS = {}
# M x N is REDUCED (131072 outputs per plain point: the output rounding turns a binary32 error into a one-ulp flip
# of a few tenths of a percent of the outputs, so the output-level L2 needs ~10^3 flips to be stable to ~10 %)
for K in (32, 128, 192, 2048, 4096):
    POINTS[f'fp16/mm-k{K}'] = _plain('fp16', 'matmul', K, 512, 256, 'fp16', 2e-2)
for K in (64, 256, 320, 2048, 4096):
    POINTS[f'fp16/lin-k{K}'] = _plain('fp16', 'linear', K, 512, 256, 'fp16', 2e-2)
for L in (512, 2048):
    POINTS[f'fp16/pv-l{L}'] = _plain('fp16', 'pv', L, 512, 256, 'fp16', 2e-2, d=64)
POINTS['fp16/mlp-fc2-k8192'] = _plain('fp16', 'mlp', 8192, 512, 256, 'fp16', 2e-2, d=2048)
POINTS['fp16/gpt2-fc2-k4096'] = _plain('fp16', 'gpt2', 4096, 512, 256, 'fp16', 2e-2, d=1024)
POINTS['fp16/llama-w2-k8192'] = _plain('fp16', 'llama', 8192, 512, 256, 'fp16', 2e-2, d=2048)
for K in (64, 320, 2048, 4096):
    POINTS[f'bf16/lin-k{K}'] = _plain('bf16', 'linear', K, 512, 256, 'bf16', 6e-2)
POINTS['bf16/mm-k2048'] = _plain('bf16', 'matmul', 2048, 512, 256, 'bf16', 6e-2)
POINTS['bf16/pv-l2048'] = _plain('bf16', 'pv', 2048, 512, 256, 'bf16', 6e-2, d=64)
POINTS['bf16/llama-w2-k8192'] = _plain('bf16', 'llama', 8192, 512, 256, 'bf16', 6e-2, d=2048)
for fmt in ('e4m3', 'e5m2'):
    POINTS[f'{fmt}/xl-k2048'] = dict(fmt=fmt, kind='block', K=2048, M=256, N=256, out='fp16', tol=4e-3)
    POINTS[f'{fmt}/xl-k6144'] = dict(fmt=fmt, kind='block', K=6144, M=128, N=256, out='fp16', tol=4e-3)


def _torch():
    import torch
    return torch


def _t64(t):
    return t.to(_torch().float64).numpy()


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _rnd_in(fmt, x):
    """Round float64 values to the element format (fp16 / bf16)."""
    return km.to_fp16_f64(x) if fmt == 'fp16' else km.round_bf16_f64(x)


def build_plain(p, seed):
    torch = _torch()
    from archbench.ops import gen, randn
    K, M, N, fmt = p['K'], p['M'], p['N'], p['fmt']
    kind = p['kind']
    if kind == 'matmul':
        A = _t64(randn(gen(seed), (M, K), 'fp16'))
        B = _t64(randn(gen(seed + 1), (K, N), 'fp16'))
    elif kind == 'linear':
        g = gen(seed)
        W = randn(g, (N, K), 'fp32', K ** -0.5).to(torch.float16)
        A = _t64(randn(g, (M, K), 'fp16'))
        B = _t64(W).T.copy()
    elif kind == 'pv':
        g = gen(seed)
        d = p['d']
        q = _t64(randn(g, (M, d), 'fp16'))
        kk = _t64(randn(g, (K, d), 'fp16'))
        s = (q @ kk.T) / np.sqrt(d)
        s = np.exp(s - s.max(axis=1, keepdims=True))
        A = _rnd_in('fp16', s / s.sum(axis=1, keepdims=True))
        B = _t64(randn(g, (K, N), 'fp16'))
    elif kind in ('mlp', 'gpt2', 'llama'):
        g = gen(seed)
        d, hid = p['d'], K
        if kind == 'mlp':
            w1 = _t64(randn(g, (hid, d), 'fp32', d ** -0.5).to(torch.float16))
            b1 = _t64(randn(g, (hid,), 'fp32', d ** -0.5).to(torch.float16))
            w2 = randn(g, (N, hid), 'fp32', hid ** -0.5).to(torch.float16)
        elif kind == 'gpt2':
            w1 = _t64(randn(g, (hid, d), 'fp32', 0.02).to(torch.float16))
            b1 = _t64(randn(g, (hid,), 'fp32', 0.02).to(torch.float16))
            w2 = randn(g, (N, hid), 'fp32', 0.02).to(torch.float16)
        else:
            w1 = _t64(randn(g, (hid, d), 'fp32', 0.02).to(torch.float16))
            w3 = _t64(randn(g, (hid, d), 'fp32', 0.02).to(torch.float16))
            w2 = randn(g, (N, hid), 'fp32', 0.02).to(torch.float16)
        x = _t64(randn(g, (M, d), 'fp16'))
        if kind == 'llama':
            g1 = km.to_fp16_f64(x @ w1.T)
            g3 = km.to_fp16_f64(x @ w3.T)
            A = km.to_fp16_f64(km.to_fp16_f64(_silu(g1)) * g3)
        else:
            h = km.to_fp16_f64(km.to_fp16_f64(x @ w1.T) + b1)
            A = km.to_fp16_f64(_gelu_tanh(h))
        B = _t64(w2).T.copy()
    else:
        raise ValueError(kind)
    if fmt == 'bf16':
        A, B = km.round_bf16_f64(A), km.round_bf16_f64(B)
    return A, B


def _round_out(acc, out):
    if out == 'fp16':
        return km.to_fp16_f64(acc)
    return km.round_bf16_f64(acc)


def metrics(acc32, acc64, out_dtype, tol):
    out = _round_out(acc32, out_dtype)
    ref = _round_out(acc64, out_dtype)
    return metrics_pair(acc32, acc64, out, ref, tol)


def _ulp(ref, out_dtype):
    """Spacing of the output grid at the exact values (float64 array)."""
    e = np.frexp(np.abs(ref))[1] - 1                    # floor(log2|v|)
    if out_dtype == 'fp16':
        return np.exp2(np.maximum(e, -14) - 10.0)
    return np.exp2(np.maximum(e, -126) - 7.0)


def expected_out_err2(acc32, acc64, ref, out_dtype):
    """Noise-free expectation of sum (out - ref)^2 over the position of the exact value in the output grid.
    With e = acc - exact and u the grid spacing: an error smaller than u flips the rounding with probability |e|/u
    and then costs u, so E = |e| u; a larger one costs e^2 + u^2/6. The measured out_rel_l2 is one realisation of
    this (a few hundred flips in a few hundred thousand outputs: +-10 % for FP16, more for BF16, whose outputs flip
    ten times less often); this is its mean."""
    e = np.abs(acc32.astype(np.float64) - acc64)
    u = _ulp(ref, out_dtype)
    return float(np.where(e <= u, e * u, e * e + u * u / 6.0).sum())


def metrics_pair(acc32, acc64, out, ref, tol, out_dtype=None):
    d = out - ref
    nref = float(np.linalg.norm(ref))
    nacc = float(np.linalg.norm(acc64))
    dacc = acc32.astype(np.float64) - acc64
    exp = {}
    if out_dtype is not None:
        exp['out_exp_rel_l2'] = float(np.sqrt(expected_out_err2(acc32, acc64, ref, out_dtype))) / nref
    return dict(out_rel_l2=float(np.linalg.norm(d)) / nref, **exp,
                out_tol=float(np.abs(d).max()) / max(float(np.abs(ref).max()), 1.0),
                out_flips=int((d != 0).sum()),
                out_max_abs=float(np.abs(d).max()),
                acc_rel_l2=float(np.linalg.norm(dacc)) / nacc,
                acc_bias=float(dacc.mean()) / (nacc / np.sqrt(acc64.size)),
                acc_max_rel=float(np.abs(dacc).max()) / float(np.abs(acc64).max()),
                tol=tol, n_out=int(acc64.size))


# ---- block-wise FP8 ------------------------------------------------------------------------------

E5M2_MAX = 57344.0


def build_block(p, seed):
    """-> dict(qf (M,KB,128) float64 activation codes, wf (N,KB,128) weight codes, a_s (M,KB) f32, sb (N,KB) f32,
    acc64 (M,N) exact pre-rounding, out_ref (M,N) float64 fp16 values)."""
    torch = _torch()
    from archbench import xl
    from archbench.ops import gen
    fmt, K, M, N = p['fmt'], p['K'], p['M'], p['N']
    g = gen(seed)
    kb = K // 128
    if fmt == 'e4m3':
        w, s = xl._fp8_weight(g, N, K)
        x = xl._randn(g, (M, K), 1.0, xl.FP16)
        q, a_s = xl._quantise(x.to(xl.FP16).float())
        out_ref_t = xl._block_matmul_cpu(x, w, s, exact=True)
    else:
        nb = -(-N // 128)
        w = torch.empty((N, K), dtype=torch.float8_e5m2)
        s = torch.empty((nb, kb), dtype=torch.float32)
        std = K ** -0.5
        for i in range(nb):
            r0, r1 = i * 128, min(N, (i + 1) * 128)
            blk = (torch.randn((r1 - r0, K), generator=g) * std).reshape(r1 - r0, kb, 128)
            sc = blk.abs().amax(dim=(0, 2)) / E5M2_MAX
            w[r0:r1] = (blk / sc[None, :, None]).to(torch.float8_e5m2).reshape(r1 - r0, K)
            s[i] = sc
        x = xl._randn(g, (M, K), 1.0, xl.FP16)
        xf = x.to(xl.FP16).float().reshape(M, kb, 128)
        a_s = xf.abs().amax(dim=-1) / E5M2_MAX
        q = (xf / a_s.clamp(min=1e-12).unsqueeze(-1)).to(torch.float8_e5m2)
        out_ref_t = None
    sb = s.repeat_interleave(128, dim=0)[:N]
    wq = w.reshape(N, kb, 128)
    qf, wf = _t64(q), _t64(wq)
    a64, b64 = _t64(a_s), _t64(sb)
    acc64 = np.zeros((M, N))
    for j in range(kb):
        acc64 = acc64 + (qf[:, j, :] @ wf[:, j, :].T) * a64[:, j][:, None] * b64[:, j][None, :]
    if out_ref_t is not None:
        out_ref = _t64(out_ref_t)
        # torch rounds binary64 to fp16 through binary32 (double rounding, about 6e-5 of the outputs): the point's own
        # reference keeps that, and so does this study (the same few outputs differ for every candidate)
        mism = int((km.to_fp16_f64(acc64) != out_ref).sum())
        assert mism <= 3e-4 * out_ref.size, f'own binary64 accumulator differs from xl._block_matmul_cpu(exact=True) in {mism} outputs'
        print(f'  own correctly rounded fp16 reference vs the point\'s torch reference: {mism} of {out_ref.size} outputs differ')
    else:
        out_ref = km.to_fp16_f64(acc64)
    return dict(qf=qf, wf=wf, a_s=a_s.numpy(), sb=sb.numpy(), acc64=acc64, out_ref=out_ref, kb=kb)


def _run(A, B, fmt, cfg):
    """cfg: a lane Cfg, None (sequential, increasing k) or 'rev' (sequential, decreasing k)."""
    if cfg is None:
        return km.seq_gemm(A, B, fmt)
    if cfg == 'rev':
        return km.seq_gemm(np.ascontiguousarray(A[:, ::-1]), np.ascontiguousarray(B[::-1]), fmt)
    return km.lane_gemm(A, B, cfg)


def block_candidate(d, fmt, cfg):
    """Binary32 accumulator of the block-wise product with the per-block dot taken by `cfg` (None: sequential),
    and the dot-level error: blk_rel_l2 = sqrt(sum_blocks ||part - exact||^2 / sum_blocks ||exact||^2) over the
    K-block partial sums BEFORE the scales are applied (what the lane changes; the scale products and the
    binary32 accumulation of the 16 or 48 blocks that follow are common to every candidate)."""
    qf, wf, a_s, sb = d['qf'], d['wf'], d['a_s'], d['sb']
    M, N = qf.shape[0], wf.shape[0]
    acc = np.zeros((M, N), dtype=np.float32)
    e2 = r2 = 0.0
    for j in range(d['kb']):
        A = np.ascontiguousarray(qf[:, j, :])
        B = np.ascontiguousarray(wf[:, j, :].T)
        part = _run(A, B, fmt, cfg).astype(np.float32)
        ex = A @ B
        e2 += float(np.square(part.astype(np.float64) - ex).sum())
        r2 += float(np.square(ex).sum())
        acc = acc + part * a_s[:, j][:, None] * sb[:, j][None, :]
    return acc, dict(blk_rel_l2=float(np.sqrt(e2 / r2)))


# ---- job -----------------------------------------------------------------------------------------

def run_job(job, seed, out_path, quick=False):
    p = POINTS[job]
    fmt = p['fmt']
    t0 = time.time()
    recs = []

    def emit(tag, cfg, m, secs):
        r = dict(job=job, fmt=fmt, kind=p['kind'], K=p['K'], seed=seed, cfg=tag, secs=round(secs, 2), **m)
        if cfg is not None:
            r.update(G=cfg.G, W=cfg.W, rnd=cfg.rnd, variant=cfg.variant)
        recs.append(r)
        with open(out_path, 'a') as f:
            f.write(json.dumps(r) + '\n')

    if p['kind'] == 'block':
        d = build_block(p, seed)
        acc64, out_ref = d['acc64'], d['out_ref']

        def cand(cfg):
            return block_candidate(d, fmt, cfg)      # (acc32, extra metrics)
    else:
        A, B = build_plain(p, seed)
        acc64 = km.exact_gemm(A, B)
        out_ref = _round_out(acc64, p['out'])

        def cand(cfg):
            return _run(A, B, fmt, cfg).astype(np.float32), {}
    print(f'{job} seed {seed}: data built in {time.time() - t0:.1f} s, {acc64.size} outputs, K = {p["K"]}', flush=True)
    t1 = time.time()
    acc, ex = cand(None)
    out = _round_out(acc.astype(np.float64), p['out'])
    emit('seq', None, dict(metrics_pair(acc, acc64, out, out_ref, p['tol'], p['out']), **ex), time.time() - t1)
    # control for the noise of the gate metric: the sequential contract in DECREASING k has the same accuracy
    # (statistically) as the increasing-k contract, so its ratio to `seq` shows what a ratio of 1 looks like
    t1 = time.time()
    acc, ex = cand('rev')
    out = _round_out(acc.astype(np.float64), p['out'])
    emit('seqrev', None, dict(metrics_pair(acc, acc64, out, out_ref, p['tol'], p['out']), **ex), time.time() - t1)
    for cfg in lane_configs(quick):
        t1 = time.time()
        acc, ex = cand(cfg)
        out = _round_out(acc.astype(np.float64), p['out'])
        emit(cfg.tag(), cfg, dict(metrics_pair(acc, acc64, out, out_ref, p['tol'], p['out']), **ex), time.time() - t1)
    print(f'{job} seed {seed}: {len(recs)} candidates in {time.time() - t0:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--job')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only-ws', help='comma-separated W values: run only these windows (variants acc_after, sum_fma), '
                                      'plus seq and seqrev; the fine sweep below the registered menu')
    ap.add_argument('--cfg-filter', help='regular expression: run only the lane configurations whose tag matches '
                                         '(seq and seqrev always run)')
    a = ap.parse_args()
    if a.cfg_filter:
        _FILTER.append(a.cfg_filter)
    if a.only_ws:
        _ONLY.update(ws=[int(x) for x in a.only_ws.split(',')], variants=['acc_after', 'sum_fma'])
    if a.list:
        for k in POINTS:
            print(k)
        return
    run_job(a.job, a.seed, a.out, a.quick)


if __name__ == '__main__':
    main()
