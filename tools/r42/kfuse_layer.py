#!/usr/bin/env python3
"""kfuse_layer.py -- the xl whole-layer points with their matrix products replaced by the emulated lane (R42-kfuse).

  python kfuse_layer.py --point model.fp8_layer_decode/xl/fp8-b1-p2048-layers1 --seed 0 \
         --cfgs seq,seqrev,sum_fma.G4.W24.trunc,acc_in.G16.W8.trunc --attn seq --out F.jsonl

The module is the point's own (`archbench.registry.get(id).make(seed)`, the code of origin/R42-xl archbench/xl.py,
unchanged); only two functions it calls are replaced for the duration of a candidate run:
  xl.linear_fp8                         the Hub block-wise W8A8 kernel's CPU twin -> this file's `emu_linear`: the
                                        activation quantisation is xl._quantise (the kernel's), every K block of 128
                                        is one dot product from a zero accumulator taken by the ENGINE (sequential or
                                        a lane configuration), then scaled by the row and block scales and added to the
                                        binary32 running sum, exactly as the kernel (and xl._block_matmul_cpu) do;
  torch.nn.functional.scaled_dot_product_attention
                                        -> `emu_sdpa`: scores = q k^T through the attention ENGINE (fp16 operands, K = 128,
                                        binary32 accumulate), times the scale, rounded to fp16; causal mask; softmax in
                                        binary32; probabilities rounded to fp16; P V through the engine (K = the cache
                                        length); rounded to fp16. The same fp16 stage boundaries for every candidate.
Everything else (RMSNorm, RoPE, SwiGLU, residuals) is the module's own torch CPU code in fp16.

The reference is the point's own `reference(module, inputs)` (xl.xl_reference: binary64, exact block-wise FP8
arithmetic, fp16 stage boundaries), the metric is archbench.platforms.common.compare (max|got - ref| / max(max|ref|, 1)
over all outputs) against the point's frozen tolerance, plus relative L2 per output tensor and combined.

ENGINE spec: `seq` (ISA sec 9C.1, increasing k), `seqrev` (control: decreasing k), or a lane tag
`variant.G<g>.W<w>.<rnd>` (kfuse_model.Cfg.tag()). The projections use --cfgs; the attention matmuls use --attn
(default `seq`: the attention is not what the FP8 lane is for).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kfuse_model as km  # noqa: E402


def parse_cfg(tag):
    variant, g, w, rnd = tag.split('.')
    return km.Cfg(int(g[1:]), int(w[1:]), rnd, variant)


def make_engine(spec, fmt):
    if spec == 'seq':
        return lambda A, B: km.seq_gemm(A, B, fmt)
    if spec == 'seqrev':
        return lambda A, B: km.seq_gemm(np.ascontiguousarray(A[:, ::-1]), np.ascontiguousarray(B[::-1]), fmt)
    cfg = parse_cfg(spec)
    return lambda A, B: km.lane_gemm(A, B, cfg)


def emu_linear_factory(engine, xl, torch):
    cache = {}

    def emu_linear(x, w, s):
        m = xl._MODE[0]
        if m != 'emu':
            return _orig_linear(x, w, s)
        K = x.shape[-1]
        N = w.shape[0]
        kb = K // 128
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K).to(xl.FP16).float()
        q, a_s = xl._quantise(x2)
        qf = q.to(torch.float64).numpy()
        key = id(w)
        if key not in cache:
            sb = s.repeat_interleave(128, dim=0)[:N]
            cache[key] = (w.reshape(N, kb, 128).to(torch.float64).numpy(), sb.numpy())
        wf, sb = cache[key]
        a_s = a_s.numpy()
        M = x2.shape[0]
        acc = np.zeros((M, N), dtype=np.float32)
        for j in range(kb):
            A = np.ascontiguousarray(qf[:, j, :])
            B = np.ascontiguousarray(wf[:, j, :].T)
            part = engine(A, B).astype(np.float32)
            acc = acc + part * a_s[:, j][:, None] * sb[:, j][None, :]
        return torch.from_numpy(acc).to(xl.FP16).reshape(*lead, N)

    _orig_linear = xl.linear_fp8
    return emu_linear


def emu_sdpa_factory(engine, torch, F_orig):
    def emu_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if q.dtype == torch.float64:
            return F_orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        assert attn_mask is None and dropout_p == 0.0 and not kw
        b, nq, s, hd = q.shape
        kv = k.shape[2]
        sc = (1.0 / np.sqrt(hd)) if scale is None else scale
        out = torch.empty((b, nq, s, hd), dtype=torch.float16)
        mask = None
        if is_causal:
            mask = torch.ones((s, kv), dtype=torch.bool).tril(diagonal=kv - s)
        for bi in range(b):
            for h in range(nq):
                Q = q[bi, h].to(torch.float64).numpy()
                Kt = np.ascontiguousarray(k[bi, h].to(torch.float64).numpy().T)
                V = v[bi, h].to(torch.float64).numpy()
                S = (engine(Q, Kt).astype(np.float32) * np.float32(sc)).astype(np.float16)
                St = torch.from_numpy(S.astype(np.float32))
                if mask is not None:
                    St = St.masked_fill(~mask, float('-inf'))
                P = torch.softmax(St, dim=-1).to(torch.float16).to(torch.float64).numpy()
                O = engine(P, V).astype(np.float32).astype(np.float16)
                out[bi, h] = torch.from_numpy(O)
        return out
    return emu_sdpa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--point', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cfgs', required=True, help='comma-separated engine specs for the FP8 projections')
    ap.add_argument('--attn', default='seq', help='engine spec of the attention matmuls')
    ap.add_argument('--out', required=True)
    ap.add_argument('--refdir', default=None)
    a = ap.parse_args()

    import torch
    from archbench import xl, registry
    from archbench.platforms.common import compare, flatten
    xl._bind_kernels = lambda: None                        # no Hub/Triton kernels on this host: the CPU path only
    torch.set_num_threads(1)
    P = registry.get(a.point)
    t0 = time.time()
    module, inputs = P.make(a.seed)
    module = module.eval()
    refdir = a.refdir or os.path.dirname(os.path.abspath(a.out))
    os.makedirs(refdir, exist_ok=True)
    rf = os.path.join(refdir, f'ref_{a.point.replace("/", "_")}_s{a.seed}.pt')
    if os.path.exists(rf):
        ref = torch.load(rf)
    else:
        with torch.no_grad():
            ref = tuple(t.detach().cpu() for t in flatten(P.reference(module, inputs)))
        torch.save(ref, rf + f'.tmp{os.getpid()}')
        os.replace(rf + f'.tmp{os.getpid()}', rf)
    print(f'{a.point} seed {a.seed}: module and binary64 reference ready in {time.time() - t0:.0f} s', flush=True)
    F_orig = torch.nn.functional.scaled_dot_product_attention
    attn_engine = make_engine(a.attn, 'fp16')
    for spec in a.cfgs.split(','):
        t1 = time.time()
        emu_lin = emu_linear_factory(make_engine(spec, 'e4m3'), xl, torch)
        orig_lin = xl.linear_fp8
        xl.linear_fp8 = emu_lin
        torch.nn.functional.scaled_dot_product_attention = emu_sdpa_factory(attn_engine, torch, F_orig)
        try:
            with xl.mode('emu'), torch.no_grad():
                got = tuple(t.detach().cpu() for t in flatten(module(*inputs)))
        finally:
            xl.linear_fp8 = orig_lin
            torch.nn.functional.scaled_dot_product_attention = F_orig
        cmp = compare(got, ref, P.tolerance)
        per = []
        d2 = r2 = 0.0
        for g, r in zip(got, ref):
            d = g.to(torch.float64) - r
            per.append(dict(rel_l2=float(d.norm() / r.norm()), max_abs=float(d.abs().max()),
                            tol_metric=float(d.abs().max() / max(float(r.abs().max()), 1.0)), n=int(r.numel())))
            d2 += float(d.pow(2).sum())
            r2 += float(r.pow(2).sum())
        rec = dict(point=a.point, seed=a.seed, proj=spec, attn=a.attn, tol=P.tolerance, max_rel_err=cmp['max_rel_err'],
                   ok=bool(cmp['ok']), rel_l2=float(np.sqrt(d2 / r2)), per_output=per, secs=round(time.time() - t1, 1))
        with open(a.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f'  {spec:26s} attn={a.attn}: tol-metric {cmp["max_rel_err"]:.3g} (tol {P.tolerance}) rel_l2 {rec["rel_l2"]:.3g} '
              f'[{rec["secs"]} s]', flush=True)


if __name__ == '__main__':
    main()
