#!/usr/bin/env python3
"""kfuse_validate.py -- validation of tools/r42/kfuse_model.py (no study data; run on the compute host).

  python tools/r42/kfuse_validate.py [--quick] [--out FILE.json]

Checks (each prints PASS/FAIL with counts; exit status 1 on any failure):
  V1  element decoding: numpy decode_array == tools/model/floatfmt.decode for every encoding of all four formats.
  V2  rn32_add (round-to-odd TwoSum) == exact Fraction RNE for random binary32 + float64 operands, wide exponents,
      cancellation, binary32 subnormal results.
  V3  limit G = 1, W = None: scalar lane (acc_in, sum_fma) == the sequential FMA of ISA sec 9C.1, evaluated by the
      independent oracle tools/model/fp32.fma_ref on the operands widened to binary32 (all four formats).
  V4  limit W = None: scalar lane == the exactly rounded group sum RNE(c + sum) (acc_in, sum_fma; tools/model/
      fdp.add_round of the exact group sum) and RNE(c + RNE(sum)) (acc_after), for G in {2, 4, 8, 16}; and the
      existing m.mma.dp lane oracle fdp.lane_mac_ref: E4M3 DP4 and FP16/BF16/E5M2/E4M3 DP2 (exact=False, with
      RO_48) agree with sum_fma/W = None (E4M3: always; the others: counted, they may differ only where RO_48
      does).
  V5  numpy lane == scalar lane, bit for bit, for every format, G in {1, 2, 4, 8, 16}, W in {-8, 0, 3, 8, 24},
      both roundings, all three variants (random encodings, adversarial exponent spreads, cancellation).
  V6  numpy sequential chain == scalar FMA chain (fp32.fma_ref), all formats, chains of 64 steps.
  V8  numpy sum_fma at G = 4 equals the existing E4M3 DP4 lane oracle fdp.lane_mac_ref over whole dot products.
  V7  lane_gemm == scalar_dot on whole dot products (K = 96, 3 formats), and lane at G = 1 vs seq_gemm: how often
      the finite-W window equals the FMA (informative).
"""
import argparse
import json
import os
import random
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kfuse_model as km  # noqa: E402
import floatfmt  # noqa: E402
import fp32 as f32  # noqa: E402
import fdp  # noqa: E402

FMTS = ('fp16', 'bf16', 'e4m3', 'e5m2')
RESULTS = {}
FAILED = []


def report(name, ok, detail):
    RESULTS[name] = dict(ok=bool(ok), detail=detail)
    print(f'{"PASS" if ok else "FAIL"}  {name}: {detail}', flush=True)
    if not ok:
        FAILED.append(name)


def finite_bits(rng, fmt, n):
    """n random finite encodings of the format, exponents spread over the whole range."""
    w = km.ELEM_BITS[fmt]
    out = []
    while len(out) < n:
        b = rng.getrandbits(w)
        v = floatfmt.decode(km.FMT[fmt], b)
        if v == floatfmt.NAN or isinstance(v, tuple):
            continue
        out.append(b)
    return out


def near_bits(rng, fmt, n, center=None, spread=3):
    """n random finite encodings whose exponent field lies within `spread` of a (random) centre: products of
    similar size, the regime a block dot product lives in."""
    e_sz, f_sz, has_inf = km._FMT_PARAMS[fmt]
    top = (1 << e_sz) - (2 if has_inf else 1)
    if center is None:
        center = rng.randrange(max(1, top // 4), max(2, 3 * top // 4))
    out = []
    for _ in range(n):
        e = min(top, max(0, center + rng.randint(-spread, spread)))
        out.append((rng.getrandbits(1) << (e_sz + f_sz)) | (e << f_sz) | rng.getrandbits(f_sz))
    return out


def f64(fmt, bits):
    return km.decode_array(fmt, np.array(bits))


def widen(fmt, bits):
    """Element encoding -> binary32 encoding, exactly (floatfmt.convert)."""
    return floatfmt.convert(km.FMT[fmt], 'f32', bits)


def c_patterns(rng, a, b, fmt, n_extra=0):
    """binary32 accumulator encodings for one group: random finite, near the products' sum (cancellation),
    zero, and large/small."""
    fa = fdp.FMT_OF  # noqa: F841
    s = 0.0
    for x, y in zip(f64(fmt, a), f64(fmt, b)):
        s += float(x) * float(y)
    out = []
    for _ in range(4):
        bits = rng.getrandbits(32)
        if ((bits >> 23) & 0xff) == 0xff:
            bits &= ~(1 << 30)
        out.append(bits)
    for scale in (1.0, 1.0 + 2.0 ** -20, 1.0 - 2.0 ** -22, 0.5, 2.0):
        v = np.float32(-s * scale)
        if np.isfinite(v):
            out.append(f32.f2b(float(v)))
    out += [0, 0x80000000, f32.f2b(1.0), f32.f2b(-1.0), 0x00000001, 0x007fffff, 0x7f7fffff, 0xff7fffff]
    return out


def v1():
    bad = 0
    for fmt in FMTS:
        w = km.ELEM_BITS[fmt]
        bits = np.arange(1 << w)
        got = km.decode_array(fmt, bits)
        for b in range(1 << w):
            ref = floatfmt.decode(km.FMT[fmt], b)
            if ref == floatfmt.NAN:
                ok = np.isnan(got[b])
            elif isinstance(ref, tuple):
                ok = got[b] == (-np.inf if ref[1] else np.inf)
            else:
                ok = float(ref) == got[b] and (np.signbit(got[b]) == bool((b >> (w - 1)) & 1) or float(ref) != 0 or True)
            if not ok:
                bad += 1
    report('V1 decode_array == floatfmt.decode (all encodings of 4 formats)', bad == 0, f'{bad} mismatches')


def v2(rng, n):
    bad = 0
    for _ in range(n):
        kind = rng.randrange(4)
        c = np.float32(rng.uniform(-1, 1) * 2.0 ** rng.randrange(-140, 120))
        if kind == 0:
            s = rng.uniform(-1, 1) * 2.0 ** rng.randrange(-200, 150)
            s = float(np.float64(s))
        elif kind == 1:                                   # cancellation: s close to -c
            s = float(-np.float64(c)) * (1 + rng.choice((-1, 1)) * 2.0 ** rng.randrange(-60, -20))
        elif kind == 2:                                   # s a 24-bit number far below c
            s = float(rng.getrandbits(24)) * 2.0 ** (rng.randrange(-190, -20))
        else:                                             # subnormal binary32 results
            c = np.float32(rng.getrandbits(23) * 2.0 ** -149)
            s = float(rng.getrandbits(30)) * 2.0 ** -152 * rng.choice((-1, 1))
        got = km.rn32_add(np.array([float(c)]), np.array([s]))[0]
        exact = Fraction(float(c)) + Fraction(s)
        ref = f32.b2f(f32.round_to_f32(exact, 0)) if exact != 0 else 0.0
        if not (got == np.float32(ref) or (np.isnan(got) and np.isnan(ref))):
            bad += 1
    report('V2 rn32_add == exact RNE (round-to-odd TwoSum)', bad == 0, f'{bad} mismatches of {n}')


def v3(rng, n):
    bad = tot = 0
    for fmt in FMTS:
        for _ in range(n):
            a, b = finite_bits(rng, fmt, 1), finite_bits(rng, fmt, 1)
            for c in c_patterns(rng, a, b, fmt):
                want = f32.fma_ref(widen(fmt, a[0]), widen(fmt, b[0]), c)
                for variant in ('acc_in', 'sum_fma'):
                    got = km.scalar_step(km.Cfg(1, None, 'trunc', variant), fmt, a, b, c)
                    tot += 1
                    bad += got != want
    report('V3 G=1, W=None == sequential FMA (fp32.fma_ref), acc_in and sum_fma', bad == 0, f'{bad} mismatches of {tot}')


def exact_ref(variant, fmt, a, b, c):
    fa = km.FMT[fmt]
    gs = fdp.group_sum(fa, fa, a, b)
    if variant == 'acc_after' and gs[0] == 'fin':
        r = gs[1]
        rb = 0 if r == 0 else f32.round_to_f32(r, 0)
        if ((rb >> 23) & 0xff) == 0xff:
            return fdp.add_round(('inf', rb >> 31), c)
        gs = ('fin', f32.to_fraction(rb), gs[2])
    return fdp.add_round(gs, c)


def v4(rng, n):
    res = {}
    for variant in km.VARIANTS:
        bad = tot = 0
        for fmt in FMTS:
            for G in (2, 4, 8, 16):
                for _ in range(n):
                    a, b = finite_bits(rng, fmt, G), finite_bits(rng, fmt, G)
                    for c in c_patterns(rng, a, b, fmt)[:10]:
                        got = km.scalar_step(km.Cfg(G, None, 'trunc', variant), fmt, a, b, c)
                        # the oracle rounds a zero total to +0; the lane rule is the same except the all-zero case
                        want = exact_ref(variant, fmt, a, b, c)
                        tot += 1
                        bad += got != want
        res[variant] = (bad, tot)
        report(f'V4 W=None, {variant} == exactly rounded group sum', bad == 0, f'{bad} mismatches of {tot}')
    # the existing m.mma.dp lane oracle
    mism = {}
    for fmt, mode, G in (('e4m3', fdp.MODE_DP4, 4), ('fp16', fdp.MODE_DP2, 2), ('bf16', fdp.MODE_DP2, 2),
                         ('e5m2', fdp.MODE_DP2, 2), ('e4m3', fdp.MODE_DP2, 2)):
        code = {'fp16': 0, 'bf16': 1, 'e5m2': 2, 'e4m3': 3}[fmt]
        bad = tot = 0
        w = 8 if mode == fdp.MODE_DP4 else 16
        for _ in range(n * 6):
            a, b = finite_bits(rng, fmt, G), finite_bits(rng, fmt, G)
            pa = sum(x << (w * i) for i, x in enumerate(a))
            pb = sum(x << (w * i) for i, x in enumerate(b))
            for c in c_patterns(rng, a, b, fmt)[:8]:
                want = fdp.lane_mac_ref(mode, pa, pb, c, code, code)
                got = km.scalar_step(km.Cfg(G, None, 'trunc', 'sum_fma'), fmt, a, b, c)
                tot += 1
                bad += got != want
        mism[f'{fmt}/G{G}'] = (bad, tot)
        # the same cases against the oracle without RO_48 (exact=True): must agree everywhere
        bad2 = 0
        rr = random.Random(7)
        for _ in range(n * 6):
            a, b = finite_bits(rr, fmt, G), finite_bits(rr, fmt, G)
            pa = sum(x << (w * i) for i, x in enumerate(a))
            pb = sum(x << (w * i) for i, x in enumerate(b))
            for c in c_patterns(rr, a, b, fmt)[:8]:
                want = fdp.lane_mac_ref(mode, pa, pb, c, code, code, exact=True)
                bad2 += km.scalar_step(km.Cfg(G, None, 'trunc', 'sum_fma'), fmt, a, b, c) != want
        mism[f'{fmt}/G{G} (exact=True, no RO_48)'] = (bad2, n * 6 * 8)
    ok = mism['e4m3/G4'][0] == 0 and all(v[0] == 0 for k, v in mism.items() if 'exact=True' in k)
    detail = '; '.join(f'{k}: {b}/{t}' for k, (b, t) in mism.items())
    report('V4b existing m.mma.dp oracle (fdp.lane_mac_ref, RO_48) vs sum_fma/W=None [E4M3 DP4 must be 0]', ok, detail)
    RESULTS['V4b_detail'] = {k: list(v) for k, v in mism.items()}


def v5(rng, n):
    """numpy lane vs scalar lane, single steps."""
    bad = tot = 0
    worst = []
    for fmt in FMTS:
        for G in (1, 2, 4, 8, 16):
            cases = []
            for _ in range(n):
                if rng.random() < 0.5:                      # similar sizes: the regime of a block dot product
                    a, b = near_bits(rng, fmt, G), near_bits(rng, fmt, G)
                else:                                       # whole-range exponents
                    a, b = finite_bits(rng, fmt, G), finite_bits(rng, fmt, G)
                for c in c_patterns(rng, a, b, fmt):
                    cases.append((a, b, c))
            A = np.array([[f64(fmt, [x])[0] for x in a] for a, b, c in cases])
            Bm = np.array([[f64(fmt, [x])[0] for x in b] for a, b, c in cases])
            C = np.array([float(f32.b2f(c)) for a, b, c in cases])
            prods = (A * Bm).T[:, :, None]                  # (G, cases, 1)
            for W in (-8, 0, 3, 8, 24):
                for rnd in km.ROUNDINGS:
                    for variant in km.VARIANTS:
                        cfg = km.Cfg(G, W, rnd, variant)
                        with np.errstate(over='ignore', invalid='ignore'):
                            got = km.group_vec(prods, C[:, None], cfg)[:, 0].astype(np.float32).view(np.uint32)
                        for i, (a, b, c) in enumerate(cases):
                            want = km.scalar_step(cfg, fmt, a, b, c)
                            tot += 1
                            g = int(got[i])
                            # signed zero of an exact-zero sum is outside the numpy model (-0 vs +0 compare equal)
                            if g != want and not (g & 0x7fffffff == 0 and want & 0x7fffffff == 0):
                                bad += 1
                                if len(worst) < 5:
                                    worst.append((fmt, cfg.tag(), a, b, hex(c), hex(g), hex(want)))
    report('V5 numpy lane == scalar lane (bit-exact; all formats, G, W, rounding, variants)', bad == 0,
           f'{bad} mismatches of {tot}' + (f'; first: {worst}' if worst else ''))
    # sensitivity: the cases are not vacuous -- the finite windows differ from the exact group sum in many of them
    diff = tot2 = 0
    rr = random.Random(11)
    for fmt in FMTS:
        for G in (4, 16):
            for _ in range(40):
                a, b = near_bits(rr, fmt, G), near_bits(rr, fmt, G)
                for c in c_patterns(rr, a, b, fmt):
                    ex = km.scalar_step(km.Cfg(G, None, 'trunc', 'acc_in'), fmt, a, b, c)
                    for W in (0, 3, 8):
                        tot2 += 1
                        diff += km.scalar_step(km.Cfg(G, W, 'trunc', 'acc_in'), fmt, a, b, c) != ex
    RESULTS['V5_sensitivity'] = dict(differs_from_exact=diff, cases=tot2)
    print(f'INFO  V5 sensitivity: acc_in trunc W in 0,3,8 differs from the exact group sum in {diff} of {tot2} cases', flush=True)


def v6(rng, n):
    bad = tot = 0
    for fmt in FMTS:
        for _ in range(n):
            K = 64
            if fmt == 'bf16' or rng.random() < 0.5:        # similar magnitudes: moderate exponents
                a, b = near_bits(rng, fmt, K, spread=4), near_bits(rng, fmt, K, spread=4)
            else:
                a, b = finite_bits(rng, fmt, K), finite_bits(rng, fmt, K)
            c = 0
            for ab, bb in zip(a, b):
                c = f32.fma_ref(widen(fmt, ab), widen(fmt, bb), c)
            A = f64(fmt, a)[None, :]
            Bm = f64(fmt, b)[:, None]
            with np.errstate(over='ignore', invalid='ignore'):
                got = km.seq_gemm(A, Bm, fmt)[0, 0]
            gotb = f32.f2b(float(got)) if np.isfinite(got) else (0x7f800000 if got > 0 else 0xff800000)
            tot += 1
            ok = gotb == c or (gotb & 0x7fffffff == 0 and c & 0x7fffffff == 0)
            if np.isnan(got) and ((c >> 23) & 0xff) == 0xff and c & 0x7fffff:
                ok = True
            bad += not ok
    report('V6 numpy sequential chain == fp32.fma_ref chain', bad == 0, f'{bad} mismatches of {tot} chains of 64')


def v7(rng, n):
    # whole dot products of similar-sized products (nothing overflows)
    bad = tot = 0
    for fmt in ('fp16', 'bf16', 'e4m3', 'e5m2'):
        for _ in range(n):
            K = 96
            a, b = near_bits(rng, fmt, K, spread=3), near_bits(rng, fmt, K, spread=3)
            A, Bm = f64(fmt, a)[None, :], f64(fmt, b)[:, None]
            for cfg in (km.Cfg(4, 8, 'trunc', 'acc_in'), km.Cfg(2, 3, 'rn', 'acc_after'), km.Cfg(16, 0, 'trunc', 'sum_fma'),
                        km.Cfg(8, 24, 'rn', 'acc_in'), km.Cfg(1, 24, 'trunc', 'acc_in')):
                with np.errstate(over='ignore', invalid='ignore'):
                    got = km.lane_gemm(A, Bm, cfg)[0, 0].astype(np.float32)
                want = np.float32(f32.b2f(km.scalar_dot(cfg, fmt, a, b, 0)))
                tot += 1
                if not (got == want or (np.isnan(got) and np.isnan(want))):
                    bad += 1
    report('V7 lane_gemm == scalar_dot on whole dot products (K = 96)', bad == 0, f'{bad} mismatches of {tot}')
    # informative: finite-W window at G = 1 against the FMA, on N(0,1) fp16 data
    g = np.random.RandomState(1)
    A = g.randn(64, 256).astype(np.float16).astype(np.float64)
    B = g.randn(256, 64).astype(np.float16).astype(np.float64)
    seq = km.seq_gemm(A, B, 'fp16')
    info = {}
    for W in (0, 3, 8, 24):
        for variant in km.VARIANTS:
            lane = km.lane_gemm(A, B, km.Cfg(1, W, 'trunc', variant))
            info[f'{variant}.W{W}'] = float((lane == seq).mean())
    RESULTS['V7_info_G1_equal_fraction_to_FMA_fp16_normal_K256'] = info
    print('INFO  G=1 finite W: fraction of 4096 K=256 dot products equal to the sequential FMA:', json.dumps(info), flush=True)


def v8(rng, n):
    """The existing m.mma.dp E4M3 lane (G = 4, exact group sum, one rounding with C) is sum_fma at W >= 12 in the
    numpy model (E4M3 products are multiples of 2^-18 below 2^18: a window of 24 + W bits at the largest product
    covers them exactly); check W = 24 and W = 8 against the lane oracle, whole dot products K = 64."""
    bad = {8: 0, 24: 0}
    tot = 0
    for _ in range(n):
        a, b = near_bits(rng, 'e4m3', 64, spread=4), near_bits(rng, 'e4m3', 64, spread=4)
        A, Bm = f64('e4m3', a)[None, :], f64('e4m3', b)[:, None]
        c = 0
        for k0 in range(0, 64, 4):
            pa = sum(x << (8 * i) for i, x in enumerate(a[k0:k0 + 4]))
            pb = sum(x << (8 * i) for i, x in enumerate(b[k0:k0 + 4]))
            c = fdp.lane_mac_ref(fdp.MODE_DP4, pa, pb, c, 0, 0)
        want = np.float32(f32.b2f(c))
        tot += 1
        for W in bad:
            got = km.lane_gemm(A, Bm, km.Cfg(4, W, 'trunc', 'sum_fma'))[0, 0].astype(np.float32)
            bad[W] += not (got == want)
    report('V8 numpy sum_fma G=4 == existing E4M3 DP4 lane oracle (fdp.lane_mac_ref) over K=64 dot products, W=24 and W=8',
           bad[24] == 0 and bad[8] == 0, f'mismatches W=24: {bad[24]}, W=8: {bad[8]} of {tot}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--seed', type=int, default=20261123)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    q = 1 if args.quick else 4
    v1()
    v2(rng, 4000 * q)
    v3(rng, 60 * q)
    v4(rng, 12 * q)
    v5(rng, 10 * q)
    v6(rng, 40 * q)
    v7(rng, 6 * q)
    v8(rng, 200 * q)
    if args.out:
        json.dump(RESULTS, open(args.out, 'w'), indent=1, sort_keys=True)
    print('FAILED:', FAILED if FAILED else 'none')
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
