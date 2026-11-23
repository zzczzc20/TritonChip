#!/usr/bin/env python3
"""kfuse_model.py -- bit-accurate model of a K-fused block dot-product lane (R42-kfuse).

The question (doc/r42_plan.md, "Numerics decision for the wide core"): a DSP-based lane that takes G
products per step, aligns them (and the incoming fp32 accumulator) to the group's maximum exponent, adds
them in a wide fixed-point tree and rounds ONCE per group, instead of the contract of ISA sec 9C.1 (one
RNE binary32 FMA per k step, increasing k). This file defines the lane's arithmetic exactly, twice:

  * a scalar model in Python integers / Fractions (`scalar_step`): the definition, any W, W = None for an
    unbounded window; specials (NaN, Inf, signed zero) follow tools/model/fdp.py;
  * a numpy model (`group_vec`, `lane_gemm`, `seq_gemm`): the same arithmetic on float64 arrays, used for the
    accuracy study. It is exact (all quantities are integers below 2^53 or exact products), bit-identical to
    the scalar model for finite data (kfuse_validate.py checks that), and handles W in [-24, 24] for
    G <= 16 (24 + W + ceil(log2(G + 1)) <= 53 bits).

THE LANE (one step = one group of G products, k increasing, accumulator c in binary32)

  terms      exact products p_i = a_i * b_i (never rounded), i < G; the zero products are dropped.
  window     E = max over the terms of floor(log2 |term|); the fixed point has its least significant bit at
             L = E - 23 - W: the leading position of the largest term and 23 + W positions below it, i.e.
             the 24 bits of a binary32 significand plus W guard bits. W = None: L = the lowest bit of any
             term (nothing is ever shifted out).
  align      every term is shifted to the fixed point; the bits below L are dropped per term, by
               'trunc'  truncation of the magnitude (toward zero; the sign-magnitude shifter of an
                        NVIDIA-style tree), or
               'rn'     round to nearest of the magnitude, ties away from zero (add the first dropped bit).
  sum        a wide integer sum of the aligned terms (carry-propagate, exact).
  round      ONE round-to-nearest-even to binary32 (subnormals kept, overflow to infinity), per group.

  variants (where the accumulator c enters)
    'acc_in'     c is a term of the same window (E includes c): one tree, one rounding per group.
    'acc_after'  the group sum is formed from the products alone, rounded to binary32, and then added to c
                 by an ordinary binary32 add (two roundings per group).
    'sum_fma'    the group sum is formed from the products alone (window anchored at the largest product),
                 kept at full width, and added to c exactly with ONE rounding. This is the structure of the
                 existing m.mma.dp lane (ISA sec 9C.4: RO_48 group sum, then align with C, round); with
                 W = None it IS that lane for E4M3 x E4M3 (its group sum is exact) and its exact-sum
                 limit otherwise.

  Limiting cases (checked by kfuse_validate.py): G = 1 and W = None: the sequential FMA of sec 9C.1, bit
  for bit (acc_in, sum_fma); W = None: the exactly rounded group sum RNE(c + sum) (acc_in, sum_fma) and
  RNE(c + RNE(sum)) (acc_after).

Sequential reference = `seq_gemm`: C[k+1] = RNE_f32(a_k * b_k + C[k]), k increasing (ISA sec 9C.1).
Exact reference = binary64 (`A @ B` in float64; the products are exact, the sums are rounded at 2^-53).
"""
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
import fdp  # noqa: E402
import fp32 as fp32ref  # noqa: E402

FMT = {'fp16': 'f16', 'bf16': 'bf16', 'e4m3': 'e4m3', 'e5m2': 'e5m2'}     # model name -> tools/model name
ELEM_BITS = {'fp16': 16, 'bf16': 16, 'e4m3': 8, 'e5m2': 8}
PROD_SIG_BITS = {'fp16': 22, 'bf16': 16, 'e4m3': 8, 'e5m2': 6}            # significand bits of an exact product
VARIANTS = ('acc_in', 'acc_after', 'sum_fma')
ROUNDINGS = ('trunc', 'rn')


@dataclass(frozen=True)
class Cfg:
    G: int
    W: Optional[int]
    rnd: str = 'trunc'
    variant: str = 'acc_in'

    def __post_init__(self):
        assert self.G >= 1 and self.rnd in ROUNDINGS and self.variant in VARIANTS

    def tag(self):
        w = 'inf' if self.W is None else str(self.W)
        return f'{self.variant}.G{self.G}.W{w}.{self.rnd}'


def tree_width_bits(cfg):
    """Width of the fixed-point adder (bits, sign included) and its parts, for a Cfg with finite W.
    window = 24 + W (the 24 binary32 significand bits plus W guard bits, anchored at the largest term),
    carry = ceil(log2(terms)) with terms = G + 1 for 'acc_in' (the accumulator is a term) and G otherwise."""
    terms = cfg.G + (1 if cfg.variant == 'acc_in' else 0)
    carry = (terms - 1).bit_length()
    window = 24 + cfg.W
    return dict(window=window, carry=carry, sign=1, total=window + carry + 1, terms=terms)


# =====================================================================================================
# scalar model: the definition
# =====================================================================================================

def _decompose(v):
    """Nonzero dyadic Fraction -> (sign, m, q) with |v| = m * 2^q, m a positive integer."""
    d = v.denominator
    q = -(d.bit_length() - 1)
    assert d == 1 << -q
    return (1 if v < 0 else 0), abs(v.numerator), q


def _shift(m, s, rnd):
    """m * 2^s as an integer, the bits below 2^0 dropped by `rnd` (magnitude)."""
    if s >= 0:
        return m << s
    s = -s
    if rnd == 'trunc':
        return m >> s
    return (m + (1 << (s - 1))) >> s            # 'rn': add the first dropped bit, ties away from zero


def _aligned_sum(values, W, rnd):
    """values: nonzero Fractions. -> (S, L): the integer sum of the aligned terms and the weight 2^L of its lsb."""
    dec = [_decompose(v) for v in values]
    top = max(q + m.bit_length() - 1 for _, m, q in dec)
    L = min(q for _, _, q in dec) if W is None else top - 23 - W
    S = 0
    for s, m, q in dec:
        t = _shift(m, q - L, rnd)
        S += -t if s else t
    return S, L


def _pow2(e):
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def scalar_step(cfg, fmt, a_bits, b_bits, c_bits, fmt_b=None):
    """One group: a_bits, b_bits = lists of G element encodings, c_bits = binary32 encoding -> binary32 encoding."""
    fa, fb = FMT[fmt], FMT[fmt_b or fmt]
    assert len(a_bits) == len(b_bits) <= cfg.G
    gs = fdp.group_sum(fa, fb, a_bits, b_bits)
    if gs[0] != 'fin':
        return fdp.add_round(gs, c_bits)
    kc, sc, _ = fp32ref.classify(c_bits)
    if kc in ('nan', 'inf'):
        return fdp.add_round(('fin', Fraction(0), 0), c_bits)
    cv = fp32ref.to_fraction(c_bits)
    zs = gs[2]                                          # 1: every product is a negative zero
    prods = []
    for ab, bb in zip(a_bits, b_bits):
        va, vb = fdp.element(fa, ab)[2], fdp.element(fb, bb)[2]
        p = va * vb
        if p != 0:
            prods.append(p)

    def zero_result(c_zero):
        # nothing nonzero entered: IEEE sign of a sum of zeros (negative only if all are negative zeros)
        return (zs & sc) << 31 if c_zero else 0

    if cfg.variant == 'acc_in':
        terms = prods + ([cv] if cv != 0 else [])
        if not terms:
            return zero_result(cv == 0)
        S, L = _aligned_sum(terms, cfg.W, cfg.rnd)
        if S == 0:
            return 0
        return fp32ref.round_to_f32(Fraction(S) * _pow2(L), 0)
    if not prods:
        if cv == 0:
            return zero_result(True)
        return fp32ref.round_to_f32(cv, 0)
    S, L = _aligned_sum(prods, cfg.W, cfg.rnd)
    r = Fraction(S) * _pow2(L)
    if cfg.variant == 'sum_fma':
        total = r + cv
        return 0 if total == 0 else fp32ref.round_to_f32(total, 0)
    # acc_after: the group sum rounded to binary32 first, then an ordinary binary32 add
    rb = 0 if S == 0 else fp32ref.round_to_f32(r, 0)
    if ((rb >> 23) & 0xff) == 0xff:                     # the group sum overflowed binary32
        return fdp.add_round(('inf', rb >> 31), c_bits)
    total = fp32ref.to_fraction(rb) + cv
    return 0 if total == 0 else fp32ref.round_to_f32(total, 0)


def scalar_dot(cfg, fmt, a_bits, b_bits, c_bits=0, fmt_b=None):
    """A whole dot product through the lane: groups of cfg.G in increasing k (the last group short)."""
    c = c_bits
    for k0 in range(0, len(a_bits), cfg.G):
        c = scalar_step(cfg, fmt, a_bits[k0:k0 + cfg.G], b_bits[k0:k0 + cfg.G], c, fmt_b)
    return c


def scalar_fma_chain(fmt, a_bits, b_bits, c_bits=0, fmt_b=None):
    """The sequential contract of ISA sec 9C.1, from the independent oracle tools/model/fp32.py."""
    fa, fb = FMT[fmt], FMT[fmt_b or fmt]
    c = c_bits
    for ab, bb in zip(a_bits, b_bits):
        ka, sa, va = fdp.element(fa, ab)
        kb, sb, vb = fdp.element(fb, bb)
        if 'nan' in (ka, kb) or 'inf' in (ka, kb):
            c = fdp.add_round(fdp.group_sum(fa, fb, [ab], [bb]), c)
            continue
        p = va * vb
        cv = fp32ref.to_fraction(c)
        kc, sc, _ = fp32ref.classify(c)
        if kc in ('nan', 'inf'):
            c = fdp.add_round(('fin', Fraction(0), 0), c)
            continue
        total = p + cv
        if total == 0:
            if p == 0 and cv == 0:
                c = (((sa ^ sb) & sc) << 31)
            else:
                c = 0
        else:
            c = fp32ref.round_to_f32(total, 0)
    return c


# =====================================================================================================
# numpy model
# =====================================================================================================

def _ebiased(x):
    """Biased binary64 exponent field of a contiguous float64 array (0 for zero)."""
    return (x.view(np.int64) >> 52) & 0x7FF


def _p2(e):
    """2^e as float64 for an int64 array e in [-1022, 1023], exactly, by building the bits."""
    return ((e + 1023) << 52).view(np.float64)


def _quantise_term(t, rnd):
    if rnd == 'trunc':
        return np.trunc(t)
    return np.trunc(t + 0.5 * np.sign(t))


def rn32_add(c, s):
    """RNE_f32(c + s) for float64 arrays c, s (exactly representable operands, any exponents), single rounding:
    the binary64 sum is made round-to-odd with a TwoSum error term, then rounded to binary32."""
    x = c + s
    bb = x - c
    err = (c - (x - bb)) + (s - bb)
    xi = x.view(np.int64)
    fix = (err != 0) & ((xi & 1) == 0)
    up = (err > 0) == (x > 0)
    xi = xi + np.where(fix, np.where(up, 1, -1), 0)
    return xi.view(np.float64).astype(np.float32)


def group_vec(prods, c, cfg):
    """One lane step for float64 arrays: prods (G, ...) exact products, c (...) binary32 values (as float64).
    Returns the new accumulator as float64 holding binary32 values."""
    W = cfg.W
    assert W is not None and W <= 24 and 24 + W + cfg.G.bit_length() <= 53, 'numpy model: finite W only'
    eb = _ebiased(prods).max(axis=0)
    if cfg.variant == 'acc_in':
        eb = np.maximum(eb, _ebiased(c))
    L = np.maximum(eb - 1023, -300) - (23 + W)
    t = _quantise_term(prods * _p2(-L), cfg.rnd).sum(axis=0)
    if cfg.variant == 'acc_in':
        t = t + _quantise_term(c * _p2(-L), cfg.rnd)
    s = t * _p2(L)
    if cfg.variant == 'acc_in':
        return s.astype(np.float32).astype(np.float64)
    if cfg.variant == 'acc_after':
        r = s.astype(np.float32).astype(np.float64)
        return (c + r).astype(np.float32).astype(np.float64)
    return rn32_add(c, s).astype(np.float64)


def _tiles(M, N, tile):
    tn = min(N, max(1, tile))
    tm = max(1, min(M, tile // tn))
    for m0 in range(0, M, tm):
        for n0 in range(0, N, tn):
            yield m0, min(M, m0 + tm), n0, min(N, n0 + tn)


def lane_gemm(A, B, cfg, c0=None, tile=8192):
    """C = chain of lane steps over K in increasing order. A (M, K), B (K, N) float64 exact element values,
    c0 (M, N) float64 holding binary32 values or None (zero). Returns float64 holding binary32 values."""
    M, K = A.shape
    N = B.shape[1]
    G = cfg.G
    Kp = -(-K // G) * G
    if Kp != K:
        A = np.concatenate([A, np.zeros((M, Kp - K))], axis=1)
        B = np.concatenate([B, np.zeros((Kp - K, N))], axis=0)
    out = np.empty((M, N), dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        for m0, m1, n0, n1 in _tiles(M, N, tile):
            Ag = np.ascontiguousarray(A[m0:m1].reshape(m1 - m0, Kp // G, G).transpose(1, 2, 0))   # (groups, G, Mt)
            Bg = np.ascontiguousarray(B[:, n0:n1].reshape(Kp // G, G, n1 - n0))                    # (groups, G, Nt)
            c = np.zeros((m1 - m0, n1 - n0)) if c0 is None else np.array(c0[m0:m1, n0:n1], dtype=np.float64)
            for g in range(Kp // G):
                prods = Ag[g][:, :, None] * Bg[g][:, None, :]
                c = group_vec(prods, c, cfg)
            out[m0:m1, n0:n1] = c
    return out


def seq_gemm(A, B, fmt, c0=None, tile=8192):
    """The sequential contract: one RNE binary32 FMA per k, increasing k. The product is exact in binary64;
    for FP16, E4M3 and E5M2 it is also a binary32 number, so the binary64 sum rounded to binary32 is the single
    rounding (double rounding of an addition of two binary32 numbers through binary64 is innocuous); BF16
    products can leave the binary32 range, and take the round-to-odd path."""
    M, K = A.shape
    N = B.shape[1]
    out = np.empty((M, N), dtype=np.float64)
    simple = fmt != 'bf16'
    with np.errstate(over='ignore', invalid='ignore'):
        for m0, m1, n0, n1 in _tiles(M, N, tile):
            Ac = np.ascontiguousarray(A[m0:m1].T)
            Br = np.ascontiguousarray(B[:, n0:n1])
            c = np.zeros((m1 - m0, n1 - n0)) if c0 is None else np.array(c0[m0:m1, n0:n1], dtype=np.float64)
            for k in range(K):
                p = Ac[k][:, None] * Br[k][None, :]
                if simple:
                    c = (c + p).astype(np.float32).astype(np.float64)
                else:
                    c = rn32_add(c, p).astype(np.float64)
            out[m0:m1, n0:n1] = c
    return out


def exact_gemm(A, B):
    """The binary64 reference: float64 matrix product (exact products, binary64 accumulation)."""
    return A @ B


# ---- element conversions -------------------------------------------------------------------------

def round_bf16_f64(x):
    """RNE of float64 values to bfloat16 (returned as float64), by bit manipulation, normal range."""
    xi = np.ascontiguousarray(x, dtype=np.float64).view(np.int64)
    lsb = (xi >> 45) & 1
    xi = (xi + (1 << 44) - 1 + lsb) & ~np.int64((1 << 45) - 1)
    return xi.view(np.float64)


def to_fp16_f64(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


_FMT_PARAMS = {'fp16': (5, 10, True), 'bf16': (8, 7, True), 'e4m3': (4, 3, False), 'e5m2': (5, 2, True)}


def decode_array(fmt, bits):
    """Element encodings (integer array) -> float64 values; NaN for NaN, +-inf for infinity. Vectorised,
    exhaustively checked against tools/model/floatfmt.decode in kfuse_validate.py."""
    e_sz, f_sz, has_inf = _FMT_PARAMS[fmt]
    bits = np.asarray(bits, dtype=np.int64) & ((1 << (1 + e_sz + f_sz)) - 1)
    sign = (bits >> (e_sz + f_sz)) & 1
    exp = (bits >> f_sz) & ((1 << e_sz) - 1)
    frac = bits & ((1 << f_sz) - 1)
    bias = (1 << (e_sz - 1)) - 1
    normal = (1.0 + frac / float(1 << f_sz)) * np.exp2((exp - bias).astype(np.float64))
    sub = (frac / float(1 << f_sz)) * 2.0 ** (1 - bias)
    v = np.where(exp == 0, sub, normal)
    top = exp == (1 << e_sz) - 1
    if has_inf:
        v = np.where(top, np.where(frac == 0, np.inf, np.nan), v)
    else:
        v = np.where(top & (frac == (1 << f_sz) - 1), np.nan, v)
    return np.where(sign == 1, -v, v)
