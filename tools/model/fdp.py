"""
fdp.py -- exact reference for m.mma.dp, the packed dot-product MAC lane.

A FEASIBILITY prototype (evidence/r41/pmac/lane/): not an ISA opcode yet.

m.mma (ISA sec 9C.1) accumulates one product per k with one rounding each:

    C <- RNE_f32( exact(A[m][k]) * exact(B[k][n]) + C )            k = 0, 1, ...

m.mma.dp is a different reduction order, so it is a different opcode. For
each group g of G consecutive k (G = 4 for 8-bit inputs, G = 2 for 16-bit
inputs), in increasing g order:

    C <- RNE_f32( C + RO_48( sum_{k in g} exact(A[m][k]) * exact(B[k][n]) ) )

- The group sum is exact (rational arithmetic), then RO_48 rounds it to 48
  significant bits with round-to-odd: truncate toward zero and, if anything
  was discarded, set the last kept bit. The exponent is unbounded: the
  intermediate neither overflows nor underflows (BF16 x BF16 sums reach
  2^257 and 2^-266).
- RNE_f32 is the binary32 FMA's rounding: nearest even, subnormals kept, no
  flush to zero, overflow to infinity.
- A and B formats may differ within one width class: E4M3 x E5M2, FP16 x BF16.
- Specials: a NaN operand, 0 * Inf, or Inf products of opposite sign in one
  group give NaN. Otherwise an Inf product gives Inf, and then IEEE Inf + C
  (C = -Inf gives NaN). A NaN C gives NaN. E4M3 has no Inf; S.1111.111 is its
  only NaN. Every NaN result is the canonical quiet NaN 0x7fc00000, as the
  binary32 FMA returns.
- Signed zeros: a group sum that is exactly zero is -0 when every product is a
  zero of negative sign, else +0 (IEEE's sum of zeros, and RNE's +0 for an
  exact cancellation). C + R follows IEEE: an exact zero is -0 only when both
  are -0.

Everything is computed with Fractions and rounded explicitly, so this is an
oracle for the RTL rather than a restatement of it. Element decoding uses
floatfmt.decode, not the hardware's decoding, and the final rounding is
fp32.round_to_f32 (the binary32 FMA's oracle).

Lane interface, as rtl/core/tnca_pmac_lane.v takes it:

    mode   0 FP32: one element per operand, widened by `fmt`  (= tnca_fpu_multi)
           1 DP2:  two 16-bit elements per 32-bit operand, lowest k in [15:0]
           2 DP4:  four 8-bit elements per 32-bit operand, lowest k in [7:0]
    alt_a, alt_b   the per-operand format in a DP mode:
           DP4: 0 = E4M3, 1 = E5M2;   DP2: 0 = FP16, 1 = BF16
"""
import os
import sys
from fractions import Fraction
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from floatfmt import NAN, decode  # noqa: E402
from fp32 import QNAN, classify, round_to_f32, to_fraction  # noqa: E402

MODE_FP32, MODE_DP2, MODE_DP4 = 0, 1, 2
RO_BITS = 48

FMT_OF = {
    MODE_DP4: ('e4m3', 'e5m2'),
    MODE_DP2: ('f16', 'bf16'),
}
WIDTH_OF = {MODE_DP4: 8, MODE_DP2: 16}


def ilog2(a: Fraction) -> int:
    """floor(log2(a)) of a positive Fraction, exactly."""
    # with e = bitlen(num) - bitlen(den): 2^(e-1) < a < 2^(e+1)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    return e - 1 if a < pow2(e) else e


def pow2(e: int) -> Fraction:
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def ro(v: Fraction, p: int = RO_BITS) -> Fraction:
    """Round to p significant bits, round to odd, unbounded exponent."""
    if v == 0:
        return v
    a = abs(v)
    e = ilog2(a)
    assert pow2(e) <= a < pow2(e + 1)
    q = e - (p - 1)                       # weight of the last kept bit
    n = a / pow2(q)
    m = n.numerator // n.denominator
    if m * n.denominator != n.numerator:  # something was discarded: set the odd bit
        m |= 1
    r = m * pow2(q)
    return -r if v < 0 else r


@lru_cache(maxsize=None)
def element(fmt: str, bits: int):
    """(kind, sign, value): kind in 'nan', 'inf', 'zero', 'fin'."""
    w = 8 if fmt in ('e4m3', 'e5m2') else 16
    bits &= (1 << w) - 1
    sign = (bits >> (w - 1)) & 1
    v = decode(fmt, bits)
    if v == NAN:
        return ('nan', sign, None)
    if isinstance(v, tuple):
        return ('inf', v[1], None)
    if v == 0:
        return ('zero', sign, Fraction(0))
    return ('fin', sign, v)


def group_sum(fa: str, fb: str, a_elems, b_elems):
    """One group: ('nan',) | ('inf', sign) | ('fin', exact Fraction, zero_sign)."""
    nan = False
    inf_signs = set()
    total = Fraction(0)
    all_neg_zero = True
    for ab, bb in zip(a_elems, b_elems):
        ka, sa, va = element(fa, ab)
        kb, sb, vb = element(fb, bb)
        s = sa ^ sb
        if ka == 'nan' or kb == 'nan':
            nan = True
            continue
        if (ka == 'inf' and kb == 'zero') or (kb == 'inf' and ka == 'zero'):
            nan = True
            continue
        if ka == 'inf' or kb == 'inf':
            inf_signs.add(s)
            continue
        p = va * vb
        total += p
        if not (p == 0 and s == 1):
            all_neg_zero = False
    if nan or len(inf_signs) > 1:
        return ('nan',)
    if inf_signs:
        return ('inf', inf_signs.pop())
    zs = 1 if (total == 0 and all_neg_zero) else 0
    return ('fin', total, zs)


def unpack(mode: int, word: int):
    w = WIDTH_OF[mode]
    return [(word >> (w * i)) & ((1 << w) - 1) for i in range(32 // w)]


def add_round(r_kind, c_bits: int, sub_c: int = 0) -> int:
    """RNE_f32(C + R) for an already RO-rounded group result R."""
    cb = (c_bits ^ (sub_c << 31)) & 0xffffffff
    kc, sc, _ = classify(cb)
    if r_kind[0] == 'nan' or kc == 'nan':
        return QNAN
    if r_kind[0] == 'inf':
        si = r_kind[1]
        if kc == 'inf' and sc != si:
            return QNAN
        return (si << 31) | (0xff << 23)
    if kc == 'inf':
        return (sc << 31) | (0xff << 23)
    _, r, zs = r_kind
    cv = to_fraction(cb)
    total = r + cv
    if total == 0:
        if r == 0 and cv == 0:
            return (zs & sc) << 31
        return 0                          # exact cancellation: +0
    return round_to_f32(total, 0)


def lane_dp_ref(mode: int, a: int, b: int, c: int, alt_a: int, alt_b: int,
                sub_c: int = 0, exact: bool = False) -> int:
    """Expected y of tnca_pmac_lane in a DP mode (one group).

    exact=True skips RO_48 (the exactly rounded RNE_f32(C + sum)), for
    measuring where the two differ; the lane implements exact=False."""
    fa = FMT_OF[mode][alt_a]
    fb = FMT_OF[mode][alt_b]
    g = group_sum(fa, fb, unpack(mode, a), unpack(mode, b))
    if g[0] == 'fin' and not exact:
        g = ('fin', ro(g[1]), g[2])
    return add_round(g, c, sub_c)


# The shared MAC lane (xls/tnca_mac.x, rtl/core/tnca_mac_lane.v, R41 G2):
# DP2 takes a 2-bit element code per operand, each element in a 16-bit slot
# (an 8-bit element in the slot's low byte, the high byte ignored); DP4 is
# E4M3 x E4M3 only.
MAC_CODE16 = ('f16', 'bf16', 'e5m2', 'e4m3')


def lane_mac_ref(mode: int, a: int, b: int, c: int, fa: int = 0, fb: int = 0,
                 sub_c: int = 0, exact: bool = False) -> int:
    """Expected y of tnca_mac_lane in a packed mode (one group)."""
    if mode == MODE_DP4:
        fma, fmb = 'e4m3', 'e4m3'
        ae, be = unpack(MODE_DP4, a), unpack(MODE_DP4, b)
    elif mode == MODE_DP2:
        fma, fmb = MAC_CODE16[fa], MAC_CODE16[fb]
        ae, be = unpack(MODE_DP2, a), unpack(MODE_DP2, b)
        if fa >= 2:
            ae = [x & 0xff for x in ae]
        if fb >= 2:
            be = [x & 0xff for x in be]
    else:
        raise ValueError('lane_mac_ref: packed modes only')
    g = group_sum(fma, fmb, ae, be)
    if g[0] == 'fin' and not exact:
        g = ('fin', ro(g[1]), g[2])
    return add_round(g, c, sub_c)


def mma_dp_ref(fa: str, fb: str, a_bits, b_bits, c_f32, M, N, K):
    """Reference for a whole m.mma.dp: C[MxN] (+)= A[MxK] . B[KxN], groups of
    G consecutive k in increasing order. a_bits is row-major MxK, b_bits KxN."""
    # ISA sec 9C.4 (approved 2026-10-29): four products only for E4M3 x E4M3,
    # two for every other pair of formats
    G = 4 if (fa == 'e4m3' and fb == 'e4m3') else 2
    assert K % G == 0
    out = list(c_f32)
    for m in range(M):
        for n in range(N):
            acc = out[m * N + n]
            for k0 in range(0, K, G):
                ae = [a_bits[m * K + k] for k in range(k0, k0 + G)]
                be = [b_bits[k * N + n] for k in range(k0, k0 + G)]
                gs = group_sum(fa, fb, ae, be)
                if gs[0] == 'fin':
                    gs = ('fin', ro(gs[1]), gs[2])
                acc = add_round(gs, acc)
            out[m * N + n] = acc
    return out


# ---------------------------------------------------------------------------
# Self-checks of the semantics, independent of any RTL: `python3 fdp.py`
# ---------------------------------------------------------------------------

def _e4m3_exactness():
    """Every E4M3 x E4M3 group sum is an integer multiple of 2^-18 below
    2^38 in magnitude, so it has at most 38 significant bits and RO_48 is the
    identity: the lane's E4M3 result is the exactly rounded RNE_f32(C + sum).
    Checked over all 256 encodings, not argued from the format table."""
    vals = []
    for x in range(256):
        k, s, v = element('e4m3', x)
        if k in ('zero', 'fin'):
            vals.append(v)
            assert (v * (1 << 9)).denominator == 1, x        # unit 2^-9
    vmax = max(abs(v) for v in vals)
    assert vmax == 448
    # every product is a multiple of 2^-18 (2^-9 * 2^-9); |sum of 4| <= 4*448^2
    bound = 4 * vmax * vmax * (1 << 18)
    assert bound < (1 << 38), bound
    return vmax, bound


def _widths():
    """The fixed-point field each FP8 pair needs for an exact 4-product sum:
    (unit exponent, magnitude bits)."""
    out = {}
    for fa in ('e4m3', 'e5m2'):
        for fb in ('e4m3', 'e5m2'):
            va = [element(fa, x)[2] for x in range(256) if element(fa, x)[0] in ('zero', 'fin')]
            vb = [element(fb, x)[2] for x in range(256) if element(fb, x)[0] in ('zero', 'fin')]
            unit = min(abs(x) for x in va if x) * min(abs(x) for x in vb if x)
            top = max(abs(x) for x in va) * max(abs(x) for x in vb) * 4
            n = top / unit
            assert n.denominator == 1
            out[(fa, fb)] = (ilog2(unit), int(n).bit_length())
    return out


if __name__ == '__main__':
    vmax, bound = _e4m3_exactness()
    print(f"E4M3 x E4M3: |value| <= {vmax}, |sum of 4| * 2^18 <= {bound} < 2^38: RO_48 exact")
    for (fa, fb), (u, bits) in _widths().items():
        print(f"{fa} x {fb}: unit 2^{u}, exact 4-sum magnitude needs {bits} bits")
    # RO_48 basics
    assert ro(Fraction((1 << 48) + 1)) == (1 << 48) + 2
    assert ro(Fraction((1 << 47) + 1)) == (1 << 47) + 1
    x = Fraction((1 << 49) + 1)                     # 50 bits: drop 2 bits, set the odd bit (weight 4)
    assert ro(x) == (1 << 49) + 4, ro(x)
    assert ro(Fraction((1 << 49) + 4)) == (1 << 49) + 4
    assert ro(-x) == -((1 << 49) + 4)
    assert ro(Fraction((1 << 49) + 8)) == (1 << 49) + 8
    # an E5M2 cancellation where RO_48 and exact rounding differ
    # 57344*57344 = 2^31.6...; 2^-16 * 2^-16 = 2^-32: sum = big - big + tiny
    a = 0x7b | (0x01 << 8)                          # e5m2: 57344, 2^-16
    b = 0x7b | (0x01 << 8)
    c = 0xcf440000                                  # -(57344^2) in binary32 = -3288334336
    print("RO48 vs exact, E5M2 cancellation:",
          hex(lane_dp_ref(MODE_DP4, a, b, c, 1, 1)),
          hex(lane_dp_ref(MODE_DP4, a, b, c, 1, 1, exact=True)))
    print("fdp.py self-checks pass")
