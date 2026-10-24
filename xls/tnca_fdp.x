// Packed dot-product ("NV-style") MAC lane -- R41 FEASIBILITY PROTOTYPE.
//
// Not integrated: nothing in the core instantiates it. It exists to measure
// what FP8 at 4x and FP16/BF16 at 2x the binary32 MAC rate cost per lane
// (evidence/r41/pmac/lane/README.md).
//
// One lane computes, per cycle, a group of G products into its binary32
// accumulator (m.mma.dp, a new opcode -- a different reduction order than
// ISA sec 9C.1's m.mma, so it may not be that opcode made faster):
//
//     C <- RNE_f32( C + RO_48( sum_{k in group} exact(a_k) * exact(b_k) ) )
//
// G = 4 for 8-bit inputs (DP4: E4M3, E5M2, either per operand) and G = 2 for
// 16-bit inputs (DP2: FP16, BF16, either per operand). RO_48 is round-to-odd
// at 48 significant bits with an unbounded exponent; RNE_f32 is the binary32
// FMA's own rounding. tools/model/fdp.py is the exact oracle.
//
// THE STRUCTURE
//
// The binary32 FMA (tnca_fma.x) is split at its multiplier: fma32_mul forms
// the exact product (pm, xp, flags), fma32_align_p aligns any such product
// with c, and fma32_normalise / fma32_round finish. A DP mode replaces only
// the product: the group sum, rounded to odd at 48 bits, IS a 48-bit
// significand with an exponent, so the back end -- align, add, normalise,
// RNE -- is the binary32 one, unchanged and shared, and RNE_f32(C + RO_48(S))
// is exactly what it computes. Binary32 mode is fma32_mul itself: bit for bit
// the current lane.
//
// Each front end is two stage functions, F1 and F2, so the lane wrapper
// (rtl/core/tnca_pmac_lane.v) can place a register between them, after them,
// or neither:
//
//   F1  DP4: decode, four small exact products, each placed by its ABSOLUTE
//       exponent into a fixed-point field wide enough to hold the sum exactly
//       (no alignment, no sticky), signed sum.
//       DP2: decode, two exact 11x11 products, the smaller-scale one shifted
//       onto the larger in a 64-bit field with a sticky bit, signed sum
//       (the FMA's borrow: a sticky bit of an opposite-sign term takes one
//       unit off the magnitude and stays as the sticky bit).
//       Binary32: fma32_mul (the 24x24 product, two DSP48E2).
//   F2  |sum|, then RO_48 -- and here the field is never normalised. The back
//       end aligns VALUES, so a product below 2^48 can go in unnormalised;
//       only a sum with more than 48 significant bits has bits to drop, and
//       then its leading one is in the field's top 18 bits. So RO_48 is an
//       18-bit leading-zero count, a right shift by r = 0..18 and an OR into
//       the last bit: no 66-bit normaliser in front of the alignment shifter.
//       The mode selects the binary32 product or this one.
//
// WHY THE FIELDS ARE THESE WIDTHS (fdp.py checks them over every encoding)
//
//   E4M3 x E4M3: every value is a multiple of 2^-9 below 448, every product a
//     multiple of 2^-18 below 448^2, so |sum of 4| * 2^18 < 2^38: 38 bits, and
//     RO_48 is the identity -- the lane's E4M3 result is the exactly rounded
//     RNE_f32(C + sum). That is variant V3, which needs no RO logic at all.
//   E4M3 x E5M2: unit 2^-25, 52 bits. E5M2 x E5M2: unit 2^-32, 66 bits. One
//     66-bit field with unit 2^-32 holds all four pairs (V1, V2).
//   DP2: a product of 11-bit significands is 22 bits; the larger-scale one at
//     [63:42]. A sticky bit needs a scale gap >= 43; then the reference
//     product is at least 2^11 (an FP16 subnormal times a BF16 normal is the
//     smallest product that can be the reference across such a gap; FP16 x
//     FP16 gaps never exceed 30), so the sum's leading one is at bit >= 52 and
//     its 48 significant bits are all inside the field: the sticky bit is
//     below them, as RO_48 needs.
//
// Variants (tools/xls/config.json, rtl/core/tnca_pmac_lane.v):
//   V1  binary32 + DP4 for all four FP8 pairs (66-bit field)
//   V2  V1 + DP2; products in LUTs (V2L) or in the binary32 multiplier's two
//       DSPs, re-split as two 24x12 multiplies (V2D)
//   V3  binary32 + DP4 for E4M3 x E4M3 only (38-bit field); E5M2 uses the
//       binary32 (1x) path

import std;
import tnca_fma;

pub const MODE_FP32 = u2:0;
pub const MODE_DP2 = u2:1;
pub const MODE_DP4 = u2:2;

type FmaProduct = tnca_fma::FmaProduct;

fn lsb_mask64(n: u7) -> u64 { !(all_ones!<u64>() << n) }
fn lsb_mask66(n: u5) -> uN[66] { !(all_ones!<uN[66]>() << n) }

//============================================================================
// Elements
//============================================================================

// An FP8 element: value = sig * 2^(q - 17), sig the significand with the
// hidden bit (a subnormal's fraction doubled, as fma32_mul does), q the
// stored exponent field plus a per-format offset.
pub struct E8 {
    s: u1,
    sig: u4,
    q: u5,
    zero: u1,
    inf: u1,
    nan: u1,
}

// alt = 0: E4M3 (1.4.3, no Inf, S.1111.111 is NaN). alt = 1: E5M2 (1.5.2).
pub fn dec8(x: u8, alt: u1) -> E8 {
    let s = x[7+:u1];
    // E4M3: value = sig * 2^(e - 10)
    let e4 = x[3+:u4];
    let f4 = x[0+:u3];
    let sig4 = if e4 != u4:0 { u1:1 ++ f4 } else { (f4 as u4) << u4:1 };
    // E5M2: value = sig * 2^(e - 17)
    let e5 = x[2+:u5];
    let f5 = x[0+:u2];
    let sig5 = if e5 != u5:0 { (u1:1 ++ f5) as u4 } else { (f5 as u4) << u4:1 };
    if alt == u1:0 {
        E8 { s: s, sig: sig4, q: (e4 as u5) + u5:7,
             zero: (e4 == u4:0 && f4 == u3:0) as u1, inf: u1:0,
             nan: (e4 == u4:15 && f4 == u3:7) as u1 }
    } else {
        E8 { s: s, sig: sig5, q: e5,
             zero: (e5 == u5:0 && f5 == u2:0) as u1,
             inf: (e5 == u5:31 && f5 == u2:0) as u1,
             nan: (e5 == u5:31 && f5 != u2:0) as u1 }
    }
}

// E4M3 only (V3): value = sig * 2^(q - 10), q = e.
pub fn dec8_e4m3(x: u8) -> E8 {
    let e4 = x[3+:u4];
    let f4 = x[0+:u3];
    E8 { s: x[7+:u1],
         sig: if e4 != u4:0 { u1:1 ++ f4 } else { (f4 as u4) << u4:1 },
         q: e4 as u5,
         zero: (e4 == u4:0 && f4 == u3:0) as u1, inf: u1:0,
         nan: (e4 == u4:15 && f4 == u3:7) as u1 }
}

// A 16-bit element: value = sig * 2^l, sig 11 bits (a BF16 significand at
// the top, so every product is 22 bits with the same layout).
pub struct E16 {
    s: u1,
    sig: u11,
    l: s12,
    zero: u1,
    inf: u1,
    nan: u1,
}

// alt = 0: FP16 (1.5.10). alt = 1: BF16 (1.8.7).
pub fn dec16(x: u16, alt: u1) -> E16 {
    let s = x[15+:u1];
    let eh = x[10+:u5];
    let fh = x[0+:u10];
    let sigh = if eh != u5:0 { u1:1 ++ fh } else { (fh as u11) << u11:1 };
    let eb = x[7+:u8];
    let fb = x[0+:u7];
    let sigb8 = if eb != u8:0 { u1:1 ++ fb } else { (fb as u8) << u8:1 };
    if alt == u1:0 {
        E16 { s: s, sig: sigh, l: (eh as s12) - s12:25,
              zero: (eh == u5:0 && fh == u10:0) as u1,
              inf: (eh == u5:31 && fh == u10:0) as u1,
              nan: (eh == u5:31 && fh != u10:0) as u1 }
    } else {
        E16 { s: s, sig: (sigb8 as u11) << u11:3, l: (eb as s12) - s12:137,
              zero: (eb == u8:0 && fb == u7:0) as u1,
              inf: (eb == u8:255 && fb == u7:0) as u1,
              nan: (eb == u8:255 && fb != u7:0) as u1 }
    }
}

//============================================================================
// Special cases of a group, and the sign of a zero group sum
//============================================================================

pub struct Spec {
    p_nan: u1,      // a NaN operand, 0 * Inf, or Inf products of both signs
    p_inf: u1,
    inf_sign: u1,
    zsign: u1,      // the sign of an exactly zero sum
}

// one product's (nan, inf, sign)
fn spec1(sa: u1, za: u1, ia: u1, na: u1, sb: u1, zb: u1, ib: u1, nb: u1) -> (u1, u1, u1) {
    let nan = na | nb | (ia & zb) | (ib & za);
    let inf = (ia | ib) & !nan;
    (nan, inf, sa ^ sb)
}

fn spec_combine<N: u32>(n: u1[N], i: u1[N], s: u1[N]) -> Spec {
    let any_nan = or_reduce(n as uN[N]);
    let inf_pos = for (k, acc) : (u32, u1) in u32:0..N { acc | (i[k] & !s[k]) }(u1:0);
    let inf_neg = for (k, acc) : (u32, u1) in u32:0..N { acc | (i[k] & s[k]) }(u1:0);
    // A group sum that is exactly zero is -0 only if every product is a zero
    // of negative sign. Every product negative and a zero sum already implies
    // every product is zero (no cancellation among same-sign terms), so the
    // sign bits alone decide it.
    Spec {
        p_nan: any_nan | (inf_pos & inf_neg),
        p_inf: inf_pos | inf_neg,
        inf_sign: inf_neg,
        zsign: and_reduce(s as uN[N]),
    }
}

fn spec4(a: E8[4], b: E8[4]) -> Spec {
    let r0 = spec1(a[0].s, a[0].zero, a[0].inf, a[0].nan, b[0].s, b[0].zero, b[0].inf, b[0].nan);
    let r1 = spec1(a[1].s, a[1].zero, a[1].inf, a[1].nan, b[1].s, b[1].zero, b[1].inf, b[1].nan);
    let r2 = spec1(a[2].s, a[2].zero, a[2].inf, a[2].nan, b[2].s, b[2].zero, b[2].inf, b[2].nan);
    let r3 = spec1(a[3].s, a[3].zero, a[3].inf, a[3].nan, b[3].s, b[3].zero, b[3].inf, b[3].nan);
    spec_combine<u32:4>([r0.0, r1.0, r2.0, r3.0], [r0.1, r1.1, r2.1, r3.1], [r0.2, r1.2, r2.2, r3.2])
}

fn spec2(a: E16[2], b: E16[2]) -> Spec {
    let r0 = spec1(a[0].s, a[0].zero, a[0].inf, a[0].nan, b[0].s, b[0].zero, b[0].inf, b[0].nan);
    let r1 = spec1(a[1].s, a[1].zero, a[1].inf, a[1].nan, b[1].s, b[1].zero, b[1].inf, b[1].nan);
    spec_combine<u32:2>([r0.0, r1.0], [r0.1, r1.1], [r0.2, r1.2])
}

//============================================================================
// DP4: four exact products in one fixed-point field
//============================================================================

// A product's signed contribution as (x ^ mask, carry): -F = !F + 1, so the
// four carries are added once, with the terms, instead of four negations.
fn term67(s: u1, f: uN[66]) -> (uN[67], uN[67]) {
    let m = if s == u1:1 { all_ones!<uN[67]>() } else { uN[67]:0 };
    ((f as uN[67]) ^ m, s as uN[67])
}

// All four FP8 pairs. Field unit 2^-32: E5M2 x E5M2 products have pos >= 0
// and P a multiple of 4 at pos 0 (doubled subnormal fractions), so the
// field's bottom two bits of unit 2^-34 are always zero and are dropped.
fn dp4_field(a: E8, b: E8) -> uN[66] {
    let p = std::umul(a.sig, b.sig);                // u8, exact
    let pos = (a.q as u6) + (b.q as u6);            // value = p * 2^(pos - 34)
    (((p as uN[68]) << pos) >> uN[68]:2) as uN[66]
}

pub struct Dp4Sum {
    sum: sN[67],        // the exact signed group sum, unit 2^-32
    sp: Spec,
}

pub fn dp4_sum(a: u32, b: u32, alt_a: u1, alt_b: u1) -> Dp4Sum {
    let ea = [dec8(a[0+:u8], alt_a), dec8(a[8+:u8], alt_a), dec8(a[16+:u8], alt_a), dec8(a[24+:u8], alt_a)];
    let eb = [dec8(b[0+:u8], alt_b), dec8(b[8+:u8], alt_b), dec8(b[16+:u8], alt_b), dec8(b[24+:u8], alt_b)];
    let (t0, c0) = term67(ea[0].s ^ eb[0].s, dp4_field(ea[0], eb[0]));
    let (t1, c1) = term67(ea[1].s ^ eb[1].s, dp4_field(ea[1], eb[1]));
    let (t2, c2) = term67(ea[2].s ^ eb[2].s, dp4_field(ea[2], eb[2]));
    let (t3, c3) = term67(ea[3].s ^ eb[3].s, dp4_field(ea[3], eb[3]));
    let sum = t0 + t1 + t2 + t3 + (c0 + c1 + c2 + c3);
    Dp4Sum { sum: sum as sN[67], sp: spec4(ea, eb) }
}

//============================================================================
// DP2: two exact 11x11 products, aligned with a sticky bit
//============================================================================

pub struct Dp2Sum {
    sum: sN[67],        // the signed sum, unit 2^(lb - 42); truncated if sticky
    sticky: u1,      // bits of the smaller-scale product below the field
    lb: s12,         // scale of the reference product (its value = P * 2^lb)
    sp: Spec,
}

// The alignment from the two products, however they were multiplied.
fn dp2_align(ea: E16[2], eb: E16[2], p0: u22, p1: u22) -> Dp2Sum {
    let l0 = ea[0].l + eb[0].l;
    let l1 = ea[1].l + eb[1].l;
    let z0 = ea[0].zero == u1:1 || eb[0].zero == u1:1;
    let z1 = ea[1].zero == u1:1 || eb[1].zero == u1:1;
    let sg0 = ea[0].s ^ eb[0].s;
    let sg1 = ea[1].s ^ eb[1].s;
    // A zero product is never the reference: its scale means nothing.
    let ref1 = if z0 { true } else if z1 { false } else { l1 > l0 };
    let d01 = l0 - l1;
    let d10 = l1 - l0;
    let dabs = if ref1 { d10 } else { d01 };
    let d = if dabs > s12:64 { u7:64 } else { dabs as u7 };
    let pb = if ref1 { p1 } else { p0 };
    let ps = if ref1 { p0 } else { p1 };
    let sb = if ref1 { sg1 } else { sg0 };
    let ss = if ref1 { sg0 } else { sg1 };
    let lb = if ref1 { l1 } else { l0 };
    let bf = (pb as u64) << u64:42;                     // [63:42]
    let sf = (ps as u64) << u64:42;
    let sh = sf >> d;
    let sticky = (sf & lsb_mask64(d)) != u64:0;
    // exact = sb*B + ss*(Ssh + eps), 0 < eps < 1 iff sticky. With a sticky bit
    // the reference term is far larger (B >= 2^53 > Ssh + 1), so for opposite
    // signs floor(|exact|) = B - Ssh - 1: the signed sum takes one more unit
    // in the small term's direction and the sticky bit stays.
    let (t0, c0) = term67(sb, bf as uN[66]);
    let (t1, c1) = term67(ss, sh as uN[66]);
    let borrow = sticky && (sb != ss);
    let (t2, c2) = if borrow { term67(ss, uN[66]:1) } else { (uN[67]:0, uN[67]:0) };
    let sum = t0 + t1 + t2 + (c0 + c1 + c2);
    Dp2Sum { sum: sum as sN[67], sticky: sticky as u1, lb: lb, sp: spec2(ea, eb) }
}

fn dec2(a: u32, b: u32, alt_a: u1, alt_b: u1) -> (E16[2], E16[2]) {
    ([dec16(a[0+:u16], alt_a), dec16(a[16+:u16], alt_a)],
     [dec16(b[0+:u16], alt_b), dec16(b[16+:u16], alt_b)])
}

pub fn dp2_sum(a: u32, b: u32, alt_a: u1, alt_b: u1) -> Dp2Sum {
    let (ea, eb) = dec2(a, b, alt_a, alt_b);
    dp2_align(ea, eb, std::umul(ea[0].sig, eb[0].sig), std::umul(ea[1].sig, eb[1].sig))
}

//============================================================================
// RO_48 and the product handed to the back end
//============================================================================

// Round a magnitude below 2^66 to 48 significant bits, to odd. A sum with at
// most 48 significant bits is returned as is (unnormalised: the back end
// aligns values); otherwise its leading one is in [65:48] and r = msb - 47.
// sticky_in is a part below the field (DP2), only ever present when the
// leading one is at bit 47 or above.
pub fn ro48(t: uN[66], sticky_in: u1) -> (u48, u5) {
    let hi = t[48+:u18];
    let r = u5:18 - (clz(hi) as u5);
    let kept = (t >> r) as u48;
    let lost = ((t & lsb_mask66(r)) != uN[66]:0) || sticky_in == u1:1;
    (kept | (lost as u48), r)
}

// The DP group as a product: |sum| rounded to odd, value = pm * 2^(xp - 46),
// xp = xpb - 254 = xpb0 - 254 + r.
fn dp_product(sum: sN[67], sticky: u1, xpb0: s12, sp: Spec) -> FmaProduct {
    let neg = sum < sN[67]:0;
    let t = (if neg { -sum } else { sum }) as uN[66];
    let (pm, r) = ro48(t, sticky);
    let t_zero = t == uN[66]:0;
    FmaProduct {
        pm: pm,
        xpb: xpb0 + (r as s12),
        sp: if sp.p_inf == u1:1 { sp.inf_sign } else if t_zero { sp.zsign } else { neg as u1 },
        p_zero: t_zero as u1,
        p_inf: sp.p_inf,
        p_nan: sp.p_nan,
    }
}

//============================================================================
// Lane stage functions: F1 (products and sum) | F2 (|sum|, RO_48, select)
//============================================================================

// What F1 hands F2 (V1, V2): the binary32 product, and the DP sum.
pub struct FdpMid {
    fp: FmaProduct,  // mode FP32
    dp: u1,          // a DP mode: F2 takes the group, not fp
    sum: sN[67],
    sticky: u1,
    xpb0: s12,       // the group's xp + 254 at r = 0
    sp: Spec,
}

// DP4: T * 2^-32 = (T >> r) * 2^(r - 32) = pm * 2^(xp - 46): xp = 14 + r
const DP4_XPB0 = s12:268;
// DP2: T * 2^(lb - 42) = pm * 2^(xp - 46): xp = lb + 4 + r
const DP2_XP_OFF = s12:4;

// V1: binary32 + DP4 (all FP8 pairs)
pub fn fdp_v1_f1(mode: u2, a: u32, b: u32, alt_a: u1, alt_b: u1, ftz: u1) -> FdpMid {
    let s = dp4_sum(a, b, alt_a, alt_b);
    FdpMid { fp: tnca_fma::fma32_mul(a, b, ftz), dp: (mode != MODE_FP32) as u1,
             sum: s.sum, sticky: u1:0, xpb0: DP4_XPB0, sp: s.sp }
}

// V2 with LUT multipliers: binary32 + DP4 + DP2. One signed adder serves both
// sums after a select of its terms -- written here as a select of the sums;
// the synthesis shares or duplicates as it finds cheaper.
pub fn fdp_v2l_f1(mode: u2, a: u32, b: u32, alt_a: u1, alt_b: u1, ftz: u1) -> FdpMid {
    let g4 = dp4_sum(a, b, alt_a, alt_b);
    let g2 = dp2_sum(a, b, alt_a, alt_b);
    let is2 = mode == MODE_DP2;
    FdpMid { fp: tnca_fma::fma32_mul(a, b, ftz), dp: (mode != MODE_FP32) as u1,
             sum: if is2 { g2.sum } else { g4.sum },
             sticky: if is2 { g2.sticky } else { u1:0 },
             xpb0: if is2 { g2.lb + DP2_XP_OFF + s12:254 } else { DP4_XPB0 },
             sp: if is2 { g2.sp } else { g4.sp } }
}

// V2 with the DSPs shared: the binary32 24x24 product as two 24x12 products,
// P = ma * mb[11:0] and Q = ma * mb[23:12] + (P >> 12) (the second DSP's
// post-adder), product = Q ++ P[11:0]. In DP2 mode each DSP takes one 11x11
// product and the post-adder input is zero. Identical binary32 product: the
// split is exact.
pub fn fdp_v2d_f1(mode: u2, a: u32, b: u32, alt_a: u1, alt_b: u1, ftz: u1) -> FdpMid {
    let is2 = mode == MODE_DP2;
    let (e2a, e2b) = dec2(a, b, alt_a, alt_b);

    // fma32_mul, with the multiplier re-split (the rest verbatim)
    let ea = a[23+:u8];
    let eb = b[23+:u8];
    let fa = a[0+:u23];
    let fb = b[0+:u23];
    let a_e0 = ea == u8:0;
    let b_e0 = eb == u8:0;
    let a_zero = a_e0 && (fa == u23:0 || ftz == u1:1);
    let b_zero = b_e0 && (fb == u23:0 || ftz == u1:1);
    let a_inf = ea == u8:0xff && fa == u23:0;
    let b_inf = eb == u8:0xff && fb == u23:0;
    let a_nan = ea == u8:0xff && fa != u23:0;
    let b_nan = eb == u8:0xff && fb != u23:0;
    let ma = (!a_e0) ++ fa;
    let mb = (!b_e0) ++ fb;
    let lo_a = if is2 { e2a[0].sig as u24 } else { ma };
    let lo_b = if is2 { e2b[0].sig as u12 } else { mb[0+:u12] };
    let hi_a = if is2 { e2a[1].sig as u24 } else { ma };
    let hi_b = if is2 { e2b[1].sig as u12 } else { mb[12+:u12] };
    let plo = std::umul(lo_a, lo_b);                                  // u36
    let phi = (std::umul(hi_a, hi_b) as u37) + (if is2 { u37:0 } else { (plo >> u36:12) as u37 });
    let pm_raw = ((phi as u48) << u48:12) | ((plo as u48) & u48:0xfff);
    let p_zero_mag = a_zero || b_zero;
    let p_dbl = (a_e0 as u2) + (b_e0 as u2);
    let pm = if p_zero_mag { u48:0 } else { pm_raw << p_dbl };
    let inf_times_zero = (a_inf && b_zero) || (b_inf && a_zero);
    let fp = FmaProduct {
        pm: pm,
        xpb: (ea as s12) + (eb as s12),
        sp: a[31+:u1] ^ b[31+:u1],
        p_zero: p_zero_mag as u1,
        p_inf: ((a_inf || b_inf) && !inf_times_zero) as u1,
        p_nan: (a_nan || b_nan || inf_times_zero) as u1,
    };

    let g2 = dp2_align(e2a, e2b, plo as u22, phi as u22);
    let g4 = dp4_sum(a, b, alt_a, alt_b);
    FdpMid { fp: fp, dp: (mode != MODE_FP32) as u1,
             sum: if is2 { g2.sum } else { g4.sum },
             sticky: if is2 { g2.sticky } else { u1:0 },
             xpb0: if is2 { g2.lb + DP2_XP_OFF + s12:254 } else { DP4_XPB0 },
             sp: if is2 { g2.sp } else { g4.sp } }
}

pub fn fdp_f2(m: FdpMid) -> FmaProduct {
    if m.dp == u1:0 { m.fp } else { dp_product(m.sum, m.sticky, m.xpb0, m.sp) }
}

// V3: binary32 + DP4 for E4M3 x E4M3 only. Field unit 2^-18, 38 bits: the sum
// is exact and has at most 38 significant bits, so RO_48 is the identity and
// there is no leading-zero count: pm = |sum| << 10, xp = 18.
pub struct FdpMid3 {
    fp: FmaProduct,
    dp: u1,
    sum: s39,
    sp: Spec,
}

fn term39(s: u1, f: u38) -> (u39, u39) {
    let m = if s == u1:1 { all_ones!<u39>() } else { u39:0 };
    ((f as u39) ^ m, s as u39)
}

fn dp4e_field(a: E8, b: E8) -> u38 {
    let p = std::umul(a.sig, b.sig);                // u8
    let pos = (a.q as u6) + (b.q as u6);            // 0..30, value = p * 2^(pos - 20)
    (((p as u40) << pos) >> u40:2) as u38
}

pub fn fdp_v3_f1(mode: u2, a: u32, b: u32, alt_a: u1, alt_b: u1, ftz: u1) -> FdpMid3 {
    let ea = [dec8_e4m3(a[0+:u8]), dec8_e4m3(a[8+:u8]), dec8_e4m3(a[16+:u8]), dec8_e4m3(a[24+:u8])];
    let eb = [dec8_e4m3(b[0+:u8]), dec8_e4m3(b[8+:u8]), dec8_e4m3(b[16+:u8]), dec8_e4m3(b[24+:u8])];
    let (t0, c0) = term39(ea[0].s ^ eb[0].s, dp4e_field(ea[0], eb[0]));
    let (t1, c1) = term39(ea[1].s ^ eb[1].s, dp4e_field(ea[1], eb[1]));
    let (t2, c2) = term39(ea[2].s ^ eb[2].s, dp4e_field(ea[2], eb[2]));
    let (t3, c3) = term39(ea[3].s ^ eb[3].s, dp4e_field(ea[3], eb[3]));
    let sum = t0 + t1 + t2 + t3 + (c0 + c1 + c2 + c3);
    FdpMid3 { fp: tnca_fma::fma32_mul(a, b, ftz), dp: (mode != MODE_FP32) as u1,
              sum: sum as s39, sp: spec4(ea, eb) }
}

pub fn fdp_v3_f2(m: FdpMid3) -> FmaProduct {
    let neg = m.sum < s39:0;
    let t = (if neg { -m.sum } else { m.sum }) as u38;
    let t_zero = t == u38:0;
    let dp = FmaProduct {
        pm: (t as u48) << u48:10,
        xpb: s12:272,                                // xp = 18
        sp: if t_zero { m.sp.zsign } else { neg as u1 },
        p_zero: t_zero as u1,
        p_inf: u1:0,
        p_nan: m.sp.p_nan,
    };
    if m.dp == u1:0 { m.fp } else { dp }
}

//============================================================================
// Whole lanes, unregistered: what every cut placement computes
//============================================================================

fn lane_back(p: FmaProduct, mode: u2, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> u32 {
    // a DP mode has no mul_only and no ftz
    let fp = mode == MODE_FP32;
    let mo = if fp { mul_only } else { u1:0 };
    let fz = if fp { ftz } else { u1:0 };
    tnca_fma::fma32_round(tnca_fma::fma32_normalise(tnca_fma::fma32_align_p(p, c, sub_c, mo, fz)))
}

// a, b: in FP32 mode the binary32 operands (already widened); in a DP mode
// the packed elements, lowest k in the low bits.
pub fn lane_v1(mode: u2, a: u32, b: u32, alt_a: u1, alt_b: u1, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> u32 {
    let fz = if mode == MODE_FP32 { ftz } else { u1:0 };
    lane_back(fdp_f2(fdp_v1_f1(mode, a, b, alt_a, alt_b, fz)), mode, c, sub_c, mul_only, ftz)
}

pub fn lane_v2l(mode: u2, a: u32, b: u32, alt_a: u1, alt_b: u1, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> u32 {
    let fz = if mode == MODE_FP32 { ftz } else { u1:0 };
    lane_back(fdp_f2(fdp_v2l_f1(mode, a, b, alt_a, alt_b, fz)), mode, c, sub_c, mul_only, ftz)
}

pub fn lane_v2d(mode: u2, a: u32, b: u32, alt_a: u1, alt_b: u1, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> u32 {
    let fz = if mode == MODE_FP32 { ftz } else { u1:0 };
    lane_back(fdp_f2(fdp_v2d_f1(mode, a, b, alt_a, alt_b, fz)), mode, c, sub_c, mul_only, ftz)
}

pub fn lane_v3(mode: u2, a: u32, b: u32, alt_a: u1, alt_b: u1, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> u32 {
    let fz = if mode == MODE_FP32 { ftz } else { u1:0 };
    lane_back(fdp_v3_f2(fdp_v3_f1(mode, a, b, alt_a, alt_b, fz)), mode, c, sub_c, mul_only, ftz)
}

//============================================================================
// Tests (expected values from tools/model/fdp.py)
//============================================================================

#[test]
fn e4m3_small_sum() {
    // [1, 2] . [1, 1] = 3
    let y = lane_v1(MODE_DP4, u32:0x4038, u32:0x3838, u1:0, u1:0, u32:0, u1:0, u1:0, u1:0);
    assert_eq(y, u32:0x40400000);
    assert_eq(lane_v3(MODE_DP4, u32:0x4038, u32:0x3838, u1:0, u1:0, u32:0, u1:0, u1:0, u1:0), y);
    assert_eq(lane_v2l(MODE_DP4, u32:0x4038, u32:0x3838, u1:0, u1:0, u32:0, u1:0, u1:0, u1:0), y);
    assert_eq(lane_v2d(MODE_DP4, u32:0x4038, u32:0x3838, u1:0, u1:0, u32:0, u1:0, u1:0, u1:0), y);
}

#[test]
fn fp32_mode_is_the_fma() {
    let a = u32:0x3fc00001;
    let b = u32:0xc0490fdb;
    let c = u32:0x3f800000;
    let want = tnca_fma::fma32(a, b, c, u1:0, u1:0, u1:0);
    assert_eq(lane_v1(MODE_FP32, a, b, u1:0, u1:0, c, u1:0, u1:0, u1:0), want);
    assert_eq(lane_v2l(MODE_FP32, a, b, u1:0, u1:0, c, u1:0, u1:0, u1:0), want);
    assert_eq(lane_v2d(MODE_FP32, a, b, u1:0, u1:0, c, u1:0, u1:0, u1:0), want);
    assert_eq(lane_v3(MODE_FP32, a, b, u1:0, u1:0, c, u1:0, u1:0, u1:0), want);
    // the binary32 subnormal/ftz/mul_only cases go through unchanged
    let tiny = u32:0x00000001;
    assert_eq(lane_v2d(MODE_FP32, u32:0x3f800000, tiny, u1:0, u1:0, u32:0, u1:0, u1:0, u1:1), u32:0);
    assert_eq(lane_v2d(MODE_FP32, u32:0x40000000, u32:0x40000000, u1:0, u1:0, u32:0x7fc00001, u1:0, u1:1, u1:0), u32:0x40800000);
}

// (mode, a, b, c, alt_a, alt_b) -> y, every variant that supports the case
fn check(mode: u2, a: u32, b: u32, c: u32, alt_a: u1, alt_b: u1, want: u32) -> () {
    if mode == MODE_DP4 {
        assert_eq(lane_v1(mode, a, b, alt_a, alt_b, c, u1:0, u1:0, u1:0), want);
    } else { };
    assert_eq(lane_v2l(mode, a, b, alt_a, alt_b, c, u1:0, u1:0, u1:0), want);
    assert_eq(lane_v2d(mode, a, b, alt_a, alt_b, c, u1:0, u1:0, u1:0), want);
    if mode == MODE_DP4 && alt_a == u1:0 && alt_b == u1:0 {
        assert_eq(lane_v3(mode, a, b, alt_a, alt_b, c, u1:0, u1:0, u1:0), want);
    } else { };
}

#[test]
fn round_to_odd_is_visible() {
    // E5M2: 57344^2 + 2^-32 - 57344^2. RO_48 keeps 2^-16 as the odd bit;
    // exact rounding would give 2^-32 (0x2f800000).
    check(MODE_DP4, u32:0x017b, u32:0x017b, u32:0xcf440000, u1:1, u1:1, u32:0x37800000);
    // FP16: 65504^2 - 2^-48 (sticky, borrow) - 65504^2: -2^-16 (exact -2^-48)
    check(MODE_DP2, u32:0x80017bff, u32:0x00017bff, u32:0xcf7fc004, u1:0, u1:0, u32:0xb7800000);
    check(MODE_DP2, u32:0x00017bff, u32:0x00017bff, u32:0xcf7fc004, u1:0, u1:0, u32:0x37800000);
    // BF16: 2^127 + 2^-266 (sticky, same sign) - 2^127: 2^80 (exact: 0)
    check(MODE_DP2, u32:0x00017f00, u32:0x00013f80, u32:0xff000000, u1:1, u1:1, u32:0x67800000);
}

#[test]
fn specials() {
    check(MODE_DP4, u32:0x7f003838, u32:0x38383838, u32:0, u1:0, u1:0, u32:0x7fc00000);          // E4M3 NaN
    check(MODE_DP4, u32:0x0000007c, u32:0x00000000, u32:0x3f800000, u1:1, u1:1, u32:0x7fc00000);  // inf * 0
    check(MODE_DP4, u32:0x3c3c3c7c, u32:0x3c3c3c3c, u32:0x3f800000, u1:1, u1:1, u32:0x7f800000);  // inf + finite
    check(MODE_DP4, u32:0x0000fc7c, u32:0x00003c3c, u32:0, u1:1, u1:1, u32:0x7fc00000);          // +inf - inf
    check(MODE_DP4, u32:0x0000007c, u32:0x0000003c, u32:0xff800000, u1:1, u1:1, u32:0x7fc00000);  // inf - inf (C)
    check(MODE_DP4, u32:0x80808080, u32:0x00000000, u32:0x80000000, u1:0, u1:0, u32:0x80000000);  // -0 sum, -0 C
    check(MODE_DP4, u32:0x0000b838, u32:0x00003838, u32:0x80000000, u1:0, u1:0, u32:0x00000000);  // cancel: +0
}

#[test]
fn ranges_and_formats() {
    check(MODE_DP2, u32:0xff007f00, u32:0x3f804000, u32:0, u1:1, u1:1, u32:0x7f000000);   // 2^128 - 2^127, fused
    check(MODE_DP2, u32:0x00007f00, u32:0x00007f00, u32:0, u1:1, u1:1, u32:0x7f800000);   // 2^254: inf
    check(MODE_DP2, u32:0x00000080, u32:0x00003580, u32:0, u1:1, u1:1, u32:0x00000008);   // 2^-146 subnormal
    check(MODE_DP2, u32:0xbc000001, u32:0x3c000001, u32:0x3f800000, u1:0, u1:0, u32:0x27800000); // 1 + (2^-48 - 1)
    check(MODE_DP2, u32:0x3c003c00, u32:0x3f804000, u32:0x3f800000, u1:0, u1:1, u32:0x40800000); // FP16 x BF16
    check(MODE_DP4, u32:0x38383838, u32:0x3c3c3c3c, u32:0, u1:0, u1:1, u32:0x40800000);   // E4M3 x E5M2
    check(MODE_DP4, u32:0x7e7e7e7e, u32:0x7e7e7e7e, u32:0, u1:0, u1:0, u32:0x49440000);   // 4 * 448^2
    check(MODE_DP4, u32:0x00000001, u32:0x00000001, u32:0, u1:1, u1:1, u32:0x2f800000);   // 2^-32
}
