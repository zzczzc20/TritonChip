// Shared multi-precision matrix MAC lane (R41 G2, author's direction of
// 2026-10-29): one datapath per lane that does, per cycle,
//
//   mode 0  FP32:  1 MAC -- C <- RNE_f32(exact(a) * exact(b) + C), the matrix
//                  lane of today (tnca_fpu_multi: widen by fmt, then the
//                  binary32 FMA), bit for bit;
//   mode 1  DP2:   2 MACs -- C <- RNE_f32(C + RO_48(a0*b0 + a1*b1)), each
//                  operand word holding two 16-bit slots, each element FP16,
//                  BF16, E5M2 or E4M3 (an 8-bit element in its slot's low
//                  byte);
//   mode 2  DP4:   4 MACs -- C <- RNE_f32(C + sum of four E4M3 x E4M3
//                  products), exact (RO_48 is the identity there).
//
// The packed modes are the proposed m.mma.dp (doc/isa, R41 G2 proposal):
// a group of G products is summed exactly, rounded to odd at 48 bits, and
// added to C with the binary32 FMA's single rounding. tools/model/fdp.py is
// the exact oracle (lane_mac_ref).
//
// STRUCTURE
//
// Every mode forms one "product" -- a 48-bit significand with an exponent
// (tnca_fma::FmaProduct) -- and hands it to the binary32 FMA's back end,
// which is shared and unchanged in function: tnca_fma::fma32_align_p,
// tnca_fmad::fma32d_normalise (proved equal to tnca_fma::fma32_normalise),
// tnca_fma::fma32_round. Mode 0's product is tnca_fma::fma32_mul, so mode 0
// is tnca_fma::fma32 of the widened operands.
//
// Where the work goes (the area aim: the shared lane costs no more than
// today's FP32-only lane):
//   - DP4 (E4M3): each element is a small fixed-point integer (a significand
//     shifted by its exponent: E4M3 values are multiples of 2^-9 below 448),
//     so each product is one 20 x 17-bit integer multiply and the four sum
//     exactly in a DSP48E2 cascade: four DSPs and almost no LUTs. The split
//     of the exponent between the two multiplier ports keeps the B operand
//     inside the DSP's 18-bit signed port (below).
//   - DP2: two 11 x 11 products (DSPs), the smaller-scale one aligned onto
//     the larger in a 64-bit field with a sticky bit (tnca_fdp's dp2_align),
//     RO_48 with a tree leading-zero count.
//   - The back end's 89-bit leading-zero count is the tree of tnca_lzc.
//
// Stage functions (rtl/core/tnca_mac_lane.v registers them): F1 (decode,
// products, group sums) | F2 (|sum|, RO_48, mode select) | B0 align | B1
// normalise | B2 round.

import std;
import tnca_fma;
import tnca_fmad;
import tnca_fdp;
import tnca_lzc;

pub const MODE_FP32 = u2:0;
pub const MODE_DP2 = u2:1;
pub const MODE_DP4 = u2:2;

// DP2 element codes (per operand)
pub const F16 = u2:0;
pub const BF16 = u2:1;
pub const E5M2 = u2:2;
pub const E4M3 = u2:3;

//============================================================================
// DP4: E4M3 x E4M3 through a DSP cascade
//============================================================================

// An E4M3 element as an integer significand m and a shift s: value =
// m * 2^s * 2^-9, m < 16, s <= 14 (exponent field 15 with fraction 7 is NaN,
// so 15 * 2^14 never occurs: |value| * 2^9 <= 14 * 2^14 < 2^18).
fn e4m3_ms(x: u8) -> (u4, u4) {
    let e = x[3+:u4];
    let f = x[0+:u3];
    if e == u4:0 { (u1:0 ++ f, u4:0) } else { (u1:1 ++ f, e - u4:1) }
}

// One product as two DSP operands with a * b * 2^18 = A * B exactly: B keeps
// at most 13 of the shift (< 2^17, inside the 18-bit signed port), A takes
// the rest (< 2^20) and the product's sign.
fn dp4_operands(a: u8, b: u8) -> (s27, s18, s48) {
    let (ma, sa) = e4m3_ms(a);
    let (mb, sb) = e4m3_ms(b);
    let sb2 = if sb > u4:13 { u4:13 } else { sb };
    let sa2 = (sa as u5) + ((sb - sb2) as u5);
    let afix = ((ma as u20) << sa2) as s27;
    let bfix = ((mb as u17) << sb2) as s18;
    let neg = a[7+:u1] ^ b[7+:u1];
    // -x = !x + 1: the multiplier takes !x (one XOR level in front of the DSP,
    // no carry chain), and !x * B = -x*B - B, so +B is added back after, with
    // the product, where the DSP's C port takes it.
    (if neg == u1:1 { !afix } else { afix }, bfix, if neg == u1:1 { bfix as s48 } else { s48:0 })
}

pub struct Dp4Fix {
    sum: s48,              // the exact group sum * 2^18; |sum| < 2^38
    sp: tnca_fdp::Spec,
}

pub fn dp4_e4m3(a: u32, b: u32) -> Dp4Fix {
    let (a0, b0, k0) = dp4_operands(a[0+:u8], b[0+:u8]);
    let (a1, b1, k1) = dp4_operands(a[8+:u8], b[8+:u8]);
    let (a2, b2, k2) = dp4_operands(a[16+:u8], b[16+:u8]);
    let (a3, b3, k3) = dp4_operands(a[24+:u8], b[24+:u8]);
    // a cascade: ((p0 + p1) + p2) + p3, each product a 27 x 18 multiply
    let p0 = (std::smul(a0, b0) as s48) + k0;
    let p1 = (std::smul(a1, b1) as s48) + k1;
    let p2 = (std::smul(a2, b2) as s48) + k2;
    let p3 = (std::smul(a3, b3) as s48) + k3;
    let ea = [tnca_fdp::dec8_e4m3(a[0+:u8]), tnca_fdp::dec8_e4m3(a[8+:u8]),
              tnca_fdp::dec8_e4m3(a[16+:u8]), tnca_fdp::dec8_e4m3(a[24+:u8])];
    let eb = [tnca_fdp::dec8_e4m3(b[0+:u8]), tnca_fdp::dec8_e4m3(b[8+:u8]),
              tnca_fdp::dec8_e4m3(b[16+:u8]), tnca_fdp::dec8_e4m3(b[24+:u8])];
    Dp4Fix { sum: ((p0 + p1) + p2) + p3, sp: tnca_fdp::spec4(ea, eb) }
}

//============================================================================
// DP2: two 16-bit-slot elements per operand
//============================================================================

// E4M3 as a 16-bit-class element: value = sig * 2^l with an 11-bit
// significand (the 4-bit E4M3 significand at the top, like BF16's).
fn e4m3_as_e16(x: u8) -> tnca_fdp::E16 {
    let e = tnca_fdp::dec8_e4m3(x);                // value = sig * 2^(q - 10)
    tnca_fdp::E16 { s: e.s, sig: (e.sig as u11) << u11:7, l: (e.q as s12) - s12:17,
                    zero: e.zero, inf: e.inf, nan: e.nan }
}

pub fn dec16x(x: u16, code: u2) -> tnca_fdp::E16 {
    if code == F16 { tnca_fdp::dec16(x, u1:0) }
    else if code == BF16 { tnca_fdp::dec16(x, u1:1) }
    // E5M2 is FP16's top byte: the same value, exactly
    else if code == E5M2 { tnca_fdp::dec16(x[0+:u8] ++ u8:0, u1:0) }
    else { e4m3_as_e16(x[0+:u8]) }
}

// tnca_fdp::dp2_align with one adder. The original sums three terms and three
// carries (the reference, the shifted smaller product, a borrow unit) in a
// 67-bit field: sum = sb*B + ss*Ssh + (borrow ? ss : 0), sx*v meaning -v for
// sx = 1. Here the reference is only 22 bits at [63:42], so its signed value
// is a small negation (22 bits) with zeros below, and -(Ssh + 1) = ~Ssh folds
// the borrow into the one carry-in: sum = tb + (ss ? ~Ssh : Ssh) + (ss ^ borrow).
// dp2_x_is_ref proves it equal to tnca_fdp::dp2_align for every operand pair.
pub struct Dp2Mid {
    pb: u22,                 // the reference product (larger scale)
    ps: u22,                 // the smaller
    d: u7,                   // the scale difference, capped at 64
    sb: u1,
    ss: u1,
    lb: s12,
    sp: tnca_fdp::Spec,
}

// Products' scales and the reference choice (before the shift).
pub fn dp2_pre(ea: tnca_fdp::E16[2], eb: tnca_fdp::E16[2], p0: u22, p1: u22) -> Dp2Mid {
    let l0 = ea[0].l + eb[0].l;
    let l1 = ea[1].l + eb[1].l;
    let z0 = ea[0].zero == u1:1 || eb[0].zero == u1:1;
    let z1 = ea[1].zero == u1:1 || eb[1].zero == u1:1;
    let sg0 = ea[0].s ^ eb[0].s;
    let sg1 = ea[1].s ^ eb[1].s;
    let ref1 = if z0 { true } else if z1 { false } else { l1 > l0 };
    let d01 = l0 - l1;
    let d10 = l1 - l0;
    let dabs = if ref1 { d10 } else { d01 };
    Dp2Mid {
        pb: if ref1 { p1 } else { p0 },
        ps: if ref1 { p0 } else { p1 },
        d: if dabs > s12:64 { u7:64 } else { dabs as u7 },
        sb: if ref1 { sg1 } else { sg0 },
        ss: if ref1 { sg0 } else { sg1 },
        lb: if ref1 { l1 } else { l0 },
        sp: tnca_fdp::spec2(ea, eb),
    }
}

// The shift of the smaller product and the one addition.
pub fn dp2_post(m: Dp2Mid) -> tnca_fdp::Dp2Sum {
    let sf = (m.ps as u64) << u64:42;
    let sh = sf >> m.d;
    let sticky = (sf & tnca_fdp::lsb_mask64(m.d)) != u64:0;
    let borrow = sticky && (m.sb != m.ss);
    let pbs = m.pb as sN[25];
    let tbh = if m.sb == u1:1 { -pbs } else { pbs };
    let tb = (tbh as sN[67]) << u32:42;
    let ts = if m.ss == u1:1 { !(sh as uN[67]) } else { sh as uN[67] };
    let cs = (m.ss ^ (borrow as u1)) as uN[67];
    let sum = (tb as uN[67]) + ts + cs;
    tnca_fdp::Dp2Sum { sum: sum as sN[67], sticky: sticky as u1, lb: m.lb, sp: m.sp }
}

pub fn dp2_align1(ea: tnca_fdp::E16[2], eb: tnca_fdp::E16[2], p0: u22, p1: u22) -> tnca_fdp::Dp2Sum {
    dp2_post(dp2_pre(ea, eb, p0, p1))
}

fn dp2_decode(a: u32, b: u32, fa: u2, fb: u2) -> (tnca_fdp::E16[2], tnca_fdp::E16[2]) {
    ([dec16x(a[0+:u16], fa), dec16x(a[16+:u16], fa)],
     [dec16x(b[0+:u16], fb), dec16x(b[16+:u16], fb)])
}

pub fn dp2_x(a: u32, b: u32, fa: u2, fb: u2) -> tnca_fdp::Dp2Sum {
    let (ea, eb) = dp2_decode(a, b, fa, fb);
    dp2_align1(ea, eb, std::umul(ea[0].sig, eb[0].sig), std::umul(ea[1].sig, eb[1].sig))
}

// The original (three adders), the reference of dp2_x_is_ref.
pub fn dp2_x_orig(a: u32, b: u32, fa: u2, fb: u2) -> tnca_fdp::Dp2Sum {
    let (ea, eb) = dp2_decode(a, b, fa, fb);
    tnca_fdp::dp2_align(ea, eb, std::umul(ea[0].sig, eb[0].sig), std::umul(ea[1].sig, eb[1].sig))
}

//============================================================================
// RO_48 and the DP2 product (tnca_fdp's, with the tree leading-zero count)
//============================================================================

fn lsb_mask66(n: u5) -> uN[66] { !(all_ones!<uN[66]>() << n) }

fn ro48(t: uN[66], sticky_in: u1) -> (u48, u5) {
    let hi = t[48+:u18];
    let r = u5:18 - (tnca_lzc::lzc(hi) as u5);
    let kept = (t >> r) as u48;
    let lost = ((t & lsb_mask66(r)) != uN[66]:0) || sticky_in == u1:1;
    (kept | (lost as u48), r)
}

// DP2: T * 2^(lb - 42) = pm * 2^(xp - 46): xp = lb + 4 + r
const DP2_XP_OFF = s12:4;

fn dp2_product(g: tnca_fdp::Dp2Sum) -> tnca_fma::FmaProduct {
    let neg = g.sum < sN[67]:0;
    let t = (if neg { -g.sum } else { g.sum }) as uN[66];
    let (pm, r) = ro48(t, g.sticky);
    let t_zero = t == uN[66]:0;
    tnca_fma::FmaProduct {
        pm: pm,
        xpb: g.lb + DP2_XP_OFF + s12:254 + (r as s12),
        sp: if g.sp.p_inf == u1:1 { g.sp.inf_sign } else if t_zero { g.sp.zsign } else { neg as u1 },
        p_zero: t_zero as u1,
        p_inf: g.sp.p_inf,
        p_nan: g.sp.p_nan,
    }
}

// DP4 (E4M3, exact): T * 2^-18 = pm * 2^(xp - 46) with pm = T << 10: xp = 18
fn dp4_product(g: Dp4Fix) -> tnca_fma::FmaProduct {
    let neg = g.sum < s48:0;
    let t = (if neg { -g.sum } else { g.sum }) as u38;
    let t_zero = t == u38:0;
    tnca_fma::FmaProduct {
        pm: (t as u48) << u48:10,
        xpb: s12:272,
        sp: if t_zero { g.sp.zsign } else { neg as u1 },
        p_zero: t_zero as u1,
        p_inf: u1:0,                              // E4M3 has no infinity
        p_nan: g.sp.p_nan,
    }
}

//============================================================================
// Stage functions
//============================================================================

pub struct MacMid1 {
    mode: u2,
    fp: tnca_fma::FmaProduct,   // mode 0
    d4: Dp4Fix,                 // mode 2
    d2: Dp2Mid,                 // mode 1
}

pub struct MacMid {
    mode: u2,
    fp: tnca_fma::FmaProduct,   // mode 0
    d4: Dp4Fix,                 // mode 2
    d2: tnca_fdp::Dp2Sum,       // mode 1
}

// S1: decode, the products (and DP4's whole sum), the DP2 reference choice.
// In mode 0, a and b are already widened to binary32 by the lane wrapper
// (as tnca_fpu_multi does).
pub fn mac_s1(mode: u2, a: u32, b: u32, fa: u2, fb: u2, ftz: u1) -> MacMid1 {
    let (ea, eb) = dp2_decode(a, b, fa, fb);
    MacMid1 { mode: mode, fp: tnca_fma::fma32_mul(a, b, ftz), d4: dp4_e4m3(a, b),
              d2: dp2_pre(ea, eb, std::umul(ea[0].sig, eb[0].sig), std::umul(ea[1].sig, eb[1].sig)) }
}

// S2: DP2's shift and addition; the other modes' results pass.
pub fn mac_s2(m: MacMid1) -> MacMid {
    MacMid { mode: m.mode, fp: m.fp, d4: m.d4, d2: dp2_post(m.d2) }
}

// F1 = S1 then S2 (the two stages unregistered).
pub fn mac_f1(mode: u2, a: u32, b: u32, fa: u2, fb: u2, ftz: u1) -> MacMid {
    mac_s2(mac_s1(mode, a, b, fa, fb, ftz))
}

// F2: the mode's product.
pub fn mac_f2(m: MacMid) -> tnca_fma::FmaProduct {
    if m.mode == MODE_DP2 { dp2_product(m.d2) }
    else if m.mode == MODE_DP4 { dp4_product(m.d4) }
    else { m.fp }
}

// B0: alignment with C. A packed mode has no mul_only and no ftz.
pub fn mac_b0(p: tnca_fma::FmaProduct, mode: u2, c: u32, sub_c: u1, mul_only: u1, ftz: u1)
    -> tnca_fma::FmaAligned {
    let fp = mode == MODE_FP32;
    tnca_fma::fma32_align_p(p, c, sub_c, if fp { mul_only } else { u1:0 }, if fp { ftz } else { u1:0 })
}

// The whole lane, unregistered (a, b widened in mode 0).
pub fn mac_lane(mode: u2, a: u32, b: u32, fa: u2, fb: u2, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> u32 {
    let p = mac_f2(mac_f1(mode, a, b, fa, fb, ftz));
    tnca_fma::fma32_round(tnca_fmad::fma32d_normalise(mac_b0(p, mode, c, sub_c, mul_only, ftz)))
}

//============================================================================
// Checks
//============================================================================

// Mode 0 is the binary32 FMA (on widened operands), for every input.
#[quickcheck]
fn fp32_mode_is_fma32(a: u32, b: u32, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> bool {
    mac_lane(MODE_FP32, a, b, u2:0, u2:0, c, sub_c, mul_only, ftz) ==
        tnca_fma::fma32(a, b, c, sub_c, mul_only, ftz)
}

// The DSP fixed-point E4M3 group sum equals tnca_fdp's LUT field sum (V3,
// verified against the oracle), for every pair of operand words.
#[quickcheck]
fn dp4_sum_is_v3(a: u32, b: u32) -> bool {
    let v3 = tnca_fdp::fdp_v3_f1(MODE_DP4, a, b, u1:0, u1:0, u1:0);
    let mine = dp4_e4m3(a, b);
    (mine.sum as s39) == v3.sum && mine.sp.p_nan == v3.sp.p_nan && mine.sp.zsign == v3.sp.zsign
}

// The same, one product at a time (16 input bits: the solver proves this
// in seconds where the four-product form above times out). Each DSP product
// A * B is V3's signed field term; both group sums are the plain sum of their
// four terms (|sum| < 2^38: no wrap in s48 or in V3's s39), so this lemma and
// the shared spec4 give dp4_sum_is_v3.
#[quickcheck(exhaustive)]
fn dp4_term_is_v3(a: u8, b: u8) -> bool {
    let (x, y, k) = dp4_operands(a, b);
    let ea = tnca_fdp::dec8_e4m3(a);
    let eb = tnca_fdp::dec8_e4m3(b);
    let f = tnca_fdp::dp4e_field(ea, eb) as s48;
    let want = if (ea.s ^ eb.s) == u1:1 { -f } else { f };
    ((std::smul(x, y) as s48) + k) == want
}

// DP2 with FP16/BF16 codes is tnca_fdp's DP2 (verified against the oracle).
#[quickcheck]
fn dp2_fp16_bf16_is_v2(a: u32, b: u32, fa: u1, fb: u1) -> bool {
    let mine = dp2_x(a, b, fa as u2, fb as u2);
    let v2 = tnca_fdp::dp2_sum(a, b, fa, fb);
    mine == v2
}

// The one-adder DP2 sum equals the original for every operand pair and every
// element code (all four codes per operand).
#[quickcheck]
fn dp2_x_is_ref(a: u32, b: u32, fa: u2, fb: u2) -> bool {
    dp2_x(a, b, fa, fb) == dp2_x_orig(a, b, fa, fb)
}

#[test]
fn small_cases() {
    // DP4: 1*1 + 1*1 + 1*1 + 1*1 = 4 (E4M3 1.0 = 0x38), C = 0
    assert_eq(mac_lane(MODE_DP4, u32:0x38383838, u32:0x38383838, u2:0, u2:0, u32:0, u1:0, u1:0, u1:0), u32:0x40800000);
    // DP2 FP16: 1*2 + 3*1 = 5 (+0)
    assert_eq(mac_lane(MODE_DP2, u32:0x42003c00, u32:0x3c004000, F16, F16, u32:0, u1:0, u1:0, u1:0), u32:0x40a00000);
    // DP2 E4M3 in 16-bit slots: 1*1 + 1*1 = 2
    assert_eq(mac_lane(MODE_DP2, u32:0x00380038, u32:0x00380038, E4M3, E4M3, u32:0, u1:0, u1:0, u1:0), u32:0x40000000);
    // DP2 E5M2 (1.0 = 0x3c): 1*1 + 1*1 = 2
    assert_eq(mac_lane(MODE_DP2, u32:0x003c003c, u32:0x003c003c, E5M2, E5M2, u32:0, u1:0, u1:0, u1:0), u32:0x40000000);
}
