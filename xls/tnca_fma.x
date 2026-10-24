// binary32 fused multiply-add, with subnormals -- generated, not hand-pipelined.
//
//     y = RNE( a * b + (sub_c ? -c : c) )     single rounding
//
// mul_only = 1 returns RNE(a * b) and ignores c entirely; ftz = 1 flushes
// subnormal inputs and results to a zero of the same sign, as an explicit
// mode (ISA sec 8.2, 12.4, 12.7). This is the arithmetic of every FP result in
// the core: the vector FPU builds vf.add/vf.sub from it with b = 1.0, and the
// matrix unit accumulates through it (tnca_fpu_multi).
//
// WHY THIS IS NOT apfloat::fma
//
// XLS's apfloat library flushes subnormals to zero (measured: apfloat::fma(
// +0.0, -0.0, 1e-45) -> 0.0, want 1e-45), and this ISA forbids that outside
// the ftz mode -- which is why the FMA used to be hand-written Verilog with
// two hand-placed register stages (doc/xls.md). What XLS does not get wrong
// is everything else: the schedule. So the arithmetic is written here, once,
// with subnormal inputs and outputs handled the way rtl/core/tnca_fp32_fma.v
// handled them, and XLS places the pipeline registers for whatever clock or
// stage count the target needs (tools/xls/config.json). A faster clock is a
// regeneration, not a redesign. The exception since R41 Stage 2 is the
// latency-2 instance, whose two cuts are placed for the FPGA (below).
//
// The algorithm is the hand-written one, so the two can be compared bit for
// bit and the exact Fraction oracle that verified one verifies the other
// (tb_fp32_fma, fma-deep: 400 000 vectors). Since R41 Stage 2 some steps are
// re-expressed for the FPGA (below); the function is the same, bit for bit:
//
//   unpack, classify; 24x24 product P; addend MC = mc << 23, both at scale
//   2^(e-46); shift the one with the smaller exponent right onto the other's
//   scale with a sticky bit, in an 88-bit field
//       [87:80] carry headroom  [79:32] operands  [31:0] guard/sticky
//   add or subtract (comparing aligned magnitudes, since equal exponents do
//   not order the significands); leading-one search; normalise or denormalise
//   to 24 bits plus guard and sticky; round to nearest even; pack.
//
// WHERE THE CUTS FALL ON AN FPGA (R41 Stage 2)
//
// The function is written as three stage functions -- fma32_align,
// fma32_normalise, fma32_round -- and fma32 is the three composed. The
// latency-2 instance (the image's FP_LAT) is the three generated
// combinational and registered by rtl/core/tnca_fp32_fma.v at the two cuts,
// FmaAligned and FmaNormalised: the cuts the U280 unit OOC measures as
// balanced. XLS schedules against asap7, which prices the 24x24 multiplier
// (two DSP48E2 slices on the FPGA) at several times its FPGA cost; its own
// three-stage schedule put the alignment shifter in the middle stage once the
// rest was shortened. The latency-3 instance is still fma32 scheduled by XLS.
//
// The arithmetic is written so that each stage is also short on the U280,
// without changing a single result bit. Each equivalence is spelled out where
// it is used, and the exact oracle (tb_fp32_fma, fma-deep) and a miter against
// the previous generated RTL (evidence/r41/stage2/candidates/F03/miter) check
// them.
//
//   F03, the first stage's inputs. Exponent fields are used as stored, a
//     subnormal's fraction being doubled instead of its exponent raised to 1,
//     so the exponent difference, its magnitude (two subtractions side by
//     side) and the alignment amount never wait on a zero test. The multiplier
//     takes the raw fraction fields with the hidden bit (exponent != 0); a
//     subnormal factor's doubling and an ftz-flushed factor's zero are applied
//     to the product after the DSP, where they fold into the alignment select.
//     Before, five LUT levels of zero detection and ftz decode sat between the
//     operand registers and the DSP, and a zero test fed the exponent adders.
//   F09, the last stage (combinational into the consumer's register: the
//     vector writeback, the accumulator file). One right shift of mag with a
//     zero appended yields both the significand and the guard bit; the sticky
//     mask is taken from the shift amount itself, with no decrement in front;
//     rounding is carry-select (sig + 1 and the all-ones test are formed beside
//     the decision), and so are the biased exponent and the overflow test.
//   F08, the middle stage. The borrow of the shifted-out sticky bit selects
//     between big - small and big - small - 1, formed side by side, instead of
//     a second 89-bit carry chain behind the latest bit of the first stage.
//     Everything after the leading-zero count is one compare or one
//     subtraction from it: the subnormal test, the biased exponent and both
//     normalising shifts are rewritten in terms of lz and exp_common, and the
//     subnormal shift does not depend on lz at all.

const QNAN = u32:0x7fc00000;
const E_MIN = s12:-126;

fn lsb_mask88(n: u8) -> uN[88] { !(all_ones!<uN[88]>() << n) }
fn lsb_mask90(n: u12) -> uN[90] { !(all_ones!<uN[90]>() << n) }

// The two cuts of the latency-2 instance (rtl/core/tnca_fp32_fma.v registers
// them; see "WHERE THE CUTS FALL"): what the alignment stage hands the
// add/normalise stage, and what that hands the rounding stage. fma32 below is
// the three stages composed, unregistered: the function every instance
// computes, and the one the latency-3 instance is scheduled from by XLS.
pub struct FmaAligned {
    small_sh: uN[80],    // the smaller-exponent operand's field, shifted onto the other's scale
    big_hi: u48,         // the larger-exponent operand's field, bits [79:32]; [31:0] are 0
    sticky_al: u1,       // bits of the smaller operand shifted out below bit 0
    s_big: u1,
    s_small: u1,
    sp: u1,
    sc: u1,
    exp_common: s12,     // value(field) = field * 2^(exp_common - 78)
    nan_res: u1,
    inf_res: u1,
    inf_sign: u1,
    mul_only: u1,
    ftz: u1,
}

pub struct FmaNormalised {
    mag: uN[81],         // the sum's magnitude, below 2^81; the shifted-out eps is sticky_al
    any: u1,             // mag != 0
    subnormal_res: u1,
    shn_left: u1,        // the normalising shift if the result is normal ...
    shn_mag: u7,
    shs_left: u1,        // ... and if it is subnormal
    shs_mag: u12,
    eb0: s12,            // the biased exponent before rounding
    sticky_al: u1,
    res_sign: u1,
    nan_res: u1,
    inf_res: u1,
    inf_sign: u1,
    ftz: u1,
}

// What the first stage's multiplier side hands its alignment side: the exact
// product as a 48-bit significand, an exponent and the product's own special
// cases. fma32_mul forms it from two binary32 factors; the packed dot-product
// lane (xls/tnca_fdp.x, R41 feasibility study) forms one from a group of
// narrow products, rounded to odd at 48 bits, with an exponent range beyond
// binary32's. fma32_align_p takes either, so both share align, normalise and
// round. Any pm in [0, 2^48) is valid, normalised or not, and xp may be any
// value whose sums below stay in s12 (|xp| < ~700): the alignment works on
// values, not on encodings (see "WHY THE RESULT IS UNCHANGED" in F03).
pub struct FmaProduct {
    pm: u48,          // value = pm * 2^(xp - 46), xp = xpb - 254
    xpb: s12,         // the product's exponent biased by 254: ea + eb for binary32 factors
    sp: u1,           // its sign, and the sign of an infinite product
    p_zero: u1,       // the product is zero: it is never the reference operand
    p_inf: u1,
    p_nan: u1,        // a NaN factor, inf * 0 (or, in a dot product, inf - inf)
}

// First stage, multiplier side: unpack and classify a and b, multiply.
pub fn fma32_mul(a: u32, b: u32, ftz: u1) -> FmaProduct {

    //------------------------------------------------------------------ unpack
    let sa = a[31+:u1];
    let sb = b[31+:u1];
    let ea = a[23+:u8];
    let eb = b[23+:u8];
    let fa = a[0+:u23];
    let fb = b[0+:u23];

    let a_e0 = ea == u8:0;
    let b_e0 = eb == u8:0;

    // ftz flushes subnormal inputs before anything else looks at them
    let a_zero = a_e0 && (fa == u23:0 || ftz == u1:1);
    let b_zero = b_e0 && (fb == u23:0 || ftz == u1:1);

    let a_inf = ea == u8:0xff && fa == u23:0;
    let b_inf = eb == u8:0xff && fb == u23:0;
    let a_nan = ea == u8:0xff && fa != u23:0;
    let b_nan = eb == u8:0xff && fb != u23:0;

    // Significands as 24-bit integers and exponents as the stored fields:
    // an operand is m * 2^(x - 23) with x = (biased exponent field) - 127, so
    // a subnormal (field 0, true scale 2^-149) carries its fraction doubled,
    // m = f << 1. Every operand's value is the usual one, so is every sum and
    // the rounded result; what changes is that the exponent arithmetic below
    // reads the exponent fields directly and never waits on a zero test of
    // them (the "subnormal is at 1 - 127" substitution put one in front of the
    // adders).
    //
    // The multiplier's operands are the stored fraction fields with the
    // hidden bit (biased exponent != 0) and nothing else: no zero test and no
    // ftz decode between the operand registers and the DSP. A subnormal
    // factor's doubling is applied to the product (a shift by 0, 1 or 2 after
    // the DSP, which folds into the alignment select), and so is the zero of a
    // factor flushed by ftz. A subnormal fraction is below 2^23, so the doubled
    // product still fits 48 bits.
    let ma = (!a_e0) ++ fa;
    let mb = (!b_e0) ++ fb;

    let sp = sa ^ sb;
    // a factor is zero exactly when it is a zero or an ftz-flushed subnormal
    let p_zero_mag = a_zero || b_zero;
    let p_dbl = (a_e0 as u2) + (b_e0 as u2);
    let pm_raw = (ma as u48) * (mb as u48);
    let pm = if p_zero_mag { u48:0 } else { pm_raw << p_dbl };   // value = pm * 2^(xp-46)

    let inf_times_zero = (a_inf && b_zero) || (b_inf && a_zero);
    let p_inf = (a_inf || b_inf) && !inf_times_zero;

    FmaProduct {
        pm: pm,
        xpb: (ea as s12) + (eb as s12),           // xp + 254
        sp: sp,
        p_zero: p_zero_mag as u1,
        p_inf: p_inf as u1,
        p_nan: (a_nan || b_nan || inf_times_zero) as u1,
    }
}

// First stage, alignment side: the addend, the exponent difference, the
// alignment shift with its sticky bit.
pub fn fma32_align_p(p: FmaProduct, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> FmaAligned {
    let sc = c[31+:u1] ^ sub_c;
    let ec = c[23+:u8];
    let fc = c[0+:u23];
    let c_e0 = ec == u8:0;
    let c_zero = (c_e0 && (fc == u23:0 || ftz == u1:1)) || mul_only == u1:1;
    let c_inf = ec == u8:0xff && fc == u23:0 && mul_only == u1:0;
    let c_nan = ec == u8:0xff && fc != u23:0 && mul_only == u1:0;
    let mc = if c_zero { u24:0 } else if c_e0 { (fc as u24) << u32:1 } else { u1:1 ++ fc };

    let sp = p.sp;
    let pm = p.pm;
    let p_zero_mag = p.p_zero == u1:1;
    let mcw = (mc as u47) << u32:23;             // value = mcw * 2^(xc-46)

    // xp and xc from the stored fields; dexp = xp - xc and its magnitude,
    // both differences side by side rather than a negation behind the
    // subtraction.
    let kab = p.xpb;                              // xp + 254
    let kcb = (ec as s12) + s12:127;              // xc + 254
    let xp = kab - s12:254;
    let xc = (ec as s12) - s12:127;
    let dexp = kab - kcb;
    let ndexp = kcb - kab;
    let d_neg = kab < kcb;
    let dabs = if d_neg { ndexp } else { dexp };
    let shamt = if dabs > s12:80 { u8:80 } else { dabs as u8 };

    // A zero significand is never the reference operand: its exponent means
    // nothing, and with mul_only the addend's exponent is whatever c held.
    // Zero-ness is read from the operands, not the product (ma*mb == 0 iff a
    // factor is 0), so the multiplier is not in front of this decision. mc is
    // 0 exactly when c_zero.
    let shift_c = if c_zero { true } else if p_zero_mag { false } else { !d_neg };

    let exp_common = if shift_c { xp } else { xc };
    let s_big = if shift_c { sp } else { sc };
    let s_small = if shift_c { sc } else { sp };

    let p_inf = p.p_inf == u1:1;
    let inf_minus_inf = p_inf && c_inf && (sp != sc);
    let nan_res = p.p_nan == u1:1 || c_nan || inf_minus_inf;
    let inf_res = p_inf || c_inf;
    let inf_sign = if p_inf { sp } else { sc };

    //------------------------------------------------------- align, add/sub
    let p_field = (pm as uN[88]) << u32:32;         // [79:32]
    let c_field = (mcw as uN[88]) << u32:32;        // [78:32]
    let big = if shift_c { p_field } else { c_field };
    let small = if shift_c { c_field } else { p_field };
    let small_sh = small >> shamt;
    let sticky_al = (small & lsb_mask88(shamt)) != uN[88]:0;

    FmaAligned {
        small_sh: small_sh as uN[80],
        big_hi: (big >> u32:32) as u48,
        sticky_al: sticky_al as u1,
        s_big: s_big,
        s_small: s_small,
        sp: sp,
        sc: sc,
        exp_common: exp_common,
        nan_res: nan_res as u1,
        inf_res: inf_res as u1,
        inf_sign: inf_sign,
        mul_only: mul_only,
        ftz: ftz,
    }
}

// First stage: unpack, classify, multiply, align -- the two halves above.
pub fn fma32_align(a: u32, b: u32, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> FmaAligned {
    fma32_align_p(fma32_mul(a, b, ftz), c, sub_c, mul_only, ftz)
}

// Second stage: add or subtract, count leading zeros, and everything that
// follows the count.
pub fn fma32_normalise(s: FmaAligned) -> FmaNormalised {
    let small_sh = s.small_sh as uN[88];
    let big = (s.big_hi as uN[88]) << u32:32;
    let sticky_al = s.sticky_al == u1:1;
    let s_big = s.s_big;
    let s_small = s.s_small;
    let sp = s.sp;
    let sc = s.sc;
    let exp_common = s.exp_common;
    let mul_only = s.mul_only;

    let same_sign = s_big == s_small;
    let sum_add = (big as uN[89]) + (small_sh as uN[89]);
    // big - (small_sh + eps): the borrow accounts for the shifted-out eps.
    // Carry-select: big - small_sh and big - small_sh - 1 (= big + !small_sh)
    // are formed side by side and the late sticky bit only selects, instead
    // of a second carry chain behind it. The sign of the first is
    // small_sh > big.
    let dif = (big as uN[89]) - (small_sh as uN[89]);
    let dif_m1 = (big as uN[89]) + !(small_sh as uN[89]);
    let small_wins = dif[88+:u1] == u1:1;
    let sub_bs = if sticky_al { dif_m1 } else { dif };
    // (small_sh + eps) - big: eps stays below bit 0 and is the new sticky
    let sub_sb = (small_sh as uN[89]) - (big as uN[89]);
    let mag = if same_sign { sum_add } else if small_wins { sub_sb } else { sub_bs };
    let res_sign_mag = if same_sign { s_big } else if small_wins { s_small } else { s_big };

    //------------------------------------------------------------ normalise
    let any = mag != uN[89]:0;
    // mag's leading one is at 88 - lz and value(mag) = mag * 2^(exp_common - 78),
    // so normalised it is 1.f * 2^e_norm, e_norm = exp_common + 10 - lz. Every
    // quantity that follows the leading-zero count is one compare or one
    // subtraction from it, the rest is formed from exp_common beside the count:
    let lz = clz(mag) as s12;
    //   e_norm < E_MIN, against a bound that does not wait for lz
    let subnormal_res = any && lz > exp_common + s12:136;
    //   the biased exponent before rounding: E_MIN + 127 if subnormal
    let eb0 = if subnormal_res { E_MIN + s12:127 } else { (exp_common + s12:137) - lz };
    //   the normalising shift (right if positive). Normal: msb - 23 = 65 - lz.
    //   Subnormal: that plus E_MIN - e_norm = -71 - exp_common, which does not
    //   depend on lz at all. Each magnitude is formed beside its sign.
    let shn_left = lz > s12:65;
    let shn_mag = if shn_left { lz - s12:65 } else { s12:65 - lz };
    let shs = s12:-71 - exp_common;
    let shs_left = shs < s12:0;
    let shs_mag = if shs_left { -shs } else { shs };

    // exact zero (operands annihilated, or both zero): +0 unless both addends
    // shared a sign
    let exact_zero = !any;
    let sign_zero = if sp == sc { sp } else { u1:0 };
    let res_sign = if mul_only == u1:1 { sp } else if exact_zero { sign_zero } else { res_sign_mag };

    FmaNormalised {
        mag: mag as uN[81],
        any: any as u1,
        subnormal_res: subnormal_res as u1,
        shn_left: shn_left as u1,
        shn_mag: shn_mag as u7,
        shs_left: shs_left as u1,
        shs_mag: shs_mag as u12,
        eb0: eb0,
        sticky_al: s.sticky_al,
        res_sign: res_sign,
        nan_res: s.nan_res,
        inf_res: s.inf_res,
        inf_sign: s.inf_sign,
        ftz: s.ftz,
    }
}

// Third stage: normalising shift, round to nearest even, assemble. The
// result is combinational: the consumer registers it.
pub fn fma32_round(s: FmaNormalised) -> u32 {
    let mag = s.mag as uN[89];
    let any = s.any == u1:1;
    let subnormal_res = s.subnormal_res == u1:1;
    let sh_left = if subnormal_res { s.shs_left == u1:1 } else { s.shn_left == u1:1 };
    let sh_mag = if subnormal_res { s.shs_mag } else { s.shn_mag as u12 };
    let eb0 = s.eb0;
    let sticky_al = s.sticky_al == u1:1;
    let res_sign = s.res_sign;
    let ftz = s.ftz;
    let nan_res = s.nan_res == u1:1;
    let inf_res = s.inf_res == u1:1;
    let inf_sign = s.inf_sign;

    //---------------------------------------------------- round, assemble
    // One shifter gives both the significand and the guard: mag with a zero
    // appended below it, shifted right, puts mag[sh-1] at bit 0 (and 0 there
    // for sh = 0). A left shift is at most 23 whenever mag != 0, so 24 bits of
    // it are the significand and there is no guard. Amounts past the field
    // give 0, as the clamp to 100 did.
    let mag_x = (mag as uN[90]) << u32:1;
    let t = mag_x >> sh_mag;
    let sig = if sh_left { (mag << sh_mag)[0+:u24] } else { t[1+:u24] };
    let guard = if sh_left { u1:0 } else { t[0+:u1] };
    // the bits below the guard, mag[sh-2:0] = mag_x[sh-1:1], masked straight
    // from the shift amount: no decrement in front of the mask
    let rest = if sh_left { false } else { (mag_x & lsb_mask90(sh_mag)) != uN[90]:0 };
    let sticky = rest || sticky_al;

    // Round to nearest even, carry-select: the incremented significand and
    // the all-ones test are formed beside the decision, which only selects.
    let round_up = guard == u1:1 && (sticky || sig[0+:u1] == u1:1);
    let sig_ones = sig == all_ones!<u24>();
    let carry_out = round_up && sig_ones;
    let sig_f = if !round_up { sig } else if sig_ones { u24:0x800000 } else { sig + u24:1 };
    let res_is_norm = sig_f[23+:u1] == u1:1;

    // the biased exponent and the overflow test, both candidates beside the
    // carry: e_biased = eb0 + carry_out
    let e_biased = if carry_out { eb0 + s12:1 } else { eb0 };
    let overflow = any && (if carry_out { eb0 >= s12:254 } else { eb0 >= s12:255 });

    // exact zero (operands annihilated, or both zero): its sign was settled
    // in the second stage. Rounded-to-zero keeps the magnitude's sign.
    let exact_zero = !any;
    // sig_f == 0 exactly when sig is 0 and nothing rounded up
    let result_zero = exact_zero || (sig == u24:0 && !round_up);

    let normal_result =
        if overflow { res_sign ++ u8:0xff ++ u23:0 }
        else if result_zero { res_sign ++ u8:0 ++ u23:0 }
        else if !res_is_norm {
            if ftz == u1:1 { res_sign ++ u8:0 ++ u23:0 } else { res_sign ++ u8:0 ++ sig_f[0+:u23] }
        } else { res_sign ++ (e_biased as u8) ++ sig_f[0+:u23] };

    if nan_res { QNAN }
    else if inf_res { inf_sign ++ u8:0xff ++ u23:0 }
    else { normal_result }
}

pub fn fma32(a: u32, b: u32, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> u32 {
    fma32_round(fma32_normalise(fma32_align(a, b, c, sub_c, mul_only, ftz)))
}

// The cases that made apfloat unusable here, and the rules around zero.
#[test]
fn subnormals_survive() {
    let tiny = u32:0x00000001;                       // 2^-149
    // fma(+0, -0, tiny) = tiny, not 0 (apfloat gives 0)
    assert_eq(fma32(u32:0x00000000, u32:0x80000000, tiny, u1:0, u1:0, u1:0), tiny);
    // tiny + 0 = tiny
    assert_eq(fma32(u32:0x3f800000, tiny, u32:0, u1:0, u1:0, u1:0), tiny);
    // ...and flushed only when asked
    assert_eq(fma32(u32:0x3f800000, tiny, u32:0, u1:0, u1:0, u1:1), u32:0);
    // 2^-149 - 2^-148 rounds to... nothing to round: -2^-149 exactly
    assert_eq(fma32(u32:0x3f800000, tiny, u32:0x00000002, u1:1, u1:0, u1:0), u32:0x80000001);
}

#[test]
fn ordinary_and_special() {
    // 1.5 * 2 + 1 = 4
    assert_eq(fma32(u32:0x3fc00000, u32:0x40000000, u32:0x3f800000, u1:0, u1:0, u1:0), u32:0x40800000);
    // x - x = +0
    assert_eq(fma32(u32:0x3f800000, u32:0x40400000, u32:0x40400000, u1:1, u1:0, u1:0), u32:0);
    // -0 + -0 = -0
    assert_eq(fma32(u32:0x80000000, u32:0x3f800000, u32:0x80000000, u1:0, u1:0, u1:0), u32:0x80000000);
    // inf * 0 is NaN
    assert_eq(fma32(u32:0x7f800000, u32:0, u32:0x3f800000, u1:0, u1:0, u1:0), QNAN);
    // mul_only ignores a NaN addend
    assert_eq(fma32(u32:0x40000000, u32:0x40000000, u32:0x7fc00001, u1:0, u1:1, u1:0), u32:0x40800000);
    // overflow to infinity
    assert_eq(fma32(u32:0x7f7fffff, u32:0x40000000, u32:0, u1:0, u1:0, u1:0), u32:0x7f800000);
}
