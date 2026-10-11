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
// regeneration, not a redesign.
//
// The algorithm is the hand-written one, step for step, so the two can be
// compared bit for bit and the exact Fraction oracle that verified one
// verifies the other (tb_fp32_fma, fma-deep: 400 000 vectors):
//
//   unpack, classify; 24x24 product P; addend MC = mc << 23, both at scale
//   2^(e-46); shift the one with the smaller exponent right onto the other's
//   scale with a sticky bit, in an 88-bit field
//       [87:80] carry headroom  [79:32] operands  [31:0] guard/sticky
//   add or subtract (comparing aligned magnitudes, since equal exponents do
//   not order the significands); leading-one search; normalise or denormalise
//   to 24 bits plus guard and sticky; round to nearest even; pack.

const QNAN = u32:0x7fc00000;
const E_MIN = s12:-126;

fn lsb_mask88(n: u8) -> u88 { !(all_ones!<u88>() << n) }
fn lsb_mask89(n: u8) -> u89 { !(all_ones!<u89>() << n) }

pub fn fma32(a: u32, b: u32, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> u32 {
    //------------------------------------------------------------------ unpack
    let sa = a[31+:u1];
    let sb = b[31+:u1];
    let sc = c[31+:u1] ^ sub_c;
    let ea = a[23+:u8];
    let eb = b[23+:u8];
    let ec = c[23+:u8];
    let fa = a[0+:u23];
    let fb = b[0+:u23];
    let fc = c[0+:u23];

    let a_sub = ea == u8:0 && fa != u23:0;
    let b_sub = eb == u8:0 && fb != u23:0;
    let c_sub = ec == u8:0 && fc != u23:0;

    // ftz flushes subnormal inputs before anything else looks at them
    let a_zero = ea == u8:0 && (fa == u23:0 || ftz == u1:1);
    let b_zero = eb == u8:0 && (fb == u23:0 || ftz == u1:1);
    let c_zero = (ec == u8:0 && (fc == u23:0 || ftz == u1:1)) || mul_only == u1:1;

    let a_inf = ea == u8:0xff && fa == u23:0;
    let b_inf = eb == u8:0xff && fb == u23:0;
    let c_inf = ec == u8:0xff && fc == u23:0 && mul_only == u1:0;
    let a_nan = ea == u8:0xff && fa != u23:0;
    let b_nan = eb == u8:0xff && fb != u23:0;
    let c_nan = ec == u8:0xff && fc != u23:0 && mul_only == u1:0;

    // significands as 24-bit integers: value = mant * 2^(exp - 23)
    let ma = if a_zero { u24:0 } else if a_sub { fa as u24 } else { u1:1 ++ fa };
    let mb = if b_zero { u24:0 } else if b_sub { fb as u24 } else { u1:1 ++ fb };
    let mc = if c_zero { u24:0 } else if c_sub { fc as u24 } else { u1:1 ++ fc };

    // unbiased exponents; a subnormal has e = 1 - 127
    let xa = if a_sub { s12:-126 } else { (ea as s12) - s12:127 };
    let xb = if b_sub { s12:-126 } else { (eb as s12) - s12:127 };
    let xc = if c_sub { s12:-126 } else { (ec as s12) - s12:127 };

    let sp = sa ^ sb;
    let pm = (ma as u48) * (mb as u48);         // value = pm * 2^(xa+xb-46)
    let xp = xa + xb;
    let mcw = (mc as u47) << u32:23;             // value = mcw * 2^(xc-46)

    let dexp = (xp as s13) - (xc as s13);
    let dabs = if dexp < s13:0 { (-dexp) as u13 } else { dexp as u13 };
    let shamt = if dabs > u13:80 { u8:80 } else { dabs as u8 };

    // A zero significand is never the reference operand: its exponent means
    // nothing, and with mul_only the addend's exponent is whatever c held.
    // Zero-ness is read from the operands, not the product (ma*mb == 0 iff a
    // factor is 0), so the multiplier is not in front of this decision.
    let c_zero_mag = mc == u24:0;
    let p_zero_mag = ma == u24:0 || mb == u24:0;
    let shift_c = if c_zero_mag { true } else if p_zero_mag { false } else { dexp >= s13:0 };

    let exp_common = if shift_c { xp } else { xc };
    let s_big = if shift_c { sp } else { sc };
    let s_small = if shift_c { sc } else { sp };

    let inf_times_zero = (a_inf && b_zero) || (b_inf && a_zero);
    let p_inf = (a_inf || b_inf) && !inf_times_zero;
    let inf_minus_inf = p_inf && c_inf && (sp != sc);
    let nan_res = a_nan || b_nan || c_nan || inf_times_zero || inf_minus_inf;
    let inf_res = p_inf || c_inf;
    let inf_sign = if p_inf { sp } else { sc };

    //------------------------------------------------------- align, add/sub
    let p_field = (pm as u88) << u32:32;         // [79:32]
    let c_field = (mcw as u88) << u32:32;        // [78:32]
    let big = if shift_c { p_field } else { c_field };
    let small = if shift_c { c_field } else { p_field };
    let small_sh = small >> shamt;
    let sticky_al = (small & lsb_mask88(shamt)) != u88:0;

    let same_sign = s_big == s_small;
    let small_wins = small_sh > big;
    let sum_add = (big as u89) + (small_sh as u89);
    // big - (small_sh + eps): the borrow accounts for the shifted-out eps
    let sub_bs = (big as u89) - (small_sh as u89) - (sticky_al as u89);
    // (small_sh + eps) - big: eps stays below bit 0 and is the new sticky
    let sub_sb = (small_sh as u89) - (big as u89);
    let mag = if same_sign { sum_add } else if small_wins { sub_sb } else { sub_bs };
    let res_sign_mag = if same_sign { s_big } else if small_wins { s_small } else { s_big };

    //------------------------------------------- normalise, round, assemble
    let any = mag != u89:0;
    let msb = if any { u8:88 - (clz(mag) as u8) } else { u8:0 };
    // value(mag) = mag * 2^(exp_common - 78); normalised = 1.f * 2^e_norm
    let e_norm = (msb as s12) + exp_common - s12:78;
    let sh_norm = (msb as s12) - s12:23;
    let subnormal_res = e_norm < E_MIN && any;
    let sh_extra = if subnormal_res { E_MIN - e_norm } else { s12:0 };
    let sh_tot = sh_norm + sh_extra;
    let sh_left = sh_tot < s12:0;
    let sh_mag = if sh_left { (-sh_tot) as u12 } else { sh_tot as u12 };
    let sh_r = if sh_mag > u12:100 { u8:100 } else { sh_mag as u8 };

    let shifted = if sh_left { mag << sh_r } else { mag >> sh_r };
    let sh_r1 = if sh_r == u8:0 { u8:0 } else { sh_r - u8:1 };
    let mag_g = mag >> sh_r1;

    let sig = shifted[0+:u24];
    let guard = if sh_left || sh_r == u8:0 { u1:0 } else { mag_g[0+:u1] };
    let rest = if sh_left || sh_r < u8:2 { false } else { (mag & lsb_mask89(sh_r1)) != u89:0 };
    let sticky = rest || sticky_al;

    let round_up = guard == u1:1 && (sticky || sig[0+:u1] == u1:1);
    let sig_r = (sig as u25) + (round_up as u25);
    let carry_out = sig_r[24+:u1] == u1:1;
    let sig_f = if carry_out { sig_r[1+:u24] } else { sig_r[0+:u24] };
    let e_f = (if subnormal_res { E_MIN } else { e_norm }) + (if carry_out { s12:1 } else { s12:0 });
    let res_is_norm = sig_f[23+:u1] == u1:1;

    let e_biased = e_f + s12:127;
    let overflow = any && e_biased >= s12:255;

    // exact zero (operands annihilated, or both zero): +0 unless both addends
    // shared a sign. Rounded-to-zero keeps the magnitude's sign.
    let exact_zero = !any;
    let result_zero = exact_zero || sig_f == u24:0;
    let sign_zero = if sp == sc { sp } else { u1:0 };
    let res_sign = if mul_only == u1:1 { sp } else if exact_zero { sign_zero } else { res_sign_mag };

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
