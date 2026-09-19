// TNCA floating-point formats.
//
// Five formats, and one of them does not follow IEEE rules:
//
//   fp32      1.8.23   IEEE-754 binary32
//   bf16      1.8.7    bfloat16: binary32's exponent range, 8 fewer fraction bits
//   fp16      1.5.10   IEEE-754 binary16
//   e5m2      1.5.2    OCP FP8 E5M2 -- IEEE-like: has inf and NaN
//   e4m3      1.4.3    OCP FP8 E4M3 -- NO infinity; bexp=15 with fraction<7
//                      are ordinary numbers, reaching 448
//
// That last line is why this file exists rather than a handful of calls into
// apfloat. apfloat models an IEEE float, where the all-ones exponent means
// inf or NaN. In E4M3 the all-ones exponent is mostly *numbers*: only
// S.1111.111 is NaN, there is no infinity at all, and the largest finite
// value is 448 rather than the 240 an IEEE reading of the same bits would
// give. Handing E4M3 to a generic IEEE routine silently loses the top of the
// range, which is exactly the kind of quiet wrongness this project refuses
// elsewhere.
//
// Everything here converts through binary32. That is not laziness: it is what
// the matrix unit already does for FP16 and BF16 inputs (ISA sec 9C.1 widens
// exactly and rounds once), and it keeps the narrow formats' arithmetic
// identical to the wide path that is already verified against an exact
// oracle.

import apfloat;

pub type F32 = apfloat::APFloat<u32:8, u32:23>;

// Format selector, as the datapath carries it.
pub const FMT_F32: u3 = u3:0;
pub const FMT_BF16: u3 = u3:1;
pub const FMT_F16: u3 = u3:2;
pub const FMT_E5M2: u3 = u3:3;
pub const FMT_E4M3: u3 = u3:4;

pub const F32_QNAN: u32 = u32:0x7fc0_0000;
// OCP FP8 E4M3: the single NaN encoding is S.1111.111.
pub const E4M3_NAN: u8 = u8:0x7f;
// The largest finite E4M3 magnitude, S.1111.110 = 448.
pub const E4M3_MAX: u8 = u8:0x7e;

// ---------------------------------------------------------------------------
// E4M3 -> binary32
// ---------------------------------------------------------------------------
// Subnormals are m * 2^-9 for m in 1..7. Normalising a three-bit fraction is
// seven cases, so it is written as seven cases: a mux the synthesiser folds
// flat, and a reader can check by eye against the table in the OCP spec.
fn e4m3_sub_to_f32(sign: u1, m: u3) -> u32 {
    let (exp, frac) = match m {
        u3:1 => (u8:118, u23:0),                 // 2^-9
        u3:2 => (u8:119, u23:0),                 // 2^-8
        u3:3 => (u8:119, u23:0x40_0000),         // 1.5 * 2^-8
        u3:4 => (u8:120, u23:0),                 // 2^-7
        u3:5 => (u8:120, u23:0x20_0000),         // 1.25 * 2^-7
        u3:6 => (u8:120, u23:0x40_0000),         // 1.5  * 2^-7
        _    => (u8:120, u23:0x60_0000),         // 1.75 * 2^-7
    };
    sign ++ exp ++ frac
}

pub fn e4m3_to_f32(x: u8) -> u32 {
    let sign = x[7+:u1];
    let e = x[3+:u4];
    let m = x[0+:u3];

    let is_nan = (e == u4:15) && (m == u3:7);
    let is_zero = (e == u4:0) && (m == u3:0);

    // Normal: 2^(e-7) * (1 + m/8). The bias difference is 127 - 7 = 120, and
    // e never reaches a value that would overflow binary32.
    let norm = sign ++ (u8:120 + (e as u8)) ++ ((m as u23) << u23:20);

    if is_nan { F32_QNAN }
    else if is_zero { sign ++ u31:0 }
    else if e == u4:0 { e4m3_sub_to_f32(sign, m) }
    else { norm }
}

// ---------------------------------------------------------------------------
// E5M2 -> binary32
// ---------------------------------------------------------------------------
// E5M2 *is* IEEE-shaped, so this is the ordinary widening: bias 15 -> 127,
// fraction 2 -> 23, with inf and NaN carried across.
fn e5m2_sub_to_f32(sign: u1, m: u2) -> u32 {
    // Subnormals are m * 2^-16 for m in 1..3.
    let (exp, frac) = match m {
        u2:1 => (u8:111, u23:0),                 // 2^-16
        u2:2 => (u8:112, u23:0),                 // 2^-15
        _    => (u8:112, u23:0x40_0000),         // 1.5 * 2^-15
    };
    sign ++ exp ++ frac
}

pub fn e5m2_to_f32(x: u8) -> u32 {
    let sign = x[7+:u1];
    let e = x[2+:u5];
    let m = x[0+:u2];

    let is_special = e == u5:31;
    let is_zero = (e == u5:0) && (m == u2:0);
    let norm = sign ++ (u8:112 + (e as u8)) ++ ((m as u23) << u23:21);

    if is_special {
        if m == u2:0 { sign ++ u8:255 ++ u23:0 } else { F32_QNAN }
    } else if is_zero { sign ++ u31:0 }
    else if e == u5:0 { e5m2_sub_to_f32(sign, m) }
    else { norm }
}

// ---------------------------------------------------------------------------
// FP16 / BF16 -> binary32
// ---------------------------------------------------------------------------
pub fn bf16_to_f32(x: u16) -> u32 {
    // Exact by construction: bf16 is binary32 truncated, so widening is a
    // shift. NaN stays NaN because the exponent and the fraction's top bits
    // are carried unchanged.
    x ++ u16:0
}

fn f16_sub_to_f32(sign: u1, m: u10) -> u32 {
    // Subnormal fp16 is m * 2^-24. With k the position of m's leading one,
    // the value is 2^(k-24) * (1 + f), so the binary32 exponent is
    // 127 + k - 24 = 103 + k. Counting leading zeros in a 16-bit container
    // gives k = 9 - (clz - 6), hence 112 - lz below.
    let lz = clz(m as u16) - u16:6;           // 0..9 for m in 512..1
    let shifted = (m as u16) << (lz as u16);  // leading 1 now at bit 9
    let frac = ((shifted as u23) & u23:0x1ff) << u23:14;
    let exp = u8:112 - (lz as u8);
    sign ++ exp ++ frac
}

pub fn f16_to_f32(x: u16) -> u32 {
    let sign = x[15+:u1];
    let e = x[10+:u5];
    let m = x[0+:u10];

    let is_special = e == u5:31;
    let is_zero = (e == u5:0) && (m == u10:0);
    let norm = sign ++ (u8:112 + (e as u8)) ++ ((m as u23) << u23:13);

    if is_special {
        if m == u10:0 { sign ++ u8:255 ++ u23:0 } else { F32_QNAN }
    } else if is_zero { sign ++ u31:0 }
    else if e == u5:0 { f16_sub_to_f32(sign, m) }
    else { norm }
}

// ---------------------------------------------------------------------------
// binary32 -> narrow, round to nearest even
// ---------------------------------------------------------------------------
// One routine for all four narrow formats. The parameter that matters is
// HAS_INF: with it, the all-ones exponent is inf/NaN and overflow produces
// infinity; without it (E4M3) the all-ones exponent is mostly numbers,
// overflow saturates to the largest finite value, and only the all-ones
// fraction is NaN.
//
// Writing the difference as a parameter rather than as a separate E4M3 path
// is deliberate. The two formats differ in one rule; a reader can see that
// here, and there is one rounding implementation to get right rather than
// two to keep in agreement.
//
// Overflow in E4M3 saturates rather than producing NaN. The OCP spec permits
// either; saturation is what makes an FP8 tensor useful, and it is stated
// here rather than discovered.
pub fn narrow_from_f32<E: u32, F: u32, HAS_INF: bool,
                       W: u32 = {u32:1 + E + F}>(x: u32) -> uN[W] {
    const BIAS: s32 = ((s32:1 << (E - u32:1)) - s32:1);
    // Largest unbiased exponent the format holds as a NUMBER.
    const MAX_UNB: s32 = if HAS_INF { ((s32:1 << E) - s32:2) - BIAS }
                         else { ((s32:1 << E) - s32:1) - BIAS };
    const MIN_UNB: s32 = s32:1 - BIAS;          // smallest normal
    const MAX_BEXP: uN[E] = if HAS_INF { (uN[E]:1 << E) - uN[E]:2 }
                            else { (uN[E]:1 << E) - uN[E]:1 };
    // Largest finite fraction: all ones, except E4M3 reserves all-ones for NaN.
    const MAX_FRAC: uN[F] = if HAS_INF { (uN[F]:1 << F) - uN[F]:1 }
                            else { (uN[F]:1 << F) - uN[F]:2 };

    let sign = x[31+:u1];
    let e32 = x[23+:u8];
    let f32b = x[0+:u23];

    let is_nan = (e32 == u8:255) && (f32b != u23:0);
    let is_inf = (e32 == u8:255) && (f32b == u23:0);
    let is_zero = (e32 == u8:0) && (f32b == u23:0);

    // A binary32 subnormal is NOT always below the target's range. BF16 has
    // binary32's exponent range, so its subnormals *are* binary32
    // subnormals, and flushing them would lose the bottom of the format --
    // which is what the exhaustive test caught. Normalising the input first
    // makes one path serve every case: shift the fraction up until the
    // implicit one appears and pay for it in the exponent. For FP16 and both
    // FP8s the normalised exponent then lands far below their smallest
    // subnormal, and the shift-past-the-end check below flushes it, which is
    // the right answer there.
    let sub_shift = clz(f32b as u32) - u32:8;
    let sub_sig = (f32b as u32) << sub_shift;
    let sub_unb = s32:-126 - (sub_shift as s32);

    let is_sub_in = (e32 == u8:0) && (f32b != u23:0);
    let unb = if is_sub_in { sub_unb } else { (e32 as s32) - s32:127 };
    // 24-bit significand with the implicit one.
    let sig = if is_sub_in { sub_sig } else { u32:1 << u32:23 | (f32b as u32) };

    // How far right to shift to leave F fraction bits. A subnormal result
    // needs the extra shift that MIN_UNB - unb represents.
    let sub_extra = if unb < MIN_UNB { (MIN_UNB - unb) as u32 } else { u32:0 };
    let shift = (u32:23 - F) + sub_extra;

    // Round to nearest, ties to even, on the discarded low bits.
    let capped = if shift > u32:31 { u32:31 } else { shift };
    let kept = sig >> capped;
    let round_bit = if capped == u32:0 { u1:0 }
                    else { ((sig >> (capped - u32:1)) & u32:1) as u1 };
    let sticky = if capped <= u32:1 { u1:0 }
                 else { ((sig & ((u32:1 << (capped - u32:1)) - u32:1)) != u32:0) as u1 };
    let round_up = round_bit & (sticky | ((kept & u32:1) as u1));
    let rounded = kept + (round_up as u32);

    // Rounding may carry into the next binade, which is why the exponent is
    // derived from the rounded significand rather than decided beforehand.
    let carried = (rounded >> (F + u32:1)) != u32:0;
    let sub_carried = (sub_extra != u32:0) && ((rounded >> F) != u32:0);

    let out_unb = if carried { unb + s32:1 } else { unb };
    let frac_norm = if carried { (rounded >> u32:1) as uN[F] }
                    else { rounded as uN[F] };

    let overflow = (out_unb > MAX_UNB) ||
                   ((out_unb == MAX_UNB) && !HAS_INF && (frac_norm > MAX_FRAC));
    let bexp_norm = ((out_unb + BIAS) as uN[E]);

    let sub_result = sign ++ (if sub_carried { uN[E]:1 } else { uN[E]:0 })
                          ++ (rounded as uN[F]);
    let norm_result = sign ++ bexp_norm ++ frac_norm;
    let zero = sign ++ uN[E]:0 ++ uN[F]:0;
    let nan = uN[1]:0 ++ ((uN[E]:1 << E) - uN[E]:1) ++ ((uN[F]:1 << F) - uN[F]:1);
    let sat = sign ++ MAX_BEXP ++ MAX_FRAC;
    let infinity = sign ++ ((uN[E]:1 << E) - uN[E]:1) ++ uN[F]:0;

    if is_nan { nan }
    else if is_inf { if HAS_INF { infinity } else { sat } }
    else if is_zero { zero }
    else if unb < MIN_UNB {
        // Subnormal or under it. A rounded-up value may reach the smallest
        // normal, which sub_carried detects and the encoding handles for free.
        if shift > u32:31 { zero } else { sub_result }
    }
    else if overflow { if HAS_INF { infinity } else { sat } }
    else { norm_result }
}

pub fn f32_to_e4m3(x: u32) -> u8 { narrow_from_f32<u32:4, u32:3, false>(x) }
pub fn f32_to_e5m2(x: u32) -> u8 { narrow_from_f32<u32:5, u32:2, true>(x) }
pub fn f32_to_f16(x: u32) -> u16 { narrow_from_f32<u32:5, u32:10, true>(x) }
pub fn f32_to_bf16(x: u32) -> u16 { narrow_from_f32<u32:8, u32:7, true>(x) }
