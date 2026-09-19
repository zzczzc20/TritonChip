// The special function unit: exp2, log2, rcp, rsqrt.
//
// ISA sec 8.8 specifies these as approximate with a published error bound --
// 2^-22 relative -- and states that the bound is a contract, not an
// implementation detail. It also specifies that SFU operations flush
// subnormal inputs to a zero of the same sign and flush subnormal results to
// zero. Both of those make this the one place in the datapath where XLS's
// flush-to-zero arithmetic is not merely acceptable but required.
//
// The shape is a reduction, a table-indexed quadratic, and a reconstruction,
// which is what special function units actually do. With 128 segments the
// quadratic's truncation error is around 2^-25 for all four functions, so
// the 2^-22 budget is spent on the fixed-point arithmetic rather than on the
// approximation. tools/model/gen_sfu_tables.py computes the coefficients in
// high-precision decimal and emits both these tables and the reference the
// hardware is measured against, so the two cannot drift.
//
// Everything below works in Q_.30 fixed point, which leaves six bits of
// headroom above a mantissa in [1,2) and thirty below -- seven more than
// binary32 keeps, so the rounding at the end is the only one that shows.

import tnca_sfu_tables as T;

const FRAC: u32 = u32:30;          // fixed-point fraction bits
const ONE: s64 = s64:1 << u32:30;  // 1.0 in Q_.30
const QNAN: u32 = u32:0x7fc0_0000;

// c0 + c1*u + c2*u^2, Horner, with u in Q_.30 and small.
//
// Everything here is s32. The coefficients all have magnitude below 2, so
// Q2.30 holds them exactly, and u is under 2^-8 which leaves it 22 bits. The
// products therefore need a 32x22 multiplier and a 64-bit accumulator, not
// the 64x64 that writing the table as s64 asked for -- and 64x64 was enough
// to stop yosys finishing a single SFU function in six minutes.
fn poly2(c0: s32, c1: s32, c2: s32, u: s32) -> s32 {
    let t = c1 + (((c2 as s64) * (u as s64)) >> FRAC) as s32;
    c0 + (((t as s64) * (u as s64)) >> FRAC) as s32
}

// Build a binary32 from a sign, an unbiased exponent and a mantissa in
// [1,2) held as Q_.30. Rounds to nearest even on the way down to 23 bits,
// and applies sec 8.8's flush: an exponent below the smallest normal gives a
// zero rather than a subnormal.
fn pack(sign: u1, unb: s32, mant_q30: s64) -> u32 {
    // Rounding can carry the mantissa to 2.0, which is one more binade.
    let round_add = mant_q30 + (s64:1 << u32:6);
    let carried = round_add >= (s64:1 << u32:31);
    let m = if carried { round_add >> u32:1 } else { round_add };
    let e = if carried { unb + s32:1 } else { unb };

    let frac = ((m >> u32:7) & s64:0x7f_ffff) as u23;
    let bexp = e + s32:127;

    if bexp >= s32:255 { sign ++ u8:255 ++ u23:0 }        // overflow -> inf
    else if bexp <= s32:0 { sign ++ u8:0 ++ u23:0 }       // sec 8.8: flush
    else { sign ++ (bexp as u8) ++ frac }
}

// A signed Q_.30 value to binary32, used by log2 whose result is not a
// mantissa-and-exponent but a number spanning roughly +/-150.
fn fixed_to_f32(v: s64) -> u32 {
    if v == s64:0 { u32:0 }
    else {
        let neg = v < s64:0;
        let mag = (if neg { -v } else { v }) as u64;
        let msb = u64:63 - (clz(mag) as u64);          // index of the leading 1
        let unb = (msb as s32) - (FRAC as s32);
        // Line the leading one up with bit 30.
        let mant = if msb < u64:30 { mag << (u64:30 - msb) }
                   else { mag >> (msb - u64:30) };
        pack(neg as u1, unb, mant as s64)
    }
}

// Decode the parts of a binary32 once.
struct Parts { sign: u1, bexp: u8, frac: u23 }
fn parts(x: u32) -> Parts {
    Parts { sign: x[31+:u1], bexp: x[23+:u8], frac: x[0+:u23] }
}
fn is_nan(p: Parts) -> bool { (p.bexp == u8:255) && (p.frac != u23:0) }
fn is_inf(p: Parts) -> bool { (p.bexp == u8:255) && (p.frac == u23:0) }
// sec 8.8 flushes subnormal inputs, so they are treated as a signed zero.
fn is_zeroish(p: Parts) -> bool { p.bexp == u8:0 }

// Index and offset within a segment, from a 23-bit fraction.
fn seg(frac: u23) -> (u32, s32) {
    let idx = (frac >> (u23:23 - (T::LOG2_SEGMENTS as u23))) as u32;
    // u = frac - idx/SEGMENTS, in Q_.30.
    let f30 = (frac as s32) << (FRAC - u32:23);
    (idx, f30 - ((idx as s32) << (FRAC - T::LOG2_SEGMENTS)))
}

// ---------------------------------------------------------------------------
// exp2
// ---------------------------------------------------------------------------
pub fn exp2(x: u32) -> u32 {
    let p = parts(x);
    let unb = (p.bexp as s32) - s32:127;
    let sig = (s64:1 << u32:23) | (p.frac as s64);

    // |x| as Q8.30. Only |x| < 256 can produce anything but 0 or inf, and
    // the shift is clamped so a huge exponent cannot wrap.
    let sh = unb - s32:23 + (FRAC as s32);
    let mag = if sh >= s32:0 {
                  if sh > s32:40 { s64:1 << u32:62 } else { sig << (sh as u32) }
              } else {
                  if sh < s32:-40 { s64:0 } else { sig >> ((-sh) as u32) }
              };
    let v = if p.sign { -mag } else { mag };

    // Split into an integer part and a fraction in [0,1). An arithmetic
    // shift floors, which is what makes the fraction non-negative for
    // negative arguments too.
    let i = (v >> FRAC) as s32;
    let r = v - ((i as s64) << FRAC);
    let idx = (r >> (FRAC - T::LOG2_SEGMENTS)) as u32;
    let u = (r - ((idx as s64) << (FRAC - T::LOG2_SEGMENTS))) as s32;

    let m = poly2(T::EXP2_C0[idx], T::EXP2_C1[idx], T::EXP2_C2[idx], u) as s64;

    if is_nan(p) { QNAN }
    else if is_inf(p) { if p.sign { u32:0 } else { u32:0x7f80_0000 } }
    else if is_zeroish(p) { u32:0x3f80_0000 }          // exp2(+-0) = 1
    else if i >= s32:128 { u32:0x7f80_0000 }
    else if i < s32:-150 { u32:0 }
    else { pack(u1:0, i, m) }
}

// ---------------------------------------------------------------------------
// log2
// ---------------------------------------------------------------------------
pub fn log2(x: u32) -> u32 {
    let p = parts(x);
    let unb = (p.bexp as s32) - s32:127;
    let (idx, u) = seg(p.frac);
    let lg = poly2(T::LOG2_C0[idx], T::LOG2_C1[idx], T::LOG2_C2[idx], u) as s64;
    let total = ((unb as s64) << FRAC) + lg;

    if is_nan(p) { QNAN }
    else if p.sign && !is_zeroish(p) { QNAN }          // log2 of a negative
    else if is_zeroish(p) { u32:0xff80_0000 }          // log2(+-0) = -inf
    else if is_inf(p) { if p.sign { QNAN } else { u32:0x7f80_0000 } }
    else { fixed_to_f32(total) }
}

// ---------------------------------------------------------------------------
// rcp
// ---------------------------------------------------------------------------
pub fn rcp(x: u32) -> u32 {
    let p = parts(x);
    let unb = (p.bexp as s32) - s32:127;
    let (idx, u) = seg(p.frac);
    let v = poly2(T::RCP_C0[idx], T::RCP_C1[idx], T::RCP_C2[idx], u) as s64;

    // 1/m lands in (0.5, 1]; shift it into [1,2) and pay an exponent.
    let low = v < ONE;
    let mant = if low { v << u32:1 } else { v };
    let e = if low { -unb - s32:1 } else { -unb };

    if is_nan(p) { QNAN }
    else if is_zeroish(p) { p.sign ++ u8:255 ++ u23:0 }   // 1/+-0 = +-inf
    else if is_inf(p) { p.sign ++ u31:0 }                 // 1/+-inf = +-0
    else { pack(p.sign, e, mant) }
}

// ---------------------------------------------------------------------------
// rsqrt
// ---------------------------------------------------------------------------
pub fn rsqrt(x: u32) -> u32 {
    let p = parts(x);
    let unb = (p.bexp as s32) - s32:127;
    let (idx, u) = seg(p.frac);

    // x = 2^unb * m. Halving the exponent needs it even, so an odd exponent
    // borrows a factor of two into the mantissa and uses the doubled table.
    let h = unb >> u32:1;              // arithmetic shift floors, as required
    let odd = (unb & s32:1) != s32:0;
    let v = (if odd {
        poly2(T::RSQRT2_C0[idx], T::RSQRT2_C1[idx], T::RSQRT2_C2[idx], u)
    } else {
        poly2(T::RSQRT_C0[idx], T::RSQRT_C1[idx], T::RSQRT_C2[idx], u)
    }) as s64;

    let low = v < ONE;
    let mant = if low { v << u32:1 } else { v };
    let e = if low { -h - s32:1 } else { -h };

    if is_nan(p) { QNAN }
    else if is_zeroish(p) { p.sign ++ u8:255 ++ u23:0 }   // rsqrt(+-0) = +-inf
    else if p.sign { QNAN }                               // rsqrt(x<0) = NaN
    else if is_inf(p) { u32:0 }                           // rsqrt(+inf) = +0
    else { pack(u1:0, e, mant) }
}
