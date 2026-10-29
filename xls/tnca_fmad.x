// binary32 FMA, R41 G2 area rebuild: the same function as tnca_fma.x, bit for
// bit, with cheaper implementations of the steps that cost the most LUTs on
// the U280 (evidence/r41/hbm_bw/fma_dslx/README.md).
//
// Each stage function keeps tnca_fma.x's interface (FmaAligned,
// FmaNormalised), so each is proved equal to its tnca_fma.x counterpart on
// every input by #[quickcheck] + prove_quickcheck_main (Z3), and the stages
// compose to the same fma32.
//
//   normalise: the 89-bit leading-zero count is tnca_lzc's balanced tree
//     instead of clz() (768 LUTs and 95 CARRY8 -> 82 LUTs in the unit screen).

import tnca_fma;
import tnca_lzc;

const E_MIN = s12:-126;   // as tnca_fma.x

pub fn fma32d_normalise(s: tnca_fma::FmaAligned) -> tnca_fma::FmaNormalised {
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
    let lz = tnca_lzc::lzc(mag) as s12;
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

    tnca_fma::FmaNormalised {
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

pub fn fma32d(a: u32, b: u32, c: u32, sub_c: u1, mul_only: u1, ftz: u1) -> u32 {
    tnca_fma::fma32_round(fma32d_normalise(tnca_fma::fma32_align(a, b, c, sub_c, mul_only, ftz)))
}

#[quickcheck]
fn normalise_equiv(s: tnca_fma::FmaAligned) -> bool {
    fma32d_normalise(s) == tnca_fma::fma32_normalise(s)
}

#[test]
fn same_as_fma32() {
    assert_eq(fma32d(u32:0x3fc00000, u32:0x40000000, u32:0x3f800000, u1:0, u1:0, u1:0), u32:0x40800000);
    assert_eq(fma32d(u32:0x3f800000, u32:0x00000001, u32:0x00000002, u1:1, u1:0, u1:0), u32:0x80000001);
}
