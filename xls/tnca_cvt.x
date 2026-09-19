// Conversion between every pair of TNCA float formats.
//
// ISA sec 8.5 gives the CVT major FP16 and BF16 only; this covers all five,
// including both OCP FP8 encodings. Every conversion goes through binary32,
// which makes the pairs a product of one widening and one narrowing rather
// than twenty separate routines to keep consistent with each other.
//
// The widenings are all exact -- binary32 holds every value of all four
// narrow formats -- so a round trip narrow -> f32 -> narrow is the identity,
// and the test suite checks exactly that over every bit pattern of each
// narrow format.

import tnca_formats;

// Any narrow format, in the low bits of a 32-bit word, to binary32.
pub fn widen(fmt: u3, x: u32) -> u32 {
    match fmt {
        tnca_formats::FMT_F32  => x,
        tnca_formats::FMT_BF16 => tnca_formats::bf16_to_f32(x as u16),
        tnca_formats::FMT_F16  => tnca_formats::f16_to_f32(x as u16),
        tnca_formats::FMT_E5M2 => tnca_formats::e5m2_to_f32(x as u8),
        tnca_formats::FMT_E4M3 => tnca_formats::e4m3_to_f32(x as u8),
        _ => tnca_formats::F32_QNAN,
    }
}

// binary32 to any narrow format, zero-extended into a 32-bit word.
pub fn narrow(fmt: u3, x: u32) -> u32 {
    match fmt {
        tnca_formats::FMT_F32  => x,
        tnca_formats::FMT_BF16 => tnca_formats::f32_to_bf16(x) as u32,
        tnca_formats::FMT_F16  => tnca_formats::f32_to_f16(x) as u32,
        tnca_formats::FMT_E5M2 => tnca_formats::f32_to_e5m2(x) as u32,
        tnca_formats::FMT_E4M3 => tnca_formats::f32_to_e4m3(x) as u32,
        _ => u32:0,
    }
}

// The datapath's conversion instruction: one source format, one destination.
pub fn convert(src: u3, dst: u3, x: u32) -> u32 { narrow(dst, widen(src, x)) }
