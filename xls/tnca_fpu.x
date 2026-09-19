// Multi-precision floating-point: format handling, not new arithmetic.
//
// WHY THERE IS NO apfloat ARITHMETIC HERE.
//
// XLS's apfloat library flushes subnormals to zero. Measured, not assumed:
//
//     apfloat::add(1e-45, 0.0)                     -> 0.0   (want 1e-45)
//     apfloat::fma(+0.0, -0.0, 1e-45)              -> 0.0   (want 1e-45)
//
// That is a legitimate choice for a library aimed at ML kernels, and the
// library says as much by shipping `subnormals_to_zero` as a helper. It is
// not a legitimate choice *here*: this ISA makes FTZ an explicitly encoded
// mode and says so in as many words -- flush-to-zero is never an
// implementation shortcut (sec 12.4, sec 12.7). A binary32 FMA that quietly
// flushed would break the default mode, and the hand-written
// rtl/core/tnca_fp32_fma.v already handles subnormals and is checked against
// an exact Fraction oracle over half a million vectors.
//
// So the arithmetic stays where it is verified, and XLS does what it is
// genuinely better at: the format work around it. rtl/core/tnca_fpu_multi.v
// composes them -- XLS widening, the existing FMA, XLS narrowing -- which
// gives FP8 and FP16 operands without a second arithmetic implementation for
// anybody to keep in agreement with the first.
//
// The SFU is the opposite case and lives in tnca_sfu.x: it is approximate by
// contract, and ISA sec 8.8 *specifies* that SFU operations flush subnormals.
// There, XLS's behaviour is the required behaviour.

import tnca_cvt;
import tnca_formats;

// Widen a pair of narrow operands in one go, which is what the matrix unit
// wants: two fragments in one format, an accumulator already in binary32.
pub fn widen_pair(fmt: u3, a: u32, b: u32) -> (u32, u32) {
    (tnca_cvt::widen(fmt, a), tnca_cvt::widen(fmt, b))
}
