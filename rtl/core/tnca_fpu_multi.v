//==============================================================================
// tnca_fpu_multi -- multi-precision FMA, by composition
//
// Operands arrive in any of the five formats (ISA sec 8.5 as extended:
// FP32, BF16, FP16, FP8 E5M2, FP8 E4M3), the product and the accumulation
// happen in binary32, and the result leaves in binary32 -- which is the
// matrix unit's contract verbatim (ISA sec 9C.1):
//
//     C[k+1] = RNE_f32( exact(A[k]) * exact(B[k]) + C[k] )
//
// Every one of these formats widens to binary32 exactly, so "widen then use
// the binary32 datapath" *is* that contract rather than an approximation of
// it.
//
// WHY THIS IS A WRAPPER AND NOT GENERATED ARITHMETIC.
//
// The widening and narrowing come from XLS (rtl/gen/tnca_widen_xls.v),
// verified exhaustively against an exact oracle. The arithmetic does not.
// XLS's apfloat library flushes subnormals to zero -- measured, see
// xls/tnca_fpu.x -- and this ISA makes FTZ an explicitly encoded mode rather
// than something an implementation may do on its own (sec 12.4, sec 12.7).
// tnca_fp32_fma already handles subnormals and is checked against an exact
// Fraction oracle over half a million vectors, so it keeps the arithmetic
// and XLS does the format work around it. One verified multiplier, five
// formats, and no second implementation for anyone to keep in agreement
// with the first.
//==============================================================================
`default_nettype none

module tnca_fpu_multi (
    input  wire        clk,
    input  wire        rst_n,

    // Operand format, shared by a and b. Codes match xls/tnca_formats.x.
    input  wire [2:0]  fmt,
    // a and b are right-aligned in their format's width; c is always the
    // binary32 accumulator, as the matrix unit holds it.
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,

    output wire [31:0] y
);
    wire [31:0] a_f32, b_f32;

    // Exact widening. Combinational and shallow -- measured at 11 levels,
    // against the 45 the narrowing side costs -- so it sits in front of the
    // existing FMA without moving its timing.
    tnca_widen_xls u_wa (.fmt(fmt), .x(a), .out(a_f32));
    tnca_widen_xls u_wb (.fmt(fmt), .x(b), .out(b_f32));

    tnca_fp32_fma u_fma (
        .clk(clk), .rst_n(rst_n),
        .a(a_f32), .b(b_f32), .c(c),
        .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz),
        .y(y)
    );
endmodule

`default_nettype wire
