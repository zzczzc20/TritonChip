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
// WHY THIS IS A COMPOSITION.
//
// The widening comes from XLS (rtl/gen/tnca_widen_xls.v), verified
// exhaustively against an exact oracle, and so, since R38, does the
// arithmetic: tnca_fp32_fma picks the generated instance (xls/tnca_fma.x,
// written without XLS's apfloat because that library flushes subnormals and
// this ISA does not allow it outside the ftz mode). One multiplier, five
// formats, and LAT -- the matrix unit's FMA_LAT -- choosing how deeply it is
// pipelined.
//==============================================================================
`default_nettype none

module tnca_fpu_multi #(
    parameter LAT = 2               // the FMA's latency (tnca_fp32_fma)
) (
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

    tnca_fp32_fma #(.LAT(LAT)) u_fma (
        .clk(clk), .rst_n(rst_n),
        .a(a_f32), .b(b_f32), .c(c),
        .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz),
        .y(y)
    );
endmodule

`default_nettype wire
