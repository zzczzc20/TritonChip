//==============================================================================
// tnca_fmax_mul, tnca_fmax_alignp, tnca_fmax_align -- stage 1 of the binary32 FMA (R42 area)
//
// The same functions, field for field, as xls/tnca_fma.x:
//   tnca_fmax_mul     fma32_mul      a, b, ftz            -> FmaProduct (64 bits)
//   tnca_fmax_alignp  fma32_align_p  FmaProduct, c, ...   -> FmaAligned (150 bits)
//   tnca_fmax_align   fma32_align    a, b, c, ...         -> FmaAligned (the two composed)
// tnca_mac_lane's B0 stage is fma32_align_p, so the shared lane reuses tnca_fmax_alignp.
//
//   FmaProduct = {pm[48], xpb[12], sp, p_zero, p_inf, p_nan}
//   FmaAligned = {small_sh[80], big_hi[48], sticky_al, s_big, s_small, sp, sc, exp_common[12],
//                 nan_res, inf_res, inf_sign, mul_only, ftz}
//
// What changes is the implementation of the alignment: the smaller operand goes through a staged right shifter that
// ORs the bits it drops into the sticky bit stage by stage (instead of a 88-bit mask, an AND and a reduction), and
// the shift is applied to the 48 significant bits (the 32 guard bits below them are constants). Combinational.
//==============================================================================
`default_nettype none
module tnca_fmax_align (
    input  wire [31:0]  a,
    input  wire [31:0]  b,
    input  wire [31:0]  c,
    input  wire         sub_c,
    input  wire         mul_only,
    input  wire         ftz,
    output wire [149:0] out
);
    wire [63:0] p;
    tnca_fmax_mul    u_m (.a(a), .b(b), .ftz(ftz), .p(p));
    tnca_fmax_alignp u_a (.p(p), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .out(out));
endmodule
`default_nettype wire
