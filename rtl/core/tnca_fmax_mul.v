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
module tnca_fmax_mul (
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire        ftz,
    output wire [63:0] p
);
    wire        sa = a[31], sb = b[31];
    wire [7:0]  ea = a[30:23], eb = b[30:23];
    wire [22:0] fa = a[22:0],  fb = b[22:0];
    wire a_e0 = (ea == 8'd0), b_e0 = (eb == 8'd0);
    wire a_fz = (fa == 23'd0), b_fz = (fb == 23'd0);
    wire a_zero = a_e0 & (a_fz | ftz);
    wire b_zero = b_e0 & (b_fz | ftz);
    wire a_ff = (ea == 8'hff), b_ff = (eb == 8'hff);
    wire a_inf = a_ff & a_fz, b_inf = b_ff & b_fz;
    wire a_nan = a_ff & ~a_fz, b_nan = b_ff & ~b_fz;
    wire [23:0] ma = {~a_e0, fa};
    wire [23:0] mb = {~b_e0, fb};
    wire        sp = sa ^ sb;
    wire        p_zero = a_zero | b_zero;
    wire [1:0]  p_dbl = {1'b0, a_e0} + {1'b0, b_e0};
    wire [47:0] pm_raw = ma * mb;
    wire [47:0] pm = p_zero ? 48'd0 : (pm_raw << p_dbl);
    wire inf_times_zero = (a_inf & b_zero) | (b_inf & a_zero);
    wire p_inf = (a_inf | b_inf) & ~inf_times_zero;
    wire p_nan = a_nan | b_nan | inf_times_zero;
    wire [11:0] xpb = {4'b0, ea} + {4'b0, eb};
    assign p = {pm, xpb, sp, p_zero, p_inf, p_nan};
endmodule
`default_nettype wire
