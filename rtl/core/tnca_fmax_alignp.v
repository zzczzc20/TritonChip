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
module tnca_fmax_alignp (
    input  wire [63:0]  p,
    input  wire [31:0]  c,
    input  wire         sub_c,
    input  wire         mul_only,
    input  wire         ftz,
    output wire [149:0] out
);
    wire [47:0] pm = p[63:16];
    wire signed [11:0] kab = p[15:4];
    wire sp = p[3], p_zero = p[2], p_inf = p[1], p_nan = p[0];

    wire        sc  = c[31] ^ sub_c;
    wire [7:0]  ec  = c[30:23];
    wire [22:0] fc  = c[22:0];
    wire c_e0 = (ec == 8'd0);
    wire c_fz = (fc == 23'd0);
    wire c_ff = (ec == 8'hff);
    wire c_zero = (c_e0 & (c_fz | ftz)) | mul_only;
    wire c_inf  = c_ff & c_fz & ~mul_only;
    wire c_nan  = c_ff & ~c_fz & ~mul_only;
    wire [23:0] mc = c_zero ? 24'd0 : (c_e0 ? {fc, 1'b0} : {1'b1, fc});

    wire signed [11:0] kcb  = {4'b0, ec} + 12'sd127;
    wire signed [11:0] dexp  = kab - kcb;
    wire signed [11:0] ndexp = kcb - kab;
    wire d_neg = (kab < kcb);
    wire signed [11:0] dabs = d_neg ? ndexp : dexp;
    wire [7:0] shamt = (dabs > 12'sd80) ? 8'd80 : dabs[7:0];
    wire shift_c = c_zero ? 1'b1 : (p_zero ? 1'b0 : ~d_neg);

    wire signed [11:0] xp = kab - 12'sd254;
    wire signed [11:0] xc = {4'b0, ec} - 12'sd127;
    wire [11:0] exp_common = shift_c ? xp : xc;
    wire s_big   = shift_c ? sp : sc;
    wire s_small = shift_c ? sc : sp;
    wire inf_minus_inf = p_inf & c_inf & (sp != sc);
    wire nan_res  = p_nan | c_nan | inf_minus_inf;
    wire inf_res  = p_inf | c_inf;
    wire inf_sign = p_inf ? sp : sc;

    // the two operands' 48 significant bits (field bits [79:32]); the 32 bits below are zero
    wire [47:0] c48 = {1'b0, mc, 23'd0};
    wire [47:0] big_hi   = shift_c ? pm  : c48;
    wire [47:0] small48  = shift_c ? c48 : pm;

    // staged right shift of {small48, 32'b0} by shamt (0..80), the dropped bits OR-ed into sticky as they go
    wire [79:0] v0 = {small48, 32'd0};
    wire        l6 = shamt[6] & (|v0[63:0]);
    wire [79:0] v1 = shamt[6] ? (v0 >> 64) : v0;
    wire        l5 = shamt[5] & (|v1[31:0]);
    wire [79:0] v2 = shamt[5] ? (v1 >> 32) : v1;
    wire        l4 = shamt[4] & (|v2[15:0]);
    wire [79:0] v3 = shamt[4] ? (v2 >> 16) : v2;
    wire        l3 = shamt[3] & (|v3[7:0]);
    wire [79:0] v4 = shamt[3] ? (v3 >> 8) : v3;
    wire        l2 = shamt[2] & (|v4[3:0]);
    wire [79:0] v5 = shamt[2] ? (v4 >> 4) : v4;
    wire        l1 = shamt[1] & (|v5[1:0]);
    wire [79:0] v6 = shamt[1] ? (v5 >> 2) : v5;
    wire        l0 = shamt[0] & v6[0];
    wire [79:0] v7 = shamt[0] ? (v6 >> 1) : v6;
    wire        sticky_al = l6 | l5 | l4 | l3 | l2 | l1 | l0;
    wire [79:0] small_sh = v7;

    assign out = {small_sh, big_hi, sticky_al, s_big, s_small, sp, sc, exp_common,
                  nan_res, inf_res, inf_sign, mul_only, ftz};
endmodule
`default_nettype wire
