//==============================================================================
// tnca_fmax_unitp -- the binary32 product a * 1.0, as tnca_fmax_mul forms it, without the multiplier (R42 clk2)
//
// tnca_fmax_mul(a, 1.0, ftz) and this module give the same FmaProduct, bit for bit, for every a and ftz (tb/clk2/unitp_miter,
// exhaustive over the 2^33 operand and ftz values): b = 1.0 has biased exponent 127, no fraction, so mb = 2^23 and the
// 24 x 24 multiply is a shift, b is neither zero, infinite nor NaN, and the product's classification is a's alone:
//     pm = a_zero ? 0 : ({~a_e0, fa} << 23) << a_e0        xpb = ea + 127       sp = sa        p_zero = a_zero
//     p_inf = a_inf                                         p_nan = a_nan
// The vector FPU uses it for vf.add, vf.sub and the reduction tree levels (a * 1.0 + c), so that the FMA's multiplier
// takes its operands straight from the operand registers, with no select in front of it (tnca_fmax_fma UNITP).
//==============================================================================
`default_nettype none
module tnca_fmax_unitp (
    input  wire [31:0] a,
    input  wire        ftz,
    output wire [63:0] p
);
    wire        sa = a[31];
    wire [7:0]  ea = a[30:23];
    wire [22:0] fa = a[22:0];
    wire a_e0 = (ea == 8'd0);
    wire a_fz = (fa == 23'd0);
    wire a_zero = a_e0 & (a_fz | ftz);
    wire a_ff = (ea == 8'hff);
    wire a_inf = a_ff & a_fz;
    wire a_nan = a_ff & ~a_fz;
    wire [46:0] pm_raw = {~a_e0, fa, 23'd0};
    wire [47:0] pm = a_zero ? 48'd0 : ({1'b0, pm_raw} << a_e0);
    wire [11:0] xpb = {4'b0, ea} + 12'd127;
    assign p = {pm, xpb, sa, a_zero, a_inf, a_nan};
endmodule
`default_nettype wire
