//==============================================================================
// tnca_fmax_sel -- stage 2b of the binary32 FMA: select the sum, leading-zero count, shift controls (R42 area)
//
// Second half of tnca_fmax_norm (see tnca_fmax_add for the input struct, 233 bits; the output is the FmaNormalised struct of
// tnca_fmax_norm, 122 bits). Combinational.
//
//   same sign:                    mag = {sum_hi, small_lo}
//   opposite signs, h_hi >= 0:    mag = {h_hi, neg_lo}                  (big wins; the winner's sign is s_big)
//   opposite signs, h_hi <  0:    mag = {dsub_hi, small_lo}             (small wins; s_small)
//==============================================================================
`default_nettype none
module tnca_fmax_sel (
    input  wire [232:0] c,
    output wire [121:0] out
);
    wire [48:0] sum_hi   = c[232:184];
    wire [48:0] dsub_hi  = c[183:135];
    wire [48:0] h_hi     = c[134:86];
    wire [31:0] small_lo = c[85:54];
    wire [31:0] neg_lo   = c[53:22];
    wire        sticky_al  = c[21];
    wire        s_big      = c[20];
    wire        s_small    = c[19];
    wire        sp         = c[18];
    wire        sc         = c[17];
    wire [11:0] exp_common = c[16:5];
    wire        nan_res    = c[4];
    wire        inf_res    = c[3];
    wire        inf_sign   = c[2];
    wire        mul_only   = c[1];
    wire        ftz        = c[0];

    wire same_sign = (s_big == s_small);
    wire h_neg     = h_hi[48];
    // the sum is below 2^81: the top bit of each 49-bit high word is zero for the selected result
    wire [80:0] mag = same_sign ? {sum_hi, small_lo} :
                      h_neg     ? {dsub_hi, small_lo} : {h_hi, neg_lo};
    wire small_wins = ~same_sign & h_neg;

    wire [6:0] c81;
    wire       zero;
    tnca_lzc81 u_lzc (.x(mag), .cnt(c81), .zero(zero));
    wire        any = ~zero;
    wire [6:0]  lz  = zero ? 7'd89 : (c81 + 7'd8);

    wire signed [11:0] lzs = {5'b0, lz};
    wire signed [11:0] ec  = exp_common;
    wire        subnormal_res = any & (lzs > (ec + 12'sd136));
    wire signed [11:0] eb0_n  = (ec + 12'sd137) - lzs;
    wire [11:0] eb0 = subnormal_res ? 12'sd1 : eb0_n;

    wire        shn_left = (lz > 7'd65);
    wire [6:0]  shn_mag  = shn_left ? (lz - 7'd65) : (7'd65 - lz);
    wire signed [11:0] shs = -12'sd71 - ec;
    wire        shs_left = shs[11];
    wire [11:0] shs_mag  = shs_left ? -shs : shs;

    wire sign_zero = (sp == sc) ? sp : 1'b0;
    wire res_sign_mag = small_wins ? s_small : s_big;
    wire res_sign = mul_only ? sp : (any ? res_sign_mag : sign_zero);

    assign out = {mag, any, subnormal_res, shn_left, shn_mag, shs_left, shs_mag, eb0,
                  sticky_al, res_sign, nan_res, inf_res, inf_sign, ftz};
endmodule
`default_nettype wire
