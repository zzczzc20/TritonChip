//==============================================================================
// tnca_fmax_norm -- stage 2 of the binary32 FMA: add or subtract, leading-zero count, shift controls (R42 area)
//
// Same ports and the same function, field for field, as tnca_fmad_norm_xls (xls/tnca_fmad.x, fma32d_normalise;
// the struct layout is the DSLX struct's: first field at the top):
//
//   s   = FmaAligned    {small_sh[80], big_hi[48], sticky_al, s_big, s_small, sp, sc, exp_common[12],
//                        nan_res, inf_res, inf_sign, mul_only, ftz}                          150 bits
//   out = FmaNormalised {mag[81], any, subnormal_res, shn_left, shn_mag[7], shs_left, shs_mag[12], eb0[12],
//                        sticky_al, res_sign, nan_res, inf_res, inf_sign, ftz}               122 bits
//
// The arithmetic is re-expressed, not changed. With big = big_hi << 32 and small = small_sh:
//
//   R1 = small + big           (same sign)          R1 = small - big  (opposite signs: the sum when small wins)
//   H  = big - small - sticky_al                    (opposite signs: the sum when big wins; its sign says who wins)
//   mag = (same_sign | H < 0) ? R1 : H
//
// two 80-bit add/subtracts and one 2:1 select instead of four 89-bit adders and a 3:1 select. The generated stage
// decides the winner on big - small (without the sticky borrow) and takes big - small - 1 when sticky_al is set; the
// two decisions differ only for big == small_sh with sticky_al = 1, which no operand pair reaches (sticky_al needs the
// small operand shifted by more than 32 bits, so small_sh < 2^47, while a nonzero big is at least 2^55 for every
// binary32 operand and 2^42 for every packed product: tnca_mac.x pm >= 2^10, 32 guard bits). tb_area_stages checks
// the equality on the structs stage 1 produces; the generated stage is the reference.
//
// ADD = 0: plain behavioural adders (carry chains).  ADD = 1: the same adders marked for DSP48E2 inference.
// Combinational.
//==============================================================================
`default_nettype none
module tnca_fmax_norm #(
    parameter ADD = 0
) (
    input  wire [149:0] s,
    output wire [121:0] out
);
    wire [79:0] small_sh   = s[149:70];
    wire [47:0] big_hi     = s[69:22];
    wire        sticky_al  = s[21];
    wire        s_big      = s[20];
    wire        s_small    = s[19];
    wire        sp         = s[18];
    wire        sc         = s[17];
    wire [11:0] exp_common = s[16:5];
    wire        nan_res    = s[4];
    wire        inf_res    = s[3];
    wire        inf_sign   = s[2];
    wire        mul_only   = s[1];
    wire        ftz        = s[0];

    wire same_sign = (s_big == s_small);
    wire [79:0] big = {big_hi, 32'b0};

    // ---- the two add/subtracts
    wire [80:0] r1, h;
    generate
        if (ADD == 0) begin : g_lut
            assign r1 = same_sign ? ({1'b0, small_sh} + {1'b0, big}) : ({1'b0, small_sh} - {1'b0, big});
            assign h  = {1'b0, big} - {1'b0, small_sh} - {80'b0, sticky_al};
        end else begin : g_dsp
            (* use_dsp = "yes" *) wire [80:0] r1_w = same_sign ? ({1'b0, small_sh} + {1'b0, big}) : ({1'b0, small_sh} - {1'b0, big});
            (* use_dsp = "yes" *) wire [80:0] h_w  = {1'b0, big} - {1'b0, small_sh} - {80'b0, sticky_al};
            assign r1 = r1_w;
            assign h  = h_w;
        end
    endgenerate
    wire small_wins = ~same_sign & h[80];
    wire [80:0] mag = (same_sign | h[80]) ? r1 : h;

    // ---- leading-zero count (the generated stage counts over 89 bits; mag is below 2^81, so +8)
    wire [6:0] c81;
    wire       zero;
    tnca_lzc81 u_lzc (.x(mag), .cnt(c81), .zero(zero));
    wire        any = ~zero;
    wire [6:0]  lz  = zero ? 7'd89 : (c81 + 7'd8);           // 8..88 for a nonzero sum, 89 for zero

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
