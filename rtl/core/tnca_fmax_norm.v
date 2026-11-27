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
// three 49-bit add/subtracts (the low word never carries) and one 3:1 select instead of four 89-bit adders and a 3:1 select. The generated stage
// decides the winner on big - small (without the sticky borrow) and takes big - small - 1 when sticky_al is set; the
// two decisions differ only for big == small_sh with sticky_al = 1, which no operand pair reaches (sticky_al needs the
// small operand shifted by more than 32 bits, so small_sh < 2^47, while a nonzero big is at least 2^55 for every
// binary32 operand and 2^42 for every packed product: tnca_mac.x pm >= 2^10, 32 guard bits). tb_area_stages checks
// the equality on the structs stage 1 produces; the generated stage is the reference.
//
// Written as tnca_fmax_add (the adders) followed by tnca_fmax_sel (select, count, controls) so that the two can be
// separated by a register. Combinational.
//==============================================================================
`default_nettype none
module tnca_fmax_norm #(
    parameter ADD = 0
) (
    input  wire [149:0] s,
    output wire [121:0] out
);
    // combinational composition of the two halves; the adders are plain carry chains here (ADD is kept for the earlier
    // encodings of BE: the marked adders only infer a DSP48E2 when a register follows them, see tnca_fmax_add REG)
    wire [232:0] c;
    tnca_fmax_add #(.REG(0)) u_add (.clk(1'b0), .s(s), .out(c));
    tnca_fmax_sel u_sel (.c(c), .out(out));
endmodule
`default_nettype wire
