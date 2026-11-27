//==============================================================================
// tnca_fmax_add -- stage 2a of the binary32 FMA: the add/subtracts (R42 area)
//
// First half of tnca_fmax_norm, split at the adders so that a register can follow them (a DSP48E2 takes its P register
// there). Input: FmaAligned (150 bits, see tnca_fmax_norm). With big = big_hi << 32, small = small_sh:
//
//   sum_hi  = big_hi + small_hi               (49 bits)   same sign: sum = {sum_hi, small_lo}
//   dsub_hi = small_hi - big_hi               (49 bits)   opposite signs, small wins: {dsub_hi, small_lo}
//   h_hi    = big_hi - small_hi - bor         (49 bits)   opposite signs, big wins: {h_hi, neg_lo}; sign = h_hi[48]
//   neg_lo  = (-(small_lo + sticky_al)) mod 2^32,  bor = (small_lo != 0) | sticky_al
//   small_lo = small_sh[31:0]
//
// (big_lo = 0, so neither sum nor the winner-small difference has a carry or borrow out of the low word.)
//
//   out = {sum_hi[49], dsub_hi[49], h_hi[49], small_lo[32], neg_lo[32],
//          sticky_al, s_big, s_small, sp, sc, exp_common[12], nan_res, inf_res, inf_sign, mul_only, ftz}     233 bits
//
// REG = 1: the outputs are registers (clk), the three 49-bit results marked use_dsp so that Vivado infers one DSP48E2 each,
// the other fields passing through plain registers; REG = 0: combinational.
//==============================================================================
`default_nettype none
module tnca_fmax_add #(
    parameter REG = 0
) (
    input  wire         clk,
    input  wire [149:0] s,
    output wire [232:0] out
);
    wire [79:0] small_sh   = s[149:70];
    wire [47:0] big_hi     = s[69:22];
    wire [31:0] small_lo   = small_sh[31:0];
    wire [47:0] small_hi   = small_sh[79:32];
    wire        sticky_al  = s[21];
    wire [21:0] flags      = s[21:0];            // sticky_al .. ftz, unchanged

    wire bor = (|small_lo) | sticky_al;
    wire [31:0] neg_lo_w = ~small_lo + {31'b0, ~sticky_al};

    generate
        if (REG != 0) begin : g_reg
            (* use_dsp = "yes" *) reg [48:0] sum_q;
            (* use_dsp = "yes" *) reg [48:0] dsub_q;
            (* use_dsp = "yes" *) reg [48:0] h_q;
            reg [31:0] lo_q, neg_q;
            reg [21:0] fl_q;
            always @(posedge clk) begin
                sum_q  <= {1'b0, big_hi} + {1'b0, small_hi};
                dsub_q <= {1'b0, small_hi} - {1'b0, big_hi};
                h_q    <= {1'b0, big_hi} - {1'b0, small_hi} - {48'b0, bor};
                lo_q   <= small_lo;
                neg_q  <= neg_lo_w;
                fl_q   <= flags;
            end
            assign out = {sum_q, dsub_q, h_q, lo_q, neg_q, fl_q};
        end else begin : g_comb
            wire [48:0] sum_w  = {1'b0, big_hi} + {1'b0, small_hi};
            wire [48:0] dsub_w = {1'b0, small_hi} - {1'b0, big_hi};
            wire [48:0] h_w    = {1'b0, big_hi} - {1'b0, small_hi} - {48'b0, bor};
            assign out = {sum_w, dsub_w, h_w, small_lo, neg_lo_w, flags};
        end
    endgenerate
endmodule
`default_nettype wire
