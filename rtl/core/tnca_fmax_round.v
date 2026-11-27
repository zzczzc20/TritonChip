//==============================================================================
// tnca_fmax_round -- stage 3 of the binary32 FMA: normalising shift, round to nearest even, pack (R42 area)
//
// Same ports and the same function as tnca_fma_round_xls (xls/tnca_fma.x, fma32_round), written out so that the
// shifter can be restructured. Combinational; the consumer registers y.
//
//   s = FmaNormalised {mag[81], any, subnormal_res, shn_left, shn_mag[7], shs_left, shs_mag[12], eb0[12],
//                      sticky_al, res_sign, nan_res, inf_res, inf_sign, ftz}                  122 bits
//==============================================================================
`default_nettype none
module tnca_fmax_round #(
    parameter SHIFT = 0
) (
    input  wire [121:0] s,
    output wire [31:0]  out
);
    wire [80:0] mag       = s[121:41];
    wire        any       = s[40];
    wire        subn      = s[39];
    wire        shn_left  = s[38];
    wire [6:0]  shn_mag   = s[37:31];
    wire        shs_left  = s[30];
    wire [11:0] shs_mag   = s[29:18];
    wire signed [11:0] eb0 = s[17:6];
    wire        sticky_al = s[5];
    wire        res_sign  = s[4];
    wire        nan_res   = s[3];
    wire        inf_res   = s[2];
    wire        inf_sign  = s[1];
    wire        ftz       = s[0];

    wire        sh_left = subn ? shs_left : shn_left;
    wire [11:0] sh_mag  = subn ? shs_mag : {5'b0, shn_mag};

    wire [89:0] mag_x90 = {8'b0, mag, 1'b0};        // mag with a zero appended below (bit 0 is the guard position's right neighbour)
    wire [88:0] mag89   = {8'b0, mag};
    wire [23:0] sig;
    wire        guard, rest;
    generate
        if (SHIFT == 0) begin : g_beh
            // behavioural: one shifter, a mask and a reduction
            wire [89:0] t = (sh_mag >= 12'd90) ? 90'b0 : (mag_x90 >> sh_mag);
            wire [88:0] ml = (sh_mag >= 12'd89) ? 89'b0 : (mag89 << sh_mag);
            wire [89:0] mask = (sh_mag >= 12'd90) ? {90{1'b1}} : ((90'd1 << sh_mag) - 90'd1);
            assign sig   = sh_left ? ml[23:0] : t[24:1];
            assign guard = sh_left ? 1'b0 : t[0];
            assign rest  = sh_left ? 1'b0 : ((mag_x90 & mask) != 90'b0);
        end else begin : g_stg
            // staged: the right shifter drops bits stage by stage and ORs the dropped bits into a sticky as it goes;
            // the left case (at most 23 for a nonzero sum) is a 24-bit left shifter. Same function, no mask.
            wire big = (sh_mag >= 12'd90);                 // shifted past the field: nothing remains
            wire [6:0] k = sh_mag[6:0];
            wire [89:0] v0 = mag_x90;
            wire        l6 = k[6] & (|v0[63:0]);
            wire [89:0] v1 = k[6] ? (v0 >> 64) : v0;
            wire        l5 = k[5] & (|v1[31:0]);
            wire [89:0] v2 = k[5] ? (v1 >> 32) : v1;
            wire        l4 = k[4] & (|v2[15:0]);
            wire [89:0] v3 = k[4] ? (v2 >> 16) : v2;
            wire        l3 = k[3] & (|v3[7:0]);
            wire [89:0] v4 = k[3] ? (v3 >> 8) : v3;
            wire        l2 = k[2] & (|v4[3:0]);
            wire [89:0] v5 = k[2] ? (v4 >> 4) : v4;
            wire        l1 = k[1] & (|v5[1:0]);
            wire [89:0] v6 = k[1] ? (v5 >> 2) : v5;
            wire        l0 = k[0] & v6[0];
            wire [89:0] v7 = k[0] ? (v6 >> 1) : v6;
            wire [24:0] t  = big ? 25'b0 : v7[24:0];
            wire        lost = l6 | l5 | l4 | l3 | l2 | l1 | l0;
            wire        any_mag = |mag;
            wire        rest_r = big ? any_mag : lost;
            // left shift of the low 24 bits by sh (< 24 for a nonzero sum; past 23 the window is zero anyway)
            wire [23:0] w0 = mag[23:0];
            wire [23:0] w1 = sh_mag[4] ? (w0 << 16) : w0;
            wire [23:0] w2 = sh_mag[3] ? (w1 << 8) : w1;
            wire [23:0] w3 = sh_mag[2] ? (w2 << 4) : w2;
            wire [23:0] w4 = sh_mag[1] ? (w3 << 2) : w3;
            wire [23:0] w5 = sh_mag[0] ? (w4 << 1) : w4;
            wire        lbig = |sh_mag[11:5];
            wire [23:0] ml = lbig ? 24'b0 : w5;
            assign sig   = sh_left ? ml : t[24:1];
            assign guard = sh_left ? 1'b0 : t[0];
            assign rest  = sh_left ? 1'b0 : rest_r;
        end
    endgenerate
    wire        sticky = rest | sticky_al;

    wire round_up   = guard & (sticky | sig[0]);
    wire sig_ones   = &sig;
    wire carry_out  = round_up & sig_ones;
    wire [23:0] sig_inc = sig + 24'd1;
    wire [23:0] sig_f   = ~round_up ? sig : (sig_ones ? 24'h800000 : sig_inc);
    wire res_is_norm = sig_f[23];

    wire signed [11:0] e_biased = carry_out ? (eb0 + 12'sd1) : eb0;
    wire overflow = any & (carry_out ? (eb0 >= 12'sd254) : (eb0 >= 12'sd255));
    wire exact_zero  = ~any;
    wire result_zero = exact_zero | ((sig == 24'd0) & ~round_up);

    wire [31:0] normal_result =
        overflow     ? {res_sign, 8'hff, 23'h0} :
        result_zero  ? {res_sign, 31'h0} :
        ~res_is_norm ? (ftz ? {res_sign, 31'h0} : {res_sign, 8'h00, sig_f[22:0]}) :
                       {res_sign, e_biased[7:0], sig_f[22:0]};
    assign out = nan_res ? 32'h7fc00000 :
                 inf_res ? {inf_sign, 8'hff, 23'h0} : normal_result;
endmodule
`default_nettype wire
