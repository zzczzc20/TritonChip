//==============================================================================
// tnca_fp32_fma -- IEEE 754 binary32 fused multiply-add, single rounding
//
//     y = round_RNE( a * b + (sub_c ? -c : c) )
//
// with mul_only=1 returning round_RNE(a * b) and the addend ignored entirely.
// The vector FP unit builds vf.add / vf.sub from this by passing b = 1.0, which
// is exact, so every FP arithmetic result in the core goes through one rounding
// in one place.
//
// PIPELINE
// --------
// Two register stages; `y` is valid two cycles after a/b/c are presented, and a
// new operation may be presented every cycle. The cuts are:
//
//   S0  unpack, classify, 24x24 multiply, exponent difference
//   --  register
//   S1  alignment shift, add/subtract, magnitude ordering
//   --  register
//   S2  leading-one search, normalise, round, assemble
//
// The combinational version of this module measured 207 gate levels, which is
// roughly 3-5 ns of logic -- about a quarter of the 1 GHz target. The cuts are
// placed either side of the two things that dominate that: the multiply array
// and the align/add, and then the normalise/round.
//
// Handles subnormal inputs and subnormal results. ftz=1 flushes subnormal
// inputs and results to zero of the same sign -- an explicit mode, per ISA
// sec 8.2, never an implicit shortcut.
//
// ALIGNMENT SCHEME
// ----------------
// Both the 48-bit product P and the addend significand MC = mant_c << 23 are
// expressed at scale 2^(e-46), where e is the respective exponent sum. The
// operand with the smaller exponent is shifted right (with a sticky bit) onto
// the other's scale, so the guard field below the operands only has to absorb
// near-equal-exponent cases -- which are exactly the cases that can cancel.
//
//   acc[87:0]:  [87:80] carry headroom
//               [79:32] operand field (48 bits)
//               [31:0]  guard/sticky field
//   value(acc) = acc * 2^(exp_common - 78)
//==============================================================================

module tnca_fp32_fma (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,       // negate the addend (vf.fms, vf.sub)
    input  wire        mul_only,    // ignore the addend (vf.mul)
    input  wire        ftz,
    output reg  [31:0] y
);

    localparam [31:0] QNAN = 32'h7fc00000;

    //==========================================================================
    // S0 -- unpack, classify, multiply
    //==========================================================================
    wire        sa_ = a[31], sb_ = b[31], sc_raw = c[31];
    wire [7:0]  ea_ = a[30:23], eb_ = b[30:23], ec_ = c[30:23];
    wire [22:0] fa_ = a[22:0],  fb_ = b[22:0],  fc_ = c[22:0];

    wire sc_ = sc_raw ^ sub_c;

    wire a_sub = (ea_ == 8'd0) && (fa_ != 23'd0);
    wire b_sub = (eb_ == 8'd0) && (fb_ != 23'd0);
    wire c_sub = (ec_ == 8'd0) && (fc_ != 23'd0);

    // ftz flushes subnormal inputs before anything else looks at them
    wire a_zero = (ea_ == 8'd0) && ((fa_ == 23'd0) || ftz);
    wire b_zero = (eb_ == 8'd0) && ((fb_ == 23'd0) || ftz);
    wire c_zero = ((ec_ == 8'd0) && ((fc_ == 23'd0) || ftz)) || mul_only;

    wire a_inf  = (ea_ == 8'hff) && (fa_ == 23'd0);
    wire b_inf  = (eb_ == 8'hff) && (fb_ == 23'd0);
    wire c_inf  = (ec_ == 8'hff) && (fc_ == 23'd0) && !mul_only;

    wire a_nan  = (ea_ == 8'hff) && (fa_ != 23'd0);
    wire b_nan  = (eb_ == 8'hff) && (fb_ != 23'd0);
    wire c_nan  = (ec_ == 8'hff) && (fc_ != 23'd0) && !mul_only;

    // Significands as 24-bit integers; value = mant * 2^(exp - 23).
    // Subnormals use the IEEE minimum exponent with a zero leading bit.
    wire [23:0] ma = a_zero ? 24'd0 : (a_sub ? {1'b0, fa_} : {1'b1, fa_});
    wire [23:0] mb = b_zero ? 24'd0 : (b_sub ? {1'b0, fb_} : {1'b1, fb_});
    wire [23:0] mc = c_zero ? 24'd0 : (c_sub ? {1'b0, fc_} : {1'b1, fc_});

    // Unbiased exponents, with the subnormal convention e = 1 - 127.
    wire signed [11:0] xa = a_sub ? -12'sd126 : ($signed({4'd0, ea_}) - 12'sd127);
    wire signed [11:0] xb = b_sub ? -12'sd126 : ($signed({4'd0, eb_}) - 12'sd127);
    wire signed [11:0] xc = c_sub ? -12'sd126 : ($signed({4'd0, ec_}) - 12'sd127);

    wire        s0_sp = sa_ ^ sb_;
    wire [47:0] s0_pm = ma * mb;                 // value = pm * 2^(xa+xb-46)
    wire signed [11:0] s0_xp = xa + xb;

    // Addend restated at the same -46 scale: MC = mc << 23.
    wire [46:0] s0_mcw = {mc, 23'd0};            // value = mcw * 2^(xc-46)

    wire signed [12:0] dexp = $signed({s0_xp[11], s0_xp}) - $signed({xc[11], xc});

    // Shift amount, clamped: past 80 bits of displacement the smaller operand
    // can only contribute a sticky bit.
    wire [12:0] dabs = dexp[12] ? (~dexp + 13'd1) : dexp;
    wire [7:0]  s0_shamt = (dabs > 13'd80) ? 8'd80 : dabs[7:0];

    // Shift the operand with the smaller exponent. A zero significand must never
    // be chosen as the reference operand: its exponent field is meaningless, and
    // with mul_only the addend's exponent is whatever the caller left in c.
    wire c_zero_mag = (s0_mcw == 47'd0);
    wire p_zero_mag = (s0_pm  == 48'd0);
    wire s0_shift_c = c_zero_mag ? 1'b1 : (p_zero_mag ? 1'b0 : ~dexp[12]);

    wire signed [11:0] s0_exp_common = s0_shift_c ? s0_xp : xc;
    wire        s0_s_big   = s0_shift_c ? s0_sp : sc_;
    wire        s0_s_small = s0_shift_c ? sc_ : s0_sp;

    // Special-case resolution is settled here, where the classification lives.
    wire inf_times_zero = (a_inf & b_zero) | (b_inf & a_zero);
    wire p_inf          = (a_inf | b_inf) & ~inf_times_zero;
    wire inf_minus_inf  = p_inf & c_inf & (s0_sp != sc_);

    wire s0_nan_res  = a_nan | b_nan | c_nan | inf_times_zero | inf_minus_inf;
    wire s0_inf_res  = p_inf | c_inf;
    wire s0_inf_sign = p_inf ? s0_sp : sc_;

    //==========================================================================
    // Register stage 0
    //==========================================================================
    reg [47:0]        p0_pm;
    reg [46:0]        p0_mcw;
    reg [7:0]         p0_shamt;
    reg               p0_shift_c;
    reg               p0_s_big, p0_s_small;
    reg signed [11:0] p0_exp_common;
    reg               p0_sp, p0_sc;
    reg               p0_mul_only, p0_ftz;
    reg               p0_nan_res, p0_inf_res, p0_inf_sign;

    always @(posedge clk) begin
        if (!rst_n) begin
            p0_pm <= 48'd0; p0_mcw <= 47'd0; p0_shamt <= 8'd0;
            p0_shift_c <= 1'b0; p0_s_big <= 1'b0; p0_s_small <= 1'b0;
            p0_exp_common <= 12'sd0; p0_sp <= 1'b0; p0_sc <= 1'b0;
            p0_mul_only <= 1'b0; p0_ftz <= 1'b0;
            p0_nan_res <= 1'b0; p0_inf_res <= 1'b0; p0_inf_sign <= 1'b0;
        end else begin
            p0_pm         <= s0_pm;
            p0_mcw        <= s0_mcw;
            p0_shamt      <= s0_shamt;
            p0_shift_c    <= s0_shift_c;
            p0_s_big      <= s0_s_big;
            p0_s_small    <= s0_s_small;
            p0_exp_common <= s0_exp_common;
            p0_sp         <= s0_sp;
            p0_sc         <= sc_;
            p0_mul_only   <= mul_only;
            p0_ftz        <= ftz;
            p0_nan_res    <= s0_nan_res;
            p0_inf_res    <= s0_inf_res;
            p0_inf_sign   <= s0_inf_sign;
        end
    end

    //==========================================================================
    // S1 -- align, add/subtract
    //==========================================================================
    wire [87:0] p_field = {8'd0, p0_pm,  32'd0};      // [79:32]
    wire [87:0] c_field = {9'd0, p0_mcw, 32'd0};      // [78:32]

    wire [87:0] big_v   = p0_shift_c ? p_field : c_field;
    wire [87:0] small_v = p0_shift_c ? c_field : p_field;

    wire [87:0] small_sh = small_v >> p0_shamt;
    // Sticky: anything shifted past bit 0 of the 88-bit field.
    wire        sticky_al = |(small_v & ~({88{1'b1}} << p0_shamt));

    wire same_sign = (p0_s_big == p0_s_small);

    // Equal exponents do NOT imply |big| >= |small|: the product significand
    // spans [1,4) while the addend's spans [1,2), so at dexp == 0 the addend can
    // still be the larger magnitude. The subtract path therefore compares the
    // aligned magnitudes rather than trusting the exponent ordering.
    wire small_wins = (small_sh > big_v);

    wire [88:0] sum_add = {1'b0, big_v} + {1'b0, small_sh};
    // big - (small_sh + eps), where eps in [0,1) is the shifted-out remainder:
    // the borrow accounts for eps and leaves (1 - eps) as the new sticky.
    wire [88:0] sub_bs  = {1'b0, big_v} - {1'b0, small_sh} - {88'd0, sticky_al};
    // (small_sh + eps) - big: eps stays below bit 0 and is the new sticky.
    wire [88:0] sub_sb  = {1'b0, small_sh} - {1'b0, big_v};

    wire [88:0] s1_mag = same_sign ? sum_add : (small_wins ? sub_sb : sub_bs);
    wire        s1_res_sign_mag = same_sign ? p0_s_big
                                            : (small_wins ? p0_s_small : p0_s_big);

    //==========================================================================
    // Register stage 1
    //==========================================================================
    reg [88:0]        p1_mag;
    reg signed [11:0] p1_exp_common;
    reg               p1_res_sign_mag, p1_sticky_op;
    reg               p1_sp, p1_sc, p1_mul_only, p1_ftz;
    reg               p1_nan_res, p1_inf_res, p1_inf_sign;

    always @(posedge clk) begin
        if (!rst_n) begin
            p1_mag <= 89'd0; p1_exp_common <= 12'sd0;
            p1_res_sign_mag <= 1'b0; p1_sticky_op <= 1'b0;
            p1_sp <= 1'b0; p1_sc <= 1'b0; p1_mul_only <= 1'b0; p1_ftz <= 1'b0;
            p1_nan_res <= 1'b0; p1_inf_res <= 1'b0; p1_inf_sign <= 1'b0;
        end else begin
            p1_mag          <= s1_mag;
            p1_exp_common   <= p0_exp_common;
            p1_res_sign_mag <= s1_res_sign_mag;
            p1_sticky_op    <= sticky_al;
            p1_sp           <= p0_sp;
            p1_sc           <= p0_sc;
            p1_mul_only     <= p0_mul_only;
            p1_ftz          <= p0_ftz;
            p1_nan_res      <= p0_nan_res;
            p1_inf_res      <= p0_inf_res;
            p1_inf_sign     <= p0_inf_sign;
        end
    end

    //==========================================================================
    // S2 -- normalise, round, assemble
    //==========================================================================
    // The leading-one search is the single deepest thing in this stage. As a
    // scan loop it is 89 chained muxes; tnca_lzc does it in O(log W).
    wire [7:0] lzc89;
    wire [6:0] msb7;
    wire       any;
    tnca_lzc #(.W(89), .LOGW(7)) u_lzc (
        .v(p1_mag), .clz(lzc89), .msb_idx(msb7), .any(any)
    );
    wire [7:0] msb = {1'b0, msb7};

    // value(mag) = mag * 2^(exp_common - 78); normalised value = 1.f * 2^E
    wire signed [11:0] e_norm = $signed({4'd0, msb}) + p1_exp_common - 12'sd78;

    // Right shift that leaves 24 significant bits, plus guard and round.
    wire signed [11:0] sh_norm_s = $signed({4'd0, msb}) - 12'sd23;

    // Subnormal results need a larger right shift so the exponent stops at -126.
    wire signed [11:0] e_min = -12'sd126;
    wire subnormal_res = (e_norm < e_min) && any;
    wire signed [11:0] sh_extra = subnormal_res ? (e_min - e_norm) : 12'sd0;
    wire signed [11:0] sh_tot_s = sh_norm_s + sh_extra;

    wire sh_left = (sh_tot_s < 0);
    wire [11:0] sh_mag = sh_left ? (-sh_tot_s) : sh_tot_s;
    wire [7:0]  sh_r = (sh_mag > 12'd100) ? 8'd100 : sh_mag[7:0];

    wire [88:0] shifted = sh_left ? (p1_mag << sh_r) : (p1_mag >> sh_r);

    // Guard and sticky are taken by shifting rather than by bit-select, so a
    // shift amount past the width of `mag` (which happens for results far below
    // the subnormal range) yields 0 instead of an out-of-range X.
    wire [7:0]  sh_r1 = (sh_r == 8'd0) ? 8'd0 : (sh_r - 8'd1);
    wire [88:0] mag_g = p1_mag >> sh_r1;

    wire [23:0] sig   = shifted[23:0];
    wire        guard = sh_left ? 1'b0 : ((sh_r == 8'd0) ? 1'b0 : mag_g[0]);
    wire        rest  = sh_left ? 1'b0
                                : ((sh_r < 8'd2) ? 1'b0
                                                 : |(p1_mag & ~({89{1'b1}} << sh_r1)));
    wire        sticky = rest | p1_sticky_op;

    // round to nearest even
    wire round_up = guard & (sticky | sig[0]);
    wire [24:0] sig_r = {1'b0, sig} + {24'd0, round_up};

    // A rounding carry out of bit 23 bumps the exponent.
    wire carry_out = sig_r[24];
    wire [23:0] sig_f = carry_out ? sig_r[24:1] : sig_r[23:0];
    wire signed [11:0] e_f = (subnormal_res ? e_min : e_norm)
                           + (carry_out ? 12'sd1 : 12'sd0);

    // A subnormal that rounded up to 1.0 becomes the smallest normal, which the
    // encoding below produces naturally because sig_f[23] is then set.
    wire res_is_norm = sig_f[23];

    wire signed [11:0] e_biased = e_f + 12'sd127;
    wire overflow  = any && (e_biased >= 12'sd255);

    // Two different zeros, and they do not get the same sign:
    //
    //   exact zero      -- the operands annihilated, or both were zero. IEEE
    //                      RNE gives +0 unless both addends shared a sign.
    //   rounded-to-zero -- the exact result was a nonzero magnitude that
    //                      underflowed. It keeps that magnitude's sign, so
    //                      (2^-150 - 2^-149) is -0, not +0.
    wire exact_zero  = !any;
    wire result_zero = exact_zero || (sig_f == 24'd0);
    wire sign_zero   = (p1_sp == p1_sc) ? p1_sp : 1'b0;
    wire res_sign    = p1_mul_only ? p1_sp
                     : (exact_zero ? sign_zero : p1_res_sign_mag);

    wire [31:0] normal_result =
        overflow                 ? {res_sign, 8'hff, 23'd0} :
        result_zero              ? {res_sign, 8'd0, 23'd0} :
        (!res_is_norm)           ? (p1_ftz ? {res_sign, 8'd0, 23'd0}
                                           : {res_sign, 8'd0, sig_f[22:0]}) :
                                   {res_sign, e_biased[7:0], sig_f[22:0]};

    always @(*) begin
        if (p1_nan_res)      y = QNAN;
        else if (p1_inf_res) y = {p1_inf_sign, 8'hff, 23'd0};
        else                 y = normal_result;
    end

    wire _unused = &{1'b0, lzc89, e_norm[11], 1'b0};

endmodule
