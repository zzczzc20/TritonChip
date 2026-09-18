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
// Combinational. Handles subnormal inputs and subnormal results. ftz=1 flushes
// subnormal inputs and results to zero of the same sign -- an explicit mode, per
// ISA sec 8.2, never an implicit shortcut.
//
// Alignment scheme
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
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,       // negate the addend (vf.fms, vf.sub)
    input  wire        mul_only,    // ignore the addend (vf.mul)
    input  wire        ftz,
    output reg  [31:0] y
);

    localparam [31:0] QNAN = 32'h7fc00000;

    // ---- unpack --------------------------------------------------------------
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
    wire c_zero = (ec_ == 8'd0) && ((fc_ == 23'd0) || ftz) || mul_only;

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

    // ---- product -------------------------------------------------------------
    wire        sp = sa_ ^ sb_;
    wire [47:0] pm = ma * mb;                 // value = pm * 2^(xa+xb-46)
    wire signed [11:0] xp = xa + xb;

    // Addend restated at the same -46 scale: MC = mc << 23.
    wire [46:0] mcw = {mc, 23'd0};            // value = mcw * 2^(xc-46)
    wire signed [11:0] xcw = xc;

    wire p_zero = (pm == 48'd0);

    // ---- align ---------------------------------------------------------------
    wire signed [12:0] dexp = $signed({xp[11], xp}) - $signed({xcw[11], xcw});

    // Shift amount, clamped: past 80 bits of displacement the smaller operand
    // can only contribute a sticky bit.
    wire [12:0] dabs = dexp[12] ? (~dexp + 13'd1) : dexp;
    wire [7:0]  shamt = (dabs > 13'd80) ? 8'd80 : dabs[7:0];

    wire [87:0] p_field = {8'd0, pm, 32'd0};          // [79:32]
    wire [87:0] c_field = {9'd0, mcw, 32'd0};         // [78:32]

    // Shift the operand with the smaller exponent. A zero significand must never
    // be chosen as the reference operand: its exponent field is meaningless, and
    // with mul_only the addend's exponent is whatever the caller left in c.
    wire c_zero_mag = (mcw == 47'd0);
    wire p_zero_mag = (pm  == 48'd0);
    wire shift_c = c_zero_mag ? 1'b1 : (p_zero_mag ? 1'b0 : ~dexp[12]);

    wire [87:0] big_v   = shift_c ? p_field : c_field;
    wire [87:0] small_v = shift_c ? c_field : p_field;

    wire [87:0] small_sh = small_v >> shamt;
    // Sticky: anything shifted past bit 0 of the 88-bit field.
    wire        sticky_al = |(small_v & ~({88{1'b1}} << shamt));

    wire signed [11:0] exp_common = shift_c ? xp : xcw;

    wire        s_big   = shift_c ? sp : sc_;
    wire        s_small = shift_c ? sc_ : sp;

    // ---- add / subtract ------------------------------------------------------
    wire same_sign = (s_big == s_small);

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

    wire [88:0] acc_raw = same_sign ? sum_add : (small_wins ? sub_sb : sub_bs);

    wire [88:0] mag = acc_raw;

    wire sticky_op  = sticky_al;
    wire res_sign_mag = same_sign ? s_big : (small_wins ? s_small : s_big);

    // ---- normalise -----------------------------------------------------------
    integer i;
    reg [7:0] msb;          // index of the highest set bit of mag, 0 if none
    reg       any;
    always @(*) begin
        msb = 8'd0;
        any = 1'b0;
        for (i = 0; i < 89; i = i + 1) begin
            if (mag[i]) begin
                msb = i[7:0];
                any = 1'b1;
            end
        end
    end

    // value(mag) = mag * 2^(exp_common - 78); normalised value = 1.f * 2^E
    wire signed [11:0] e_norm = $signed({4'd0, msb}) + exp_common - 12'sd78;

    // Right shift that leaves 24 significant bits, plus guard and round.
    // Target: bit msb lands at bit 23 of the significand.
    wire signed [11:0] sh_norm_s = $signed({4'd0, msb}) - 12'sd23;

    // Subnormal results need a larger right shift so the exponent stops at -126.
    wire signed [11:0] e_min = -12'sd126;
    wire subnormal_res = (e_norm < e_min) && any;
    wire signed [11:0] sh_extra = subnormal_res ? (e_min - e_norm) : 12'sd0;
    wire signed [11:0] sh_tot_s = sh_norm_s + sh_extra;

    wire sh_left = (sh_tot_s < 0);
    wire [11:0] sh_mag = sh_left ? (-sh_tot_s) : sh_tot_s;
    wire [7:0]  sh_r = (sh_mag > 12'd100) ? 8'd100 : sh_mag[7:0];

    wire [88:0] shifted = sh_left ? (mag << sh_r) : (mag >> sh_r);
    wire        lost    = sh_left ? 1'b0 : (|(mag & ~({89{1'b1}} << sh_r)));

    // Guard and sticky are taken by shifting rather than by bit-select, so a
    // shift amount past the width of `mag` (which happens for results far below
    // the subnormal range) yields 0 instead of an out-of-range X.
    wire [7:0]  sh_r1 = (sh_r == 8'd0) ? 8'd0 : (sh_r - 8'd1);
    wire [88:0] mag_g = mag >> sh_r1;

    wire [23:0] sig   = shifted[23:0];
    wire        guard = sh_left ? 1'b0 : ((sh_r == 8'd0) ? 1'b0 : mag_g[0]);
    wire        rest  = sh_left ? 1'b0
                                : ((sh_r < 8'd2) ? 1'b0
                                                 : |(mag & ~({89{1'b1}} << sh_r1)));
    wire        sticky = rest | sticky_op;

    // round to nearest even
    wire round_up = guard & (sticky | sig[0]);
    wire [24:0] sig_r = {1'b0, sig} + {24'd0, round_up};

    // A rounding carry out of bit 23 bumps the exponent.
    wire carry_out = sig_r[24];
    wire [23:0] sig_f = carry_out ? sig_r[24:1] : sig_r[23:0];
    wire signed [11:0] e_f = (subnormal_res ? e_min : e_norm) + (carry_out ? 12'sd1 : 12'sd0);

    // A subnormal that rounded up to 1.0 becomes the smallest normal, which the
    // encoding below produces naturally because sig_f[23] is then set.
    wire res_is_norm = sig_f[23];

    wire signed [11:0] e_biased = e_f + 12'sd127;
    wire overflow  = any && (e_biased >= 12'sd255);
    wire underflow = any && (!res_is_norm) && (e_biased <= 12'sd0);

    // ---- result sign ---------------------------------------------------------
    // Two different zeros, and they do not get the same sign:
    //
    //   exact zero      -- the operands annihilated, or both were zero. IEEE
    //                      RNE gives +0 unless both addends shared a sign.
    //   rounded-to-zero -- the exact result was a nonzero magnitude that
    //                      underflowed. It keeps that magnitude's sign, so
    //                      (2^-150 - 2^-149) is -0, not +0.
    wire exact_zero  = !any;
    wire result_zero = exact_zero || (sig_f == 24'd0);
    wire sign_zero   = (sp == sc_) ? sp : 1'b0;
    wire res_sign    = mul_only ? sp
                     : (exact_zero ? sign_zero : res_sign_mag);

    // ---- assemble ------------------------------------------------------------
    wire [31:0] normal_result =
        overflow                 ? {res_sign, 8'hff, 23'd0} :
        result_zero              ? {res_sign, 8'd0, 23'd0} :
        (!res_is_norm)           ? (ftz ? {res_sign, 8'd0, 23'd0}
                                        : {res_sign, 8'd0, sig_f[22:0]}) :
                                   {res_sign, e_biased[7:0], sig_f[22:0]};

    // ---- special cases -------------------------------------------------------
    wire inf_times_zero = (a_inf & b_zero) | (b_inf & a_zero);
    wire p_inf          = (a_inf | b_inf) & ~inf_times_zero;
    wire inf_minus_inf  = p_inf & c_inf & (sp != sc_);

    always @(*) begin
        if (a_nan | b_nan | c_nan | inf_times_zero | inf_minus_inf)
            y = QNAN;
        else if (p_inf)
            y = {sp, 8'hff, 23'd0};
        else if (c_inf)
            y = {sc_, 8'hff, 23'd0};
        else
            y = normal_result;
    end

    // silence lint on unused bits
    wire _unused = &{1'b0, underflow, lost, e_norm[11], 1'b0};

endmodule
