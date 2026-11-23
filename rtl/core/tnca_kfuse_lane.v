//==============================================================================
// tnca_kfuse_lane -- K-fused E4M3 x E4M3 dot-product lane on DSP48E2 (R42)
//
// R42-kfuse-rtl SCREEN. Nothing in the core instantiates this module; it exists
// to measure what a DSP-mapped K-fused lane costs (evidence/r42/kfuse_rtl/).
//
// WHAT IT COMPUTES. One group of G E4M3 x E4M3 products plus the binary32
// accumulator c, bit for bit the model tools/r42/kfuse_model.py, variant
// 'sum_fma', rounding 'trunc', window W:
//
//   terms   exact products p_i = a_i * b_i, never rounded; zero products dropped
//   window  top = max_i floor(log2 |p_i|) over the non-zero products;
//           L = top - 23 - W   (the fixed point's least significant bit)
//   align   t_i = sign(p_i) * floor(|p_i| / 2^L)   (magnitude truncation)
//   sum     S = sum_i t_i   (exact integer, |S| < G * 2^(24+W))
//   round   y = RNE_binary32(S * 2^L + c)   ONE rounding per group, exact add
//
// Specials as the model (tools/model/fdp.py add_round): a NaN element -> every
// product of the group is dropped and the result is the canonical NaN
// 0x7fc00000; c NaN -> 0x7fc00000; c = +-Inf -> c (E4M3 has no Inf, a finite
// group cannot change it); an exact zero sum is +0 unless every product is a
// negative zero and c is -0 (then -0); a group sum or total that cancels
// exactly gives +0.
//
// HOW THE G PRODUCTS GO ONTO DSP48E2 (the structure, and why it is not the
// plain "shifted mantissa x mantissa, PCIN cascade" of the brief)
//
//   1. DSP1 (one per product) multiplies the two raw 4-bit mantissas (hidden
//      bit = exponent != 0): P1 = ma * mb, 8 bits, exact. The operands need no
//      logic but the hidden-bit OR.
//   2. The window anchor needs the exact leading-one position of every
//      product, floor(log2 p_i) = X_i - 20 + msb(P1_i), X_i = ea + eb (the
//      effective exponents). The msb of a product is not known before the
//      multiplication (it is msb(ma)+msb(mb) or one more), which is why the
//      product comes first and the alignment second. T_i = X_i + msb(P1_i);
//      Tmax = max T_i is a compare-select tree in LUTs.
//   3. DSP2 (one per product) aligns AND truncates: A = one-hot 2^sa, B = P1
//      (shifted left 8 for the largest shifts), so the multiplier output is
//      P1 * 2^sh at a grid 7 bits finer than the window's least significant
//      bit. The sign is a two's-complement A (-2^sa for a negative product,
//      SIGN_MODE 1, the default; SIGN_MODE 0 writes it as C -+ A*B for the
//      post-adder's ALUMODE, which Vivado 2023.2 maps to two DSP48E2 per
//      product). The C port carries +127 for negative products, so that the
//      plain right shift by 7 of P (wiring, P[47:7]) is magnitude truncation
//      toward zero: floor((127 - X)/128) = -floor(X/128). The cascade (PCIN)
//      cannot sum the products: a truncation is a per-term operation, and PCIN
//      adds carry across the dropped bits.
//   4. The G truncated terms are summed by a binary tree. TREE 0: G-1 DSP48E2
//      as two-input adders (A:B + C) with a P register per level, no LUT. TREE
//      1: the same tree in LUT carry chains.
//   5. The back end is R41's: |S| and its sign are packed as a binary32 FMA
//      "product" (pm = |S| unnormalised, xpb from Tmax) and fed to the
//      unchanged generated align / normalise / round stages
//      (tnca_mac_b0_xls, tnca_fmad_norm_xls, tnca_fma_round_xls), which are
//      exact for any pm and exponent (xls/tnca_fma.x, FmaProduct).
//
// PARAMETERS
//   G          products per step: 2, 4, 8 or 16
//   W          guard bits of the window: 0..3 (window = 24 + W bits; the shift
//              range is 30 + W, which the 26-bit one-hot port and a single
//              8-bit pre-shift cover up to W = 3)
//   CUTS       optional registers, in addition to the ones the DSP48E2 stages
//              need (decode/operand regs, DSP1 P, DSP2 A/B and P, one P per
//              tree level):
//                [0] after msb + T        [1] after the Tmax tree
//                [2] in the middle of the Tmax tree (G >= 8)
//                [3] after |S| / FmaProduct packing
//                [4] after B0 (align)     [5] after B1 (normalise)
//              B2 (round) is combinational into the consumer's register, as in
//              tnca_mac_lane.
//   IN_REG     1: operand registers inside the lane (as tnca_mac_lane)
//   C_LATE     0: c is presented with a and b and delayed inside the lane;
//              1: c is presented LAT_FRONT cycles after a and b, when the group
//              sum reaches the align stage (the accumulator read can then be
//              late; the loop a row's accumulator sees is only the back end)
//   SIGN_MODE  1: sign in the A operand (1 DSP48E2 per DSP2); 0: add/subtract form (2 DSP48E2)
//   TREE       0: DSP adder tree; 1: LUT adder tree
//
// Latency: LAT = LAT_FRONT + CUTS[4] + CUTS[5] (localparams below); one group
// per cycle, no stalls, no reset of data registers (valid has a synchronous
// reset). Operand k is element [8k +: 8] of a and b.
//==============================================================================
`default_nettype none

//------------------------------------------------------------------------------
// N-cycle delay of a W-bit word (flip-flops, never an SRL)
//------------------------------------------------------------------------------
module tnca_kfuse_dly #(
    parameter integer W = 1,
    parameter integer N = 1
) (
    input  wire         clk,
    input  wire [W-1:0] d,
    output wire [W-1:0] q
);
    generate
        if (N == 0) begin : g0
            assign q = d;
        end else begin : gn
            (* shreg_extract = "no" *) reg [W-1:0] r [0:N-1];
            integer i;
            always @(posedge clk) begin
                r[0] <= d;
                for (i = 1; i < N; i = i + 1) r[i] <= r[i-1];
            end
            assign q = r[N-1];
        end
    endgenerate
endmodule

//------------------------------------------------------------------------------
// max of G 6-bit values; MIDREG puts a register after the second level (G >= 8),
// ENDREG one after the last. Latency MIDREG*(G >= 8) + ENDREG.
//------------------------------------------------------------------------------
module tnca_kfuse_maxtree #(
    parameter integer G      = 8,
    parameter integer MIDREG = 0,
    parameter integer ENDREG = 1
) (
    input  wire           clk,
    input  wire [6*G-1:0] t,
    output wire [5:0]     m
);
    function integer clog2;
        input integer v;
        integer i;
        begin
            clog2 = 0;
            for (i = 1; i < v; i = i * 2) clog2 = clog2 + 1;
        end
    endfunction
    function integer dep;            // floor(log2 k)
        input integer k;
        integer i;
        begin
            dep = 0;
            for (i = 2; i <= k; i = i * 2) dep = dep + 1;
        end
    endfunction
    localparam integer LG   = clog2(G);
    localparam integer MIDD = LG - 2;                        // depth of the nodes that get the middle register
    localparam integer MID  = (MIDREG != 0) && (LG >= 3);

    wire [5:0] mxv [1:2*G-1];       // node value from its children as consumed
    wire [5:0] mxs [1:2*G-1];       // node value as consumed by its parent
    genvar k;
    generate
        for (k = G; k < 2*G; k = k + 1) begin : g_leaf
            assign mxv[k] = t[6*(k-G) +: 6];
            assign mxs[k] = mxv[k];
        end
        for (k = 1; k < G; k = k + 1) begin : g_node
            assign mxv[k] = (mxs[2*k] > mxs[2*k+1]) ? mxs[2*k] : mxs[2*k+1];
            if (MID && dep(k) == MIDD) begin : g_r
                reg [5:0] q;
                always @(posedge clk) q <= mxv[k];
                assign mxs[k] = q;
            end else begin : g_w
                assign mxs[k] = mxv[k];
            end
        end
        if (ENDREG != 0) begin : g_end
            reg [5:0] q;
            always @(posedge clk) q <= mxs[1];
            assign m = q;
        end else begin : g_noend
            assign m = mxs[1];
        end
    endgenerate
endmodule

//------------------------------------------------------------------------------
// one level of the adder tree: N signed WI-bit terms -> N/2 signed (WI+1)-bit
// sums, registered. DSP=1: DSP48E2 adders (A:B + C, P register).
//------------------------------------------------------------------------------
module tnca_kfuse_addlvl #(
    parameter integer N   = 2,
    parameter integer WI  = 25,
    parameter integer DSP = 1
) (
    input  wire                    clk,
    input  wire [N*WI-1:0]         x,
    output wire [(N/2)*(WI+1)-1:0] y
);
    genvar i;
    generate
        for (i = 0; i < N/2; i = i + 1) begin : g
            wire signed [WI-1:0] p = x[(2*i)*WI +: WI];
            wire signed [WI-1:0] q = x[(2*i+1)*WI +: WI];
            if (DSP != 0) begin : g_dsp
                (* use_dsp = "yes" *) reg signed [WI:0] s;
                always @(posedge clk) s <= p + q;
                assign y[i*(WI+1) +: WI+1] = s;
            end else begin : g_lut
                (* use_dsp = "no" *) reg signed [WI:0] s;
                always @(posedge clk) s <= p + q;
                assign y[i*(WI+1) +: WI+1] = s;
            end
        end
    endgenerate
endmodule

//------------------------------------------------------------------------------
// the whole tree, G in {2, 4, 8, 16}; latency log2(G)
//------------------------------------------------------------------------------
module tnca_kfuse_addtree #(
    parameter integer G   = 8,
    parameter integer TW  = 25,
    parameter integer DSP = 1,
    parameter integer LG  = (G == 2) ? 1 : (G == 4) ? 2 : (G == 8) ? 3 : 4   // do not override
) (
    input  wire                      clk,
    input  wire [G*TW-1:0]           t,
    output wire [TW+LG-1:0]          s
);
    generate
        if (G == 2) begin : g2
            tnca_kfuse_addlvl #(.N(2), .WI(TW), .DSP(DSP)) l1 (.clk(clk), .x(t), .y(s));
        end else if (G == 4) begin : g4
            wire [2*(TW+1)-1:0] u1;
            tnca_kfuse_addlvl #(.N(4), .WI(TW),   .DSP(DSP)) l1 (.clk(clk), .x(t),  .y(u1));
            tnca_kfuse_addlvl #(.N(2), .WI(TW+1), .DSP(DSP)) l2 (.clk(clk), .x(u1), .y(s));
        end else if (G == 8) begin : g8
            wire [4*(TW+1)-1:0] u1;
            wire [2*(TW+2)-1:0] u2;
            tnca_kfuse_addlvl #(.N(8), .WI(TW),   .DSP(DSP)) l1 (.clk(clk), .x(t),  .y(u1));
            tnca_kfuse_addlvl #(.N(4), .WI(TW+1), .DSP(DSP)) l2 (.clk(clk), .x(u1), .y(u2));
            tnca_kfuse_addlvl #(.N(2), .WI(TW+2), .DSP(DSP)) l3 (.clk(clk), .x(u2), .y(s));
        end else begin : g16
            wire [8*(TW+1)-1:0] u1;
            wire [4*(TW+2)-1:0] u2;
            wire [2*(TW+3)-1:0] u3;
            tnca_kfuse_addlvl #(.N(16), .WI(TW),   .DSP(DSP)) l1 (.clk(clk), .x(t),  .y(u1));
            tnca_kfuse_addlvl #(.N(8),  .WI(TW+1), .DSP(DSP)) l2 (.clk(clk), .x(u1), .y(u2));
            tnca_kfuse_addlvl #(.N(4),  .WI(TW+2), .DSP(DSP)) l3 (.clk(clk), .x(u2), .y(u3));
            tnca_kfuse_addlvl #(.N(2),  .WI(TW+3), .DSP(DSP)) l4 (.clk(clk), .x(u3), .y(s));
        end
    endgenerate
endmodule

//------------------------------------------------------------------------------
// one product: decode, DSP1, T = X + msb(P1), the shift decode and DSP2.
//   time (cycles after a8/b8): c0 regs +1, P1 +2, t at +2+D0, the shift decode
//   at +2+DWAIT (tmax must be valid then), term (DSP2 P) at +2+DWAIT+2.
//------------------------------------------------------------------------------
module tnca_kfuse_prod #(
    parameter integer W         = 0,
    parameter integer D0        = 1,     // register after T
    parameter integer DWAIT     = 2,     // cycles from P1 to the availability of tmax
    parameter integer SIGN_MODE = 1,
    parameter integer TW        = 25 + W
) (
    input  wire          clk,
    input  wire [7:0]    a8,
    input  wire [7:0]    b8,
    input  wire [5:0]    tmax,
    output wire [5:0]    t,
    output wire [TW-1:0] term,
    output wire          nan,
    output wire          nzn
);
    wire       ha = |a8[6:3];
    wire       hb = |b8[6:3];
    wire [3:0] ma = {ha, a8[2:0]};
    wire [3:0] mb = {hb, b8[2:0]};
    wire [3:0] ea = {a8[6:4], a8[3] | ~ha};               // effective exponent: 0 counts as 1
    wire [3:0] eb = {b8[6:4], b8[3] | ~hb};
    wire [4:0] xs = {1'b0, ea} + {1'b0, eb};
    wire       sg = a8[7] ^ b8[7];
    wire       za = ~ha & ~(|a8[2:0]);
    wire       zb = ~hb & ~(|b8[2:0]);
    assign nan = (&a8[6:0]) | (&b8[6:0]);
    assign nzn = (za | zb) & sg;

    // c0: DSP1's A and B registers, and what rides beside
    reg [3:0] ma_q, mb_q;
    reg [4:0] x_q;
    reg       sg_q;
    always @(posedge clk) begin
        ma_q <= ma; mb_q <= mb; x_q <= xs; sg_q <= sg;
    end

    // DSP1: P1 = ma * mb
    (* use_dsp = "yes" *) reg [7:0] p1_q;
    always @(posedge clk) p1_q <= ma_q * mb_q;
    wire [5:0] x1_sg1;
    tnca_kfuse_dly #(.W(6), .N(1)) u_d1 (.clk(clk), .d({x_q, sg_q}), .q(x1_sg1));

    // T = X + msb(P1), 0 for a zero product
    wire       nz = |p1_q;
    wire [2:0] msb = p1_q[7] ? 3'd7 : p1_q[6] ? 3'd6 : p1_q[5] ? 3'd5 : p1_q[4] ? 3'd4 :
                     p1_q[3] ? 3'd3 : p1_q[2] ? 3'd2 : p1_q[1] ? 3'd1 : 3'd0;
    wire [5:0] t_d = nz ? ({1'b0, x1_sg1[5:1]} + {3'b000, msb}) : 6'd0;
    generate
        if (D0 != 0) begin : g_t
            reg [5:0] t_r;
            always @(posedge clk) t_r <= t_d;
            assign t = t_r;
        end else begin : g_nt
            assign t = t_d;
        end
    endgenerate

    // {p1, x, sg} delayed to the time Tmax is available
    wire [13:0] bd;
    tnca_kfuse_dly #(.W(14), .N(DWAIT)) u_db (.clk(clk), .d({p1_q, x1_sg1}), .q(bd));

    // shift decode: sh = X + 30 + W - Tmax in [-5, 30 + W]
    wire [7:0] p1_d = bd[13:6];
    wire [4:0] x_d  = bd[5:1];
    wire       sg_d = bd[0];
    wire [7:0] shv  = {3'b000, x_d} + 8'd30 + W - {2'b00, tmax};
    wire       shok = ~shv[7] & (|p1_d);
    wire       hi   = shv[5] | shv[4];                              // sh >= 16
    wire [5:0] sa   = shv[5:0] - (hi ? 6'd8 : 6'd0);               // 0..25
    wire [15:0] b2  = hi ? {p1_d, 8'd0} : {8'd0, p1_d};

    wire signed [47:0] p2;
    generate
        if (SIGN_MODE == 0) begin : g_s0
            // ALUMODE add/subtract, written as P = C -+ A*B with C = 127 for a negative product.
            // (Vivado 2023.2 maps this to TWO DSP48E2: the multiplier, then an add/subtract DSP.)
            wire [25:0] a2 = shok ? (26'd1 << sa[4:0]) : 26'd0;
            reg  [25:0] a2_q;
            reg  [15:0] b2_q;
            reg         neg_q;
            always @(posedge clk) begin
                a2_q <= a2; b2_q <= b2; neg_q <= sg_d;
            end
            wire [41:0] m = a2_q * b2_q;
            (* use_dsp = "yes" *) reg signed [47:0] p2_q;
            always @(posedge clk) p2_q <= neg_q ? (48'd127 - {6'd0, m}) : {6'd0, m};
            assign p2 = p2_q;
        end else begin : g_s1
            // two's-complement A: -2^sa for a negative product; C = {7{neg}}
            wire signed [26:0] a2 = !shok ? 27'd0 :
                                    sg_d  ? (27'h7ffffff << sa[4:0]) : (27'd1 << sa[4:0]);
            reg  signed [26:0] a2_q;
            reg  signed [17:0] b2_q;
            reg         [6:0]  c2_q;
            always @(posedge clk) begin
                a2_q <= a2; b2_q <= {2'b00, b2}; c2_q <= {7{sg_d}};
            end
            wire signed [47:0] mm = a2_q * b2_q;
            (* use_dsp = "yes" *) reg signed [47:0] p2_q;
            always @(posedge clk) p2_q <= mm + $signed({41'd0, c2_q});
            assign p2 = p2_q;
        end
    endgenerate
    // magnitude truncation: drop the 7 fractional bits
    assign term = p2[7 +: TW];
endmodule

//------------------------------------------------------------------------------
// the lane
//------------------------------------------------------------------------------
module tnca_kfuse_lane #(
    parameter integer G         = 8,
    parameter integer W         = 0,
    parameter [7:0]   CUTS      = 8'b00111011,
    parameter integer IN_REG    = 1,
    parameter integer C_LATE    = 0,
    parameter integer SIGN_MODE = 1,
    parameter integer TREE      = 0
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    input  wire [8*G-1:0] a,
    input  wire [8*G-1:0] b,
    input  wire [31:0]    c,
    output wire [31:0]    y,
    output wire           valid_out
);
    function integer clog2;
        input integer v;
        integer i;
        begin
            clog2 = 0;
            for (i = 1; i < v; i = i * 2) clog2 = clog2 + 1;
        end
    endfunction

    localparam integer LG  = clog2(G);
    localparam integer TW  = 25 + W;                 // one aligned term, signed
    localparam integer SW  = TW + LG;                // the group sum, signed
    localparam integer N0  = IN_REG;
    localparam integer D0  = CUTS[0] ? 1 : 0;        // after msb + T
    localparam integer DM  = (CUTS[2] && LG >= 3) ? 1 : 0;
    localparam integer D1  = CUTS[1] ? 1 : 0;        // after the Tmax tree
    localparam integer DP  = CUTS[3] ? 1 : 0;        // after FmaProduct packing
    localparam integer DB0 = CUTS[4] ? 1 : 0;
    localparam integer DB1 = CUTS[5] ? 1 : 0;
    // times, in cycles after the operands are presented to the lane
    localparam integer TC0 = N0 + 1;                 // decode registers (DSP1 A and B)
    localparam integer TP1 = TC0 + 1;                // DSP1 P
    localparam integer TT3 = TP1 + D0 + DM + D1;     // Tmax available
    localparam integer TP2 = TT3 + 2;                // DSP2 P (terms)
    localparam integer TS  = TP2 + LG;               // group sum S
    localparam integer TPK = TS + DP;                // FmaProduct
    localparam integer LAT_FRONT = TPK;
    localparam integer LAT = TPK + DB0 + DB1;

    //--------------------------------------------------------------------------
    // operand registers
    //--------------------------------------------------------------------------
    wire [8*G-1:0] i_a, i_b;
    wire [31:0]    i_c;
    wire           i_v;
    generate
        if (IN_REG != 0) begin : g_in
            reg [8*G-1:0] a_q, b_q;
            reg [31:0]    c_q;
            reg           v_q;
            always @(posedge clk) begin
                a_q <= a; b_q <= b; c_q <= c;
                if (!rst_n) v_q <= 1'b0; else v_q <= valid_in;
            end
            assign i_a = a_q; assign i_b = b_q; assign i_c = c_q; assign i_v = v_q;
        end else begin : g_nin
            assign i_a = a; assign i_b = b; assign i_c = c; assign i_v = valid_in;
        end
    endgenerate

    //--------------------------------------------------------------------------
    // per product: decode, DSP1 (raw mantissa product), T, then (after Tmax)
    // the shift decode and DSP2
    //--------------------------------------------------------------------------
    wire [G-1:0]    nan_vec;     // an operand of product i is NaN
    wire [G-1:0]    nzn_vec;     // product i is a negative zero
    wire [6*G-1:0]  t_vec;       // T_i, at TP1 + D0
    wire [5:0]      tmax;        // at TT3
    wire [G*TW-1:0] term_vec;    // aligned truncated terms, at TP2

    genvar gi;
    generate
        for (gi = 0; gi < G; gi = gi + 1) begin : g_p
            tnca_kfuse_prod #(.W(W), .D0(D0), .DWAIT(D0 + DM + D1), .SIGN_MODE(SIGN_MODE)) u_prod (
                .clk(clk), .a8(i_a[8*gi +: 8]), .b8(i_b[8*gi +: 8]), .tmax(tmax),
                .t(t_vec[6*gi +: 6]), .term(term_vec[TW*gi +: TW]),
                .nan(nan_vec[gi]), .nzn(nzn_vec[gi]));
        end
    endgenerate

    //--------------------------------------------------------------------------
    // group flags: NaN, all-negative-zero
    //--------------------------------------------------------------------------
    wire nan_any = |nan_vec;
    wire nzn_all = &nzn_vec;

    //--------------------------------------------------------------------------
    // Tmax
    //--------------------------------------------------------------------------
    tnca_kfuse_maxtree #(.G(G), .MIDREG(DM), .ENDREG(D1)) u_max (.clk(clk), .t(t_vec), .m(tmax));

    //--------------------------------------------------------------------------
    // adder tree
    //--------------------------------------------------------------------------
    wire signed [SW-1:0] s_sum;
    tnca_kfuse_addtree #(.G(G), .TW(TW), .DSP(TREE == 0 ? 1 : 0)) u_tree (.clk(clk), .t(term_vec), .s(s_sum));

    //--------------------------------------------------------------------------
    // flags, Tmax and valid carried to the packing stage (TPK-DP)
    //--------------------------------------------------------------------------
    wire [5:0] tmax_s;
    tnca_kfuse_dly #(.W(6), .N(2 + LG)) u_dt (.clk(clk), .d(tmax), .q(tmax_s));
    wire nan_s, nzn_s, v_s;
    // nan_any, nzn_all (from i_a, i_b) and i_v are at time N0; the group sum is at TS
    tnca_kfuse_dly #(.W(3), .N(TS - N0)) u_df (.clk(clk), .d({nan_any, nzn_all, i_v}),
                                               .q({nan_s, nzn_s, v_s}));

    //--------------------------------------------------------------------------
    // |S|, the sign, FmaProduct
    //--------------------------------------------------------------------------
    wire             s_neg  = s_sum[SW-1];
    wire [SW-1:0]    s_abs  = s_neg ? (~s_sum + 1'b1) : s_sum;
    wire             s_zero = ~(|s_sum);
    wire [47:0]      pm     = {{(48-SW){1'b0}}, s_abs};
    wire [11:0]      xpb    = {6'd0, tmax_s} + 12'd257 - W;               // value = pm * 2^(xpb - 254 - 46)
    wire             sp     = s_zero ? nzn_s : s_neg;
    wire [63:0]      pkt_d  = {pm, xpb, sp, s_zero, 1'b0, nan_s};

    wire [63:0] pkt_o;
    wire [31:0] c_pk;
    wire        v_pk;
    generate
        if (DP != 0) begin : g_pk
            reg [63:0] q;
            always @(posedge clk) q <= pkt_d;
            assign pkt_o = q;
        end else begin : g_npk
            assign pkt_o = pkt_d;
        end
        if (C_LATE != 0) begin : g_cl
            assign c_pk = c;
        end else begin : g_ce
            // c enters IN_REG at N0 and must be at the align stage at TPK
            tnca_kfuse_dly #(.W(32), .N(TPK - N0)) u_dc (.clk(clk), .d(i_c), .q(c_pk));
        end
        tnca_kfuse_dly #(.W(1), .N(TPK - TS)) u_dv (.clk(clk), .d(v_s), .q(v_pk));
    endgenerate

    //--------------------------------------------------------------------------
    // back end: align | B0 cut | normalise | B1 cut | round -> y
    //--------------------------------------------------------------------------
    wire [149:0] al_d, al_o;
    wire [121:0] nm_d, nm_o;
    tnca_mac_b0_xls     u_b0 (.p(pkt_o), .mode(2'd2), .c(c_pk), .sub_c(1'b0), .mul_only(1'b0),
                              .ftz(1'b0), .out(al_d));
    tnca_fmad_norm_xls  u_b1 (.s(al_o), .out(nm_d));
    tnca_fma_round_xls  u_b2 (.s(nm_o), .out(y));

    wire v_b0, v_b1;
    generate
        if (DB0 != 0) begin : g_b0
            reg [149:0] q;
            always @(posedge clk) q <= al_d;
            assign al_o = q;
            tnca_kfuse_dly #(.W(1), .N(1)) u_v (.clk(clk), .d(v_pk), .q(v_b0));
        end else begin : g_nb0
            assign al_o = al_d;
            assign v_b0 = v_pk;
        end
        if (DB1 != 0) begin : g_b1
            reg [121:0] q;
            always @(posedge clk) q <= nm_d;
            assign nm_o = q;
            tnca_kfuse_dly #(.W(1), .N(1)) u_v (.clk(clk), .d(v_b0), .q(v_b1));
        end else begin : g_nb1
            assign nm_o = nm_d;
            assign v_b1 = v_b0;
        end
    endgenerate
    assign valid_out = v_b1;
endmodule
`default_nettype wire
