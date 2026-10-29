//==============================================================================
// tnca_mac_lane -- the shared multi-precision matrix MAC lane (R41 G2)
//
// xls/tnca_mac.x: mode 0 FP32 (= tnca_fpu_multi, bit for bit), mode 1 DP2 (two
// 16-bit-slot products per cycle: FP16, BF16, E5M2, E4M3 per operand), mode 2
// DP4 (four E4M3 x E4M3 products per cycle), the packed modes being the
// proposed m.mma.dp. The stage functions are generated (rtl/gen/tnca_mac_*);
// this wrapper widens mode 0's operands by fmt, as tnca_fpu_multi does, and
// places registers at the cuts CUTS selects:
//   CUTS[0] after F1 (products and sums)  CUTS[1] after F2 (the product)
//   CUTS[2] after B0 (align)              CUTS[3] after B1 (normalise)
// B2 (round) is combinational into the consumer's register, as the FMA's.
// The latency is IN_REG + popcount(CUTS); today's lane (tnca_fpu_multi with
// IN_REG=1, LAT=2) is CUTS = 4'b1100.
//==============================================================================
`default_nettype none
module tnca_mac_lane #(
    parameter [3:0] CUTS   = 4'b1110,
    parameter       IN_REG = 1
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,           // 0 FP32, 1 DP2, 2 DP4
    input  wire [2:0]  fmt,            // mode 0: the operands' format (tnca_fpu_multi)
    input  wire [1:0]  fa,             // DP2: a's element code (0 FP16, 1 BF16, 2 E5M2, 3 E4M3)
    input  wire [1:0]  fb,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output wire [31:0] y
);
    localparam W_MID   = 202;   // MacMid
    localparam W_PROD  = 64;    // FmaProduct
    localparam W_ALIGN = 150;   // FmaAligned
    localparam W_NORM  = 122;   // FmaNormalised
    localparam W_K     = 37;    // {mode, c, sub_c, mul_only, ftz}

    wire [31:0] a_f32, b_f32;
    tnca_widen_xls u_wa (.fmt(fmt), .x(a), .out(a_f32));
    tnca_widen_xls u_wb (.fmt(fmt), .x(b), .out(b_f32));
    wire        fp32 = (mode == 2'd0);
    wire [31:0] a_in = fp32 ? a_f32 : a;
    wire [31:0] b_in = fp32 ? b_f32 : b;
    wire [W_K-1:0] k_in = {mode, c, sub_c, mul_only, ftz};

    wire [31:0] i_a, i_b;
    wire [1:0]  i_fa, i_fb;
    wire [W_K-1:0] i_k;
    generate
        if (IN_REG != 0) begin : g_in_reg
            reg [31:0] a_q, b_q;
            reg [1:0]  fa_q, fb_q;
            reg [W_K-1:0] k_q;
            always @(posedge clk) begin
                a_q <= a_in; b_q <= b_in; fa_q <= fa; fb_q <= fb; k_q <= k_in;
            end
            assign i_a = a_q; assign i_b = b_q; assign i_fa = fa_q; assign i_fb = fb_q; assign i_k = k_q;
        end else begin : g_in_wire
            assign i_a = a_in; assign i_b = b_in; assign i_fa = fa; assign i_fb = fb; assign i_k = k_in;
        end
    endgenerate

    wire [W_MID-1:0]   f1_d, f1_o;
    wire [W_PROD-1:0]  f2_d, f2_o;
    wire [W_ALIGN-1:0] al_d, al_o;
    wire [W_NORM-1:0]  nm_d, nm_o;
    wire [W_K-1:0]     k1, k2;

    tnca_mac_f1_xls u_f1 (.mode(i_k[36:35]), .a(i_a), .b(i_b), .fa(i_fa), .fb(i_fb),
                          .ftz(i_k[0]), .out(f1_d));
    tnca_mac_f2_xls u_f2 (.m(f1_o), .out(f2_d));
    tnca_mac_b0_xls u_b0 (.p(f2_o), .mode(k2[36:35]), .c(k2[34:3]), .sub_c(k2[2]),
                          .mul_only(k2[1]), .ftz(k2[0]), .out(al_d));
    tnca_fmad_norm_xls u_b1 (.s(al_o), .out(nm_d));
    tnca_fma_round_xls u_b2 (.s(nm_o), .out(y));

    generate
        if (CUTS[0]) begin : g_c0
            reg [W_MID-1:0] q; reg [W_K-1:0] kq;
            always @(posedge clk) begin q <= f1_d; kq <= i_k; end
            assign f1_o = q; assign k1 = kq;
        end else begin : g_n0
            assign f1_o = f1_d; assign k1 = i_k;
        end
        if (CUTS[1]) begin : g_c1
            reg [W_PROD-1:0] q; reg [W_K-1:0] kq;
            always @(posedge clk) begin q <= f2_d; kq <= k1; end
            assign f2_o = q; assign k2 = kq;
        end else begin : g_n1
            assign f2_o = f2_d; assign k2 = k1;
        end
        if (CUTS[2]) begin : g_c2
            reg [W_ALIGN-1:0] q;
            always @(posedge clk) q <= al_d;
            assign al_o = q;
        end else begin : g_n2
            assign al_o = al_d;
        end
        if (CUTS[3]) begin : g_c3
            reg [W_NORM-1:0] q;
            always @(posedge clk) q <= nm_d;
            assign nm_o = q;
        end else begin : g_n3
            assign nm_o = nm_d;
        end
    endgenerate
endmodule
`default_nettype wire
