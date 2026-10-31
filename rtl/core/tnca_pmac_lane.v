//==============================================================================
// tnca_pmac_lane -- packed dot-product ("NV-style") MAC lane
//
// R41 FEASIBILITY PROTOTYPE. Nothing in the core instantiates this module; it
// exists to measure what a packed lane costs (evidence/r41/pmac/lane/).
//
//   mode 0  FP32  y = RNE_f32( widen(fmt, a) * widen(fmt, b) + c )
//                 bit for bit tnca_fpu_multi (m.mma via widening, row ops)
//   mode 1  DP2   y = RNE_f32( c + RO_48( a[15:0]*b[15:0] + a[31:16]*b[31:16] ) )
//                 alt_a/alt_b: 0 = FP16, 1 = BF16, per operand    (VARIANT 2)
//   mode 2  DP4   y = RNE_f32( c + RO_48( sum_{i<4} a[8i+:8]*b[8i+:8] ) )
//                 alt_a/alt_b: 0 = E4M3, 1 = E5M2, per operand    (VARIANT 1, 2;
//                 VARIANT 3: E4M3 x E4M3 only, alt ignored)
//
// Products are exact, the group sum is exact before RO_48 (round to odd at 48
// significant bits, unbounded exponent), and the last rounding is the binary32
// FMA's. tools/model/fdp.py is the exact oracle; xls/tnca_fdp.x says how and
// why. In a DP mode mul_only and ftz are ignored (taken as 0); sub_c negates
// c as in every mode.
//
// THE PIPELINE, AND WHERE THE REGISTERS GO
//
//   [widen, mode select] IN_REG [F1] c0 [F2] c1 [align] c2 [normalise] c3 [round] -> y
//
//   F1     DP products and their exact signed sum; the binary32 24x24 product
//   F2     |sum|, RO_48, the mode's product select       (xls/tnca_fdp.x)
//   align  addend, exponent difference, alignment shift  (fma32_align_p)
//   normalise, round                                     (the binary32 FMA's)
//
// CUTS[i] places a register at ci. The binary32 FMA's latency-2 instance is
// CUTS = 4'b1100 (c2 = its FmaAligned cut, c3 = its FmaNormalised cut), so a
// lane with CUTS = 4'b1100 and IN_REG = 1 has the current lane's latency (3);
// each further cut adds one cycle in every mode. y is combinational from the
// round stage, like tnca_fp32_fma's: the consumer (the accumulator file)
// registers it. c, sub_c, mul_only and ftz ride alongside to the align stage.
// Registers inside the arithmetic have the generated pipeline's synchronous
// reset, as tnca_fp32_fma's do; IN_REG has none, as tnca_fpu_multi's has none.
//
// Latency: IN_REG + CUTS[0] + CUTS[1] + CUTS[2] + CUTS[3] cycles, one result
// per cycle.
//==============================================================================
`default_nettype none

module tnca_pmac_lane #(
    parameter       VARIANT = 1,       // 1: FP32 + DP4; 2: + DP2; 3: FP32 + DP4 E4M3 only
    parameter       DSP2    = 0,       // VARIANT 2: DP2 products in the binary32 multiplier's DSPs
    parameter [3:0] CUTS    = 4'b1100, // registers at c3..c0
    parameter       IN_REG  = 1
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [1:0]  mode,           // 0 FP32, 1 DP2, 2 DP4
    input  wire [2:0]  fmt,            // FP32 mode: the operands' format (tnca_fpu_multi's fmt)
    input  wire        alt_a,          // DP mode: a's format (see above)
    input  wire        alt_b,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output wire [31:0] y
);
    // xls/tnca_fdp.x / tnca_fma.x structs, flattened by XLS
    localparam W_PROD  = 64;                          // FmaProduct
    localparam W_MID   = (VARIANT == 3) ? 108 : 149;  // FdpMid3 / FdpMid
    localparam W_ALIGN = 150;                         // FmaAligned
    localparam W_NORM  = 122;                         // FmaNormalised
    localparam W_K     = 35;                          // {c, sub_c, mul_only, ftz}

    //--------------------------------------------------------------------------
    // Operand side, in front of IN_REG: FP32 mode widens exactly as
    // tnca_fpu_multi does; a DP mode takes the packed elements as they are.
    //--------------------------------------------------------------------------
    wire [31:0] a_f32, b_f32;
    tnca_widen_xls u_wa (.fmt(fmt), .x(a), .out(a_f32));
    tnca_widen_xls u_wb (.fmt(fmt), .x(b), .out(b_f32));

    wire        fp32 = (mode == 2'd0);
    wire [31:0] a_in = fp32 ? a_f32 : a;
    wire [31:0] b_in = fp32 ? b_f32 : b;
    wire [W_K-1:0] k_in = {c, sub_c, fp32 & mul_only, fp32 & ftz};

    wire [31:0]    i_a, i_b;
    wire [1:0]     i_mode;
    wire           i_alt_a, i_alt_b;
    wire [W_K-1:0] i_k;
    generate
        if (IN_REG != 0) begin : g_in_reg
            reg [31:0]    a_q, b_q;
            reg [1:0]     mode_q;
            reg           alt_a_q, alt_b_q;
            reg [W_K-1:0] k_q;
            always @(posedge clk) begin
                a_q <= a_in; b_q <= b_in; mode_q <= mode;
                alt_a_q <= alt_a; alt_b_q <= alt_b; k_q <= k_in;
            end
            assign i_a = a_q; assign i_b = b_q; assign i_mode = mode_q;
            assign i_alt_a = alt_a_q; assign i_alt_b = alt_b_q; assign i_k = k_q;
        end else begin : g_in_wire
            assign i_a = a_in; assign i_b = b_in; assign i_mode = mode;
            assign i_alt_a = alt_a; assign i_alt_b = alt_b; assign i_k = k_in;
        end
    endgenerate

    //--------------------------------------------------------------------------
    // F1 | c0 | F2 | c1
    //--------------------------------------------------------------------------
    wire [W_MID-1:0] f1_d, f1_o;
    wire [W_K-1:0]   k1;
    wire [W_PROD-1:0] f2_d, f2_o;
    wire [W_K-1:0]   k2;

    generate
        if (VARIANT == 3) begin : g_v3
            tnca_fdp_v3_f1_xls u_f1 (.mode(i_mode), .a(i_a), .b(i_b), .alt_a(i_alt_a),
                                     .alt_b(i_alt_b), .ftz(i_k[0]), .out(f1_d));
            tnca_fdp_v3_f2_xls u_f2 (.m(f1_o), .out(f2_d));
        end else begin : g_v12
            if (VARIANT == 2 && DSP2 != 0) begin : g_v2d
                tnca_fdp_v2d_f1_xls u_f1 (.mode(i_mode), .a(i_a), .b(i_b), .alt_a(i_alt_a),
                                          .alt_b(i_alt_b), .ftz(i_k[0]), .out(f1_d));
            end else if (VARIANT == 2) begin : g_v2l
                tnca_fdp_v2l_f1_xls u_f1 (.mode(i_mode), .a(i_a), .b(i_b), .alt_a(i_alt_a),
                                          .alt_b(i_alt_b), .ftz(i_k[0]), .out(f1_d));
            end else begin : g_v1
                tnca_fdp_v1_f1_xls u_f1 (.mode(i_mode), .a(i_a), .b(i_b), .alt_a(i_alt_a),
                                         .alt_b(i_alt_b), .ftz(i_k[0]), .out(f1_d));
            end
            tnca_fdp_f2_xls u_f2 (.m(f1_o), .out(f2_d));
        end

        if (CUTS[0]) begin : g_c0
            reg [W_MID-1:0] q;
            reg [W_K-1:0]   kq;
            always @(posedge clk)
                if (!rst_n) begin q <= {W_MID{1'b0}}; kq <= {W_K{1'b0}}; end
                else        begin q <= f1_d;          kq <= i_k;         end
            assign f1_o = q; assign k1 = kq;
        end else begin : g_n0
            assign f1_o = f1_d; assign k1 = i_k;
        end

        if (CUTS[1]) begin : g_c1
            reg [W_PROD-1:0] q;
            reg [W_K-1:0]    kq;
            always @(posedge clk)
                if (!rst_n) begin q <= {W_PROD{1'b0}}; kq <= {W_K{1'b0}}; end
                else        begin q <= f2_d;           kq <= k1;          end
            assign f2_o = q; assign k2 = kq;
        end else begin : g_n1
            assign f2_o = f2_d; assign k2 = k1;
        end
    endgenerate

    //--------------------------------------------------------------------------
    // align | c2 | normalise | c3 | round -- the binary32 FMA's back end
    //--------------------------------------------------------------------------
    wire [W_ALIGN-1:0] al_d, al_o;
    wire [W_NORM-1:0]  nm_d, nm_o;

    tnca_fma_alignp_xls u_al (.p(f2_o), .c(k2[34:3]), .sub_c(k2[2]),
                              .mul_only(k2[1]), .ftz(k2[0]), .out(al_d));
    tnca_fma_norm_xls   u_nm (.s(al_o), .out(nm_d));
    tnca_fma_round_xls  u_rd (.s(nm_o), .out(y));

    generate
        if (CUTS[2]) begin : g_c2
            reg [W_ALIGN-1:0] q;
            always @(posedge clk)
                if (!rst_n) q <= {W_ALIGN{1'b0}};
                else        q <= al_d;
            assign al_o = q;
        end else begin : g_n2
            assign al_o = al_d;
        end

        if (CUTS[3]) begin : g_c3
            reg [W_NORM-1:0] q;
            always @(posedge clk)
                if (!rst_n) q <= {W_NORM{1'b0}};
                else        q <= nm_d;
            assign nm_o = q;
        end else begin : g_n3
            assign nm_o = nm_d;
        end
    endgenerate
endmodule

`default_nettype wire
