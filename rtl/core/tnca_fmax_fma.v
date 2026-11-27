//==============================================================================
// tnca_fmax_fma -- binary32 fused multiply-add from the R42 area stages, with selectable pipeline cuts
//
//     y = round_RNE( a * b + (sub_c ? -c : c) )      mul_only: round_RNE(a * b); ftz: subnormals flushed
//
// The same function, bit for bit, as tnca_fp32_fma / xls/tnca_fma.x (tb/area, make fma-deep-area). Five combinational
// stages, optional registers between them:
//
//     A  tnca_fmax_mul      unpack, classify, the 24 x 24 multiplier (two DSP48E2)       -> FmaProduct (64 bits)
//     B  tnca_fmax_alignp   exponent difference, swap, staged right shift with sticky    -> FmaAligned (150)
//     C  tnca_fmax_add      the three 49-bit add/subtracts of the 81-bit sum              -> 233 bits
//     D  tnca_fmax_sel      select, leading-zero count, shift controls                   -> FmaNormalised (122)
//     E  tnca_fmax_round    window shift, round to nearest even, pack                    -> y (combinational)
//
// CUTS[i] puts a register behind stage A, B, C, D (bit 0..3); the latency is popcount(CUTS), y is combinational from
// stage E as in tnca_fp32_fma. 4'b1010 (behind B and D) is the latency-2 instance; 4'b1110 is latency 3, and there the
// register behind C is the P register of three inferred DSP48E2 (the adders), the one placement at which Vivado maps an
// adder onto a DSP (an adder followed by logic, or a dynamic add/subtract, stays in carry chains: evidence/r42/area).
// XLS_ALIGN = 1 replaces A and B by the generated tnca_fma_align_xls (one stage, CUTS[0] must be 0).
// Registers behind A, B, D reset to zero like tnca_fp32_fma's; the adder registers have no reset.
//==============================================================================
`default_nettype none
module tnca_fmax_fma #(
    parameter [3:0] CUTS      = 4'b1010,
    parameter       SHIFT     = 1,        // tnca_fmax_round SHIFT
    parameter       XLS_ALIGN = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output wire [31:0] y
);
    // ---- A (and B in one piece when XLS_ALIGN)
    wire [63:0]  p_d;
    wire [149:0] al_d;
    wire [149:0] al_o;
    wire [63:0]  p_o;
    wire [31:0]  c_o;
    wire         sc_o, mo_o, fz_o;
    generate
        if (XLS_ALIGN != 0) begin : g_xa
            tnca_fma_align_xls u_ab (.a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .out(al_d));
            assign p_d = 64'b0; assign p_o = 64'b0; assign c_o = 32'b0; assign sc_o = 1'b0; assign mo_o = 1'b0; assign fz_o = 1'b0;
        end else begin : g_na
            tnca_fmax_mul u_a (.a(a), .b(b), .ftz(ftz), .p(p_d));
            if (CUTS[0]) begin : g_c0
                reg [63:0] pq; reg [31:0] cq; reg sq, mq, fq;
                always @(posedge clk)
                    if (!rst_n) begin pq <= 64'b0; cq <= 32'b0; sq <= 1'b0; mq <= 1'b0; fq <= 1'b0; end
                    else begin pq <= p_d; cq <= c; sq <= sub_c; mq <= mul_only; fq <= ftz; end
                assign p_o = pq; assign c_o = cq; assign sc_o = sq; assign mo_o = mq; assign fz_o = fq;
            end else begin : g_n0
                assign p_o = p_d; assign c_o = c; assign sc_o = sub_c; assign mo_o = mul_only; assign fz_o = ftz;
            end
            tnca_fmax_alignp u_b (.p(p_o), .c(c_o), .sub_c(sc_o), .mul_only(mo_o), .ftz(fz_o), .out(al_d));
        end
        // ---- cut behind B
        if (CUTS[1]) begin : g_c1
            reg [149:0] q;
            always @(posedge clk) if (!rst_n) q <= 150'b0; else q <= al_d;
            assign al_o = q;
        end else begin : g_n1
            assign al_o = al_d;
        end
    endgenerate

    // ---- C: the adders, registered (DSP48E2 P registers) when CUTS[2]
    wire [232:0] cs_o;
    tnca_fmax_add #(.REG(CUTS[2] ? 1 : 0)) u_c (.clk(clk), .s(al_o), .out(cs_o));

    // ---- D, cut behind D, E
    wire [121:0] nm_d, nm_o;
    tnca_fmax_sel u_d (.c(cs_o), .out(nm_d));
    generate
        if (CUTS[3]) begin : g_c3
            reg [121:0] q;
            always @(posedge clk) if (!rst_n) q <= 122'b0; else q <= nm_d;
            assign nm_o = q;
        end else begin : g_n3
            assign nm_o = nm_d;
        end
    endgenerate
    tnca_fmax_round #(.SHIFT(SHIFT)) u_e (.s(nm_o), .out(y));
endmodule
`default_nettype wire
