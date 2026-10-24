// pmac_ooc_top -- one MAC lane between registers, for out-of-context synthesis
// and place-and-route on the U280 part.
//
// The w_* registers stand for what drives a lane in tnca_matrix (the operand
// latches, the accumulator read) and y_q for the accumulator file's write, so
// every path of the lane is register to register: the operand side into
// IN_REG (widening, mode select), each stage, and the round stage into the
// accumulator. sub_c and ftz are tied to 0 as tnca_matrix ties them.
//
//   LANE 0: the current lane, tnca_fpu_multi #(LAT 2, IN_REG 1)  (V0)
//   LANE 1: tnca_pmac_lane #(VARIANT, DSP2, CUTS, IN_REG 1)
module pmac_ooc_top #(
    parameter       LANE    = 1,
    parameter       VARIANT = 1,
    parameter       DSP2    = 0,
    parameter [3:0] CUTS    = 4'b1100
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,
    input  wire [2:0]  fmt,
    input  wire        alt_a,
    input  wire        alt_b,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        mul_only,
    output reg  [31:0] y_q
);
    reg [1:0]  w_mode;
    reg [2:0]  w_fmt;
    reg        w_alt_a, w_alt_b, w_mul_only;
    reg [31:0] w_a, w_b, w_c;
    always @(posedge clk) begin
        w_mode <= mode; w_fmt <= fmt; w_alt_a <= alt_a; w_alt_b <= alt_b;
        w_a <= a; w_b <= b; w_c <= c; w_mul_only <= mul_only;
    end

    wire [31:0] y;
    generate
        if (LANE == 0) begin : g_cur
            tnca_fpu_multi #(.LAT(2), .IN_REG(1)) u_lane (
                .clk(clk), .rst_n(rst_n), .fmt(w_fmt), .a(w_a), .b(w_b), .c(w_c),
                .sub_c(1'b0), .mul_only(w_mul_only), .ftz(1'b0), .y(y));
        end else begin : g_new
            tnca_pmac_lane #(.VARIANT(VARIANT), .DSP2(DSP2), .CUTS(CUTS), .IN_REG(1)) u_lane (
                .clk(clk), .rst_n(rst_n), .mode(w_mode), .fmt(w_fmt), .alt_a(w_alt_a), .alt_b(w_alt_b),
                .a(w_a), .b(w_b), .c(w_c), .sub_c(1'b0), .mul_only(w_mul_only), .ftz(1'b0), .y(y));
        end
    endgenerate

    always @(posedge clk) y_q <= y;
endmodule
