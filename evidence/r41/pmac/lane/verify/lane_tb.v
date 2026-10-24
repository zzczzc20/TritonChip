// lane_tb -- every lane variant beside the current lane (r41-pmac base,
// renamed *_ref by mkref.sh), for lane_check.cpp.
//
// The four variants share one set of inputs; the reference lane has its own,
// so a DP group with one nonzero product can be checked against the binary32
// FMA on that product's exactly widened operands. Latencies: variants
// 1 + popcount(CUTS), reference 3 (IN_REG + FP_LAT 2).
module lane_tb #(
    parameter [3:0] CUTS = 4'b1100
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,
    input  wire [2:0]  fmt,
    input  wire        alt_a,
    input  wire        alt_b,
    input  wire [31:0] a, b, c,
    input  wire        sub_c, mul_only, ftz,
    input  wire [2:0]  rfmt,
    input  wire [31:0] ra, rb, rc,
    input  wire        rsub, rmo, rftz,
    output wire [31:0] y1, y2l, y2d, y3, yref
);
    tnca_pmac_lane #(.VARIANT(1), .CUTS(CUTS)) u_v1 (
        .clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt), .alt_a(alt_a), .alt_b(alt_b),
        .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y1));
    tnca_pmac_lane #(.VARIANT(2), .DSP2(0), .CUTS(CUTS)) u_v2l (
        .clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt), .alt_a(alt_a), .alt_b(alt_b),
        .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y2l));
    tnca_pmac_lane #(.VARIANT(2), .DSP2(1), .CUTS(CUTS)) u_v2d (
        .clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt), .alt_a(alt_a), .alt_b(alt_b),
        .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y2d));
    tnca_pmac_lane #(.VARIANT(3), .CUTS(CUTS)) u_v3 (
        .clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt), .alt_a(alt_a), .alt_b(alt_b),
        .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y3));
    tnca_fpu_multi_ref #(.LAT(2), .IN_REG(1)) u_ref (
        .clk(clk), .rst_n(rst_n), .fmt(rfmt), .a(ra), .b(rb), .c(rc),
        .sub_c(rsub), .mul_only(rmo), .ftz(rftz), .y(yref));
endmodule
