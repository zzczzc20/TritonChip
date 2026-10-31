// mac_tb -- tnca_mac_lane (at CUTS) beside the current matrix lane
// (tnca_fpu_multi IN_REG 1 LAT 2, modules renamed *_ref by mkref.sh), for
// mac_check.cpp. The reference has its own inputs, so a packed group holding
// one nonzero product can be checked against the binary32 FMA on that
// product's exactly widened operands. Latencies: lane 1 + popcount(CUTS),
// reference 3.
module mac_tb #(
    parameter [4:0] CUTS = 5'b11101
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,
    input  wire [2:0]  fmt,
    input  wire [1:0]  fa, fb,
    input  wire [31:0] a, b, c,
    input  wire        sub_c, mul_only, ftz,
    input  wire [2:0]  rfmt,
    input  wire [31:0] ra, rb, rc,
    input  wire        rsub, rmo, rftz,
    output wire [31:0] y, yref
);
    tnca_mac_lane #(.CUTS(CUTS), .IN_REG(1)) u_lane (
        .clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt), .fa(fa), .fb(fb),
        .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y));
    tnca_fpu_multi_ref #(.LAT(2), .IN_REG(1)) u_ref (
        .clk(clk), .rst_n(rst_n), .fmt(rfmt), .a(ra), .b(rb), .c(rc),
        .sub_c(rsub), .mul_only(rmo), .ftz(rftz), .y(yref));
endmodule
