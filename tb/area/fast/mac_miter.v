// mac_miter -- the shared MAC lane with the R42 area stages (BE) against the generated stages (BE = 0), all modes.
// Same CUTS and IN_REG, same inputs; the C++ driver (mac_miter.cpp) compares y every cycle.
module mac_miter #(parameter BE = 1, parameter [4:0] CUTS = 5'b11101) (
    input  wire        clk, rst_n,
    input  wire [1:0]  mode, fa, fb,
    input  wire [2:0]  fmt,
    input  wire [31:0] a, b, c,
    input  wire        sub_c, mul_only, ftz,
    output wire [31:0] y_ref, y_new,
    output wire [149:0] al_ref, al_new,   // B0 output (FmaAligned) of the two lanes, before the cut register
    output wire [121:0] nm_ref, nm_new    // normalise output (FmaNormalised)
);
    tnca_mac_lane #(.CUTS(CUTS), .IN_REG(1), .BE(0))  u_ref (.clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt), .fa(fa), .fb(fb),
        .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_ref));
    tnca_mac_lane #(.CUTS(CUTS), .IN_REG(1), .BE(BE)) u_new (.clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt), .fa(fa), .fb(fb),
        .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_new));
    assign al_ref = u_ref.al_d; assign al_new = u_new.al_d;
    assign nm_ref = u_ref.nm_d; assign nm_new = u_new.nm_d;
endmodule
