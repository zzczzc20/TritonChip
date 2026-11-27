// fma_cyc_miter -- tnca_fmax_fma (CUTS, SHIFT, XLS_ALIGN) against the generated tnca_fp32_fma IMPL 1 at the same latency
// (LAT = popcount(CUTS), 2 or 3), every cycle. Plain tnca_fp32_fma BE = 0 is the reference; the dut is instantiated directly.
module fma_cyc_miter #(
    parameter [3:0] CUTS = 4'b1010, parameter SHIFT = 1, parameter XLS_ALIGN = 0, parameter LAT = 2
) (
    input  wire        clk, rst_n,
    input  wire [31:0] a, b, c,
    input  wire        sub_c, mul_only, ftz,
    output wire [31:0] y_ref, y_new
);
    tnca_fp32_fma #(.LAT(LAT), .IMPL(1)) u_ref (.clk(clk), .rst_n(rst_n), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_ref));
    tnca_fmax_fma #(.CUTS(CUTS), .SHIFT(SHIFT), .XLS_ALIGN(XLS_ALIGN)) u_new (.clk(clk), .rst_n(rst_n), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_new));
endmodule
