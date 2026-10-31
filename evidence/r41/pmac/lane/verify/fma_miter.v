// fma_miter -- F03's miter (fma_miter.cpp, copied unchanged), reference =
// tnca_fp32_fma at r41-pmac's base 0b0a3cc (ref/, renamed *_ref by mkref.sh),
// candidate = tnca_fp32_fma over the rtl/gen this tree regenerated from the
// refactored xls/tnca_fma.x (fma32_align = fma32_align_p . fma32_mul).
module fma_miter (
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] a, b, c,
    input  wire        sub_c, mul_only, ftz,
    output wire [31:0] y_ref2, y_new2, y_ref3, y_new3
);
    tnca_fp32_fma_ref #(.LAT(2)) u_r2 (.clk(clk), .rst_n(!rst), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_ref2));
    tnca_fp32_fma_ref #(.LAT(3)) u_r3 (.clk(clk), .rst_n(!rst), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_ref3));
    tnca_fp32_fma #(.LAT(2)) u_n2 (.clk(clk), .rst_n(!rst), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_new2));
    tnca_fp32_fma #(.LAT(3)) u_n3 (.clk(clk), .rst_n(!rst), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_new3));
endmodule
