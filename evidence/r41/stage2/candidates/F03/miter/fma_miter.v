// fma_miter -- the candidate FMA against the previous generated RTL.
// ref_l2/ref_l3 are r41s2's rtl/gen/tnca_fma_xls.v and tnca_fma_l3_xls.v (the
// XLS-scheduled instances of latency 2 and 3), modules renamed *_ref. new_l2
// and new_l3 are the candidate's tnca_fp32_fma at LAT 2 and 3, i.e. whatever
// rtl/core/tnca_fp32_fma.v builds from the candidate's rtl/gen. Same inputs,
// same latency, so the outputs must agree every cycle.
module fma_miter (
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] a, b, c,
    input  wire        sub_c, mul_only, ftz,
    output wire [31:0] y_ref2, y_new2, y_ref3, y_new3
);
    tnca_fma_xls_ref    u_r2 (.clk(clk), .rst(rst), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .out(y_ref2));
    tnca_fma_l3_xls_ref u_r3 (.clk(clk), .rst(rst), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .out(y_ref3));
    tnca_fp32_fma #(.LAT(2)) u_n2 (.clk(clk), .rst_n(!rst), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_new2));
    tnca_fp32_fma #(.LAT(3)) u_n3 (.clk(clk), .rst_n(!rst), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_new3));
endmodule
