// unitp_miter (R42 clk2): tnca_fmax_unitp(a, ftz) against tnca_fmax_mul(a, 1.0, ftz)
module unitp_miter (
    input  wire [31:0] a, input wire ftz,
    output wire [63:0] y_ref, y_new
);
    tnca_fmax_mul   u_ref (.a(a), .b(32'h3f800000), .ftz(ftz), .p(y_ref));
    tnca_fmax_unitp u_new (.a(a), .ftz(ftz), .p(y_new));
endmodule
