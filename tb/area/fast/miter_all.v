// miter_all -- the R42 area FMA stages against the generated ones, combinational, for the Verilator driver
// (fast_miter.cpp). FMA chain from a, b, c; round stage on an arbitrary FmaNormalised struct s.
module miter_all #(parameter ADD = 0, parameter SHIFT = 0) (
    input  wire [31:0]  a, b, c,
    input  wire         sub_c, mul_only, ftz,
    input  wire [121:0] s,
    output wire [149:0] al_ref_o, al_new_o,
    output wire [121:0] nm_ref, nm_new,
    output wire [31:0]  y_ref, y_new,
    output wire [31:0]  yr_ref, yr_new
);
    wire [149:0] al, al_new;
    tnca_fma_align_xls u_al (.a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .out(al));
    tnca_fmax_align u_an (.a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .out(al_new));
    tnca_fmad_norm_xls u_nr (.s(al), .out(nm_ref));
    assign al_ref_o = al; assign al_new_o = al_new;
    tnca_fmax_norm #(.ADD(ADD)) u_nn (.s(al_new), .out(nm_new));
    tnca_fma_round_xls u_rr (.s(nm_ref), .out(y_ref));
    tnca_fmax_round #(.SHIFT(SHIFT)) u_rn (.s(nm_new), .out(y_new));
    tnca_fma_round_xls u_r2 (.s(s), .out(yr_ref));
    tnca_fmax_round #(.SHIFT(SHIFT)) u_n2 (.s(s), .out(yr_new));
endmodule
