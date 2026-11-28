// cvt_int_miter (R42 clk2) -- tnca_cvt_int_p with REG = 0 (the two halves combinational) against tnca_cvt_int, which tb/area/fast/cvt_int_miter
// proved equal to cvt_int_f. Driven by tb/area/fast/cvt_int_miter.cpp (same top name and ports).
module cvt_int_miter (
    input  wire [31:0] x, input wire [7:0] fn, input wire [2:0] rm,
    output wire [31:0] y_ref, y_new
);
    tnca_cvt_int u_ref (.x(x), .fn(fn), .rm(rm), .y(y_ref));
    tnca_cvt_int_p #(.REG(0)) u_new (.clk(1'b0), .x(x), .fn(fn), .rm(rm), .y(y_new));
endmodule
