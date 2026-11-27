// cvt_int_miter -- tnca_cvt_int against the vector FPU's function cvt_int_f (copied verbatim into cvt_int_ref, exp/cvt_int_ref.v)
module cvt_int_miter (
    input  wire [31:0] x, input wire [7:0] fn, input wire [2:0] rm,
    output wire [31:0] y_ref, y_new
);
    cvt_int_ref u_ref (.x(x), .fn(fn), .rm(rm), .y(y_ref));
    tnca_cvt_int u_new (.x(x), .fn(fn), .rm(rm), .y(y_new));
endmodule
