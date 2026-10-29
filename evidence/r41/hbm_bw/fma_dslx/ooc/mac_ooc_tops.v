// mac_ooc_tops -- out-of-context tops for the lane cost screens (x86 Vivado,
// tools/r41/fmax/run_unit.sh: regwrap adds a flop on every port, so IN_REG=0
// here is the matrix engine's operand register). mac_ooc_<CUTS>: tnca_mac_lane
// at that cut placement; mac_ooc_v0: today's lane, tnca_fpu_multi LAT 2.
module mac_ooc_1100(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,
    input  wire [2:0]  fmt,
    input  wire [1:0]  fa,
    input  wire [1:0]  fb,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output wire [31:0] y
);
    tnca_mac_lane #(.CUTS(4'b1100), .IN_REG(0)) u (.clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt),
        .fa(fa), .fb(fb), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y));
endmodule
module mac_ooc_1110(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,
    input  wire [2:0]  fmt,
    input  wire [1:0]  fa,
    input  wire [1:0]  fb,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output wire [31:0] y
);
    tnca_mac_lane #(.CUTS(4'b1110), .IN_REG(0)) u (.clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt),
        .fa(fa), .fb(fb), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y));
endmodule
module mac_ooc_1101(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,
    input  wire [2:0]  fmt,
    input  wire [1:0]  fa,
    input  wire [1:0]  fb,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output wire [31:0] y
);
    tnca_mac_lane #(.CUTS(4'b1101), .IN_REG(0)) u (.clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt),
        .fa(fa), .fb(fb), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y));
endmodule
module mac_ooc_1111(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,
    input  wire [2:0]  fmt,
    input  wire [1:0]  fa,
    input  wire [1:0]  fb,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output wire [31:0] y
);
    tnca_mac_lane #(.CUTS(4'b1111), .IN_REG(0)) u (.clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt),
        .fa(fa), .fb(fb), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y));
endmodule
module mac_ooc_0110(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,
    input  wire [2:0]  fmt,
    input  wire [1:0]  fa,
    input  wire [1:0]  fb,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output wire [31:0] y
);
    tnca_mac_lane #(.CUTS(4'b0110), .IN_REG(0)) u (.clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt),
        .fa(fa), .fb(fb), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y));
endmodule
module mac_ooc_v0(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [2:0]  fmt,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output wire [31:0] y
);
    tnca_fpu_multi #(.LAT(2), .IN_REG(0)) u (.clk(clk), .rst_n(rst_n), .fmt(fmt), .a(a), .b(b), .c(c),
        .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y));
endmodule
