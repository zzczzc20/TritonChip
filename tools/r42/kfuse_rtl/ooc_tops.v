// ooc_tops -- out-of-context tops for the K-fused lane screens (R42-kfuse-rtl). Every port of the lane is a
// register in the wrapper (the operand latches of tnca_matrix before it, the accumulator write after it), so
// every path is register to register. The lane is instantiated with IN_REG = 0: the wrapper's operand
// register is the lane's first register, as in tnca_matrix.
//
//   kfuse_ooc_lane  one lane                           kfuse_ooc_array  NL lanes (SHARE_A = 1: one broadcast a)
//   kfuse_ooc_mac   the R41 shared lane tnca_mac_lane (CUTS 5'b11101, the image's MAC_CUTS 29), mode a port
//   kfuse_ooc_mac4  the same lane with mode tied to DP4 (E4M3 x E4M3, four products)
module kfuse_ooc_lane #(
    parameter integer G         = 8,
    parameter integer W         = 0,
    parameter [7:0]   CUTS      = 8'b00111011,
    parameter integer C_LATE    = 0,
    parameter integer SIGN_MODE = 1,
    parameter integer TREE      = 0
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    input  wire [8*G-1:0] a,
    input  wire [8*G-1:0] b,
    input  wire [31:0]    c,
    output reg  [31:0]    y_q,
    output reg            valid_q
);
    reg [8*G-1:0] w_a, w_b;
    reg [31:0]    w_c;
    reg           w_v;
    always @(posedge clk) begin w_a <= a; w_b <= b; w_c <= c; w_v <= valid_in; end
    wire [31:0] y;
    wire        v;
    tnca_kfuse_lane #(.G(G), .W(W), .CUTS(CUTS), .IN_REG(0), .C_LATE(C_LATE), .SIGN_MODE(SIGN_MODE),
                      .TREE(TREE)) u_lane (
        .clk(clk), .rst_n(rst_n), .valid_in(w_v), .a(w_a), .b(w_b), .c(w_c), .y(y), .valid_out(v));
    always @(posedge clk) begin y_q <= y; valid_q <= v; end
endmodule

module kfuse_ooc_array #(
    parameter integer G         = 8,
    parameter integer W         = 0,
    parameter [7:0]   CUTS      = 8'b00111011,
    parameter integer C_LATE    = 0,
    parameter integer SIGN_MODE = 1,
    parameter integer TREE      = 0,
    parameter integer NL        = 16,
    parameter integer SHARE_A   = 0
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    input  wire [8*G*(SHARE_A != 0 ? 1 : NL)-1:0] a,
    input  wire [8*G*NL-1:0] b,
    input  wire [32*NL-1:0]  c,
    output reg  [32*NL-1:0]  y_q,
    output reg               valid_q
);
    localparam integer NA = (SHARE_A != 0) ? 1 : NL;
    reg [8*G*NA-1:0] w_a;
    reg [8*G*NL-1:0] w_b;
    reg [32*NL-1:0]  w_c;
    reg              w_v;
    always @(posedge clk) begin w_a <= a; w_b <= b; w_c <= c; w_v <= valid_in; end
    wire [32*NL-1:0] y;
    wire [NL-1:0]    v;
    genvar i;
    generate
        for (i = 0; i < NL; i = i + 1) begin : g_l
            tnca_kfuse_lane #(.G(G), .W(W), .CUTS(CUTS), .IN_REG(0), .C_LATE(C_LATE), .SIGN_MODE(SIGN_MODE),
                              .TREE(TREE)) u_lane (
                .clk(clk), .rst_n(rst_n), .valid_in(w_v),
                .a(SHARE_A != 0 ? w_a : w_a[8*G*i +: 8*G]), .b(w_b[8*G*i +: 8*G]), .c(w_c[32*i +: 32]),
                .y(y[32*i +: 32]), .valid_out(v[i]));
        end
    endgenerate
    always @(posedge clk) begin y_q <= y; valid_q <= v[0]; end
endmodule

module kfuse_ooc_mac #(
    parameter [4:0] CUTS = 5'b11101
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,
    input  wire [2:0]  fmt,
    input  wire [1:0]  fa,
    input  wire [1:0]  fb,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        mul_only,
    output reg  [31:0] y_q
);
    reg [1:0]  w_mode, w_fa, w_fb;
    reg [2:0]  w_fmt;
    reg        w_mul_only;
    reg [31:0] w_a, w_b, w_c;
    always @(posedge clk) begin
        w_mode <= mode; w_fmt <= fmt; w_fa <= fa; w_fb <= fb; w_a <= a; w_b <= b; w_c <= c; w_mul_only <= mul_only;
    end
    wire [31:0] y;
    tnca_mac_lane #(.CUTS(CUTS), .IN_REG(0)) u_lane (
        .clk(clk), .rst_n(rst_n), .mode(w_mode), .fmt(w_fmt), .fa(w_fa), .fb(w_fb), .a(w_a), .b(w_b), .c(w_c),
        .sub_c(1'b0), .mul_only(w_mul_only), .ftz(1'b0), .y(y));
    always @(posedge clk) y_q <= y;
endmodule

module kfuse_ooc_mac4 #(
    parameter [4:0] CUTS = 5'b11101
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    output reg  [31:0] y_q
);
    reg [31:0] w_a, w_b, w_c;
    always @(posedge clk) begin w_a <= a; w_b <= b; w_c <= c; end
    wire [31:0] y;
    tnca_mac_lane #(.CUTS(CUTS), .IN_REG(0)) u_lane (
        .clk(clk), .rst_n(rst_n), .mode(2'd2), .fmt(3'd0), .fa(2'd3), .fb(2'd3), .a(w_a), .b(w_b), .c(w_c),
        .sub_c(1'b0), .mul_only(1'b0), .ftz(1'b0), .y(y));
    always @(posedge clk) y_q <= y;
endmodule
