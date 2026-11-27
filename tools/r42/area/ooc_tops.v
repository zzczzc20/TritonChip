// ooc_tops -- out-of-context tops for the R42-area screens. Every port of the unit under test is a register of the
// wrapper (operand latches before it, a result register after it), so every path is register to register.
//
//   area_fma        tnca_fp32_fma (LAT, IMPL; IMPL 0 generated, 1 tree-LZC rebuild, 2 DSP-mapped when present)
//   area_st_align   tnca_fma_align_xls     (stage 1 of the FMA, multiplier + alignment)
//   area_st_norm    tnca_fma_norm_xls / tnca_fmad_norm_xls (stage 2, IMPL selects)
//   area_st_round   tnca_fma_round_xls     (stage 3)
//   area_mac        tnca_mac_lane (the shared multi-mode lane, mode on a port), CUTS, IN_REG 0 as in tnca_matrix
//   area_mac4       the same lane with mode tied to DP4
module area_fma #(
    parameter integer LAT  = 2,
    parameter integer IMPL = 1,
    parameter integer BE   = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,
    input  wire        mul_only,
    input  wire        ftz,
    output reg  [31:0] y_q
);
    reg [31:0] w_a, w_b, w_c;
    reg        w_s, w_m, w_f;
    always @(posedge clk) begin w_a <= a; w_b <= b; w_c <= c; w_s <= sub_c; w_m <= mul_only; w_f <= ftz; end
    wire [31:0] y;
    tnca_fp32_fma #(.LAT(LAT), .IMPL(IMPL), .BE(BE)) u_fma (
        .clk(clk), .rst_n(rst_n), .a(w_a), .b(w_b), .c(w_c), .sub_c(w_s), .mul_only(w_m), .ftz(w_f), .y(y));
    always @(posedge clk) y_q <= y;
endmodule

module area_st_align (
    input  wire         clk,
    input  wire [31:0]  a, b, c,
    input  wire         sub_c, mul_only, ftz,
    output reg  [149:0] o_q
);
    reg [31:0] w_a, w_b, w_c;
    reg        w_s, w_m, w_f;
    always @(posedge clk) begin w_a <= a; w_b <= b; w_c <= c; w_s <= sub_c; w_m <= mul_only; w_f <= ftz; end
    wire [149:0] o;
    tnca_fma_align_xls u (.a(w_a), .b(w_b), .c(w_c), .sub_c(w_s), .mul_only(w_m), .ftz(w_f), .out(o));
    always @(posedge clk) o_q <= o;
endmodule

module area_st_norm #(parameter integer IMPL = 1) (
    input  wire         clk,
    input  wire [149:0] s,
    output reg  [121:0] o_q
);
    reg [149:0] w_s;
    always @(posedge clk) w_s <= s;
    wire [121:0] o;
    generate
        if (IMPL == 1) begin : g1 tnca_fmad_norm_xls u (.s(w_s), .out(o)); end
        else           begin : g0 tnca_fma_norm_xls  u (.s(w_s), .out(o)); end
    endgenerate
    always @(posedge clk) o_q <= o;
endmodule

module area_st_round (
    input  wire         clk,
    input  wire [121:0] s,
    output reg  [31:0]  y_q
);
    reg [121:0] w_s;
    always @(posedge clk) w_s <= s;
    wire [31:0] y;
    tnca_fma_round_xls u (.s(w_s), .out(y));
    always @(posedge clk) y_q <= y;
endmodule

module area_mac #(
    parameter [4:0] CUTS = 5'b11101,
    parameter integer BE = 0
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
    tnca_mac_lane #(.CUTS(CUTS), .IN_REG(0), .BE(BE)) u_lane (
        .clk(clk), .rst_n(rst_n), .mode(w_mode), .fmt(w_fmt), .fa(w_fa), .fb(w_fb), .a(w_a), .b(w_b), .c(w_c),
        .sub_c(1'b0), .mul_only(w_mul_only), .ftz(1'b0), .y(y));
    always @(posedge clk) y_q <= y;
endmodule

// the R42 area back-end stages on their own (same struct ports as the generated ones)
module area_st_norm2 #(parameter integer ADD = 0) (
    input  wire         clk,
    input  wire [149:0] s,
    output reg  [121:0] o_q
);
    reg [149:0] w_s;
    always @(posedge clk) w_s <= s;
    wire [121:0] o;
    tnca_fmax_norm #(.ADD(ADD)) u (.s(w_s), .out(o));
    always @(posedge clk) o_q <= o;
endmodule

module area_st_round2 #(parameter integer SHIFT = 0) (
    input  wire         clk,
    input  wire [121:0] s,
    output reg  [31:0]  y_q
);
    reg [121:0] w_s;
    always @(posedge clk) w_s <= s;
    wire [31:0] y;
    tnca_fmax_round #(.SHIFT(SHIFT)) u (.s(w_s), .out(y));
    always @(posedge clk) y_q <= y;
endmodule

// the special function unit as the vector FPU instantiates it (VLEN lanes through one four-function slice)
module area_sfu #(parameter integer LANES = 1) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,
    input  wire [7:0]    func,
    input  wire [511:0]  va,
    output reg           busy_q,
    output reg           done_q,
    output reg  [511:0]  vres_q
);
    reg          w_start; reg [7:0] w_func; reg [511:0] w_va;
    always @(posedge clk) begin w_start <= start; w_func <= func; w_va <= va; end
    wire busy, done; wire [511:0] vres;
    tnca_sfu #(.VLEN(16), .ELEN(32), .LAT(6), .LANES(LANES)) u (.clk(clk), .rst_n(rst_n), .start(w_start), .func(w_func), .va(w_va),
        .busy(busy), .done(done), .vres(vres));
    always @(posedge clk) begin busy_q <= busy; done_q <= done; vres_q <= vres; end
endmodule

// one converter lane (format conversion, combinational generated stage)
module area_cvt (
    input  wire         clk,
    input  wire [31:0]  x,
    input  wire [2:0]   src,
    input  wire [2:0]   dst,
    output reg  [31:0]  y_q
);
    reg [31:0] w_x; reg [2:0] w_s, w_d;
    always @(posedge clk) begin w_x <= x; w_s <= src; w_d <= dst; end
    wire [31:0] y;
    tnca_cvt_comb_xls u (.src(w_s), .dst(w_d), .x(w_x), .out(y));
    always @(posedge clk) y_q <= y;
endmodule
