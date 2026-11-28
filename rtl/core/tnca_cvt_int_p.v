//==============================================================================
// tnca_cvt_int_p -- tnca_cvt_int cut in two (R42 clk2, tnca_vector_fpu CVTP)
//
// The same function, bit for bit, as tnca_cvt_int: stage A is everything up to the leading-zero count (int -> f32: |x|, count) or
// up to the staged shift with its guard and sticky bit (f32 -> int: classification, the special-case word, the shifted magnitude);
// stage B is the rest (left shift, rounding and packing; rounding increment, negation, saturation, selection). REG = 1 puts a
// register between them (latency 1: y is valid one cycle after x, fn, rm are); REG = 0 makes the module combinational, which is
// what tb/clk2/cvt_int_p_miter compares with tnca_cvt_int over every input word, function and rounding mode.
//==============================================================================
`default_nettype none
module tnca_cvt_int_p #(
    parameter REG = 1
) (
    input  wire        clk,
    input  wire [31:0] x,
    input  wire [7:0]  fn,
    input  wire [2:0]  rm,
    output wire [31:0] y
);
    // ================================================================ stage A
    wire is_i2f = (fn == 8'h00) | (fn == 8'h01);
    wire sgn_i  = (fn == 8'h00);
    wire cuns   = (fn == 8'h03);

    // int -> f32: magnitude and leading-zero count
    wire        neg_i  = sgn_i & x[31];
    wire [31:0] mag_i  = neg_i ? (~x + 32'd1) : x;
    wire        zero_i = (mag_i == 32'd0);
    wire [4:0]  lz;
    wire        h16 = |mag_i[31:16];
    wire [15:0] m16 = h16 ? mag_i[31:16] : mag_i[15:0];
    wire        h8  = |m16[15:8];
    wire [7:0]  m8  = h8 ? m16[15:8] : m16[7:0];
    wire        h4  = |m8[7:4];
    wire [3:0]  m4  = h4 ? m8[7:4] : m8[3:0];
    wire        h2  = |m4[3:2];
    wire [1:0]  m2  = h2 ? m4[3:2] : m4[1:0];
    wire        h1  = m2[1];
    assign lz = {~h16, ~h8, ~h4, ~h2, ~h1};

    // f32 -> int: classification, the special-case word, the shifted magnitude with guard and lost
    wire [7:0]  cexp = x[30:23];
    wire        cneg = x[31];
    wire        nan  = (cexp == 8'hff) & (x[22:0] != 23'd0);
    wire        lt1  = (cexp < 8'd127);
    wire        big  = (cexp >= (cuns ? 8'd159 : 8'd158));
    wire        nz   = (x[30:0] != 31'd0);
    wire        half_gt = (cexp == 8'd126) & (x[22:0] != 23'd0);
    reg         inc_small;
    always @(*) begin
        case (rm)
        3'd1:    inc_small = half_gt;
        3'd2:    inc_small = cneg & nz;
        3'd3:    inc_small = ~cneg & nz;
        default: inc_small = 1'b0;
        endcase
    end
    wire [31:0] y_small = ~inc_small ? 32'd0 : (cneg ? (cuns ? 32'd0 : 32'hffffffff) : 32'd1);
    wire [31:0] y_big   = cuns ? (cneg ? 32'd0 : 32'hffffffff) : (cneg ? 32'h80000000 : 32'h7fffffff);
    // the word taken in place of the shifted magnitude: NaN 0, |x| < 1 y_small, out of range y_big
    wire        spec_sel = nan | lt1 | big;
    wire [31:0] y_spec   = nan ? 32'd0 : lt1 ? y_small : y_big;

    wire [4:0]  csh = 5'd30 - cexp[4:0] + 5'd0;
    wire [32:0] vx  = {1'b1, x[22:0], 8'd0, 1'b0};
    wire        l4  = csh[4] & (|vx[15:0]);
    wire [32:0] s1  = csh[4] ? (vx >> 16) : vx;
    wire        l3  = csh[3] & (|s1[7:0]);
    wire [32:0] s2  = csh[3] ? (s1 >> 8) : s1;
    wire        l2  = csh[2] & (|s2[3:0]);
    wire [32:0] s3  = csh[2] ? (s2 >> 4) : s2;
    wire        l1  = csh[1] & (|s3[1:0]);
    wire [32:0] s4  = csh[1] ? (s3 >> 2) : s3;
    wire        l0  = csh[0] & s4[0];
    wire [32:0] s5  = csh[0] ? (s4 >> 1) : s4;
    wire        guard_a = s5[0];
    wire [31:0] cmag_a  = s5[32:1];
    wire        lost_a  = l4 | l3 | l2 | l1 | l0;

    // what crosses the cut
    wire [31:0] a_v32 = is_i2f ? mag_i : cmag_a;       // int -> f32: |x|; f32 -> int: the shifted magnitude
    wire        a_i2f = is_i2f;
    wire        a_neg = is_i2f ? neg_i : cneg;
    wire        a_zero = zero_i;
    wire [4:0]  a_lz  = lz;
    wire        a_uns = cuns;
    wire [2:0]  a_rm  = rm;
    wire        a_guard = guard_a;
    wire        a_lost  = lost_a;
    wire        a_sel   = spec_sel;
    wire [31:0] a_spec  = y_spec;

    reg [31:0] q_v32; reg q_i2f, q_neg, q_zero; reg [4:0] q_lz; reg q_uns; reg [2:0] q_rm; reg q_guard, q_lost, q_sel; reg [31:0] q_spec;
    generate
        if (REG != 0) begin : g_reg
            always @(posedge clk) begin
                q_v32 <= a_v32; q_i2f <= a_i2f; q_neg <= a_neg; q_zero <= a_zero; q_lz <= a_lz; q_uns <= a_uns; q_rm <= a_rm;
                q_guard <= a_guard; q_lost <= a_lost; q_sel <= a_sel; q_spec <= a_spec;
            end
        end else begin : g_comb
            always @(*) begin
                q_v32 = a_v32; q_i2f = a_i2f; q_neg = a_neg; q_zero = a_zero; q_lz = a_lz; q_uns = a_uns; q_rm = a_rm;
                q_guard = a_guard; q_lost = a_lost; q_sel = a_sel; q_spec = a_spec;
            end
        end
    endgenerate

    // ================================================================ stage B
    // int -> f32
    wire [31:0] norm = q_v32 << q_lz;
    wire [23:0] man  = norm[31:8];
    wire        g_i  = norm[7];
    wire        st_i = |norm[6:0];
    wire        up_i = g_i & (st_i | man[0]);
    wire [24:0] man1 = {1'b0, man} + {24'd0, up_i};
    wire        cy_i = man1[24];
    wire [7:0]  exp_i = 8'd158 - {3'd0, q_lz} + {7'd0, cy_i};
    wire [31:0] y_i2f = q_zero ? 32'd0 : {q_neg, exp_i, cy_i ? 23'd0 : man1[22:0]};
    // f32 -> int
    reg         inc_f;
    always @(*) begin
        case (q_rm)
        3'd1:    inc_f = q_guard & (q_lost | q_v32[0]);
        3'd2:    inc_f = q_neg & (q_guard | q_lost);
        3'd3:    inc_f = ~q_neg & (q_guard | q_lost);
        default: inc_f = 1'b0;
        endcase
    end
    wire [32:0] cmag1 = {1'b0, q_v32} + {32'd0, inc_f};
    wire [31:0] y_s   = q_neg ? (~cmag1[31:0] + 32'd1) : (cmag1[32] | cmag1[31] ? 32'h7fffffff : cmag1[31:0]);
    wire [31:0] y_u   = q_neg ? 32'd0 : (cmag1[32] ? 32'hffffffff : cmag1[31:0]);
    wire [31:0] y_mid = q_uns ? y_u : y_s;
    wire [31:0] y_f2i = q_sel ? q_spec : y_mid;
    assign y = q_i2f ? y_i2f : y_f2i;
endmodule
`default_nettype wire
