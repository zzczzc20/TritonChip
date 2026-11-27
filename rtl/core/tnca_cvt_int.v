//==============================================================================
// tnca_cvt_int -- the vector FPU's int <-> binary32 conversion, one lane (R42 area)
//
// The same function, bit for bit, as cvt_int_f in tnca_vector_fpu.v (ISA sec 8.4: s32/u32 -> f32 rounds to nearest even;
// f32 -> s32/u32 rounds by rm (RTZ, RNE, RMI, RPI), saturates out-of-range values and infinities, NaN -> 0, a negative value
// to an unsigned integer -> 0), written with a guard bit and a running sticky bit instead of 64-bit fraction compares:
//
//   int -> f32:  |x| (32-bit negate), leading-zero count, left shift, round to nearest even on the 24-bit significand
//   f32 -> int:  {1, fraction, 8'b0} shifted right by 158 - exponent through a staged shifter that ORs what it drops into
//                a sticky bit; the dropped bit just below the integer part is the guard bit. Round-up conditions per rm
//                use (guard, sticky, lsb, sign).
//
// fn: 0 = s32 -> f32, 1 = u32 -> f32, any other value is a float -> int conversion with an unsigned result for fn = 3
// (tnca_vector_fpu only presents 0..3). Combinational. tb/area/fast/cvt_int_miter compares it with cvt_int_f over every
// input word, function and rounding mode.
//==============================================================================
`default_nettype none
module tnca_cvt_int (
    input  wire [31:0] x,
    input  wire [7:0]  fn,
    input  wire [2:0]  rm,
    output wire [31:0] y
);
    wire is_i2f = (fn == 8'h00) | (fn == 8'h01);
    wire sgn_i  = (fn == 8'h00);
    wire cuns   = (fn == 8'h03);

    // ---------------------------------------------------------------- int -> f32
    wire        neg_i  = sgn_i & x[31];
    wire [31:0] mag_i  = neg_i ? (~x + 32'd1) : x;
    wire        zero_i = (mag_i == 32'd0);
    // leading-zero count of mag_i (0..31), as a priority encoder tree
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
    wire [31:0] norm = mag_i << lz;
    wire [23:0] man  = norm[31:8];
    wire        g_i  = norm[7];
    wire        st_i = |norm[6:0];
    wire        up_i = g_i & (st_i | man[0]);
    wire [24:0] man1 = {1'b0, man} + {24'd0, up_i};
    wire        cy_i = man1[24];
    wire [7:0]  exp_i = 8'd158 - {3'd0, lz} + {7'd0, cy_i};      // 127 + (31 - lz) + carry
    wire [31:0] y_i2f = zero_i ? 32'd0 : {neg_i, exp_i, cy_i ? 23'd0 : man1[22:0]};

    // ---------------------------------------------------------------- f32 -> int
    wire [7:0]  cexp = x[30:23];
    wire        cneg = x[31];
    wire        nan  = (cexp == 8'hff) & (x[22:0] != 23'd0);
    wire        lt1  = (cexp < 8'd127);
    wire        big  = (cexp >= (cuns ? 8'd159 : 8'd158));
    wire        nz   = (x[30:0] != 31'd0);
    // |x| < 1: only the rounding mode can make it 1 (|x| > 0.5 for RNE)
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

    // 1 <= |x| < 2^32: v = {1, fraction, 8'b0} >> csh, csh = 158 - cexp in 0..31, with guard and sticky
    wire [4:0]  csh = 5'd30 - cexp[4:0] + 5'd0;                  // 158 - cexp mod 32; cexp in 127..158 -> 31..0
    wire [32:0] vx  = {1'b1, x[22:0], 8'd0, 1'b0};               // {v, 0}: bit 0 becomes the guard position after the shift
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
    wire        guard = s5[0];
    wire [31:0] cmag  = s5[32:1];
    wire        lost  = l4 | l3 | l2 | l1 | l0;                  // bits dropped below the guard bit
    reg         inc_f;
    always @(*) begin
        case (rm)
        3'd1:    inc_f = guard & (lost | cmag[0]);
        3'd2:    inc_f = cneg & (guard | lost);
        3'd3:    inc_f = ~cneg & (guard | lost);
        default: inc_f = 1'b0;
        endcase
    end
    wire [32:0] cmag1 = {1'b0, cmag} + {32'd0, inc_f};
    wire [31:0] y_s   = cneg ? (~cmag1[31:0] + 32'd1) : (cmag1[32] | cmag1[31] ? 32'h7fffffff : cmag1[31:0]);
    wire [31:0] y_u   = cneg ? 32'd0 : (cmag1[32] ? 32'hffffffff : cmag1[31:0]);
    wire [31:0] y_mid = cuns ? y_u : y_s;

    wire [31:0] y_f2i = nan ? 32'd0 : lt1 ? y_small : big ? y_big : y_mid;
    assign y = is_i2f ? y_i2f : y_f2i;
endmodule
`default_nettype wire
