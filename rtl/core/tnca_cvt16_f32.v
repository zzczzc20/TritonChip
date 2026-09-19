//==============================================================================
// tnca_cvt16_f32 -- FP16 or BF16 to binary32, exactly
//
// Both conversions are exact: every FP16 and BF16 value is representable in
// binary32, so there is no rounding here and no rounding mode to choose. That
// matters for the matrix unit's accumulation contract (ISA sec 9C.1): if the
// widening were lossy, "the product is computed exactly" would be false before
// the multiplier even saw the operands.
//
// mode 0 = FP16 (1-5-10), mode 1 = BF16 (1-8-7).
//==============================================================================

module tnca_cvt16_f32 (
    input  wire [15:0] v,
    input  wire        mode,       // 0 = FP16, 1 = BF16
    output wire [31:0] y
);

    // ---- BF16: the top half of a binary32, by construction ------------------
    wire [31:0] bf = {v, 16'd0};

    // ---- FP16 ---------------------------------------------------------------
    wire        s  = v[15];
    wire [4:0]  e  = v[14:10];
    wire [9:0]  f  = v[9:0];

    wire is_zero = (e == 5'd0)  && (f == 10'd0);
    wire is_sub  = (e == 5'd0)  && (f != 10'd0);
    wire is_inf  = (e == 5'h1f) && (f == 10'd0);
    wire is_nan  = (e == 5'h1f) && (f != 10'd0);

    // Subnormals need normalising: value = f * 2^-24, and the exponent follows
    // the position of the leading one.
    wire [4:0] lz;
    wire [3:0] msb_i;
    wire       any;
    tnca_lzc #(.W(10), .LOGW(4)) u_lzc (
        .v(f), .clz(lz), .msb_idx(msb_i), .any(any)
    );

    wire [7:0]  sub_exp  = 8'd112 - {4'd0, lz[3:0]};
    wire [9:0]  sub_shf  = f << (lz[3:0] + 4'd1);
    wire [22:0] sub_frac = {sub_shf, 13'd0};

    wire [31:0] f16 =
        is_nan  ? 32'h7fc00000 :
        is_inf  ? {s, 8'hff, 23'd0} :
        is_zero ? {s, 31'd0} :
        is_sub  ? {s, sub_exp, sub_frac}
                : {s, ({3'd0, e} + 8'd112), f, 13'd0};

    assign y = mode ? bf : f16;

    wire _unused = &{1'b0, msb_i, any, lz[4], 1'b0};

endmodule
