//==============================================================================
// tnca_lzc -- leading-zero count / most-significant-set-bit index
//
// The obvious way to write this in Verilog is
//     for (i = 0; i < W; i = i + 1) if (v[i]) idx = W-1-i;
// which is correct and synthesises to a chain of W muxes. At W = 89 that chain
// was most of the 207-level critical path through tnca_fp32_fma.
//
// This version isolates the highest set bit with one subtract
// (rev & -rev on the bit-reversed input) and converts the resulting one-hot to
// binary with LOGW independent OR trees. Depth is O(log W) instead of O(W).
//
// clz reads W when v is zero, matching the ISA's definition of s.clz/v.clz.
//==============================================================================

module tnca_lzc #(
    parameter W    = 64,
    parameter LOGW = 6            // ceil(log2(W))
) (
    input  wire [W-1:0]     v,
    output wire [LOGW:0]    clz,      // leading zeros; W when v == 0
    output wire [LOGW-1:0]  msb_idx,  // index of the highest set bit; 0 when v == 0
    output wire             any
);

    localparam PW = 1 << LOGW;        // PW >= W

    // Bit-reverse into a power-of-two field. Bit 0 of `rev` is bit W-1 of v, so
    // the position of rev's lowest set bit IS the leading-zero count.
    wire [PW-1:0] rev;
    genvar i, j;
    generate
        for (i = 0; i < PW; i = i + 1) begin : g_rev
            if (i < W) begin : g_real
                assign rev[i] = v[W-1-i];
            end else begin : g_pad
                assign rev[i] = 1'b0;
            end
        end
    endgenerate

    // Isolate the lowest set bit.
    wire [PW-1:0] iso = rev & (~rev + {{(PW-1){1'b0}}, 1'b1});

    // One-hot to binary: bit k of the position is the OR of every iso[j] whose
    // index j has bit k set. j is a genvar, so the selection is constant folding.
    wire [LOGW-1:0] pos;
    generate
        for (i = 0; i < LOGW; i = i + 1) begin : g_bit
            wire [PW-1:0] sel;
            for (j = 0; j < PW; j = j + 1) begin : g_sel
                assign sel[j] = iso[j] & (((j >> i) & 1) != 0);
            end
            assign pos[i] = |sel;
        end
    endgenerate

    assign any     = |v;
    assign clz     = any ? {1'b0, pos} : W[LOGW:0];
    assign msb_idx = any ? (W[LOGW-1:0] - 1'b1 - pos) : {LOGW{1'b0}};

endmodule
