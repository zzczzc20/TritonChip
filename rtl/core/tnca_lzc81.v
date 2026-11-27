//==============================================================================
// tnca_lzc81 -- count of leading zeros of an 81-bit value, as a balanced tree (R42 area).
//
//   cnt = number of zero bits above the most significant one of x, 0..80; cnt = 81 for x = 0 (zero = 1).
//
// The same structure as xls/tnca_lzc.x (each node: "some one below me" and the count inside it; a parent takes the
// left child's count if the left child has a one, else the left width plus the right child's), written for 81 bits
// so that the upper eight positions of the 89-bit view the generated stage counts (always zero: the sum is below
// 2^81) cost nothing. Combinational.
//==============================================================================
`default_nettype none
module tnca_lzc81 (
    input  wire [80:0] x,
    output wire [6:0]  cnt,    // 0..81
    output wire        zero
);
    // pad to 128 with the value at the top; the low 47 positions are zero, so a nonzero value's count is unchanged
    // (they sit below its leading one) and the all-zero case is the `zero` output.
    wire [127:0] p = {x, 47'b0};
    wire [63:0] v1; wire [63:0] c1;          // count width 1
    genvar i;
    generate for (i = 0; i < 64; i = i + 1) begin : g1
        assign v1[i] = p[127 - 2*i] | p[126 - 2*i];
        assign c1[i] = ~p[127 - 2*i];
    end endgenerate
    wire [31:0] v2; wire [63:0] c2;          // width 2
    generate for (i = 0; i < 32; i = i + 1) begin : g2
        assign v2[i] = v1[2*i] | v1[2*i+1];
        assign c2[2*i +: 2] = v1[2*i] ? {1'b0, c1[2*i]} : {1'b1, c1[2*i+1]};
    end endgenerate
    wire [15:0] v3; wire [47:0] c3;          // width 3
    generate for (i = 0; i < 16; i = i + 1) begin : g3
        assign v3[i] = v2[2*i] | v2[2*i+1];
        assign c3[3*i +: 3] = v2[2*i] ? {1'b0, c2[2*(2*i) +: 2]} : {1'b1, c2[2*(2*i+1) +: 2]};
    end endgenerate
    wire [7:0] v4; wire [31:0] c4;           // width 4
    generate for (i = 0; i < 8; i = i + 1) begin : g4
        assign v4[i] = v3[2*i] | v3[2*i+1];
        assign c4[4*i +: 4] = v3[2*i] ? {1'b0, c3[3*(2*i) +: 3]} : {1'b1, c3[3*(2*i+1) +: 3]};
    end endgenerate
    wire [3:0] v5; wire [19:0] c5;           // width 5
    generate for (i = 0; i < 4; i = i + 1) begin : g5
        assign v5[i] = v4[2*i] | v4[2*i+1];
        assign c5[5*i +: 5] = v4[2*i] ? {1'b0, c4[4*(2*i) +: 4]} : {1'b1, c4[4*(2*i+1) +: 4]};
    end endgenerate
    wire [1:0] v6; wire [11:0] c6;           // width 6
    generate for (i = 0; i < 2; i = i + 1) begin : g6
        assign v6[i] = v5[2*i] | v5[2*i+1];
        assign c6[6*i +: 6] = v5[2*i] ? {1'b0, c5[5*(2*i) +: 5]} : {1'b1, c5[5*(2*i+1) +: 5]};
    end endgenerate
    wire        v7 = v6[0] | v6[1];
    wire [6:0]  c7 = v6[0] ? {1'b0, c6[5:0]} : {1'b1, c6[11:6]};
    assign zero = ~v7;
    assign cnt  = v7 ? c7 : 7'd81;
endmodule
`default_nettype wire
