//==============================================================================
// calibration.v -- reference structures for reading the depth numbers
//
// There is no process library here, so `synth.py` reports gate levels after
// mapping to a generic two-level gate set. That mapping has no carry-lookahead
// or parallel-prefix adder cells, so every wide addition comes out as ripple
// carry and its depth grows linearly with width. A real standard-cell flow
// implements the same addition as a parallel-prefix tree at roughly logarithmic
// depth.
//
// These modules measure that floor directly. A design whose depth sits near the
// floor for the arithmetic it contains is not necessarily slow -- the metric is.
// A design several times above the floor has structure worth fixing.
//
// Nothing here is part of the chip.
//==============================================================================

module calib_add32 (input wire [31:0] a, input wire [31:0] b, output wire [31:0] y);
    assign y = a + b;
endmodule

module calib_add64 (input wire [63:0] a, input wire [63:0] b, output wire [63:0] y);
    assign y = a + b;
endmodule

module calib_add89 (input wire [88:0] a, input wire [88:0] b, output wire [88:0] y);
    assign y = a + b;
endmodule

module calib_mul24 (input wire [23:0] a, input wire [23:0] b, output wire [47:0] y);
    assign y = a * b;
endmodule

module calib_mul32 (input wire [31:0] a, input wire [31:0] b, output wire [63:0] y);
    assign y = a * b;
endmodule

// A 16-to-1 crossbar of 32-bit words: the shuffle network's shape.
module calib_xbar16 (
    input  wire [16*32-1:0] v,
    input  wire [3:0]       sel,
    output wire [31:0]      y
);
    assign y = v[sel*32 +: 32];
endmodule
