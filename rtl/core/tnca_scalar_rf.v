//==============================================================================
// tnca_scalar_rf -- scalar register file, NUM_RCTX x 64 x 64 bit
//
// 3 read ports + 1 write port. Reads are asynchronous (read in RD, use in EX)
// and are write-first: a read that collides with the concurrent write returns
// the new value. That bypass is what lets the scoreboard release a destination
// in the same cycle it is written, instead of costing an extra bubble.
//
// S0 reads as zero and discards writes (ISA sec 1.1).
//==============================================================================
`include "tnca_defs.vh"

module tnca_scalar_rf #(
    parameter NUM_RCTX  = 2,
    parameter RCTX_W    = 1
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // read port A
    input  wire [RCTX_W-1:0]     ra_rctx,
    input  wire [5:0]            ra_idx,
    output wire [63:0]           ra_data,
    // read port B
    input  wire [RCTX_W-1:0]     rb_rctx,
    input  wire [5:0]            rb_idx,
    output wire [63:0]           rb_data,
    // read port C
    input  wire [RCTX_W-1:0]     rc_rctx,
    input  wire [5:0]            rc_idx,
    output wire [63:0]           rc_data,

    // write port
    input  wire                  w_en,
    input  wire [RCTX_W-1:0]     w_rctx,
    input  wire [5:0]            w_idx,
    input  wire [63:0]           w_data
);

    localparam DEPTH = NUM_RCTX * 64;

    reg [63:0] mem [0:DEPTH-1];

    wire [RCTX_W+5:0] wa = {w_rctx, w_idx};
    wire              w_real = w_en & (w_idx != 6'd0);

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1)
                mem[i] <= 64'd0;
        end else if (w_real) begin
            mem[wa] <= w_data;
        end
    end

    // Asynchronous, write-first reads. S0 forced to zero.
    wire [RCTX_W+5:0] aa = {ra_rctx, ra_idx};
    wire [RCTX_W+5:0] ab = {rb_rctx, rb_idx};
    wire [RCTX_W+5:0] ac = {rc_rctx, rc_idx};

    assign ra_data = (ra_idx == 6'd0)            ? 64'd0 :
                     (w_real && (aa == wa))      ? w_data : mem[aa];
    assign rb_data = (rb_idx == 6'd0)            ? 64'd0 :
                     (w_real && (ab == wa))      ? w_data : mem[ab];
    assign rc_data = (rc_idx == 6'd0)            ? 64'd0 :
                     (w_real && (ac == wa))      ? w_data : mem[ac];

endmodule
