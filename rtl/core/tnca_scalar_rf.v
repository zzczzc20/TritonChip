//==============================================================================
// tnca_scalar_rf -- scalar register file, NUM_RCTX x 64 x 64 bit
//
// 7 read ports + 1 write port (A-C operands, D-G a transfer's coordinates). Reads are asynchronous (read in RD, use in EX)
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
    // read port D -- a transfer's coordinates come from CONSECUTIVE scalar
    // registers (ISA sec 9E), so the second one needs its own read. The scalar
    // file is small enough that a fourth port is cheaper than a second cycle.
    input  wire [RCTX_W-1:0]     rd_rctx,
    input  wire [5:0]            rd_idx,
    output wire [63:0]           rd_data,

    // read ports E, F, G -- the third to fifth coordinates of a rank-3..5
    // transfer (Sb+2..Sb+4). Same reasoning as port D: five consecutive
    // reads in one cycle cost three more read muxes on a 64-entry file,
    // where a multi-cycle read would cost a sequencer in the issue path.
    input  wire [RCTX_W-1:0]     re_rctx,
    input  wire [5:0]            re_idx,
    output wire [63:0]           re_data,
    input  wire [5:0]            rf_idx,
    output wire [63:0]           rf_data,
    input  wire [5:0]            rg_idx,
    output wire [63:0]           rg_data,

    // write port
    input  wire                  w_en,
    input  wire [RCTX_W-1:0]     w_rctx,
    input  wire [5:0]            w_idx,
    input  wire [63:0]           w_data
);

    localparam DEPTH = NUM_RCTX * 64;

    // The array has no reset, so it maps to distributed RAM, which the tool
    // replicates once per read port (each copy: the one write, one
    // asynchronous read). With a reset loop it was 8 192 flip-flops and ~25 k
    // LUTs of seven 128:1 read multiplexers (timing-012). A valid bit per
    // register stands in for the reset: cleared by it, set by a write, and a
    // register whose bit is clear reads as zero -- as the reset left it.
    reg [63:0]       mem   [0:DEPTH-1];
    reg [DEPTH-1:0]  valid;

    wire [RCTX_W+5:0] wa = {w_rctx, w_idx};
    wire              w_real = w_en & (w_idx != 6'd0);

    always @(posedge clk)
        if (w_real)
            mem[wa] <= w_data;

    always @(posedge clk) begin
        if (!rst_n)
            valid <= {DEPTH{1'b0}};
        else if (w_real)
            valid[wa] <= 1'b1;
    end

    wire [RCTX_W+5:0] aa = {ra_rctx, ra_idx};
    wire [RCTX_W+5:0] ab = {rb_rctx, rb_idx};
    wire [RCTX_W+5:0] ac = {rc_rctx, rc_idx};
    wire [RCTX_W+5:0] ad = {rd_rctx, rd_idx};
    wire [RCTX_W+5:0] ae = {re_rctx, re_idx};
    wire [RCTX_W+5:0] af = {re_rctx, rf_idx};
    wire [RCTX_W+5:0] ag = {re_rctx, rg_idx};

    assign ra_data = (ra_idx == 6'd0)            ? 64'd0 :
                     (w_real && (aa == wa))      ? w_data : (valid[aa] ? mem[aa] : 64'd0);
    assign rb_data = (rb_idx == 6'd0)            ? 64'd0 :
                     (w_real && (ab == wa))      ? w_data : (valid[ab] ? mem[ab] : 64'd0);
    assign rc_data = (rc_idx == 6'd0)            ? 64'd0 :
                     (w_real && (ac == wa))      ? w_data : (valid[ac] ? mem[ac] : 64'd0);
    assign rd_data = (rd_idx == 6'd0)            ? 64'd0 :
                     (w_real && (ad == wa))      ? w_data : (valid[ad] ? mem[ad] : 64'd0);
    assign re_data = (re_idx == 6'd0)            ? 64'd0 :
                     (w_real && (ae == wa))      ? w_data : (valid[ae] ? mem[ae] : 64'd0);
    assign rf_data = (rf_idx == 6'd0)            ? 64'd0 :
                     (w_real && (af == wa))      ? w_data : (valid[af] ? mem[af] : 64'd0);
    assign rg_data = (rg_idx == 6'd0)            ? 64'd0 :
                     (w_real && (ag == wa))      ? w_data : (valid[ag] ? mem[ag] : 64'd0);

endmodule
