//==============================================================================
// tnca_pred_rf -- predicate register file, NUM_RCTX x 16 x VLEN bit
//
// Three read ports: two explicit operands plus the lane-mask port every
// maskable instruction uses. P0 reads as all-ones and discards writes
// (ISA sec 1.1), which is what makes "pred=0, PN=0" the unpredicated encoding.
//==============================================================================
`include "tnca_defs.vh"

module tnca_pred_rf #(
    parameter NUM_RCTX = 2,
    parameter RCTX_W   = 1,
    parameter VLEN     = 16
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire [RCTX_W-1:0]     ra_rctx,
    input  wire [3:0]            ra_idx,
    output wire [VLEN-1:0]       ra_data,

    input  wire [RCTX_W-1:0]     rb_rctx,
    input  wire [3:0]            rb_idx,
    output wire [VLEN-1:0]       rb_data,

    // lane-mask read port
    input  wire [RCTX_W-1:0]     rm_rctx,
    input  wire [3:0]            rm_idx,
    output wire [VLEN-1:0]       rm_data,

    input  wire                  w_en,
    input  wire [RCTX_W-1:0]     w_rctx,
    input  wire [3:0]            w_idx,
    input  wire [VLEN-1:0]       w_data
);

    localparam DEPTH = NUM_RCTX * 16;

    reg [VLEN-1:0] mem [0:DEPTH-1];

    wire [RCTX_W+3:0] wa     = {w_rctx, w_idx};
    wire              w_real = w_en & (w_idx != 4'd0);

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1)
                mem[i] <= {VLEN{1'b0}};
        end else if (w_real) begin
            mem[wa] <= w_data;
        end
    end

    wire [RCTX_W+3:0] aa = {ra_rctx, ra_idx};
    wire [RCTX_W+3:0] ab = {rb_rctx, rb_idx};
    wire [RCTX_W+3:0] am = {rm_rctx, rm_idx};

    assign ra_data = (ra_idx == 4'd0)       ? {VLEN{1'b1}} :
                     (w_real && (aa == wa)) ? w_data : mem[aa];
    assign rb_data = (rb_idx == 4'd0)       ? {VLEN{1'b1}} :
                     (w_real && (ab == wa)) ? w_data : mem[ab];
    assign rm_data = (rm_idx == 4'd0)       ? {VLEN{1'b1}} :
                     (w_real && (am == wa)) ? w_data : mem[am];

endmodule
