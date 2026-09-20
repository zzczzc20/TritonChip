//==============================================================================
// tnca_acc_rf -- accumulator register file, ACC_N slots x ROWS x VLEN x FP32
//
// Separate from the vector file (arch overview sec 5.1) but reachable from the
// vector unit (sec 5.4), and the second half is not a convenience. Online
// softmax rescales O by exp(m - m') on every key block; if that round trip had
// to go accumulator -> cache -> vector -> cache -> accumulator, a design with
// no scratchpad would have lost the argument for existing.
//
// One row-wide read port and one row-wide write port, each VLEN*32 bits. The
// matrix engine and the a.* instructions share them, which is why the decoder
// puts both on the same functional unit: the contention is real and belongs in
// the issue logic rather than hidden here.
//
// At the reference configuration (16 slots x 16 rows x 16 lanes x 32 bit) this
// is 16 KiB of state. Like the vector file it is flops here and belongs in
// SRAM in silicon; see doc/synthesis.md.
//==============================================================================

module tnca_acc_rf #(
    parameter ACC_N  = 16,
    parameter ACC_W  = 4,
    parameter ROWS   = 16,
    parameter ROW_W  = 4,
    parameter VLEN   = 16,
    parameter ELEN   = 32
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire [ACC_W-1:0]         r_slot,
    input  wire [ROW_W-1:0]         r_row,
    output wire [VLEN*ELEN-1:0]     r_data,

    input  wire                     w_en,
    input  wire [ACC_W-1:0]         w_slot,
    input  wire [ROW_W-1:0]         w_row,
    input  wire [VLEN-1:0]          w_lane_en,
    input  wire [VLEN*ELEN-1:0]     w_data,

    // whole-slot zero, one row per cycle driven by the caller
    input  wire                     z_en,
    input  wire [ACC_W-1:0]         z_slot,
    input  wire [ROW_W-1:0]         z_row
);

    localparam DEPTH = ACC_N * ROWS;

    reg [VLEN*ELEN-1:0] mem [0:DEPTH-1];

    wire [ACC_W+ROW_W-1:0] ra = {r_slot, r_row};
    wire [ACC_W+ROW_W-1:0] wa = {w_slot, w_row};
    wire [ACC_W+ROW_W-1:0] za = {z_slot, z_row};

    integer i, l;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1)
                mem[i] <= {(VLEN*ELEN){1'b0}};
        end else begin
            if (z_en) begin
                mem[za] <= {(VLEN*ELEN){1'b0}};
            end else if (w_en) begin
                for (l = 0; l < VLEN; l = l + 1)
                    if (w_lane_en[l])
                        mem[wa][l*ELEN +: ELEN] <= w_data[l*ELEN +: ELEN];
            end
        end
    end

    // Write-first read, matching the other register files: the matrix engine
    // reads and writes the same row two cycles apart, and a reader that
    // collides with the writer must see the new value.
    wire hit = (w_en && (ra == wa)) || (z_en && (ra == za));

    genvar g;
    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_bypass
            assign r_data[g*ELEN +: ELEN] =
                (z_en && (ra == za))                  ? {ELEN{1'b0}} :
                (w_en && (ra == wa) && w_lane_en[g])  ? w_data[g*ELEN +: ELEN]
                                                      : mem[ra][g*ELEN +: ELEN];
        end
    endgenerate

    wire _unused = &{1'b0, hit, 1'b0};

endmodule
