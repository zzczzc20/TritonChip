//==============================================================================
// tnca_acc_rf -- accumulator register file, ACC_N slots x ROWS x VLEN x FP32
//
// Separate from the vector file (arch overview sec 5.1) but reachable from the
// vector unit (sec 5.4), and the second half is not a convenience. Online
// softmax rescales O by exp(m - m') on every key block; if that round trip had
// to go accumulator -> cache -> vector -> cache -> accumulator, a design with
// no scratchpad would have lost the argument for existing.
//
// One row-wide read port and one row-wide write port, each VLEN*32 bits; a
// read returns the row as it was before this cycle's write (see below). The
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

    // Read-old, with no write-to-read bypass. The bypass this file used to
    // have put the FMA's normalise-and-round, the write port and the next
    // multiply-add's operand selection and alignment into one cycle: it was
    // the worst clk_dev path of the U280 link (60 logic levels, 23.9 ns), and
    // it never carried a value anyone used. The matrix engine reads row m and
    // writes it FMA_LAT cycles later, while it is reading row m+FMA_LAT; the
    // next read of row m is ROWS+1 cycles after the first. A row op and
    // a.vread start only after the previous command's write has landed.
    // tnca_matrix checks that in simulation (`acc_collide`), so a change that
    // made the bypass necessary would fail loudly rather than read stale data.
    //
    // The zero port keeps its bypass: z_en and the row are registers, so it
    // is two levels of logic, not a path through the arithmetic.
    genvar g;
    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_rd
            assign r_data[g*ELEN +: ELEN] =
                (z_en && (ra == za)) ? {ELEN{1'b0}} : mem[ra][g*ELEN +: ELEN];
        end
    endgenerate

endmodule
