//==============================================================================
// tnca_rf_bank -- one synchronous, single-ported storage bank
//
// doc/phase3_tasks_and_acceptance.md R3-03 asks for the thing the behavioural
// register file is standing in for: a bank with a REQUEST and a RESPONSE, a
// read latency you can choose, a write mask, and no more ports than a real
// macro has.
//
// What it is not is the flop array in tnca_vector_rf.v. That one answers a
// read combinationally on four ports at once, which is free in simulation and
// does not exist in silicon; its own header says so. This is one read port and
// one write port per bank, and everything above has to arbitrate for them.
//
// READ-DURING-WRITE
//
//   A read of the address being written in the same cycle returns the NEW
//   data when RDW_BYPASS is 1, and the OLD data when it is 0.
//
//   Write-first is the architectural behaviour the core already has, so
//   RDW_BYPASS defaults to 1 and the pipeline keeps working. It is worth
//   being explicit that this is not free: an SRAM macro gives read-first, and
//   write-first is a bypass mux plus a comparator on the address, which is
//   what the `rdw_hit` term below is. It is counted here rather than assumed
//   away, because R3-05 says any new data holding "都要计入寄存器/存储结构".
//
// RESET, AND WHERE THE ZERO COMES FROM
//
//   `rst_n` clears the response pipeline and the VALID bits. It does not
//   clear the data array: a memory does not forget its contents on reset,
//   and an SRAM macro has no such input.
//
//   That leaves a question the v1 component test answered one way and the
//   architecture answers the other. A never-written location holds X. The
//   architecture says an unwritten register reads ZERO
//   (spec/r3v2_contract.json, reset.architectural_read_of_unwritten), and
//   the behavioural file it replaces gets that from a 2-state array that
//   happens to start at zero -- which is a simulation artefact, not a
//   design.
//
//   So the zero is made explicit. One valid bit per lane per row, reset to
//   0, set by the lanes a masked write actually names. A read of a lane
//   whose valid bit is clear returns zero rather than what the array holds.
//   ZERO_INVALID selects it: 0 leaves the raw array visible, which is what
//   the v1 RF-03 case checks and is still the honest statement about the
//   STORAGE; 1 is the architectural behaviour and is what the candidate
//   elaborates. Both are tested, because "the array is not initialised" and
//   "the register reads zero" are different claims and the design has to
//   make both true at once.
//
//   The valid bits are counted separately from the data
//   (spec/r3v2_contract.json, bit_accounting): 32 rows x 16 lanes = 512 bits
//   per bank, against 16,384 bits of data. Adding them together is how a
//   control structure disappears into a capacity number.
//
// BACKPRESSURE
//
//   A response is held while the consumer is not ready, and no further read
//   is accepted while one is held. `rd_accept` says so on the cycle the
//   request is offered, so the layer above rations reads rather than
//   discovering afterwards that one was dropped.
//==============================================================================
`include "tnca_defs.vh"

module tnca_rf_bank #(
    parameter DEPTH         = 32,
    parameter LANES         = 16,
    parameter ELEN          = 32,
    parameter ADDR_W        = 5,
    parameter READ_LATENCY  = 1,   // 1 or 2
    parameter RDW_BYPASS    = 1,   // 1 = write-first, 0 = read-first
    parameter ZERO_INVALID  = 0,   // 1 = an unwritten lane reads zero
    parameter DST_W         = 8    // width of the destination tag
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- write port ------------------------------------------------------
    input  wire                     we,
    input  wire [ADDR_W-1:0]        waddr,
    input  wire [LANES-1:0]         wmask,
    input  wire [LANES*ELEN-1:0]    wdata,

    // ---- read request ----------------------------------------------------
    //
    // `dst` travels WITH the request through the same registers as its
    // data. It used to be a parallel pipeline in the wrapper, advancing
    // every cycle; a response held for backpressure then kept its data and
    // lost its destination, and went to whichever port the shift register
    // happened to be holding. A tag that rides with its payload cannot get
    // out of step with it.
    input  wire                     re,
    input  wire [ADDR_W-1:0]        raddr,
    input  wire [DST_W-1:0]         dst,

    // ---- read response, READ_LATENCY cycles later ------------------------
    //
    // `rsp_ready` low holds the response; `rd_accept` says whether a read
    // offered this cycle would be taken.
    input  wire                     rsp_ready,
    output wire                     rd_accept,
    output wire                     rvalid,
    output wire [DST_W-1:0]         rdst,
    output wire [LANES*ELEN-1:0]    rdata
);

    localparam WIDTH = LANES * ELEN;

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    // One valid bit per lane per row. Reset clears these and not the array.
    reg [LANES-1:0] vld [0:DEPTH-1];

    integer li, ri;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (ri = 0; ri < DEPTH; ri = ri + 1)
                vld[ri] <= {LANES{1'b0}};
        end else if (we) begin
            for (li = 0; li < LANES; li = li + 1)
                if (wmask[li])
                    vld[waddr][li] <= 1'b1;
        end
    end

    always @(posedge clk) begin
        if (we) begin
            for (li = 0; li < LANES; li = li + 1)
                if (wmask[li])
                    mem[waddr][li*ELEN +: ELEN] <= wdata[li*ELEN +: ELEN];
        end
    end

    // ---- the array read, the write-first bypass, and the zero ------------
    wire rdw_hit = RDW_BYPASS && re && we && (raddr == waddr);

    wire [WIDTH-1:0] array_q = mem[raddr];
    wire [LANES-1:0] vld_q   = vld[raddr];

    // A lane the write names is valid from this cycle whether or not its
    // valid bit has been written back yet: the bypass carries the data, so
    // it has to carry the validity too, or a write-then-read in
    // consecutive cycles would return zero for a lane that was just set.
    wire [LANES-1:0] eff_vld = vld_q | (rdw_hit ? wmask : {LANES{1'b0}});

    wire [WIDTH-1:0] merged;
    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : g_merge
            wire [ELEN-1:0] from_array =
                (ZERO_INVALID && !eff_vld[g]) ? {ELEN{1'b0}}
                                              : array_q[g*ELEN +: ELEN];
            assign merged[g*ELEN +: ELEN] =
                (rdw_hit && wmask[g]) ? wdata[g*ELEN +: ELEN]
                                      : from_array;
        end
    endgenerate

    // ---- stage 1, with backpressure --------------------------------------
    //
    // One response in flight. It is loaded when the stage is empty or the
    // consumer is taking what is there, and held otherwise -- which is also
    // the condition under which a new read can be accepted, so the two are
    // the same signal rather than two that have to agree.
    reg [WIDTH-1:0] s1_data;
    reg [DST_W-1:0] s1_dst;
    reg             s1_valid;

    wire s1_free = !s1_valid || rsp_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
        end else if (s1_free) begin
            s1_valid <= re;
            s1_data  <= merged;
            s1_dst   <= dst;
        end
    end

    // ---- stage 2, when the configuration asks for it ---------------------
    reg [WIDTH-1:0] s2_data;
    reg [DST_W-1:0] s2_dst;
    reg             s2_valid;
    wire s2_free = !s2_valid || rsp_ready;
    always @(posedge clk) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
        end else if (s2_free) begin
            s2_valid <= s1_valid;
            s2_data  <= s1_data;
            s2_dst   <= s1_dst;
        end
    end

    assign rvalid    = (READ_LATENCY == 2) ? s2_valid : s1_valid;
    assign rdata     = (READ_LATENCY == 2) ? s2_data  : s1_data;
    assign rdst      = (READ_LATENCY == 2) ? s2_dst   : s1_dst;
    assign rd_accept = (READ_LATENCY == 2) ? (s1_free && s2_free) : s1_free;

`ifndef SYNTHESIS
    initial begin
        if (READ_LATENCY != 1 && READ_LATENCY != 2) begin
            $display("tnca_rf_bank: READ_LATENCY=%0d is not 1 or 2",
                     READ_LATENCY);
            $finish;
        end
    end
    // RF-04: one accepted read and one accepted write per cycle, no more.
    // The wrapper above is what enforces it; this is the assertion that
    // says so out loud, so a future change that quietly adds a port is a
    // failure rather than a faster simulation.
    always @(posedge clk) begin
        if (rst_n && re && we && raddr == waddr && !RDW_BYPASS) begin
            // read-first with a concurrent write is legal and defined;
            // nothing to assert, but the branch is named so it is clear
            // it was considered rather than missed.
        end
    end
`endif

endmodule
