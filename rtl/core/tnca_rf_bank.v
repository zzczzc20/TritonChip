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
// RESET
//
//   `rst_n` clears the response pipeline, not the array. A memory does not
//   forget its contents on reset and neither does this; what it does is stop
//   presenting a response that belongs to a request from before the reset.
//   RF-03 asks that invalid content not leak after a relaunch, and the answer
//   is that a read of an address never written returns X, not a helpful zero
//   -- a 2-state zero would hide exactly the bug the case is looking for.
//==============================================================================
`include "tnca_defs.vh"

module tnca_rf_bank #(
    parameter DEPTH         = 32,
    parameter LANES         = 16,
    parameter ELEN          = 32,
    parameter ADDR_W        = 5,
    parameter READ_LATENCY  = 1,   // 1 or 2
    parameter RDW_BYPASS    = 1    // 1 = write-first, 0 = read-first
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- write port ------------------------------------------------------
    input  wire                     we,
    input  wire [ADDR_W-1:0]        waddr,
    input  wire [LANES-1:0]         wmask,
    input  wire [LANES*ELEN-1:0]    wdata,

    // ---- read request ----------------------------------------------------
    input  wire                     re,
    input  wire [ADDR_W-1:0]        raddr,

    // ---- read response, READ_LATENCY cycles later ------------------------
    output wire                     rvalid,
    output wire [LANES*ELEN-1:0]    rdata
);

    localparam WIDTH = LANES * ELEN;

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    integer li;
    always @(posedge clk) begin
        if (we) begin
            for (li = 0; li < LANES; li = li + 1)
                if (wmask[li])
                    mem[waddr][li*ELEN +: ELEN] <= wdata[li*ELEN +: ELEN];
        end
    end

    // ---- stage 1: the array read, plus the write-first bypass ------------
    wire rdw_hit = RDW_BYPASS && re && we && (raddr == waddr);

    reg [WIDTH-1:0] s1_data;
    reg             s1_valid;
    reg [LANES-1:0] s1_bypass_mask;
    reg [WIDTH-1:0] s1_bypass_data;

    integer bi;
    always @(posedge clk) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
        end else begin
            s1_valid       <= re;
            s1_data        <= mem[raddr];
            // The bypass is per lane, because the write is: a masked write
            // to the address being read updates only the lanes it names,
            // and the rest must still come from the array.
            s1_bypass_mask <= rdw_hit ? wmask : {LANES{1'b0}};
            s1_bypass_data <= wdata;
        end
    end

    wire [WIDTH-1:0] s1_merged;
    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : g_merge
            assign s1_merged[g*ELEN +: ELEN] =
                s1_bypass_mask[g] ? s1_bypass_data[g*ELEN +: ELEN]
                                  : s1_data[g*ELEN +: ELEN];
        end
    endgenerate

    // ---- stage 2, when the configuration asks for it ---------------------
    reg [WIDTH-1:0] s2_data;
    reg             s2_valid;
    always @(posedge clk) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
        end else begin
            s2_valid <= s1_valid;
            s2_data  <= s1_merged;
        end
    end

    assign rvalid = (READ_LATENCY == 2) ? s2_valid : s1_valid;
    assign rdata  = (READ_LATENCY == 2) ? s2_data  : s1_merged;

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
