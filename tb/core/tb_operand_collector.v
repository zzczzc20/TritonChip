//==============================================================================
// tb_operand_collector -- the collector against the banked register file
//
// doc/phase3_tasks_and_acceptance.md R3-04. Directed cases, each printing what
// happened so tools/r3/flow_tests.py can check it. The bench decides nothing.
//
// +case=<n> picks the scenario:
//   1  three sources in ONE bank: they must arrive one per cycle, in order,
//      and the entry must not be ready until the last one lands (FLOW-01)
//   2  three sources in three banks: all three in one cycle (FLOW-01)
//   3  four entries all wanting the same bank: round-robin, and every entry
//      served within ENTRIES grants (FLOW-02)
//   4  a flush while reads are outstanding: the entry goes, the responses
//      are absorbed as orphans, nothing is written into a reused slot
//      (FLOW-05), and request/response conservation holds (FLOW-04)
//==============================================================================
`timescale 1ns/1ps
`include "tnca_defs.vh"

module tb_operand_collector;

    localparam VLEN = 16, ELEN = 32, WIDTH = VLEN*ELEN;
    localparam ENTRIES = 4, SRCS = 3, ENT_W = 2, NUM_PORTS = 4;
    localparam RCTX_W = 1, PHYS_W = 7, NUM_BANKS = 4, BANK_W = 2;

    // At one cycle a read accepted on edge N returns on N+1 and the
    // collector, which is always ready, consumes it there -- so a flush
    // can never land while a read is outstanding, and the orphan path is
    // unreachable. At two cycles there is a gap, and it is reachable.
    // Both are run.
    parameter RF_LAT = 1;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [1:0]                   vb;
    wire [2*PHYS_W-1:0]          vbase = {7'd64, 7'd0};

    wire [NUM_PORTS-1:0]         req_valid, req_accept, rsp_valid;
    wire [NUM_PORTS*RCTX_W-1:0]  req_rctx;
    wire [NUM_PORTS*8-1:0]       req_idx;
    wire [NUM_PORTS*WIDTH-1:0]   rsp_data;

    reg                          alloc_valid;
    reg  [ENT_W-1:0]             alloc_entry;
    reg  [SRCS-1:0]              alloc_srcs;
    reg  [SRCS*8-1:0]            alloc_idx;
    reg  [RCTX_W-1:0]            alloc_rctx;
    wire [ENTRIES-1:0]           busy, ready;
    reg  [ENTRIES-1:0]           consume;
    reg                          flush;
    wire [31:0] n_wait, n_grant, n_conflict, n_orphan;

    reg                          we;
    reg  [RCTX_W-1:0]            w_rctx;
    reg  [7:0]                   w_idx;
    reg  [VLEN-1:0]              wmask;
    reg  [WIDTH-1:0]             wdata;

    tnca_vrf_banked #(
        .PHYS_VREGS(128), .VREGS_PER_RCTX(64), .NUM_RCTX(2), .VLEN(VLEN),
        .ELEN(ELEN), .RCTX_W(RCTX_W), .PHYS_W(PHYS_W),
        .NUM_BANKS(NUM_BANKS), .BANK_W(BANK_W), .READ_LATENCY(RF_LAT),
        .NUM_PORTS(NUM_PORTS)
    ) u_rf (
        .clk(clk), .rst_n(rst_n), .vbase(vbase),
        .req_valid(req_valid), .req_rctx(req_rctx), .req_idx(req_idx),
        .req_accept(req_accept), .rsp_ready({NUM_PORTS{1'b1}}),
        .rsp_valid(rsp_valid), .rsp_data(rsp_data),
        .we(we), .w_rctx(w_rctx), .w_idx(w_idx), .wmask(wmask),
        .wdata(wdata), .w_accept()
    );

    tnca_operand_collector #(
        .ENTRIES(ENTRIES), .SRCS(SRCS), .VLEN(VLEN), .ELEN(ELEN),
        .RCTX_W(RCTX_W), .ENT_W(ENT_W), .NUM_PORTS(NUM_PORTS)
    ) u_oc (
        .clk(clk), .rst_n(rst_n),
        .alloc_valid(alloc_valid), .alloc_entry(alloc_entry),
        .alloc_srcs(alloc_srcs), .alloc_idx(alloc_idx),
        .alloc_rctx(alloc_rctx), .busy(busy),
        .req_valid(req_valid), .req_rctx(req_rctx), .req_idx(req_idx),
        .req_accept(req_accept),
        .rsp_valid(rsp_valid), .rsp_data(rsp_data),
        .ready(ready), .consume(consume), .flush(flush),
        .n_wait(n_wait), .n_grant(n_grant), .n_conflict(n_conflict),
        .n_orphan(n_orphan)
    );

    integer which, cyc, i;
    reg [1023:0] dummy;

    task write_reg;
        input [7:0] idx;
        input integer word;
        integer L;
        begin
            @(negedge clk);
            we = 1; w_rctx = 0; w_idx = idx; wmask = {VLEN{1'b1}};
            for (L = 0; L < VLEN; L = L + 1)
                wdata[L*ELEN +: ELEN] = word + L;
            @(negedge clk);
            we = 0;
        end
    endtask

    task step;
        begin
            @(negedge clk);
            $display("CYC %0d busy=%b ready=%b grant=%0d conflict=%0d wait=%0d orphan=%0d",
                     cyc, busy, ready, n_grant, n_conflict, n_wait, n_orphan);
            cyc = cyc + 1;
        end
    endtask

    initial begin
        if (!$value$plusargs("case=%d", which)) which = 1;
        alloc_valid = 0; consume = 0; flush = 0; we = 0; cyc = 0;
        alloc_rctx = 0; alloc_idx = 0; alloc_srcs = 0; alloc_entry = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;

        // Fill the registers the cases read, with distinguishable values.
        for (i = 0; i < 16; i = i + 1)
            write_reg(i[7:0], 32'h1000 + i*32'h100);

        $display("CASE %0d", which);

        if (which == 1) begin
            // v0, v4, v8 -- all bank 0 with base 0, so one per cycle
            @(negedge clk);
            alloc_valid = 1; alloc_entry = 0; alloc_srcs = 3'b111;
            alloc_idx = {8'd8, 8'd4, 8'd0};
            @(negedge clk);
            alloc_valid = 0;
            for (i = 0; i < 10; i = i + 1) step;
            $display("OPS %0d %h %h %h", 0,
                     u_oc.e_data[0][0*WIDTH +: WIDTH],
                     u_oc.e_data[0][1*WIDTH +: WIDTH],
                     u_oc.e_data[0][2*WIDTH +: WIDTH]);
        end else if (which == 2) begin
            // v0, v1, v2 -- three different banks, all in one cycle
            @(negedge clk);
            alloc_valid = 1; alloc_entry = 0; alloc_srcs = 3'b111;
            alloc_idx = {8'd2, 8'd1, 8'd0};
            @(negedge clk);
            alloc_valid = 0;
            for (i = 0; i < 8; i = i + 1) step;
            $display("OPS %0d %h %h %h", 0,
                     u_oc.e_data[0][0*WIDTH +: WIDTH],
                     u_oc.e_data[0][1*WIDTH +: WIDTH],
                     u_oc.e_data[0][2*WIDTH +: WIDTH]);
        end else if (which == 3) begin
            // Four entries, each wanting THREE sources, all in bank 0
            // (v0, v4, v8 are phys 0, 4, 8 and 4 divides all of them).
            // Twelve reads competing for one read port is real contention;
            // the first version of this case gave each entry one source,
            // and one allocation per cycle exactly matched one grant per
            // cycle, so nothing ever queued and the round robin was never
            // asked to decide anything.
            @(negedge clk);
            // The allocation cycles are printed too. They were not, and
            // the first printed cycle then carried every grant issued
            // while the entries were being allocated -- which read as
            // four grants in one cycle on a bank with one read port.
            for (i = 0; i < ENTRIES; i = i + 1) begin
                alloc_valid = 1; alloc_entry = i[ENT_W-1:0];
                alloc_srcs = 3'b111;
                alloc_idx = {8'd8, 8'd4, 8'd0};
                step;
            end
            alloc_valid = 0;
            for (i = 0; i < 20; i = i + 1) step;
        end else if (which == 4) begin
            @(negedge clk);
            alloc_valid = 1; alloc_entry = 0; alloc_srcs = 3'b111;
            alloc_idx = {8'd8, 8'd4, 8'd0};
            @(negedge clk);
            alloc_valid = 0;
            step;
            flush = 1;
            $display("FLUSH %0d", cyc);
            @(negedge clk);
            flush = 0;
            // Reuse entry 0 immediately: an absorbed response must not
            // land in it.
            alloc_valid = 1; alloc_entry = 0; alloc_srcs = 3'b001;
            alloc_idx = {8'd0, 8'd0, 8'd12};
            @(negedge clk);
            alloc_valid = 0;
            for (i = 0; i < 12; i = i + 1) step;
            $display("REUSED %h", u_oc.e_data[0][0*WIDTH +: WIDTH]);
        end

        $display("FINAL grant=%0d conflict=%0d wait=%0d orphan=%0d",
                 n_grant, n_conflict, n_wait, n_orphan);
        $display("DONE");
        $finish;
    end

    initial begin
        #500000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
