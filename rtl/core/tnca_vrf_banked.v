//==============================================================================
// tnca_vrf_banked -- the vector register file as banks, with real ports
//
// doc/phase3_tasks_and_acceptance.md R3-03. Same architectural file as
// tnca_vector_rf: PHYS_VREGS physical registers of VLEN*ELEN bits, with each
// RCTX naming VREGS_PER_RCTX of them through a programmable base, so two roles
// with equal bases alias completely and disjoint bases do not (arch overview
// sec 5.1).
//
// RF-01 asks that the banks sum to exactly the declared capacity and that the
// address mapping be invertible. Both fall out of the mapping chosen here:
//
//     bank  = phys % NUM_BANKS           row = phys / NUM_BANKS
//     phys  = row * NUM_BANKS + bank
//
// Low-order interleaving, so consecutive registers land in different banks --
// which is the case that matters, because a three-operand instruction reads
// v0, v1, v2 and those should not queue behind each other. The inverse is
// written out above because "invertible" is a property somebody has to be
// able to check, and it is checked in tb_rf_bank.v.
//
// Capacity: NUM_BANKS * BANK_DEPTH registers of VLEN*ELEN bits. At the
// reference configuration (128 x 16 x 32) that is 8 KiB, which is what the
// contract names. The elaboration below refuses a configuration where the
// banks do not sum to PHYS_VREGS, rather than quietly addressing part of the
// file.
//
// PORTS
//
// Each bank has one read and one write port (tnca_rf_bank). This module does
// NOT arbitrate: it accepts one read request per bank per cycle and reports
// which requests it took. Arbitration, the operand collector and the fairness
// policy are R3-04, and putting them here would make this module the thing it
// is meant to replace.
//==============================================================================
`include "tnca_defs.vh"

module tnca_vrf_banked #(
    parameter PHYS_VREGS     = 128,
    parameter VREGS_PER_RCTX = 64,
    parameter NUM_RCTX       = 2,
    parameter VLEN           = 16,
    parameter ELEN           = 32,
    parameter RCTX_W         = 1,
    parameter PHYS_W         = 7,
    parameter NUM_BANKS      = 4,
    parameter BANK_W         = 2,   // $clog2(NUM_BANKS)
    parameter READ_LATENCY   = 1,
    parameter NUM_PORTS      = 4,   // read requesters presented per cycle
    parameter ZERO_INVALID   = 0,   // 1 = an unwritten lane reads zero
    parameter NUM_WRITERS    = 1    // how many write clients contend
) (
    input  wire                              clk,
    input  wire                              rst_n,

    input  wire [NUM_RCTX*PHYS_W-1:0]        vbase,

    // ---- read requests, one per requester --------------------------------
    input  wire [NUM_PORTS-1:0]              req_valid,
    input  wire [NUM_PORTS*RCTX_W-1:0]       req_rctx,
    input  wire [NUM_PORTS*8-1:0]            req_idx,
    output wire [NUM_PORTS-1:0]              req_accept,

    // ---- read responses, READ_LATENCY cycles after acceptance ------------
    //
    // `rsp_ready` is per requester. A bank whose response is being held
    // accepts no further read, and req_accept says so on the cycle the
    // request is offered.
    input  wire [NUM_PORTS-1:0]              rsp_ready,
    output wire [NUM_PORTS-1:0]              rsp_valid,
    output wire [NUM_PORTS*VLEN*ELEN-1:0]    rsp_data,

    // ---- write ports -----------------------------------------------------
    //
    // Two writers, because the core has two: the writeback stage and the
    // transfer engine. Each bank still has ONE write port, so two writes
    // to the same bank in one cycle cannot both land -- `w_accept` says
    // which one did, and the loser has to hold its payload and retry.
    // Silently dropping it is the failure R3V2-FLOW-01 is about.
    input  wire [NUM_WRITERS-1:0]            we,
    input  wire [NUM_WRITERS*RCTX_W-1:0]     w_rctx,
    input  wire [NUM_WRITERS*8-1:0]          w_idx,
    input  wire [NUM_WRITERS*VLEN-1:0]       wmask,
    input  wire [NUM_WRITERS*VLEN*ELEN-1:0]  wdata,
    output wire [NUM_WRITERS-1:0]            w_accept
);

    localparam BANK_DEPTH = PHYS_VREGS / NUM_BANKS;
    localparam ROW_W      = (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH);
    localparam WIDTH      = VLEN * ELEN;

    // ---- physical address, the one place the base is applied -------------
    function [PHYS_W-1:0] phys_of;
        input [RCTX_W-1:0] rctx;
        input [7:0]        idx;
        reg   [PHYS_W-1:0] base;
        begin
            base    = vbase[rctx*PHYS_W +: PHYS_W];
            phys_of = base + idx[PHYS_W-1:0];
        end
    endfunction

    // ---- writes: one per bank, lowest writer index first -----------------
    //
    // The policy is fixed rather than fair on purpose: writer 0 is the
    // writeback stage, whose stall reaches the whole pipeline, and writer
    // 1 is the transfer engine, which already has a holding register for
    // its payload. R3V2-FLOW-02's fairness bound is about READ requesters;
    // this is a two-way conflict with a declared winner.
    wire [PHYS_W-1:0] w_phys_of [0:NUM_WRITERS-1];
    wire [BANK_W-1:0] w_bank_of [0:NUM_WRITERS-1];
    wire [ROW_W-1:0]  w_row_of  [0:NUM_WRITERS-1];

    genvar wq;
    generate
        for (wq = 0; wq < NUM_WRITERS; wq = wq + 1) begin : g_wdec
            assign w_phys_of[wq] =
                phys_of(w_rctx[wq*RCTX_W +: RCTX_W], w_idx[wq*8 +: 8]);
            assign w_bank_of[wq] = w_phys_of[wq][BANK_W-1:0];
            assign w_row_of[wq]  = w_phys_of[wq][BANK_W +: ROW_W];
        end
    endgenerate

    reg [NUM_WRITERS-1:0] w_accept_c;
    reg [NUM_BANKS-1:0]   w_bank_busy;
    reg [NUM_BANKS-1:0]   bank_we_c;
    reg [ROW_W-1:0]       bank_wrow_c  [0:NUM_BANKS-1];
    reg [VLEN-1:0]        bank_wmask_c [0:NUM_BANKS-1];
    reg [WIDTH-1:0]       bank_wdata_c [0:NUM_BANKS-1];

    integer wi2, wb;
    always @* begin
        w_accept_c  = {NUM_WRITERS{1'b0}};
        w_bank_busy = {NUM_BANKS{1'b0}};
        bank_we_c   = {NUM_BANKS{1'b0}};
        for (wb = 0; wb < NUM_BANKS; wb = wb + 1) begin
            bank_wrow_c[wb]  = {ROW_W{1'b0}};
            bank_wmask_c[wb] = {VLEN{1'b0}};
            bank_wdata_c[wb] = {WIDTH{1'b0}};
        end
        for (wi2 = 0; wi2 < NUM_WRITERS; wi2 = wi2 + 1) begin
            if (we[wi2] && !w_bank_busy[w_bank_of[wi2]]) begin
                w_accept_c[wi2]                  = 1'b1;
                w_bank_busy[w_bank_of[wi2]]      = 1'b1;
                bank_we_c[w_bank_of[wi2]]        = 1'b1;
                bank_wrow_c[w_bank_of[wi2]]      = w_row_of[wi2];
                bank_wmask_c[w_bank_of[wi2]]     =
                    wmask[wi2*VLEN +: VLEN];
                bank_wdata_c[w_bank_of[wi2]]     =
                    wdata[wi2*WIDTH +: WIDTH];
            end
        end
    end

    assign w_accept = w_accept_c;

    // ---- read requests: at most one per bank per cycle --------------------
    //
    // Two requesters that want the same bank in the same cycle are a
    // conflict, and exactly one is taken. Lower index wins, which is a
    // policy, not fairness -- R3-04 owns fairness, and this module says
    // plainly which request it took so that layer can count the rest as
    // waiting rather than as lost.
    wire [PHYS_W-1:0] req_phys [0:NUM_PORTS-1];
    wire [BANK_W-1:0] req_bank [0:NUM_PORTS-1];
    wire [ROW_W-1:0]  req_row  [0:NUM_PORTS-1];

    genvar p;
    generate
        for (p = 0; p < NUM_PORTS; p = p + 1) begin : g_decode
            assign req_phys[p] = phys_of(req_rctx[p*RCTX_W +: RCTX_W],
                                         req_idx[p*8 +: 8]);
            assign req_bank[p] = req_phys[p][BANK_W-1:0];
            assign req_row[p]  = req_phys[p][BANK_W +: ROW_W];
        end
    endgenerate

    // A port with a response nobody has taken yet must not be given a
    // second one. Without this it can have two reads outstanding in two
    // banks, both come back, and one lands on top of the other -- a silent
    // overwrite, which is the thing the contract names when it asks for
    // every pending response to have a reserved destination.
    reg  [NUM_PORTS-1:0] port_busy;
    integer pb;
    reg [7:0] pbd;
    always @* begin
        port_busy = {NUM_PORTS{1'b0}};
        for (pb = 0; pb < NUM_BANKS; pb = pb + 1) begin
            pbd = bank_rdst[pb*8 +: 8];
            if (bank_rvalid[pb] && pbd < NUM_PORTS
                    && !rsp_ready[pbd])
                port_busy[pbd] = 1'b1;
        end
    end

    reg  [NUM_PORTS-1:0] accept_c;
    reg  [NUM_BANKS-1:0] bank_taken;
    reg  [ROW_W-1:0]     bank_row_c  [0:NUM_BANKS-1];
    reg  [NUM_BANKS-1:0] bank_re_c;
    // which port each bank is answering, so the response can be routed back
    reg  [NUM_BANKS*8-1:0] bank_port_c;

    integer pi, bk;
    always @* begin
        accept_c    = {NUM_PORTS{1'b0}};
        bank_taken  = {NUM_BANKS{1'b0}};
        bank_re_c   = {NUM_BANKS{1'b0}};
        bank_port_c = {NUM_BANKS*8{1'b0}};
        for (bk = 0; bk < NUM_BANKS; bk = bk + 1)
            bank_row_c[bk] = {ROW_W{1'b0}};
        for (pi = 0; pi < NUM_PORTS; pi = pi + 1) begin
            // A bank holding a response takes no new read. Refusing here
            // rather than issuing and dropping is what makes `req_accept`
            // mean what it says.
            if (req_valid[pi] && !bank_taken[req_bank[pi]]
                    && bank_accept[req_bank[pi]]
                    && !port_busy[pi]) begin
                accept_c[pi]                 = 1'b1;
                bank_taken[req_bank[pi]]     = 1'b1;
                bank_re_c[req_bank[pi]]      = 1'b1;
                bank_row_c[req_bank[pi]]     = req_row[pi];
                bank_port_c[req_bank[pi]*8 +: 8] = pi[7:0];
            end
        end
    end

    assign req_accept = accept_c;

    // ---- the banks --------------------------------------------------------
    wire [NUM_BANKS-1:0]       bank_rvalid;
    wire [NUM_BANKS-1:0]       bank_accept;
    wire [NUM_BANKS*8-1:0]     bank_rdst;
    wire [NUM_BANKS*WIDTH-1:0] bank_rdata;

    // A bank's ready comes from the requester its response is destined
    // for, which is the port recorded when the read was accepted. That is
    // the same reservation R3V2-RF-01 asks about: a response has one
    // destination, fixed at acceptance, and the backpressure follows it.
    reg [NUM_BANKS-1:0] bank_ready_c;
    integer rq;
    reg [7:0] rdst;
    always @* begin
        bank_ready_c = {NUM_BANKS{1'b1}};
        for (rq = 0; rq < NUM_BANKS; rq = rq + 1) begin
            rdst = bank_rdst[rq*8 +: 8];
            bank_ready_c[rq] = (bank_rvalid[rq] && rdst < NUM_PORTS)
                               ? rsp_ready[rdst] : 1'b1;
        end
    end
    wire [NUM_BANKS-1:0] bank_ready = bank_ready_c;

    genvar b;
    generate
        for (b = 0; b < NUM_BANKS; b = b + 1) begin : g_bank
            tnca_rf_bank #(
                .DEPTH        (BANK_DEPTH),
                .LANES        (VLEN),
                .ELEN         (ELEN),
                .ADDR_W       (ROW_W),
                .READ_LATENCY (READ_LATENCY),
                .RDW_BYPASS   (1),
                .ZERO_INVALID (ZERO_INVALID),
                .DST_W        (8)
            ) u_bank (
                .clk    (clk),
                .rst_n  (rst_n),
                .we     (bank_we_c[b]),
                .waddr  (bank_wrow_c[b]),
                .wmask  (bank_wmask_c[b]),
                .wdata  (bank_wdata_c[b]),
                .re        (bank_re_c[b]),
                .raddr     (bank_row_c[b]),
                .dst       (bank_port_c[b*8 +: 8]),
                .rsp_ready (bank_ready[b]),
                .rd_accept (bank_accept[b]),
                .rvalid    (bank_rvalid[b]),
                .rdst      (bank_rdst[b*8 +: 8]),
                .rdata     (bank_rdata[b*WIDTH +: WIDTH])
            );
        end
    endgenerate

    // ---- route each bank's response back to the port that asked ----------
    //
    // RF-04: "任一 pending 响应有保留目的地". The destination is registered
    // with the request and travels with it, so a response can neither be
    // lost nor delivered to a port that did not ask.
    // The destination used to be a shift register here, advancing every
    // cycle in parallel with the data. It now rides inside the bank with
    // the response it belongs to, so there is nothing left to keep in
    // step -- which is the only reason the two could ever disagree.

    reg  [NUM_PORTS-1:0]           rsp_valid_c;
    reg  [NUM_PORTS*WIDTH-1:0]     rsp_data_c;
    integer rb;
    reg [7:0] dst;
    always @* begin
        rsp_valid_c = {NUM_PORTS{1'b0}};
        rsp_data_c  = {NUM_PORTS*WIDTH{1'b0}};
        for (rb = 0; rb < NUM_BANKS; rb = rb + 1) begin
            if (bank_rvalid[rb]) begin
                dst = bank_rdst[rb*8 +: 8];
                if (dst < NUM_PORTS) begin
                    rsp_valid_c[dst]                = 1'b1;
                    rsp_data_c[dst*WIDTH +: WIDTH]  =
                        bank_rdata[rb*WIDTH +: WIDTH];
                end
            end
        end
    end

    assign rsp_valid = rsp_valid_c;
    assign rsp_data  = rsp_data_c;

`ifndef SYNTHESIS
    initial begin
        if (NUM_BANKS * BANK_DEPTH != PHYS_VREGS) begin
            $display("tnca_vrf_banked: %0d banks x %0d = %0d, not PHYS_VREGS=%0d",
                     NUM_BANKS, BANK_DEPTH, NUM_BANKS*BANK_DEPTH, PHYS_VREGS);
            $finish;
        end
        if ((1 << BANK_W) != NUM_BANKS) begin
            $display("tnca_vrf_banked: BANK_W=%0d does not describe %0d banks",
                     BANK_W, NUM_BANKS);
            $finish;
        end
    end
`endif

endmodule
