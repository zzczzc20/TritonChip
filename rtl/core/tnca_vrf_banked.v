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
    parameter NUM_PORTS      = 4    // read requesters presented per cycle
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
    output wire [NUM_PORTS-1:0]              rsp_valid,
    output wire [NUM_PORTS*VLEN*ELEN-1:0]    rsp_data,

    // ---- write port ------------------------------------------------------
    input  wire                              we,
    input  wire [RCTX_W-1:0]                 w_rctx,
    input  wire [7:0]                        w_idx,
    input  wire [VLEN-1:0]                   wmask,
    input  wire [VLEN*ELEN-1:0]              wdata
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

    // ---- write ------------------------------------------------------------
    wire [PHYS_W-1:0] w_phys = phys_of(w_rctx, w_idx);
    wire [BANK_W-1:0] w_bank = w_phys[BANK_W-1:0];
    wire [ROW_W-1:0]  w_row  = w_phys[BANK_W +: ROW_W];

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
            if (req_valid[pi] && !bank_taken[req_bank[pi]]) begin
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
    wire [NUM_BANKS*WIDTH-1:0] bank_rdata;

    genvar b;
    generate
        for (b = 0; b < NUM_BANKS; b = b + 1) begin : g_bank
            tnca_rf_bank #(
                .DEPTH        (BANK_DEPTH),
                .LANES        (VLEN),
                .ELEN         (ELEN),
                .ADDR_W       (ROW_W),
                .READ_LATENCY (READ_LATENCY),
                .RDW_BYPASS   (1)
            ) u_bank (
                .clk    (clk),
                .rst_n  (rst_n),
                .we     (we && (w_bank == b[BANK_W-1:0])),
                .waddr  (w_row),
                .wmask  (wmask),
                .wdata  (wdata),
                .re     (bank_re_c[b]),
                .raddr  (bank_row_c[b]),
                .rvalid (bank_rvalid[b]),
                .rdata  (bank_rdata[b*WIDTH +: WIDTH])
            );
        end
    endgenerate

    // ---- route each bank's response back to the port that asked ----------
    //
    // RF-04: "任一 pending 响应有保留目的地". The destination is registered
    // with the request and travels with it, so a response can neither be
    // lost nor delivered to a port that did not ask.
    localparam PIPE = (READ_LATENCY == 2) ? 2 : 1;

    reg [NUM_BANKS*8-1:0] dest_q [0:PIPE-1];
    integer di;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (di = 0; di < PIPE; di = di + 1)
                dest_q[di] <= {NUM_BANKS*8{1'b0}};
        end else begin
            dest_q[0] <= bank_port_c;
            for (di = 1; di < PIPE; di = di + 1)
                dest_q[di] <= dest_q[di-1];
        end
    end

    reg  [NUM_PORTS-1:0]           rsp_valid_c;
    reg  [NUM_PORTS*WIDTH-1:0]     rsp_data_c;
    integer rb;
    reg [7:0] dst;
    always @* begin
        rsp_valid_c = {NUM_PORTS{1'b0}};
        rsp_data_c  = {NUM_PORTS*WIDTH{1'b0}};
        for (rb = 0; rb < NUM_BANKS; rb = rb + 1) begin
            if (bank_rvalid[rb]) begin
                dst = dest_q[PIPE-1][rb*8 +: 8];
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
