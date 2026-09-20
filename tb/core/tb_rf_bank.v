//==============================================================================
// tb_rf_bank -- replay a stimulus file through tnca_vrf_banked and dump what
// came back, so an independent model can check it.
//
// doc/phase3_tasks_and_acceptance.md R3-03. The bench decides nothing: it
// prints one line per accepted request and one per response, and
// tools/r3/rf_tests.py compares those against a model that has never seen
// this file. A bench that computed the expected value itself would be
// comparing the design against a second copy of the design.
//
// Stimulus format, one line per cycle:
//
//   <we> <w_rctx> <w_idx> <wmask_hex> <wdata_word> \
//   <p0_valid> <p0_rctx> <p0_idx> ... <p3_valid> <p3_rctx> <p3_idx>
//
// wdata is one 32-bit word, splatted across the lanes with the lane index
// added, so every write is distinguishable and a lane that took the wrong
// neighbour's data shows up rather than matching by luck.
//==============================================================================
`timescale 1ns/1ps
`include "tnca_defs.vh"

module tb_rf_bank;

    localparam PHYS_VREGS = 128;
    localparam VREGS_PER  = 64;
    localparam NUM_RCTX   = 2;
    localparam VLEN       = 16;
    localparam ELEN       = 32;
    localparam RCTX_W     = 1;
    localparam PHYS_W     = 7;
    localparam NUM_BANKS  = 4;
    localparam BANK_W     = 2;
    localparam NUM_PORTS  = 4;
    localparam WIDTH      = VLEN * ELEN;

    integer READ_LATENCY;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [NUM_RCTX*PHYS_W-1:0] vbase;

    reg                         we;
    reg  [RCTX_W-1:0]           w_rctx;
    reg  [7:0]                  w_idx;
    reg  [VLEN-1:0]             wmask;
    reg  [WIDTH-1:0]            wdata;

    reg  [NUM_PORTS-1:0]        req_valid;
    reg  [NUM_PORTS*RCTX_W-1:0] req_rctx;
    reg  [NUM_PORTS*8-1:0]      req_idx;
    wire [NUM_PORTS-1:0]        req_accept;
    wire [NUM_PORTS-1:0]        rsp_valid;
    wire [NUM_PORTS*WIDTH-1:0]  rsp_data;

    // Two instances, one per read latency, so a single run covers both and
    // the comparison cannot be run against whichever happened to be built.
    tnca_vrf_banked #(
        .PHYS_VREGS(PHYS_VREGS), .VREGS_PER_RCTX(VREGS_PER),
        .NUM_RCTX(NUM_RCTX), .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W),
        .PHYS_W(PHYS_W), .NUM_BANKS(NUM_BANKS), .BANK_W(BANK_W),
        .READ_LATENCY(1), .NUM_PORTS(NUM_PORTS)
    ) u_l1 (
        .clk(clk), .rst_n(rst_n), .vbase(vbase),
        .req_valid(req_valid), .req_rctx(req_rctx), .req_idx(req_idx),
        .req_accept(req_accept),
        // Always ready: this bench predates the handshake and its cases are
        // about the storage, not about backpressure. Tying it high keeps
        // the v1 semantics exactly as they were.
        .rsp_ready({NUM_PORTS{1'b1}}),
        .rsp_valid(rsp_valid), .rsp_data(rsp_data),
        .we(we), .w_rctx(w_rctx), .w_idx(w_idx), .wmask(wmask), .wdata(wdata)
    );

    wire [NUM_PORTS-1:0]       a2, v2;
    wire [NUM_PORTS*WIDTH-1:0] d2;
    tnca_vrf_banked #(
        .PHYS_VREGS(PHYS_VREGS), .VREGS_PER_RCTX(VREGS_PER),
        .NUM_RCTX(NUM_RCTX), .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W),
        .PHYS_W(PHYS_W), .NUM_BANKS(NUM_BANKS), .BANK_W(BANK_W),
        .READ_LATENCY(2), .NUM_PORTS(NUM_PORTS)
    ) u_l2 (
        .clk(clk), .rst_n(rst_n), .vbase(vbase),
        .req_valid(req_valid), .req_rctx(req_rctx), .req_idx(req_idx),
        .req_accept(a2), .rsp_ready({NUM_PORTS{1'b1}}),
        .rsp_valid(v2), .rsp_data(d2),
        .we(we), .w_rctx(w_rctx), .w_idx(w_idx), .wmask(wmask), .wdata(wdata)
    );

    integer fh, n, cyc, i, rc;
    integer f_we, f_wrctx, f_widx, f_wword, f_base0, f_base1;
    reg [VLEN-1:0] f_wmask;
    integer pv [0:NUM_PORTS-1];
    integer pr [0:NUM_PORTS-1];
    integer pidx [0:NUM_PORTS-1];
    reg [1023:0] stim;

    initial begin
        if (!$value$plusargs("stim=%s", stim)) begin
            $display("tb_rf_bank: +stim=<file> is required");
            $finish;
        end
        fh = $fopen(stim, "r");
        if (fh == 0) begin
            $display("tb_rf_bank: cannot open stimulus");
            $finish;
        end

        // The bases come first, so a case can alias two contexts or keep
        // them disjoint without recompiling.
        n = $fscanf(fh, "%d %d\n", f_base0, f_base1);
        vbase = {f_base1[PHYS_W-1:0], f_base0[PHYS_W-1:0]};
        $display("BASE %0d %0d", f_base0, f_base1);

        we = 0; req_valid = 0; cyc = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;

        forever begin
            n = $fscanf(fh, "%d %d %d %h %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                        f_we, f_wrctx, f_widx, f_wmask, f_wword,
                        pv[0], pr[0], pidx[0], pv[1], pr[1], pidx[1],
                        pv[2], pr[2], pidx[2], pv[3], pr[3], pidx[3]);
            if (n != 17) begin
                // Let the last responses drain before stopping.
                we = 0; req_valid = 0;
                repeat (4) begin
                    @(negedge clk);
                    report_responses;
                    cyc = cyc + 1;
                end
                $display("DONE %0d", cyc);
                $finish;
            end

            we     = f_we[0];
            w_rctx = f_wrctx[RCTX_W-1:0];
            w_idx  = f_widx[7:0];
            wmask  = f_wmask;
            for (i = 0; i < VLEN; i = i + 1)
                wdata[i*ELEN +: ELEN] = f_wword + i;

            for (i = 0; i < NUM_PORTS; i = i + 1) begin
                req_valid[i]              = pv[i][0];
                req_rctx[i*RCTX_W +: RCTX_W] = pr[i][RCTX_W-1:0];
                req_idx[i*8 +: 8]         = pidx[i][7:0];
            end

            #1;   // let the combinational accept settle
            for (i = 0; i < NUM_PORTS; i = i + 1)
                if (req_valid[i])
                    $display("ACC %0d %0d %0d", cyc, i, req_accept[i]);

            @(negedge clk);
            report_responses;
            cyc = cyc + 1;
        end
    end

    task report_responses;
        integer j;
        begin
            for (j = 0; j < NUM_PORTS; j = j + 1) begin
                if (rsp_valid[j])
                    $display("RSP1 %0d %0d %h", cyc, j,
                             rsp_data[j*WIDTH +: WIDTH]);
                if (v2[j])
                    $display("RSP2 %0d %0d %h", cyc, j,
                             d2[j*WIDTH +: WIDTH]);
            end
        end
    endtask

    initial begin
        #2000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
