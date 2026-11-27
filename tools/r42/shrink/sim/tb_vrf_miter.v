//==============================================================================
// tb_vrf_miter -- equivalence of tnca_vrf_banked XBAR=0 (the shipped response crossbar) against XBAR=1 (AND-OR over one-hot selects,
// every output identical) or XBAR=2 (bank-number mux: rsp_data compared only while rsp_valid), R42-shrink. Same random reads (five
// requesters, three of them presenting a physical register as in tnca_core), two writers with lane masks, random rsp_ready (the core
// ties it to 1; the unit supports holds, so both are exercised), random `clear`. Every output of both instances is compared each cycle.
// Optionally (VLD=1) instance B also has VLD_RAM=1 (valid bits in the bank RAM + reset sweep): then the first SWEEP cycles after reset are not driven.
//==============================================================================
`timescale 1ns/1ps
`ifndef VLDA
`define VLDA
`define VLDB_
`endif
module tb;
    parameter XB      = 1;
    parameter VLDB    = 0;
    parameter SEED    = 1;
    parameter CYCLES  = 300000;
    parameter P_REQ   = 70;
    parameter P_WR    = 50;
    parameter P_RDY   = 100;      // percent rsp_ready per port
    parameter NREG    = 256;      // registers touched
    parameter SWEEP   = 80;       // idle cycles after reset (valid-bit sweep when VLDB)
    localparam PHYS_VREGS = 256, NUM_RCTX = 4, VLEN = 16, ELEN = 32, W = 512;
    reg clk = 0, rst_n = 0; always #5 clk = ~clk;
    reg clear; reg [NUM_RCTX*8-1:0] vbase;
    reg [4:0] req_valid; reg [9:0] req_rctx; reg [39:0] req_idx, req_phys; reg [4:0] rsp_ready;
    reg [1:0] we; reg [3:0] w_rctx; reg [15:0] w_idx; reg [2*VLEN-1:0] wmask; reg [2*W-1:0] wdata;
    wire [4:0] a_acc, b_acc, a_rv, b_rv; wire [5*W-1:0] a_rd, b_rd; wire [1:0] a_wa, b_wa;
    tnca_vrf_banked #(.PHYS_VREGS(256), .VREGS_PER_RCTX(64), .NUM_RCTX(4), .VLEN(16), .ELEN(32), .RCTX_W(2), .PHYS_W(8), .NUM_BANKS(4), .BANK_W(2),
        .READ_LATENCY(1), .NUM_PORTS(5), .ZERO_INVALID(1), .NUM_WRITERS(2), .PHYS_IN(5'b00111), .XBAR(0) `VLDA) ua (
        .clk(clk), .rst_n(rst_n), .clear(clear), .vbase(vbase), .req_valid(req_valid), .req_rctx(req_rctx), .req_idx(req_idx), .req_phys(req_phys),
        .req_accept(a_acc), .rsp_ready(rsp_ready), .rsp_valid(a_rv), .rsp_data(a_rd), .we(we), .w_rctx(w_rctx), .w_idx(w_idx), .wmask(wmask), .wdata(wdata), .w_accept(a_wa));
    tnca_vrf_banked #(.PHYS_VREGS(256), .VREGS_PER_RCTX(64), .NUM_RCTX(4), .VLEN(16), .ELEN(32), .RCTX_W(2), .PHYS_W(8), .NUM_BANKS(4), .BANK_W(2),
        .READ_LATENCY(1), .NUM_PORTS(5), .ZERO_INVALID(1), .NUM_WRITERS(2), .PHYS_IN(5'b00111), .XBAR(XB) `VLDB_) ub (
        .clk(clk), .rst_n(rst_n), .clear(clear), .vbase(vbase), .req_valid(req_valid), .req_rctx(req_rctx), .req_idx(req_idx), .req_phys(req_phys),
        .req_accept(b_acc), .rsp_ready(rsp_ready), .rsp_valid(b_rv), .rsp_data(b_rd), .we(we), .w_rctx(w_rctx), .w_idx(w_idx), .wmask(wmask), .wdata(wdata), .w_accept(b_wa));
    reg [63:0] rng;
    function [63:0] xs; input [63:0] x; reg [63:0] y; begin y = x; y = y ^ (y << 13); y = y ^ (y >> 7); y = y ^ (y << 17); xs = y; end endfunction
    task rnd; begin rng = xs(rng); end endtask
    function integer pct; input integer p; input [63:0] r; begin pct = ((r >> 20) % 100) < p; end endfunction
    integer cyc, mism, k, nrsp, nwr, nacc, nwa; reg [4:0] av;
    reg quiet;
    initial begin
        rng = 64'hBADC0DE000000000 ^ (SEED * 64'h9E3779B97F4A7C15);
        req_valid = 0; req_rctx = 0; req_idx = 0; req_phys = 0; rsp_ready = 5'h1f; we = 0; w_rctx = 0; w_idx = 0; wmask = 0; wdata = 0; clear = 0;
        vbase = {8'd192, 8'd128, 8'd64, 8'd0};
        cyc = 0; mism = 0; nrsp = 0; nwr = 0; nacc = 0; nwa = 0; quiet = 1;
        repeat (4) @(posedge clk);
        #1 rst_n = 1;
        repeat (SWEEP) @(posedge clk);
        #1 quiet = 0;
    end
    always @(posedge clk) if (rst_n && !quiet) begin
        #1;
        for (k = 0; k < 5; k = k + 1) begin
            rnd; req_valid[k] = pct(P_REQ, rng);
            rnd; req_rctx[k*2 +: 2] = rng[5:4]; req_idx[k*8 +: 8] = (rng >> 8) % 64; req_phys[k*8 +: 8] = (rng >> 24) % NREG;
            rnd; rsp_ready[k] = pct(P_RDY, rng);
        end
        // requesters 0..2 present a physical register (PHYS_IN 00111); 3, 4 present rctx + index (idx < 64 so the base applies)
        for (k = 0; k < 2; k = k + 1) begin
            rnd; we[k] = pct(P_WR, rng);
            w_rctx[k*2 +: 2] = rng[5:4]; w_idx[k*8 +: 8] = (rng >> 8) % 64;
            rnd; wmask[k*VLEN +: VLEN] = (rng >> 12) & 16'hffff; if (((rng >> 40) & 7) == 0) wmask[k*VLEN +: VLEN] = 16'hffff;
        end
        for (k = 0; k < 2*16; k = k + 1) begin rnd; wdata[k*32 +: 32] = rng[31:0]; end
        rnd; clear = ((rng >> 30) % 500) == 0;
    end
    always @(negedge clk) if (rst_n && !quiet) begin
        cyc = cyc + 1;
        if (a_acc !== b_acc || a_wa !== b_wa || a_rv !== b_rv ||
            ((XB == 1) ? (a_rd !== b_rd) : (XB == 2) ? ((a_rd & {{W{a_rv[4]}}, {W{a_rv[3]}}, {W{a_rv[2]}}, {W{a_rv[1]}}, {W{a_rv[0]}}}) !== (b_rd & {{W{b_rv[4]}}, {W{b_rv[3]}}, {W{b_rv[2]}}, {W{b_rv[1]}}, {W{b_rv[0]}}})) : (a_rd !== b_rd))) begin
            mism = mism + 1;
            if (mism <= 3) $display("MISMATCH cycle %0d: acc %b/%b wa %b/%b rv %b/%b", cyc, a_acc, b_acc, a_wa, b_wa, a_rv, b_rv);
            if (mism >= 3) begin $display("MITER FAIL"); $finish; end
        end
        nrsp = nrsp + a_rv[0] + a_rv[1] + a_rv[2] + a_rv[3] + a_rv[4]; nacc = nacc + a_acc[0] + a_acc[1] + a_acc[2] + a_acc[3] + a_acc[4];
        nwa = nwa + a_wa[0] + a_wa[1]; nwr = nwr + we[0] + we[1];
        if (cyc >= CYCLES) begin
            $display("VRF MITER %s: XB=%0d VLDB=%0d SEED=%0d cycles=%0d mismatches=%0d | accepted reads=%0d responses=%0d writes offered=%0d accepted=%0d", mism == 0 ? "PASS" : "FAIL", XB, VLDB, SEED, cyc, mism, nacc, nrsp, nwr, nwa);
            $finish;
        end
    end
endmodule
