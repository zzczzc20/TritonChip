//==============================================================================
// tb_r3v2_vrf -- the candidate VRF's storage semantics
//
// doc/phase3_tasks_and_acceptance.md G1. Two instances of the same wrapper,
// differing only in ZERO_INVALID, because R3V2-RF-03 asks for two things
// that sound contradictory and are not:
//
//   the architecture reads zero from a register nobody wrote
//   the storage still holds X, so a 2-state zero cannot be hiding in it
//
// `u_arch` is the candidate (ZERO_INVALID=1). `u_raw` is the same design
// with the substitution off, and exists so the second claim can be checked
// rather than asserted.
//
// +case=<n>:
//   1  cold reset: every architectural read is zero, every raw read is X
//   2  a partial write: the lanes written read back, the rest still zero,
//      and the raw instance shows X in exactly those other lanes
//   3  reset with a read in flight: the response is dropped, not delivered
//      into the new epoch
//   4  backpressure: rsp_ready low holds the response and no further read
//      is accepted; when it lifts, the held value is the right one
//   5  a stimulus replay, for the per-request comparison against the model
//==============================================================================
`timescale 1ns/1ps
`include "tnca_defs.vh"

module tb_r3v2_vrf;

    localparam VLEN = 16, ELEN = 32, WIDTH = VLEN*ELEN;
    localparam RCTX_W = 1, PHYS_W = 7, NUM_BANKS = 4, BANK_W = 2;
    localparam NUM_PORTS = 4;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    wire [2*PHYS_W-1:0] vbase = {7'd64, 7'd0};

    reg  [NUM_PORTS-1:0]        req_valid, rsp_ready;
    reg  [NUM_PORTS*RCTX_W-1:0] req_rctx;
    reg  [NUM_PORTS*8-1:0]      req_idx;
    reg                         we;
    reg  [RCTX_W-1:0]           w_rctx;
    reg  [7:0]                  w_idx;
    reg  [VLEN-1:0]             wmask;
    reg  [WIDTH-1:0]            wdata;

    wire [NUM_PORTS-1:0]        a_acc, a_rv, r_acc, r_rv;
    wire [NUM_PORTS*WIDTH-1:0]  a_rd, r_rd;

`define VRF_PORTS(ACC, RV, RD) \
        .clk(clk), .rst_n(rst_n), .vbase(vbase), \
        .req_valid(req_valid), .req_rctx(req_rctx), .req_idx(req_idx), \
        .req_accept(ACC), .rsp_ready(rsp_ready), \
        .rsp_valid(RV), .rsp_data(RD), \
        .we(we), .w_rctx(w_rctx), .w_idx(w_idx), \
        .wmask(wmask), .wdata(wdata)

    tnca_vrf_banked #(
        .PHYS_VREGS(128), .VREGS_PER_RCTX(64), .NUM_RCTX(2), .VLEN(VLEN),
        .ELEN(ELEN), .RCTX_W(RCTX_W), .PHYS_W(PHYS_W),
        .NUM_BANKS(NUM_BANKS), .BANK_W(BANK_W), .READ_LATENCY(1),
        .NUM_PORTS(NUM_PORTS), .ZERO_INVALID(1)
    ) u_arch (`VRF_PORTS(a_acc, a_rv, a_rd));

    tnca_vrf_banked #(
        .PHYS_VREGS(128), .VREGS_PER_RCTX(64), .NUM_RCTX(2), .VLEN(VLEN),
        .ELEN(ELEN), .RCTX_W(RCTX_W), .PHYS_W(PHYS_W),
        .NUM_BANKS(NUM_BANKS), .BANK_W(BANK_W), .READ_LATENCY(1),
        .NUM_PORTS(NUM_PORTS), .ZERO_INVALID(0)
    ) u_raw (`VRF_PORTS(r_acc, r_rv, r_rd));

    integer which, cyc, i, n, fh;
    integer p_idx [0:NUM_PORTS-1];
    reg [1023:0] stim;

    task idle;
        begin
            req_valid = 0; we = 0;
        end
    endtask

    task do_write;
        input [7:0] idx;
        input [VLEN-1:0] mask;
        input integer word;
        integer L;
        begin
            @(negedge clk);
            we = 1; w_rctx = 0; w_idx = idx; wmask = mask;
            for (L = 0; L < VLEN; L = L + 1)
                wdata[L*ELEN +: ELEN] = word + L;
            @(negedge clk);
            we = 0;
        end
    endtask

    // The response is valid for exactly one cycle when the consumer is
    // ready, so it is sampled on the edge it appears rather than the one
    // after -- which is where the first version looked, and saw the zero
    // that `rsp_data` holds when nothing is valid.
    task read_back;
        input [7:0] idx;
        begin
            @(negedge clk);
            req_valid = 4'b0001; req_rctx = 0; req_idx = {24'd0, idx};
            @(negedge clk);
            req_valid = 0;
            $display("ARCH %0d v=%b %h", idx, a_rv[0], a_rd[0 +: WIDTH]);
            $display("RAW  %0d v=%b %h", idx, r_rv[0], r_rd[0 +: WIDTH]);
        end
    endtask

    initial begin
        if (!$value$plusargs("case=%d", which)) which = 1;
        idle; rsp_ready = {NUM_PORTS{1'b1}}; cyc = 0;
        req_rctx = 0; req_idx = 0; wmask = 0; w_idx = 0; w_rctx = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        $display("CASE %0d", which);

        if (which == 1) begin
            // Nothing written anywhere.
            read_back(8'd0);
            read_back(8'd5);
            read_back(8'd33);
        end else if (which == 2) begin
            // Four lanes of v7, then read the whole register.
            do_write(8'd7, 16'h000f, 32'h4000);
            read_back(8'd7);
        end else if (which == 3) begin
            do_write(8'd9, {VLEN{1'b1}}, 32'h5000);
            @(negedge clk);
            req_valid = 4'b0001; req_rctx = 0; req_idx = {24'd0, 8'd9};
            @(negedge clk);
            req_valid = 0;
            // The read was accepted; reset before its response is taken.
            rst_n = 0;
            $display("RESET-IN-FLIGHT");
            @(negedge clk);
            rst_n = 1;
            for (i = 0; i < 4; i = i + 1) begin
                @(negedge clk);
                $display("AFTER %0d arch_v=%b raw_v=%b", i, a_rv, r_rv);
            end
            // and the data array kept its contents across the reset
            read_back(8'd9);
        end else if (which == 4) begin
            do_write(8'd3, {VLEN{1'b1}}, 32'h6000);
            @(negedge clk);
            rsp_ready = 4'b0000;              // nobody is taking responses
            req_valid = 4'b0001; req_rctx = 0; req_idx = {24'd0, 8'd3};
            @(negedge clk);
            $display("ACCEPT-WHILE-HELD %b", a_acc);   // the request was taken on the previous edge; with the response held, no further read is accepted
            @(negedge clk);
            // the response is now held; a further read must not be taken
            $display("HELD arch_v=%b accept=%b data=%h",
                     a_rv, a_acc, a_rd[0 +: WIDTH]);
            @(negedge clk);
            $display("STILL arch_v=%b accept=%b data=%h",
                     a_rv, a_acc, a_rd[0 +: WIDTH]);
            req_valid = 0;
            rsp_ready = {NUM_PORTS{1'b1}};
            @(negedge clk);
            $display("RELEASED arch_v=%b data=%h", a_rv, a_rd[0 +: WIDTH]);
        end else if (which == 5) begin
            if (!$value$plusargs("stim=%s", stim)) begin
                $display("case 5 needs +stim="); $finish;
            end
            fh = $fopen(stim, "r");
            forever begin
                // One index PER PORT. The first version read a single
                // integer into req_idx, which is NUM_PORTS*8 bits wide, so
                // ports 1..3 always requested register 0 while the model
                // used four different ones -- and every disagreement was
                // the bench's, not the design's.
                n = $fscanf(fh, "%d %d %h %d %d %d %d %d %d %d\n",
                            we, w_idx, wmask, i, req_valid,
                            p_idx[0], p_idx[1], p_idx[2], p_idx[3],
                            rsp_ready);
                if (n != 10) begin
                    idle;
                    repeat (4) begin
                        @(negedge clk);
                        report;
                        cyc = cyc + 1;
                    end
                    $display("DONE %0d", cyc);
                    $finish;
                end
                w_rctx = 0; req_rctx = 0;
                for (n = 0; n < NUM_PORTS; n = n + 1)
                    req_idx[n*8 +: 8] = p_idx[n][7:0];
                for (n = 0; n < VLEN; n = n + 1)
                    wdata[n*ELEN +: ELEN] = i + n;
                #1;
                $display("ACC %0d %b", cyc, a_acc);
                @(negedge clk);
                report;
                cyc = cyc + 1;
            end
        end

        $display("DONE");
        $finish;
    end

    task report;
        integer j;
        begin
            for (j = 0; j < NUM_PORTS; j = j + 1)
                if (a_rv[j])
                    $display("RSP %0d %0d %h", cyc, j,
                             a_rd[j*WIDTH +: WIDTH]);
        end
    endtask

    initial begin
        #500000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
