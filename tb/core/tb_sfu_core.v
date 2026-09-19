//==============================================================================
// tb_sfu_core -- the SFU as the core sees it
//
// tb_sfu_xls checks the four generated pipelines against ISA sec 8.8's error
// bound. This checks the thing built around them: one slice shared by VLEN
// lanes, a start/busy/done handshake, and a latency that has to be exactly
// right for a lane to come back to the lane it belongs to.
//
// That last point is why this bench exists separately. A wrong LAT in
// tnca_sfu does not make results late -- the sequencer would still report
// `done` after the same number of cycles -- it makes lane i receive lane
// i+1's answer. The vectors are drawn from the same file tb_sfu_xls uses, so
// every lane of every batch has its own expected interval and a shift by one
// lane fails immediately.
//
// It also checks the handshake: `busy` must cover the whole operation and
// `done` must be exactly one cycle, or the core's completion port would
// either write twice or write nothing.
//==============================================================================
`timescale 1ns/1ps

module tb_sfu_core;
    parameter VLEN = 16;
    parameter ELEN = 32;
    parameter MAX_VECS = 65536;
    // Overridable so the claim above can be checked rather than asserted:
    // -P tb_sfu_core.LAT=5 and -P tb_sfu_core.LAT=7 both fail, in the
    // thousands, on lanes reading their neighbour's result.
    parameter LAT  = 6;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg                  start = 1'b0;
    reg  [7:0]           func  = 8'd0;
    reg  [VLEN*ELEN-1:0] va    = {(VLEN*ELEN){1'b0}};
    wire                 busy, done;
    wire [VLEN*ELEN-1:0] vres;

    tnca_sfu #(.VLEN(VLEN), .ELEN(ELEN), .LAT(LAT)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .func(func), .va(va),
        .busy(busy), .done(done), .vres(vres)
    );

    integer fh, n, fails, r, i, k, base, cyc, done_cnt, batches;
    reg [31:0] op_a [0:MAX_VECS-1];
    reg [31:0] x_a  [0:MAX_VECS-1];
    reg [31:0] lo_a [0:MAX_VECS-1];
    reg [31:0] hi_a [0:MAX_VECS-1];
    reg        nan_a[0:MAX_VECS-1];
    reg [1023:0] vecfile;
    integer op_i, nan_i;
    reg     same_op;

    function [31:0] ordered; input [31:0] b;
        ordered = b[31] ? ~b : (b | 32'h8000_0000);
    endfunction
    function is_qnan; input [31:0] b;
        is_qnan = (b[30:23] == 8'hff) && (b[22:0] != 23'd0) && b[22];
    endfunction

    // ISA func codes; the vector file numbers the ops exp2, log2, rcp, rsqrt.
    function [7:0] func_of; input integer o;
        case (o)
        0: func_of = 8'h00;   // exp2
        1: func_of = 8'h03;   // log2
        2: func_of = 8'h01;   // rcp
        default: func_of = 8'h02;   // rsqrt
        endcase
    endfunction

    reg [31:0] got;
    task check_batch; input integer b0; begin
        for (k = 0; k < VLEN; k = k + 1) begin
            if (b0 + k < n) begin
                got = vres[k*ELEN +: ELEN];
                if (nan_a[b0+k]) begin
                    if (!is_qnan(got)) begin
                        fails = fails + 1;
                        if (fails < 10)
                            $display("FAIL lane %0d op=%0d x=%h got %h want a quiet NaN",
                                     k, op_a[b0+k], x_a[b0+k], got);
                    end
                end else if (ordered(got) < ordered(lo_a[b0+k])
                          || ordered(got) > ordered(hi_a[b0+k])) begin
                    fails = fails + 1;
                    if (fails < 10)
                        $display("FAIL lane %0d op=%0d x=%h got %h want [%h, %h]",
                                 k, op_a[b0+k], x_a[b0+k], got,
                                 lo_a[b0+k], hi_a[b0+k]);
                end
            end
        end
    end endtask

    initial begin
        if (!$value$plusargs("vec=%s", vecfile)) begin
            $display("FATAL: pass +vec=<file>"); $finish;
        end
        fh = $fopen(vecfile, "r");
        if (fh == 0) begin
            $display("FATAL: cannot open %0s", vecfile); $finish;
        end
        n = 0; fails = 0; r = 5; batches = 0;
        while (r == 5 && n < MAX_VECS) begin
            r = $fscanf(fh, "%d %h %h %h %d", op_i, x_a[n], lo_a[n], hi_a[n], nan_i);
            if (r == 5) begin
                op_a[n] = op_i; nan_a[n] = (nan_i != 0);
                n = n + 1;
            end
        end
        $fclose(fh);

        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        // One batch per op-homogeneous run of VLEN vectors. The bulk of the
        // file is grouped by operation; the sec 8.8 special values at the end
        // are grouped by input instead, so a window that spans two operations
        // is stepped past rather than split.
        base = 0;
        while (base + VLEN <= n) begin
            same_op = 1'b1;
            for (i = 1; i < VLEN; i = i + 1)
                if (op_a[base + i] != op_a[base]) same_op = 1'b0;
            if (!same_op) begin
                base = base + 1;
            end else begin
                func = func_of(op_a[base]);
                for (i = 0; i < VLEN; i = i + 1)
                    va[i*ELEN +: ELEN] = x_a[base + i];
                start = 1'b1;
                @(posedge clk); #1;
                start = 1'b0;
                if (!busy) begin
                    $display("FAIL: busy did not rise after start");
                    fails = fails + 1;
                end
                cyc = 0; done_cnt = 0;
                while (done_cnt == 0 && cyc < 200) begin
                    @(posedge clk); #1;
                    cyc = cyc + 1;
                    if (done) done_cnt = done_cnt + 1;
                end
                if (done_cnt != 1) begin
                    $display("FAIL: no done within 200 cycles (batch at %0d)", base);
                    fails = fails + 1;
                end else begin
                    if (busy) begin
                        $display("FAIL: busy still high in the done cycle");
                        fails = fails + 1;
                    end
                    check_batch(base);
                    // done must be exactly one cycle wide.
                    @(posedge clk); #1;
                    if (done) begin
                        $display("FAIL: done stayed high for a second cycle");
                        fails = fails + 1;
                    end
                end
                batches = batches + 1;
                base = base + VLEN;
            end
        end

        $display("tb_sfu_core: %0d batches of %0d lanes (%0d vectors), %0d failures",
                 batches, VLEN, batches*VLEN, fails);
        if (fails == 0) $display("PASS");
        else            $display("FAIL");
        $finish;
    end
endmodule
