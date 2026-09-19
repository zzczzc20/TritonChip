//==============================================================================
// tb_fpu_multi -- multi-precision FMA against the exact oracle
//
// The unit is XLS widening in front of the hand-written binary32 FMA, so what
// this checks is the composition: that operands in FP8, FP16, BF16 or FP32
// reach the multiplier as the values they denote, and that the result obeys
// ISA sec 9C.1 -- exact product, one rounding.
//
// The reference widens with the oracle, not with the hardware's converter. If
// both used the same converter a bug in it would cancel out and this would
// pass on a wrong answer.
//
// The FMA is a two-stage pipeline accepting one operation per cycle, so
// vectors stream in and results are checked LATENCY cycles behind. Streaming
// rather than draining is the point: a stage that read a later operation's
// operands would only show up with the pipe full.
//==============================================================================
`timescale 1ns/1ps

module tb_fpu_multi;
    // Zero, and the reasoning matters. The FMA registers its result, so
    // after the edge that captures vector i the output still holds vector
    // i-1's result. check() runs before push(), so the queue head is already
    // vector i-1's expectation -- the two line up with no extra delay. A
    // non-zero value here compares the output against an older vector and
    // every row fails by one position, which is what it did.
    localparam LATENCY = 0;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg  [2:0]  fmt;
    reg  [31:0] a, b, c;
    wire [31:0] y;

    tnca_fpu_multi dut (
        .clk(clk), .rst_n(rst_n),
        .fmt(fmt), .a(a), .b(b), .c(c),
        .sub_c(1'b0), .mul_only(1'b0), .ftz(1'b0),
        .y(y)
    );

    // Expectations trail the operands by the pipeline depth.
    reg [31:0] q_want [0:LATENCY];
    reg [31:0] q_a [0:LATENCY];
    reg [2:0]  q_fmt [0:LATENCY];
    reg        q_v [0:LATENCY];

    integer fh, n, fails, r, k;
    reg [2:0]  rf;
    reg [31:0] ra, rb, rc, want;
    reg [1023:0] vecfile;

    function is_nan; input [31:0] x;
        is_nan = (x[30:23] == 8'hff) && (x[22:0] != 23'd0);
    endfunction

    task push; input [31:0] w; input [31:0] av; input [2:0] fv; input vv;
        integer j;
        begin
            for (j = LATENCY; j > 0; j = j - 1) begin
                q_want[j] = q_want[j-1]; q_a[j] = q_a[j-1];
                q_fmt[j] = q_fmt[j-1];   q_v[j] = q_v[j-1];
            end
            q_want[0] = w; q_a[0] = av; q_fmt[0] = fv; q_v[0] = vv;
        end
    endtask

    task check;
        begin
            if (q_v[LATENCY]) begin
                n = n + 1;
                if (is_nan(q_want[LATENCY])) begin
                    if (!is_nan(y)) begin
                        fails = fails + 1;
                        if (fails < 8)
                            $display("FAIL fmt=%0d a=%h got %h want a NaN",
                                     q_fmt[LATENCY], q_a[LATENCY], y);
                    end
                end else if (y !== q_want[LATENCY]) begin
                    fails = fails + 1;
                    if (fails < 8)
                        $display("FAIL fmt=%0d a=%h got %h want %h",
                                 q_fmt[LATENCY], q_a[LATENCY], y,
                                 q_want[LATENCY]);
                end
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("vec=%s", vecfile)) begin
            $display("FATAL: pass +vec=<file>"); $finish;
        end
        fh = $fopen(vecfile, "r");
        if (fh == 0) begin
            $display("FATAL: cannot open %0s", vecfile); $finish;
        end
        for (k = 0; k <= LATENCY; k = k + 1) q_v[k] = 1'b0;
        n = 0; fails = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        r = 5;
        while (r == 5) begin
            r = $fscanf(fh, "%d %h %h %h %h", rf, ra, rb, rc, want);
            if (r == 5) begin
                fmt = rf; a = ra; b = rb; c = rc;
                @(posedge clk);
                #1;
                check();
                push(want, ra, rf, 1'b1);
            end
        end
        // Drain.
        for (k = 0; k <= LATENCY; k = k + 1) begin
            @(posedge clk); #1;
            check();
            push(32'd0, 32'd0, 3'd0, 1'b0);
        end

        $fclose(fh);
        $display("tb_fpu_multi: %0d vectors, %0d failures", n, fails);
        if (fails == 0) $display("PASS");
        else            $display("FAIL");
        $finish;
    end
endmodule
