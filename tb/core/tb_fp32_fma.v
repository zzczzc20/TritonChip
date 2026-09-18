//==============================================================================
// tb_fp32_fma -- differential test of tnca_fp32_fma against the exact
// Fraction-based oracle in tools/model/fp32.py.
//
// The unit is a 2-stage pipeline accepting one operation per cycle, so the
// testbench streams vectors in and checks results LATENCY cycles later. That is
// deliberate rather than driving one vector at a time: back-to-back issue is
// what exercises the pipeline registers, and a stage that accidentally reads a
// later operation's operands only shows up when the pipe is full.
//
// Vector file format, one test per line, all hex:
//     <a> <b> <c> <ctrl> <expected>
// with ctrl[0]=sub_c, ctrl[1]=mul_only, ctrl[2]=ftz.
//
// A NaN result is compared by class, not bit pattern: the ISA only promises a
// quiet NaN, not a particular payload.
//==============================================================================
`timescale 1ns/1ps

module tb_fp32_fma;

    // Queue offset, not the DUT's cycle latency. The DUT has two register
    // stages, so a result is valid two cycles after the operands are presented;
    // but this testbench pushes the expectation *after* the edge that captured
    // the operands, so at the check point `y` belongs to the entry one slot
    // back. (The core's writeback delay is the other number: 2.)
    localparam LATENCY = 1;
    localparam MAXQ    = 8;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    reg  [31:0] a, b, c;
    reg         sub_c, mul_only, ftz;
    wire [31:0] y;

    tnca_fp32_fma dut (
        .clk(clk), .rst_n(rst_n),
        .a(a), .b(b), .c(c),
        .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz),
        .y(y)
    );

    // in-flight expectations, shifted alongside the DUT pipeline
    reg [31:0]  q_exp  [0:MAXQ-1];
    reg [31:0]  q_a    [0:MAXQ-1];
    reg [31:0]  q_b    [0:MAXQ-1];
    reg [31:0]  q_c    [0:MAXQ-1];
    reg [3:0]   q_ctrl [0:MAXQ-1];
    reg [MAXQ-1:0] q_vld;

    integer fh, r, n, fails, ctrl, k;
    // push() must not share a loop variable with its callers: Verilog tasks are
    // static, so the shift loop would reset the caller's counter and the drain
    // loop would never terminate.
    integer kk;
    reg [31:0] exp_y;
    reg [1023:0] vecfile;

    function is_nan;
        input [31:0] v;
        is_nan = (v[30:23] == 8'hff) && (v[22:0] != 23'd0);
    endfunction

    task check_output;
        begin
            if (q_vld[LATENCY]) begin
                n = n + 1;
                if (is_nan(q_exp[LATENCY])) begin
                    if (!is_nan(y)) begin
                        fails = fails + 1;
                        if (fails <= 20)
                            $display("FAIL[%0d] a=%h b=%h c=%h ctrl=%h : got %h, want a NaN",
                                     n, q_a[LATENCY], q_b[LATENCY], q_c[LATENCY],
                                     q_ctrl[LATENCY], y);
                    end
                end else if (y !== q_exp[LATENCY]) begin
                    fails = fails + 1;
                    if (fails <= 20)
                        $display("FAIL[%0d] a=%h b=%h c=%h ctrl=%h : got %h, want %h",
                                 n, q_a[LATENCY], q_b[LATENCY], q_c[LATENCY],
                                 q_ctrl[LATENCY], y, q_exp[LATENCY]);
                end
            end
        end
    endtask

    task push;
        input [31:0] pa, pb, pc;
        input [3:0]  pctrl;
        input [31:0] pexp;
        input        valid;
        begin
            for (kk = MAXQ-1; kk > 0; kk = kk - 1) begin
                q_exp[kk]  = q_exp[kk-1];
                q_a[kk]    = q_a[kk-1];
                q_b[kk]    = q_b[kk-1];
                q_c[kk]    = q_c[kk-1];
                q_ctrl[kk] = q_ctrl[kk-1];
                q_vld[kk]  = q_vld[kk-1];
            end
            q_exp[0] = pexp; q_a[0] = pa; q_b[0] = pb; q_c[0] = pc;
            q_ctrl[0] = pctrl; q_vld[0] = valid;
        end
    endtask

    initial begin
        if (!$value$plusargs("vec=%s", vecfile)) begin
            $display("FATAL: pass +vec=<file>");
            $finish;
        end
        fh = $fopen(vecfile, "r");
        if (fh == 0) begin
            $display("FATAL: cannot open %0s", vecfile);
            $finish;
        end

        n = 0; fails = 0; q_vld = {MAXQ{1'b0}};
        for (k = 0; k < MAXQ; k = k + 1) begin
            q_exp[k] = 0; q_a[k] = 0; q_b[k] = 0; q_c[k] = 0; q_ctrl[k] = 0;
        end

        repeat (3) @(posedge clk);
        #1 rst_n = 1'b1;

        r = 5;
        while (r == 5) begin
            r = $fscanf(fh, "%h %h %h %h %h", a, b, c, ctrl, exp_y);
            if (r == 5) begin
                sub_c    = ctrl[0];
                mul_only = ctrl[1];
                ftz      = ctrl[2];
                @(posedge clk);
                #1;
                push(a, b, c, ctrl[3:0], exp_y, 1'b1);
                check_output;
            end
        end

        // drain
        for (k = 0; k <= LATENCY + 2; k = k + 1) begin
            @(posedge clk);
            #1;
            push(32'd0, 32'd0, 32'd0, 4'd0, 32'd0, 1'b0);
            check_output;
        end

        $fclose(fh);
        $display("tb_fp32_fma: %0d vectors, %0d failures", n, fails);
        if (fails == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end

endmodule
