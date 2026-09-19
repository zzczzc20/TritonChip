//==============================================================================
// tb_sfu_xls -- the SFU against its published error bound
//
// ISA sec 8.8 specifies these operations by a bound, not by a result, so this
// checks membership in an interval rather than equality. The interval comes
// from tools/model/gen_sfu_vectors.py, which evaluates each function in
// 60-digit decimal and then applies the 2^-22 tolerance; a reference carrying
// only binary32 precision could not tell a result inside the band from one
// outside it.
//
// The units are seven-stage pipelines (XLS scheduled them against the asap7
// delay model at the project's 1300 ps target; see doc/xls.md), so this
// streams one vector per cycle and compares each output against the vector
// issued LATENCY cycles earlier. That also checks the pipeline itself: a unit
// that settled only when its input stopped changing would fail here, and a
// wrong LATENCY fails everything -- 5 and 7 both report 12,164 of the 16,692
// vectors outside the band, so the 6 below is measured, not assumed.
//
// LATENCY is stages minus one because the units are generated with
// --flop_inputs=false: the first stage register is fed combinationally from
// x, so the rising edge that accepts x is also the edge that fills stage 0.
//
// Comparison is on the ordered bit pattern, which for same-signed finite
// binary32 values is monotonic with magnitude -- so a numeric compare of the
// bit patterns is a compare of the values. A vector whose lo equals its hi is
// a result sec 8.8 specifies exactly rather than bounds -- a flush, a
// saturation, or one of the special values -- and the fifth column marks the
// ones that must be a quiet NaN, which is compared by class because the ISA
// promises no payload.
//==============================================================================
`timescale 1ns/1ps

module tb_sfu_xls;
    parameter LATENCY  = 6;
    parameter MAX_VECS = 65536;

    reg         clk = 1'b0;
    reg         rst = 1'b1;
    reg  [31:0] x   = 32'h0;
    wire [31:0] y_exp2, y_log2, y_rcp, y_rsqrt;
    reg  [31:0] y;

    always #5 clk = ~clk;

    tnca_sfu_exp2_xls  u_exp2 (.clk(clk), .rst(rst), .x(x), .out(y_exp2));
    tnca_sfu_log2_xls  u_log2 (.clk(clk), .rst(rst), .x(x), .out(y_log2));
    tnca_sfu_rcp_xls   u_rcp  (.clk(clk), .rst(rst), .x(x), .out(y_rcp));
    tnca_sfu_rsqrt_xls u_rsqrt(.clk(clk), .rst(rst), .x(x), .out(y_rsqrt));

    integer fh, n, fails, r, i, j;
    integer n_op [0:3];
    reg [31:0] op_a [0:MAX_VECS-1];
    reg [31:0] x_a  [0:MAX_VECS-1];
    reg [31:0] lo_a [0:MAX_VECS-1];
    reg [31:0] hi_a [0:MAX_VECS-1];
    reg        nan_a[0:MAX_VECS-1];
    integer    n_spec;
    reg [1023:0] vecfile;

    // Order binary32 bit patterns as numbers: flip the sign bit for
    // positives, invert entirely for negatives.
    function [31:0] ordered; input [31:0] b;
        ordered = b[31] ? ~b : (b | 32'h8000_0000);
    endfunction

    function is_qnan; input [31:0] b;
        is_qnan = (b[30:23] == 8'hff) && (b[22:0] != 23'd0) && b[22];
    endfunction

    task check; input integer k; begin
        case (op_a[k])
        0: y = y_exp2;
        1: y = y_log2;
        2: y = y_rcp;
        default: y = y_rsqrt;
        endcase
        if (nan_a[k]) begin
            if (!is_qnan(y)) begin
                fails = fails + 1;
                if (fails < 10)
                    $display("FAIL op=%0d x=%h got %h want a quiet NaN",
                             op_a[k], x_a[k], y);
            end
        end else if (ordered(y) < ordered(lo_a[k])
                  || ordered(y) > ordered(hi_a[k])) begin
            fails = fails + 1;
            if (fails < 10)
                $display("FAIL op=%0d x=%h got %h want [%h, %h]",
                         op_a[k], x_a[k], y, lo_a[k], hi_a[k]);
        end
    end endtask

    integer op_i, nan_i;
    initial begin
        if (!$value$plusargs("vec=%s", vecfile)) begin
            $display("FATAL: pass +vec=<file>"); $finish;
        end
        fh = $fopen(vecfile, "r");
        if (fh == 0) begin
            $display("FATAL: cannot open %0s", vecfile); $finish;
        end
        n = 0; fails = 0; r = 5; n_spec = 0;
        for (i = 0; i < 4; i = i + 1) n_op[i] = 0;
        while (r == 5 && n < MAX_VECS) begin
            r = $fscanf(fh, "%d %h %h %h %d", op_i, x_a[n], lo_a[n], hi_a[n],
                        nan_i);
            if (r == 5) begin
                op_a[n]  = op_i;
                nan_a[n] = (nan_i != 0);
                if (nan_a[n] || lo_a[n] === hi_a[n]) n_spec = n_spec + 1;
                n_op[op_i] = n_op[op_i] + 1;
                n = n + 1;
            end
        end
        $fclose(fh);

        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);

        // One vector per cycle. Output seen after edge i belongs to vector
        // i - LATENCY; the tail drains with the input held at the last value.
        for (i = 0; i < n + LATENCY; i = i + 1) begin
            x = x_a[(i < n) ? i : n - 1];
            @(posedge clk);
            #1;
            j = i - LATENCY;
            if (j >= 0) check(j);
        end

        $display("tb_sfu_xls: %0d vectors within 2^-22 (exp2=%0d log2=%0d rcp=%0d rsqrt=%0d), %0d exact, %0d outside",
                 n, n_op[0], n_op[1], n_op[2], n_op[3], n_spec, fails);
        if (fails == 0) $display("PASS");
        else            $display("FAIL");
        $finish;
    end
endmodule
