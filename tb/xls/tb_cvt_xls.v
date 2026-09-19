//==============================================================================
// tb_cvt_xls -- the XLS-generated converter against the exact oracle
//
// Exhaustive where exhaustive is possible: every bit pattern of BF16, FP16,
// E5M2 and E4M3, to every one of the five formats. Only binary32 sources are
// sampled, and then from the boundaries -- every value each narrow format can
// represent and its neighbours, every binade edge, every rounding tie.
//
// Expected values come from tools/model/floatfmt.py, which computes in exact
// Fractions rather than host floats. A reference that rounded twice would
// agree with a wrong implementation precisely where rounding is hard.
//
// NaN is compared by class, not by bit pattern: the ISA does not promise a
// particular payload, and demanding one would be testing an accident.
//
// The DUT is a four-stage pipeline (doc/xls.md), so one vector goes in per
// cycle and the expectation travels alongside it in a shift register rather
// than the whole file being held in memory -- there are 866,495 vectors. The
// shift register is also what makes this a test of the pipeline: with the
// inputs changing every cycle, a unit that only settled once its input went
// quiet would fail, and so does any other value of LATENCY.
//==============================================================================
`timescale 1ns/1ps

module tb_cvt_xls;
    // Stages minus one: --flop_inputs=false, so the edge that accepts the
    // input is also the edge that fills stage 0.
    parameter LATENCY = 3;
    parameter DEPTH   = 16;

    reg         clk = 1'b0;
    reg         rst = 1'b1;
    reg  [2:0]  src, dst;
    reg  [31:0] x;
    wire [31:0] out;

    always #5 clk = ~clk;

    tnca_cvt_xls dut(.clk(clk), .rst(rst), .src(src), .dst(dst), .x(x), .out(out));

    integer fh, n, fails, r, k;
    reg [2:0] rs, rd;
    reg [31:0] rx, want;
    reg [1023:0] vecfile;

    // The expectation pipeline, shifted in step with the DUT's.
    reg [2:0]  p_src [0:DEPTH-1];
    reg [2:0]  p_dst [0:DEPTH-1];
    reg [31:0] p_x   [0:DEPTH-1];
    reg [31:0] p_want[0:DEPTH-1];
    reg        p_val [0:DEPTH-1];

    // Format codes, matching xls/tnca_formats.x and tools/model/floatfmt.py.
    localparam F32 = 3'd0, BF16 = 3'd1, F16 = 3'd2, E5M2 = 3'd3, E4M3 = 3'd4;

    function is_nan;
        input [2:0] fmt;
        input [31:0] b;
        begin
            case (fmt)
            F32:  is_nan = (b[30:23] == 8'hff) && (b[22:0] != 23'd0);
            BF16: is_nan = (b[14:7]  == 8'hff) && (b[6:0]  != 7'd0);
            F16:  is_nan = (b[14:10] == 5'h1f) && (b[9:0]  != 10'd0);
            E5M2: is_nan = (b[6:2]   == 5'h1f) && (b[1:0]  != 2'd0);
            // E4M3 has no infinity: only the all-ones fraction is NaN.
            E4M3: is_nan = (b[6:3]   == 4'hf)  && (b[2:0]  == 3'h7);
            default: is_nan = 1'b0;
            endcase
        end
    endfunction

    task push;
        input [2:0] s; input [2:0] d; input [31:0] xv; input [31:0] w; input v;
        begin
            for (k = DEPTH-1; k > 0; k = k - 1) begin
                p_src[k] = p_src[k-1]; p_dst[k]  = p_dst[k-1];
                p_x[k]   = p_x[k-1];   p_want[k] = p_want[k-1];
                p_val[k] = p_val[k-1];
            end
            p_src[0] = s; p_dst[0] = d; p_x[0] = xv; p_want[0] = w;
            p_val[0] = v;
        end
    endtask

    task check;
        begin
            if (p_val[LATENCY]) begin
                if (is_nan(p_dst[LATENCY], p_want[LATENCY])) begin
                    if (!is_nan(p_dst[LATENCY], out)) begin
                        fails = fails + 1;
                        if (fails < 8)
                            $display("FAIL src=%0d dst=%0d x=%h got %h want a NaN",
                                     p_src[LATENCY], p_dst[LATENCY],
                                     p_x[LATENCY], out);
                    end
                end else if (out !== p_want[LATENCY]) begin
                    fails = fails + 1;
                    if (fails < 8)
                        $display("FAIL src=%0d dst=%0d x=%h got %h want %h",
                                 p_src[LATENCY], p_dst[LATENCY], p_x[LATENCY],
                                 out, p_want[LATENCY]);
                end
            end
        end
    endtask

    integer drain;
    initial begin
        if (!$value$plusargs("vec=%s", vecfile)) begin
            $display("FATAL: pass +vec=<file>"); $finish;
        end
        fh = $fopen(vecfile, "r");
        if (fh == 0) begin
            $display("FATAL: cannot open %0s", vecfile); $finish;
        end
        n = 0; fails = 0; r = 4;
        for (k = 0; k < DEPTH; k = k + 1) p_val[k] = 1'b0;
        src = F32; dst = F32; x = 32'd0;

        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);

        while (r == 4) begin
            r = $fscanf(fh, "%d %d %h %h", rs, rd, rx, want);
            if (r == 4) begin
                src = rs; dst = rd; x = rx;
                push(rs, rd, rx, want, 1'b1);
                n = n + 1;
            end else begin
                push(3'd0, 3'd0, 32'd0, 32'd0, 1'b0);
            end
            @(posedge clk);
            #1;
            check;
        end
        for (drain = 0; drain < LATENCY; drain = drain + 1) begin
            push(3'd0, 3'd0, 32'd0, 32'd0, 1'b0);
            @(posedge clk);
            #1;
            check;
        end
        $fclose(fh);
        $display("tb_cvt_xls: %0d vectors, %0d failures", n, fails);
        if (fails == 0) $display("PASS");
        else            $display("FAIL");
        $finish;
    end
endmodule
