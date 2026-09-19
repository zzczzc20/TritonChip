//==============================================================================
// tb_cvt16 -- exhaustive check of tnca_cvt16_f32
//
// FP16 and BF16 are 16 bits, so "exhaustive" means all 65,536 of each. There is
// no reason to sample when the whole input space fits in a loop.
//==============================================================================
`timescale 1ns/1ps

module tb_cvt16;
    reg [15:0] v;
    reg        mode;
    wire [31:0] y;
    tnca_cvt16_f32 dut(.v(v), .mode(mode), .y(y));

    integer fh, n, fails, r, k;
    reg [15:0] rv; integer rm; reg [31:0] exp;
    reg [1023:0] vecfile;

    function is_nan; input [31:0] x;
        is_nan = (x[30:23] == 8'hff) && (x[22:0] != 23'd0);
    endfunction

    initial begin
        if (!$value$plusargs("vec=%s", vecfile)) begin
            $display("FATAL: pass +vec=<file>"); $finish;
        end
        fh = $fopen(vecfile, "r");
        if (fh == 0) begin $display("FATAL: cannot open %0s", vecfile); $finish; end
        n = 0; fails = 0; r = 3;
        while (r == 3) begin
            r = $fscanf(fh, "%h %d %h", rv, rm, exp);
            if (r == 3) begin
                v = rv; mode = (rm != 0);
                #1;
                n = n + 1;
                if (is_nan(exp)) begin
                    if (!is_nan(y)) begin
                        fails = fails + 1;
                        if (fails < 10) $display("FAIL v=%h mode=%0d got %h want a NaN", rv, rm, y);
                    end
                end else if (y !== exp) begin
                    fails = fails + 1;
                    if (fails < 10) $display("FAIL v=%h mode=%0d got %h want %h", rv, rm, y, exp);
                end
                #1;
            end
        end
        $fclose(fh);
        $display("tb_cvt16: %0d vectors, %0d failures", n, fails);
        if (fails == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end
endmodule
