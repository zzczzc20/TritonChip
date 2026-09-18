//==============================================================================
// tb_fp32_fma -- differential test of tnca_fp32_fma against the exact
// Fraction-based oracle in tools/model/fp32.py.
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

    reg  [31:0] a, b, c;
    reg         sub_c, mul_only, ftz;
    wire [31:0] y;

    tnca_fp32_fma dut (
        .a(a), .b(b), .c(c),
        .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz),
        .y(y)
    );

    integer fh, r, n, fails, ctrl;
    reg [31:0] exp_y;
    reg [1023:0] vecfile;

    function is_nan;
        input [31:0] v;
        is_nan = (v[30:23] == 8'hff) && (v[22:0] != 23'd0);
    endfunction

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

        n = 0;
        fails = 0;
        r = 5;
        while (r == 5) begin
            r = $fscanf(fh, "%h %h %h %h %h", a, b, c, ctrl, exp_y);
            if (r == 5) begin
                sub_c    = ctrl[0];
                mul_only = ctrl[1];
                ftz      = ctrl[2];
                #1;
                n = n + 1;
                if (is_nan(exp_y)) begin
                    if (!is_nan(y)) begin
                        fails = fails + 1;
                        if (fails <= 20)
                            $display("FAIL[%0d] a=%h b=%h c=%h ctrl=%h : got %h, want a NaN",
                                     n, a, b, c, ctrl[3:0], y);
                    end
                end else if (y !== exp_y) begin
                    fails = fails + 1;
                    if (fails <= 20)
                        $display("FAIL[%0d] a=%h b=%h c=%h ctrl=%h : got %h, want %h",
                                 n, a, b, c, ctrl[3:0], y, exp_y);
                end
                #1;
            end
        end
        $fclose(fh);

        $display("tb_fp32_fma: %0d vectors, %0d failures", n, fails);
        if (fails == 0) $display("PASS");
        else            $display("FAIL");
        $finish;
    end

endmodule
