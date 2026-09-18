//==============================================================================
// tb_bitops -- exhaustive/randomised check of tnca_lzc and tnca_popc against
// the definitions in ISA sec 6 and 8.1.
//==============================================================================
`timescale 1ns/1ps

module tb_bitops;

    reg  [63:0] v64;
    reg  [31:0] v32;
    reg  [88:0] v89;

    wire [6:0]  clz64;  wire [5:0] msb64; wire any64;
    wire [5:0]  clz32;  wire [4:0] msb32; wire any32;
    wire [7:0]  clz89;  wire [6:0] msb89; wire any89;
    wire [6:0]  pc64;
    wire [5:0]  pc32;

    tnca_lzc  #(.W(64), .LOGW(6)) u_l64 (.v(v64), .clz(clz64), .msb_idx(msb64), .any(any64));
    tnca_lzc  #(.W(32), .LOGW(5)) u_l32 (.v(v32), .clz(clz32), .msb_idx(msb32), .any(any32));
    tnca_lzc  #(.W(89), .LOGW(7)) u_l89 (.v(v89), .clz(clz89), .msb_idx(msb89), .any(any89));
    tnca_popc #(.W(64), .LOGW(6)) u_p64 (.v(v64), .count(pc64));
    tnca_popc #(.W(32), .LOGW(5)) u_p32 (.v(v32), .count(pc32));

    integer i, k, fails, n;
    integer exp_clz, exp_pc;

    task check64;
        begin
            #1;
            exp_clz = 64; exp_pc = 0;
            for (k = 63; k >= 0; k = k - 1) begin
                if (v64[k] && exp_clz == 64) exp_clz = 63 - k;
                exp_pc = exp_pc + v64[k];
            end
            n = n + 1;
            if (clz64 !== exp_clz[6:0]) begin
                fails = fails + 1;
                if (fails < 10) $display("FAIL clz64 v=%h got=%0d want=%0d", v64, clz64, exp_clz);
            end
            if (any64 && (msb64 !== (63 - exp_clz))) begin
                fails = fails + 1;
                if (fails < 10) $display("FAIL msb64 v=%h got=%0d want=%0d", v64, msb64, 63-exp_clz);
            end
            if (pc64 !== exp_pc[6:0]) begin
                fails = fails + 1;
                if (fails < 10) $display("FAIL popc64 v=%h got=%0d want=%0d", v64, pc64, exp_pc);
            end
        end
    endtask

    task check32;
        begin
            #1;
            exp_clz = 32; exp_pc = 0;
            for (k = 31; k >= 0; k = k - 1) begin
                if (v32[k] && exp_clz == 32) exp_clz = 31 - k;
                exp_pc = exp_pc + v32[k];
            end
            n = n + 1;
            if (clz32 !== exp_clz[5:0]) begin
                fails = fails + 1;
                if (fails < 10) $display("FAIL clz32 v=%h got=%0d want=%0d", v32, clz32, exp_clz);
            end
            if (pc32 !== exp_pc[5:0]) begin
                fails = fails + 1;
                if (fails < 10) $display("FAIL popc32 v=%h got=%0d want=%0d", v32, pc32, exp_pc);
            end
        end
    endtask

    task check89;
        begin
            #1;
            exp_clz = 89;
            for (k = 88; k >= 0; k = k - 1)
                if (v89[k] && exp_clz == 89) exp_clz = 88 - k;
            n = n + 1;
            if (clz89 !== exp_clz[7:0]) begin
                fails = fails + 1;
                if (fails < 10) $display("FAIL clz89 v=%h got=%0d want=%0d", v89, clz89, exp_clz);
            end
            if (any89 && (msb89 !== (88 - exp_clz))) begin
                fails = fails + 1;
                if (fails < 10) $display("FAIL msb89 v=%h got=%0d want=%0d", v89, msb89, 88-exp_clz);
            end
        end
    endtask

    initial begin
        fails = 0; n = 0;

        // every single-bit position, plus zero and all-ones
        v64 = 64'd0; check64;
        v64 = {64{1'b1}}; check64;
        for (i = 0; i < 64; i = i + 1) begin
            v64 = 64'd1 << i;                 check64;
            v64 = (64'd1 << i) | 64'd1;       check64;
            v64 = {64{1'b1}} >> i;            check64;
        end
        v32 = 32'd0; check32;
        v32 = {32{1'b1}}; check32;
        for (i = 0; i < 32; i = i + 1) begin
            v32 = 32'd1 << i;                 check32;
            v32 = (32'd1 << i) | 32'd1;       check32;
            v32 = {32{1'b1}} >> i;            check32;
        end
        v89 = 89'd0; check89;
        for (i = 0; i < 89; i = i + 1) begin
            v89 = 89'd1 << i;                 check89;
            v89 = (89'd1 << i) | 89'd1;       check89;
        end

        // random
        for (i = 0; i < 20000; i = i + 1) begin
            v64 = {$random, $random}; check64;
            v32 = $random;            check32;
            v89 = {$random, $random, $random}; check89;
        end

        $display("tb_bitops: %0d checks, %0d failures", n, fails);
        if (fails == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end

endmodule
