// tb_area_stages -- the R42 area stages against the generated ones.
//
//   norm   tnca_fmax_norm (ADD parameter) against tnca_fmad_norm_xls on FmaAligned structs produced by
//          tnca_fma_align_xls from random operands (the structs the FMA's stage 1 can produce)
//   round  tnca_fmax_round against tnca_fma_round_xls on random FmaNormalised structs (every field random,
//          then structs produced by the generated norm stage from the aligned structs)
//   chain  the whole path a, b, c -> y of the new stages against the generated stages
//
// +n=<vectors per class> +seed=<seed> ; prints one line per class and PASS/FAIL. Plain iverilog.
`timescale 1ns/1ps
module tb_area_stages;
    parameter ADD = 0;
    parameter SHIFT = 0;
    reg  [31:0] a, b, c;
    reg         sub_c, mul_only, ftz;
    wire [149:0] al;
    tnca_fma_align_xls u_al (.a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .out(al));
    wire [121:0] nm_ref, nm_new;
    tnca_fmad_norm_xls u_nr (.s(al), .out(nm_ref));
    tnca_fmax_norm #(.ADD(ADD)) u_nn (.s(al), .out(nm_new));
    wire [31:0] y_ref, y_new;
    tnca_fma_round_xls u_rr (.s(nm_ref), .out(y_ref));
    tnca_fmax_round #(.SHIFT(SHIFT)) u_rn (.s(nm_new), .out(y_new));
    // round stage on arbitrary structs
    reg  [121:0] rs;
    wire [31:0] yr_ref, yr_new;
    tnca_fma_round_xls u_rr2 (.s(rs), .out(yr_ref));
    tnca_fmax_round #(.SHIFT(SHIFT)) u_rn2 (.s(rs), .out(yr_new));

    integer n, seed, cls, i;
    integer fails_n, fails_r, fails_c, fails_s, cf;
    reg [63:0] st0, st1;
    function [31:0] r32(input integer dummy);
        begin r32 = $urandom; end
    endfunction
    function [22:0] frac(input integer dummy);
        reg [3:0] k; reg [31:0] r;
        begin
            k = $urandom % 8; r = $urandom;
            case (k)
                0: frac = 0;
                1: frac = 23'h7fffff;
                2: frac = 23'd1 << ($urandom % 23);
                3: frac = 23'h7fffff ^ (23'd1 << ($urandom % 23));
                4: frac = 23'h7fffff >> ($urandom % 23);
                5: frac = (23'h7fffff << ($urandom % 23)) & 23'h7fffff;
                default: frac = r[22:0];
            endcase
        end
    endfunction
    function integer expo(input integer dummy);
        integer k;
        begin
            k = $urandom % 10;
            case (k)
                0: expo = 0;
                1: expo = 1 + $urandom % 3;
                2: expo = 254 - $urandom % 3;
                3: expo = 255;
                default: expo = 1 + $urandom % 254;
            endcase
        end
    endfunction
    function [31:0] mk(input s, input integer e, input [22:0] f);
        integer ee;
        begin
            ee = e; if (ee < 0) ee = 0; if (ee > 255) ee = 255;
            mk = {s, ee[7:0], f};
        end
    endfunction
    // truncated product of two normal operands, the encoding a fused add of its negation cancels against
    function [31:0] prodf(input [31:0] x, input [31:0] y);
        reg [47:0] pm; integer e;
        begin
            pm = {1'b1, x[22:0]} * {1'b1, y[22:0]};
            e = x[30:23] + y[30:23] - 127;
            if (pm[47]) prodf = mk(x[31]^y[31], e + 1, pm[46:24]);
            else        prodf = mk(x[31]^y[31], e,     pm[45:23]);
        end
    endfunction

    task gen(input integer cl);
        reg [31:0] r; integer ea, eb, d, dl; reg [31:0] pu; reg [2:0] ctl;
        begin
            r = $urandom; ctl = r[2:0];
            sub_c = $urandom; mul_only = ($urandom % 8) == 0; ftz = ($urandom % 4) == 0;
            case (cl)
            0: begin a = $urandom; b = $urandom; c = $urandom; end
            1: begin a = mk($urandom, expo(0), frac(0)); b = mk($urandom, expo(0), frac(0)); c = mk($urandom, expo(0), frac(0)); end
            2: begin ea = 1 + $urandom % 254; eb = 1 + $urandom % 254; d = ($urandom % 200) - 100;
                     a = mk($urandom, ea, frac(0)); b = mk($urandom, eb, frac(0)); c = mk($urandom, ea + eb - 127 + d, frac(0)); end
            3: begin ea = 20 + $urandom % 215; eb = 127 + ($urandom % 41) - 20;
                     a = mk($urandom, ea, frac(0)); b = mk($urandom, eb, $urandom);
                     pu = prodf(a, b); dl = ($urandom % 9) - 4; c = pu + dl; if ($urandom % 2) c = c ^ 32'h80000000;
                     sub_c = 0; end
            4: begin a = mk($urandom, $urandom % 40, frac(0)); b = mk($urandom, 127 - 60 + $urandom % 60, frac(0));
                     c = mk($urandom, $urandom % 4, frac(0));
                     if ($urandom % 3 == 0) begin pu = prodf(a, b); c = pu + (($urandom % 5) - 2); c[31] = c[31] ^ $urandom; end
                     sub_c = 0; end
            5: begin a = mk($urandom, 100 + $urandom % 155, 23'h7fffff ^ ($urandom % 4));
                     b = mk($urandom, 127 + ($urandom % 3) - 1, ($urandom % 2) ? 23'h0 : (23'h7fffff ^ ($urandom % 4)));
                     c = mk($urandom, expo(0), frac(0));
                     if ($urandom % 2) begin ea = a[30:23] + b[30:23] - 127 - 24 - $urandom % 4; c = mk($urandom, ea, frac(0)); end end
            default: begin
                     a = mk($urandom, 1 + $urandom % 254, ($urandom & 23'h7ff) << 12);
                     b = mk($urandom, 1 + $urandom % 254, ($urandom & 23'h7ff) << 12);
                     pu = prodf(a, b); c = pu ^ 32'h80000000;
                     if ($urandom % 2) c = mk(~pu[31], pu[30:23] - ($urandom % 30), frac(0));
                     sub_c = $urandom; end
            endcase
        end
    endtask

    task check_round_random(input integer mode);
        reg [31:0] r;
        begin
            rs[31:0] = $urandom; rs[63:32] = $urandom; rs[95:64] = $urandom; rs[121:96] = $urandom;
            case (mode)
            0: ;                                                  // everything random
            1: begin rs[40] = ($urandom % 8) != 0;                 // any mostly 1
                     rs[39] = $urandom % 4 == 0;                  // subnormal_res sometimes
                     rs[37:31] = $urandom % 30; rs[29:18] = $urandom % 110;
                     rs[17:6] = 12'd1 + $urandom % 300; end
            default: begin rs[40] = 1; rs[39] = ($urandom % 3) == 0; rs[38] = ($urandom % 3) == 0;
                     rs[37:31] = $urandom % 40; rs[29:18] = ($urandom % 130); rs[30] = ($urandom % 4) == 0;
                     rs[17:6] = 12'd250 + ($urandom % 10) - 5;
                     rs[121:41] = rs[121:41] | (81'd1 << ($urandom % 81)); end
            endcase
            #1;
        end
    endtask

    initial begin
        n = 100000; seed = 1;
        if ($value$plusargs("n=%d", n)) ;
        if ($value$plusargs("seed=%d", seed)) ;
        i = $urandom(seed);
        fails_n = 0; fails_r = 0; fails_c = 0; fails_s = 0;
        for (cls = 0; cls < 7; cls = cls + 1) begin
            cf = 0;
            for (i = 0; i < n; i = i + 1) begin
                gen(cls); #1;
                if (nm_ref !== nm_new) begin
                    cf = cf + 1; fails_n = fails_n + 1;
                    if (fails_n <= 10) $display("NORM MISMATCH cls %0d a %h b %h c %h sub %b mo %b ftz %b\n  ref %h\n  new %h", cls, a, b, c, sub_c, mul_only, ftz, nm_ref, nm_new);
                end
                if (y_ref !== y_new) begin
                    cf = cf + 1; fails_c = fails_c + 1;
                    if (fails_c <= 10) $display("CHAIN MISMATCH cls %0d a %h b %h c %h sub %b mo %b ftz %b ref %h new %h", cls, a, b, c, sub_c, mul_only, ftz, y_ref, y_new);
                end
            end
            $display("class %0d: %0d vectors, %0d mismatches (norm+chain)", cls, n, cf);
        end
        // round stage on arbitrary structs
        for (cls = 0; cls < 3; cls = cls + 1) begin
            cf = 0;
            for (i = 0; i < n; i = i + 1) begin
                check_round_random(cls);
                if (yr_ref !== yr_new) begin
                    cf = cf + 1; fails_r = fails_r + 1;
                    if (fails_r <= 10) $display("ROUND MISMATCH mode %0d s %h ref %h new %h", cls, rs, yr_ref, yr_new);
                end
            end
            $display("round mode %0d: %0d structs, %0d mismatches", cls, n, cf);
        end
        $display("tb_area_stages ADD=%0d: norm %0d chain %0d round %0d mismatches", ADD, fails_n, fails_c, fails_r);
        if (fails_n + fails_c + fails_r == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end
endmodule
