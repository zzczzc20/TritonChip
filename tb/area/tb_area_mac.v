// tb_area_mac -- the shared MAC lane with the R42 area back end (BE) against the generated back end (BE = 0), every mode:
// mode 0 FP32 (operands widened by fmt, sub_c/mul_only/ftz free), mode 1 DP2 (two 16-bit slots: FP16, BF16, E5M2, E4M3),
// mode 2 DP4 (four E4M3 x E4M3). Same CUTS and IN_REG, same inputs, outputs compared every cycle.
// +n=<vectors per mode> +seed=<seed>.
`timescale 1ns/1ps
module tb_area_mac;
    parameter BE = 1;
    parameter [4:0] CUTS = 5'b11101;
    reg clk, rst_n;
    reg [1:0] mode, fa, fb;
    reg [2:0] fmt;
    reg [31:0] a, b, c;
    reg sub_c, mul_only, ftz;
    wire [31:0] y_ref, y_new;
    tnca_mac_lane #(.CUTS(CUTS), .IN_REG(1), .BE(0))  u_ref (.clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt), .fa(fa), .fb(fb),
        .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_ref));
    tnca_mac_lane #(.CUTS(CUTS), .IN_REG(1), .BE(BE)) u_new (.clk(clk), .rst_n(rst_n), .mode(mode), .fmt(fmt), .fa(fa), .fb(fb),
        .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .y(y_new));

    integer n, seed, i, md, fails, cf, cyc, lat, k4;
    reg [15:0] t16;
    reg [31:0] ha[0:15], hb[0:15], hc[0:15];
    reg [15:0] hk[0:15];

    function [22:0] frac(input integer w);          // a fraction of w bits, structured
        reg [3:0] k; reg [22:0] r, m;
        begin
            k = $urandom % 8; r = $urandom; m = (23'd1 << w) - 23'd1;
            case (k)
                0: frac = 0;
                1: frac = m;
                2: frac = (23'd1 << ($urandom % w));
                3: frac = m ^ (23'd1 << ($urandom % w));
                default: frac = r & m;
            endcase
        end
    endfunction
    function integer pickexp(input integer emax);   // an exponent field 0..emax
        integer k;
        begin
            k = $urandom % 10;
            case (k)
                0: pickexp = 0;
                1: pickexp = 1 + ($urandom % 2);
                2: pickexp = emax - ($urandom % 2);
                3: pickexp = emax;
                4: pickexp = emax / 2 + ($urandom % 3) - 1;
                default: pickexp = 1 + ($urandom % (emax));
            endcase
        end
    endfunction
    // one element of format code fc (0 FP16, 1 BF16, 2 E5M2, 3 E4M3) in a 16-bit slot, structured
    function [15:0] elem(input [1:0] fc);
        reg s; reg [4:0] e5; reg [7:0] e8; reg [3:0] e4;
        begin
            s = $urandom;
            case (fc)
            0: begin e5 = pickexp(31); elem = {s, e5, frac(10)}; end
            1: begin e8 = pickexp(255); elem = {s, e8, frac(7)}; end
            2: begin e5 = pickexp(31); elem = {8'h00, s, e5, frac(2)}; end
            default: begin e4 = pickexp(15); elem = {8'h00, s, e4, frac(3)}; end
            endcase
            if ($urandom % 12 == 0) elem = $urandom;           // anything, garbage high bytes included
        end
    endfunction
    function [31:0] f32s(input integer dummy);       // a structured binary32
        reg [7:0] e;
        begin
            e = pickexp(255);
            f32s = {$urandom % 2 == 0 ? 1'b0 : 1'b1, e, frac(23)};
        end
    endfunction
    task gen(input integer m);
        reg [31:0] r; reg [1:0] c0; integer ex;
        begin
            mode = m; fa = $urandom; fb = $urandom; fmt = $urandom % 5;
            sub_c = ($urandom % 2); mul_only = ($urandom % 8 == 0); ftz = ($urandom % 4 == 0);
            if (m == 0) begin
                a = f32s(0); b = f32s(0);
                case (fmt) 0: ; 1,2: begin a = {16'b0, elem(fmt==1 ? 2'd1 : 2'd0)}; b = {16'b0, elem(fmt==1 ? 2'd1 : 2'd0)}; end
                           default: begin a = {24'b0, elem(fmt==3 ? 2'd2 : 2'd3)}; b = {24'b0, elem(fmt==3 ? 2'd2 : 2'd3)}; end endcase
                if ($urandom % 6 == 0) begin a = $urandom; b = $urandom; end
            end else if (m == 1) begin
                a = {elem(fa), elem(fa)}; b = {elem(fb), elem(fb)};
                if ($urandom % 3 == 0) begin a[31:16] = elem($urandom); b[31:16] = elem($urandom); end
                if ($urandom % 3 == 0) begin a[15:0] = elem($urandom); b[15:0] = elem($urandom); end
                if ($urandom % 10 == 0) begin a = $urandom; b = $urandom; end
                sub_c = 0; mul_only = 0; ftz = 0;
            end else begin
                for (k4 = 0; k4 < 4; k4 = k4 + 1) begin
                    t16 = elem(3); a[8*k4 +: 8] = t16[7:0];
                    t16 = elem(3); b[8*k4 +: 8] = t16[7:0];
                end
                if ($urandom % 10 == 0) begin a = $urandom; b = $urandom; end
                sub_c = 0; mul_only = 0; ftz = 0;
            end
            // c: structured, or near the exponent range the products reach, or special
            r = $urandom;
            case ($urandom % 6)
              0: c = $urandom;
              1: c = f32s(0);
              2: begin ex = 100 + ($urandom % 60); c = {r[31], ex[7:0], frac(23)}; end
              3: begin ex = 127 - 40 + ($urandom % 60); c = {r[31], ex[7:0], frac(23)}; end
              4: c = {r[31], 31'b0};
              default: begin ex = 137 - ($urandom % 30); c = {r[31], ex[7:0], frac(23)}; end
            endcase
        end
    endtask

    initial begin
        n = 20000; seed = 1;
        if ($value$plusargs("n=%d", n)) ;
        if ($value$plusargs("seed=%d", seed)) ;
        i = $urandom(seed);
        lat = 1;
        for (i = 0; i < 5; i = i + 1) lat = lat + CUTS[i];
        clk = 0; rst_n = 0; mode = 0; fa = 0; fb = 0; fmt = 0; a = 0; b = 0; c = 0; sub_c = 0; mul_only = 0; ftz = 0;
        #5 clk = 1; #5 clk = 0; #5 clk = 1; #5 clk = 0; rst_n = 1;
        fails = 0; cyc = 0;
        for (md = 0; md < 3; md = md + 1) begin
            cf = 0;
            for (i = 0; i < n + lat + 2; i = i + 1) begin
                if (i < n) gen(md); else begin a = 0; b = 0; c = 0; end
                ha[cyc % 16] = a; hb[cyc % 16] = b; hc[cyc % 16] = c; hk[cyc % 16] = {mode, fa, fb, fmt, sub_c, mul_only, ftz};
                #4;
                if (i >= lat && i - lat < n && y_ref !== y_new) begin
                    cf = cf + 1; fails = fails + 1;
                    if (fails <= 12) $display("MISMATCH mode %0d a %h b %h c %h k %h ref %h new %h", md,
                        ha[(cyc - lat) % 16], hb[(cyc - lat) % 16], hc[(cyc - lat) % 16], hk[(cyc - lat) % 16], y_ref, y_new);
                end
                #1 clk = 1; #5 clk = 0; cyc = cyc + 1;
            end
            $display("mode %0d: %0d vectors, %0d mismatches", md, n, cf);
        end
        $display("tb_area_mac BE=%0d CUTS=%b: %0d mismatches", BE, CUTS, fails);
        if (fails == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end
endmodule
