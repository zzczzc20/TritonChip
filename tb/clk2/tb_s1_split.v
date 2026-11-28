// R42 clk2: tnca_mac_s1a_xls then tnca_mac_s1b_xls equals tnca_mac_s1_xls on random operands, every mode and element code.
//   iverilog -g2005 -o s1split.vvp tb/clk2/tb_s1_split.v rtl/gen/tnca_mac_s1_xls.v rtl/gen/tnca_mac_s1a_xls.v rtl/gen/tnca_mac_s1b_xls.v && vvp s1split.vvp +n=2000000
`timescale 1ns/1ps
module tb_s1_split;
    reg [1:0]  mode, fa, fb;
    reg [31:0] a, b;
    reg        ftz;
    wire [186:0] o_ref, o_new;
    wire [363:0] mid;
    tnca_mac_s1_xls  u_ref (.mode(mode), .a(a), .b(b), .fa(fa), .fb(fb), .ftz(ftz), .out(o_ref));
    tnca_mac_s1a_xls u_a   (.mode(mode), .a(a), .b(b), .fa(fa), .fb(fb), .ftz(ftz), .out(mid));
    tnca_mac_s1b_xls u_b   (.m(mid), .out(o_new));
    integer n, i, bad;
    reg [31:0] seed;
    function [31:0] rnd; input [31:0] s; reg [31:0] x; begin x = s; x = x ^ (x << 13); x = x ^ (x >> 17); x = x ^ (x << 5); rnd = x; end endfunction
    initial begin
        if (!$value$plusargs("n=%d", n)) n = 200000;
        bad = 0; seed = 32'h1badcafe;
        for (i = 0; i < n; i = i + 1) begin
            seed = rnd(seed); a = seed;
            seed = rnd(seed); b = seed;
            seed = rnd(seed); mode = seed[1:0]; fa = seed[3:2]; fb = seed[5:4]; ftz = seed[6];
            // every eighth vector: special exponents (zero, subnormal, inf, nan) in the binary32 and FP8 fields
            if (i % 8 == 1) begin a[30:23] = (seed[8] ? 8'hff : 8'h00); end
            if (i % 8 == 2) begin b[30:23] = (seed[9] ? 8'hff : 8'h00); end
            if (i % 8 == 3) begin a = {a[31], 7'h0, a[23:0]}; b = {b[31], 7'h0, b[23:0]}; end
            if (i % 8 == 4) begin a[7:0] = {a[7], seed[10] ? 4'hf : 4'h0, a[2:0]}; b[7:0] = {b[7], seed[11] ? 4'hf : 4'h0, b[2:0]}; end
            #1;
            if (o_ref !== o_new) begin
                bad = bad + 1;
                if (bad < 5) $display("MISMATCH mode %0d fa %0d fb %0d ftz %0d a %h b %h ref %h new %h", mode, fa, fb, ftz, a, b, o_ref, o_new);
            end
        end
        $display("tb_s1_split: %0d vectors, %0d mismatches", n, bad);
        if (bad != 0) $display("FAIL tb_s1_split");
        else $display("PASS tb_s1_split");
        $finish;
    end
endmodule
