// vf.cvt.f32.s32 / vf.cvt.f32.u32 through tnca_vector_fpu against the ISA
// sec 8.4 text (tools/model/gen_f2i_vectors.py, exact Fractions). Every
// mismatch is counted and the first ones printed; any mismatch fails.
`timescale 1ns/1ps
`include "tnca_defs.vh"
module tb_vector_fpu_f2i;
    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;
    reg  [7:0] major, func;
    reg  [9:0] mod;
    reg  [511:0] va;
    wire is_mc, busy, red_busy, done;
    wire [511:0] vres, mc_vres;
    wire [63:0] sres, mc_sres;
    wire [15:0] pres;
    tnca_vector_fpu #(.FP_LAT(2)) dut (
        .clk(clk), .rst_n(rst_n), .major(major), .func(func), .mod(mod), .mask(16'hffff),
        .va(va), .vb(512'd0), .vc(512'd0), .fpctl(5'd0), .is_multicycle(is_mc), .start(1'b0),
        .busy(busy), .red_busy_o(red_busy), .done(done), .vres(vres), .sres(sres), .pres(pres),
        .mc_sres(mc_sres), .mc_vres(mc_vres));
    integer fin, n, i, ops, vectors, bad;
    reg [31:0] sgn, mode, raw, want;
    reg [31:0] expect [0:15];
    reg [31:0] sgn0, mode0;
    reg [8*512-1:0] path;
    initial begin
        if (!$value$plusargs("vec=%s", path)) begin $display("FATAL +vec"); $finish; end
        fin = $fopen(path, "r");
        if (fin == 0) begin $display("FATAL cannot open vectors"); $finish; end
        major = `TNCA_MAJ_CVT; func = `TNCA_CVT_F32_S32; mod = 0; va = 0;
        ops = 0; vectors = 0; bad = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        while (!$feof(fin)) begin
            n = 0;
            for (i = 0; i < 16; i = i + 1) begin
                if ($fscanf(fin, "%d %d %h %h\n", sgn, mode, raw, want) == 4) begin
                    if (i == 0) begin sgn0 = sgn; mode0 = mode; end
                    else if (sgn != sgn0 || mode != mode0) begin
                        $display("FATAL vector file groups mix (signed, mode)"); $finish;
                    end
                    va[i*32 +: 32] = raw; expect[i] = want; n = n + 1;
                end
            end
            if (n != 0 && n != 16) begin $display("FATAL partial vector group %0d", n); $finish; end
            if (n == 16) begin
                @(negedge clk);
                func = sgn0 ? `TNCA_CVT_F32_S32 : `TNCA_CVT_F32_U32;
                mod = {7'd0, mode0[2:0]};
                repeat (8) @(posedge clk);
                #1;
                for (i = 0; i < 16; i = i + 1) begin
                    if (vres[i*32 +: 32] !== expect[i]) begin
                        if (bad < 20)
                            $display("MISMATCH signed=%0d mode=%0d in=%08h isa=%08h rtl=%08h lane=%0d",
                                     sgn0, mode0, va[i*32 +: 32], expect[i], vres[i*32 +: 32], i);
                        bad = bad + 1;
                    end
                end
                ops = ops + 1; vectors = vectors + 16;
            end
        end
        if (bad != 0 || vectors == 0) begin
            $display("FAIL f2i: %0d of %0d vectors differ from ISA sec 8.4", bad, vectors);
            $fatal(1, "f2i mismatch");
        end
        $display("tb_vector_fpu_f2i: %0d vectors, 0 failures", vectors);
        $display("PASS f2i: %0d vectors in %0d vector operations, 0 differences from ISA sec 8.4", vectors, ops);
        $finish;
    end
endmodule
