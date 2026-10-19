// Offline oracle capture: drives the product's tnca_vector_fpu with recorded
// stimuli and writes its outputs. Simulation-only; never part of the TLM
// model's execution path. The model is checked against the resulting file.
`timescale 1ns/1ps
`include "tnca_defs.vh"
module tb_vector_fpu_oracle;
    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;
    reg  [7:0] major, func;
    reg  [9:0] mod;
    reg  [15:0] mask;
    reg  [511:0] va, vb, vc;
    reg  [4:0] fpctl;
    reg  start = 1'b0;
    wire is_mc, busy, red_busy, done;
    wire [511:0] vres, mc_vres;
    wire [63:0] sres, mc_sres;
    wire [15:0] pres;
    tnca_vector_fpu #(.FP_LAT(2)) dut (
        .clk(clk), .rst_n(rst_n), .major(major), .func(func), .mod(mod), .mask(mask),
        .va(va), .vb(vb), .vc(vc), .fpctl(fpctl), .is_multicycle(is_mc), .start(start),
        .busy(busy), .red_busy_o(red_busy), .done(done), .vres(vres), .sres(sres), .pres(pres),
        .mc_sres(mc_sres), .mc_vres(mc_vres));
    integer fin, fout, n, i, guard;
    reg [31:0] w;
    reg [8*512-1:0] in_path, out_path;
    reg [511:0] rv;
    reg [63:0]  rs;
    reg [15:0]  rp;
    initial begin
        if (!$value$plusargs("in=%s", in_path)) begin $display("FATAL +in"); $finish; end
        if (!$value$plusargs("out=%s", out_path)) begin $display("FATAL +out"); $finish; end
        fin = $fopen(in_path, "r"); fout = $fopen(out_path, "w");
        major = 0; func = 0; mod = 0; mask = 0; va = 0; vb = 0; vc = 0; fpctl = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        while (!$feof(fin)) begin
            n = $fscanf(fin, "%h %h %h %h", major, func, mod, mask);
            if (n == 4) begin
                for (i = 0; i < 16; i = i + 1) begin n = $fscanf(fin, "%h", w); va[i*32 +: 32] = w; end
                for (i = 0; i < 16; i = i + 1) begin n = $fscanf(fin, "%h", w); vb[i*32 +: 32] = w; end
                for (i = 0; i < 16; i = i + 1) begin n = $fscanf(fin, "%h", w); vc[i*32 +: 32] = w; end
                // tnca_core.v r_fpctl, from the same opcode.
                fpctl = {(major == `TNCA_MAJ_RED) && (func >= `TNCA_RD_FADD),
                         (major == `TNCA_MAJ_VFP) && (func == `TNCA_VF_MUL),
                         (major == `TNCA_MAJ_VFP) && ((func == `TNCA_VF_SUB) || (func == `TNCA_VF_FMS)),
                         (func == `TNCA_VF_MUL),
                         (func == `TNCA_VF_ADD) || (func == `TNCA_VF_SUB)};
                if (((major == `TNCA_MAJ_RED) && (func >= `TNCA_RD_FADD)) || (major == `TNCA_MAJ_SFU)) begin
                    @(negedge clk); start = 1'b1;
                    @(negedge clk); start = 1'b0;
                    guard = 0;
                    while (done !== 1'b1 && guard < 1000) begin @(posedge clk); #1; guard = guard + 1; end
                    rv = mc_vres; rs = mc_sres; rp = 16'd0;
                    repeat (2) @(posedge clk);
                end else begin
                    repeat (8) @(posedge clk);
                    #1;
                    rv = vres; rs = sres; rp = pres;
                end
                for (i = 0; i < 16; i = i + 1) $fwrite(fout, "%08h ", rv[i*32 +: 32]);
                $fwrite(fout, "%04h %08h\n", rp, rs[31:0]);
            end
        end
        $fclose(fout);
        $finish;
    end
endmodule
