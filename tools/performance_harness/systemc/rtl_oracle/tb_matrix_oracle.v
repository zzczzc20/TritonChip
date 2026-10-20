// Offline oracle capture for tnca_matrix (MMA in every dtype/K, accumulator
// row operations). A behavioural vector file answers the engine's reads with
// vrf_valid tied high, as the core's behavioural file does. Simulation-only;
// never part of the TLM model's execution path.
`timescale 1ns/1ps
`include "tnca_defs.vh"
module tb_matrix_oracle;
    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;
    reg         cmd_valid = 1'b0;
    reg  [1:0]  cmd_op, cmd_k, cmd_dt, cmd_rowop;
    reg  [3:0]  cmd_slot, cmd_row;
    reg  [7:0]  cmd_va, cmd_vb;
    reg  [511:0] cmd_vdata;
    reg  [31:0] cmd_scale;
    reg  [15:0] cmd_mask;
    reg  [3:0]  ard_slot, ard_row;
    wire        busy, vrf_req, done, done_bar_en, src_free;
    wire [7:0]  vrf_addr;
    wire [511:0] ard_data;
    wire [3:0]  done_slot;
    wire [4:0]  done_bar, src_free_bar;
    reg  [511:0] vrf [0:255];
    tnca_matrix #(.ABUF(`ORACLE_ABUF)) dut (
        .clk(clk), .rst_n(rst_n), .flush(1'b0),
        .cmd_valid(cmd_valid), .cmd_op(cmd_op), .cmd_slot(cmd_slot), .cmd_va(cmd_va), .cmd_vb(cmd_vb),
        .cmd_k(cmd_k), .cmd_dt(cmd_dt), .cmd_bar_en(1'b0), .cmd_bar(5'd0), .cmd_sf_en(1'b0), .cmd_sf_bar(5'd0),
        .cmd_rowop(cmd_rowop), .cmd_row(cmd_row), .cmd_vdata(cmd_vdata), .cmd_scale(cmd_scale), .cmd_mask(cmd_mask),
        .busy(busy), .vrf_addr(vrf_addr), .vrf_data(vrf[vrf_addr]), .vrf_req(vrf_req), .vrf_valid(1'b1),
        .ard_slot(ard_slot), .ard_row(ard_row), .ard_data(ard_data),
        .done(done), .done_slot(done_slot), .done_bar_en(done_bar_en), .done_bar(done_bar),
        .src_free(src_free), .src_free_bar(src_free_bar));
    integer fin, fout, n, i, r, guard, cycles;
    reg [31:0] w, idx;
    reg [8*8-1:0] kind;
    reg [8*512-1:0] in_path, out_path;
    initial begin
        if (!$value$plusargs("in=%s", in_path)) begin $display("FATAL +in"); $finish; end
        if (!$value$plusargs("out=%s", out_path)) begin $display("FATAL +out"); $finish; end
        fin = $fopen(in_path, "r"); fout = $fopen(out_path, "w");
        for (i = 0; i < 256; i = i + 1) vrf[i] = 512'd0;
        cmd_op = 0; cmd_k = 0; cmd_dt = 0; cmd_rowop = 0; cmd_slot = 0; cmd_row = 0; cmd_va = 0; cmd_vb = 0;
        cmd_vdata = 0; cmd_scale = 0; cmd_mask = 0; ard_slot = 0; ard_row = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        while (!$feof(fin)) begin
            n = $fscanf(fin, "%s", kind);
            if (n == 1) begin
                if (kind == "V") begin
                    n = $fscanf(fin, "%h", idx);
                    for (i = 0; i < 16; i = i + 1) begin n = $fscanf(fin, "%h", w); vrf[idx][i*32 +: 32] = w; end
                end else if (kind == "C") begin
                    n = $fscanf(fin, "%h %h %h %h %h %h %h %h %h %h %h", cmd_op, cmd_slot, cmd_va, cmd_vb, cmd_k, cmd_dt,
                                cmd_rowop, cmd_row, cmd_scale, cmd_mask, w);
                    for (i = 0; i < 16; i = i + 1) begin n = $fscanf(fin, "%h", w); cmd_vdata[i*32 +: 32] = w; end
                    @(negedge clk); cmd_valid = 1'b1;
                    @(negedge clk); cmd_valid = 1'b0;
                    cycles = 1; guard = 0;
                    while (busy === 1'b1 && guard < 100000) begin @(negedge clk); cycles = cycles + 1; guard = guard + 1; end
                    $fwrite(fout, "C %0d\n", cycles);
                end else if (kind == "R") begin
                    n = $fscanf(fin, "%h", idx);
                    ard_slot = idx[3:0];
                    $fwrite(fout, "R %0d", idx);
                    for (r = 0; r < 16; r = r + 1) begin
                        ard_row = r; #1;
                        for (i = 0; i < 16; i = i + 1) $fwrite(fout, " %08h", ard_data[i*32 +: 32]);
                    end
                    $fwrite(fout, "\n");
                end
            end
        end
        $fclose(fout);
        $finish;
    end
endmodule
