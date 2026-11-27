//==============================================================================
// tb_matrix_lock -- tnca_matrix at default parameters against the delivered engine, cycle for cycle
//
// R42-img: the integration of R42-mtx (OVL), R42-fp32 (MAC_F32) and R42-lineelig (ABUF 2) restructured
// rtl/core/tnca_matrix.v. Each item's default keeps the engine as it was; this bench holds that to the cycle.
// `tnca_matrix_ref` is the engine of the base commit (rtl/core/tnca_matrix.v of origin/R42 with the module
// renamed); `tnca_matrix` is the engine under test with OVL = 0 and ABUF as given (MAC_F32 may be 1: a machine
// that has the binary32 element format but is not given an m.mma.f32.async must behave as before).
// One random command list (m.zero, m.mma, m.mma.dp of every K and format, row ops, flushes), one vector-file
// model answering both from the reference's request; every output of both engines is compared in every cycle.
//
//   +seed=N  +cmds=N   VRF_MODE 0 behavioural file (valid tied high), 1 banked file (random wait)
//==============================================================================
`timescale 1ns/1ps
`include "tnca_defs.vh"

module tb_matrix_lock;
    parameter MAC_DP   = 1;
    parameter VRF_MODE = 1;
    parameter ABUF     = 1;
    parameter MAC_F32  = 0;
    localparam VW = 512;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg          flush;
    reg          cmd_valid;
    reg  [1:0]   cmd_op, cmd_k, cmd_dt, cmd_bdt, cmd_rowop;
    reg  [3:0]   cmd_slot, cmd_row;
    reg  [7:0]   cmd_va, cmd_vb;
    reg          cmd_dp, cmd_bar_en, cmd_sf_en;
    reg  [4:0]   cmd_bar, cmd_sf_bar;
    reg  [VW-1:0] cmd_vdata;
    reg  [31:0]  cmd_scale;
    reg  [15:0]  cmd_mask;
    reg  [3:0]   ard_slot, ard_row;
    reg  [VW-1:0] vmem [0:1023];

    // the vector file's answer, from the reference's request
    wire [7:0]   a_vrf_addr, b_vrf_addr;
    wire         a_vrf_req,  b_vrf_req;
    reg  [VW-1:0] vrf_data;
    reg          vrf_valid;
    reg  [VW-1:0] rsp_data;
    reg          rsp_valid, acc_p;
    integer      waitc, vseed;
    wire [VW-1:0] rd_now = vmem[{2'b00, a_vrf_addr}];
    wire rq = a_vrf_req && !rsp_valid;
    always @(*) begin
        if (VRF_MODE == 0) begin vrf_valid = 1'b1;      vrf_data = rd_now;   end
        else               begin vrf_valid = rsp_valid; vrf_data = rsp_data; end
    end
    always @(posedge clk) begin
        rsp_valid <= 1'b0;
        if (!rst_n || flush) begin
            waitc <= 0; acc_p <= 1'b0;
        end else if (rq) begin
            if (!acc_p && ($random(vseed) % 8 == 0)) begin
                acc_p <= 1'b1; waitc <= $random(vseed) & 3;
            end else if (acc_p && waitc != 0) begin
                waitc <= waitc - 1;
            end else begin
                acc_p <= 1'b0;
                rsp_valid <= 1'b1;
                rsp_data  <= rd_now;
            end
        end else acc_p <= 1'b0;
    end

    wire         a_busy, a_done, a_src_free, a_done_bar_en;
    wire [3:0]   a_done_slot;
    wire [4:0]   a_done_bar, a_src_free_bar;
    wire [VW-1:0] a_ard;
    wire         b_busy, b_done, b_src_free, b_done_bar_en;
    wire [3:0]   b_done_slot;
    wire [4:0]   b_done_bar, b_src_free_bar;
    wire [VW-1:0] b_ard;
    wire         b_pend_ok, b_slot_live, b_done_keep, b_pend_freed, b_pend_v, b_ret_v;
    wire [0:0]   b_vrf_tag, b_cur_tag, b_pend_tag, b_ret_tag;

    tnca_matrix_ref #(
        .FMA_LAT(2), .FMA_IMPL(1), .MAC_DP(MAC_DP), .MAC_CUTS(5'd29), .ABUF(ABUF == 2 ? 1 : ABUF)
    ) a (
        .clk(clk), .rst_n(rst_n), .flush(flush),
        .cmd_valid(cmd_valid), .cmd_op(cmd_op), .cmd_slot(cmd_slot),
        .cmd_va(cmd_va), .cmd_vb(cmd_vb), .cmd_k(cmd_k), .cmd_dt(cmd_dt),
        .cmd_dp(cmd_dp), .cmd_bdt(cmd_bdt),
        .cmd_bar_en(cmd_bar_en), .cmd_bar(cmd_bar),
        .cmd_sf_en(cmd_sf_en), .cmd_sf_bar(cmd_sf_bar),
        .cmd_rowop(cmd_rowop), .cmd_row(cmd_row), .cmd_vdata(cmd_vdata),
        .cmd_scale(cmd_scale), .cmd_mask(cmd_mask),
        .busy(a_busy),
        .vrf_addr(a_vrf_addr), .vrf_data(vrf_data), .vrf_req(a_vrf_req), .vrf_valid(vrf_valid),
        .ard_slot(ard_slot), .ard_row(ard_row), .ard_data(a_ard),
        .done(a_done), .done_slot(a_done_slot), .done_bar_en(a_done_bar_en), .done_bar(a_done_bar),
        .src_free(a_src_free), .src_free_bar(a_src_free_bar)
    );

    tnca_matrix #(
        .FMA_LAT(2), .FMA_IMPL(1), .MAC_DP(MAC_DP), .MAC_CUTS(5'd29), .ABUF(ABUF),
        .OVL(0), .VRF_LAT(VRF_MODE), .TAG_W(1), .MAC_F32(MAC_F32)
    ) b (
        .clk(clk), .rst_n(rst_n), .flush(flush),
        .cmd_valid(cmd_valid), .cmd_op(cmd_op), .cmd_slot(cmd_slot),
        .cmd_va(cmd_va), .cmd_vb(cmd_vb), .cmd_k(cmd_k), .cmd_dt(cmd_dt),
        .cmd_dp(cmd_dp), .cmd_bdt(cmd_bdt), .cmd_f32(1'b0),
        .cmd_bar_en(cmd_bar_en), .cmd_bar(cmd_bar),
        .cmd_sf_en(cmd_sf_en), .cmd_sf_bar(cmd_sf_bar),
        .cmd_rowop(cmd_rowop), .cmd_row(cmd_row), .cmd_vdata(cmd_vdata),
        .cmd_scale(cmd_scale), .cmd_mask(cmd_mask), .cmd_tag(1'b0),
        .busy(b_busy), .pend_ok(b_pend_ok), .cmd_slot_live(b_slot_live),
        .done_keep(b_done_keep), .pend_freed(b_pend_freed),
        .vrf_tag(b_vrf_tag), .cur_tag(b_cur_tag), .pend_v(b_pend_v), .pend_tag(b_pend_tag),
        .ret_v(b_ret_v), .ret_tag(b_ret_tag),
        .vrf_addr(b_vrf_addr), .vrf_data(vrf_data), .vrf_req(b_vrf_req), .vrf_valid(vrf_valid),
        .ard_slot(ard_slot), .ard_row(ard_row), .ard_data(b_ard),
        .done(b_done), .done_slot(b_done_slot), .done_bar_en(b_done_bar_en), .done_bar(b_done_bar),
        .src_free(b_src_free), .src_free_bar(b_src_free_bar)
    );

    //--------------------------------------------------------------------------
    // The cycle-by-cycle comparison (sampled at the falling edge: both engines settled)
    //--------------------------------------------------------------------------
    integer fails = 0, ncyc = 0, ndone = 0, nreq = 0, nmma = 0;
    always @(negedge clk) if (rst_n) begin
        ncyc = ncyc + 1;
        if (a_done) ndone = ndone + 1;
        if (a_vrf_req) nreq = nreq + 1;
        if ({a_busy, a_vrf_req, a_done, a_src_free, a_done_bar_en} !== {b_busy, b_vrf_req, b_done, b_src_free, b_done_bar_en} ||
            (a_vrf_req && (a_vrf_addr !== b_vrf_addr)) ||
            (a_done && ({a_done_slot, a_done_bar_en} !== {b_done_slot, b_done_bar_en})) ||
            (a_done && a_done_bar_en && (a_done_bar !== b_done_bar)) ||
            (a_src_free && (a_src_free_bar !== b_src_free_bar)) ||
            (a_ard !== b_ard) ||
            b_pend_ok || b_slot_live || b_done_keep || b_pend_freed || b_pend_v || b_ret_v) begin
            if (fails < 10)
                $display("FAIL @cycle %0d: busy %b/%b req %b/%b addr %h/%h done %b/%b slot %h/%h src_free %b/%b ard %s pend %b%b%b%b%b%b",
                         ncyc, a_busy, b_busy, a_vrf_req, b_vrf_req, a_vrf_addr, b_vrf_addr, a_done, b_done,
                         a_done_slot, b_done_slot, a_src_free, b_src_free, (a_ard === b_ard) ? "same" : "differs",
                         b_pend_ok, b_slot_live, b_done_keep, b_pend_freed, b_pend_v, b_ret_v);
            fails = fails + 1;
        end
    end

    //--------------------------------------------------------------------------
    // The driver
    //--------------------------------------------------------------------------
    integer seed, ncmd, ci, gap, j, b8;
    reg [31:0] rnd;
    function [31:0] f32r(input integer dummy);
        reg [31:0] x;
        begin x = $random(seed); f32r = {x[31], 4'b0111, x[26:0]}; end
    endfunction

    task issue(input [1:0] op, input [3:0] slot, input [1:0] k, input [1:0] dt, input dp, input [1:0] bdt);
        integer q;
        begin
            @(negedge clk);
            cmd_valid = 1'b1; cmd_op = op; cmd_slot = slot; cmd_k = k; cmd_dt = dt;
            cmd_dp = (MAC_DP != 0) && dp; cmd_bdt = bdt;
            rnd = $random(seed); cmd_va = rnd[7:0] % 8'd220;
            rnd = $random(seed); cmd_vb = rnd[7:0] % 8'd220;
            rnd = $random(seed);
            cmd_bar = rnd[4:0]; cmd_bar_en = (op == 2'd1); cmd_sf_en = (op == 2'd1) && rnd[5]; cmd_sf_bar = rnd[10:6];
            cmd_rowop = rnd[12:11] % 3; cmd_row = rnd[16:13];
            cmd_mask = rnd[19] ? 16'hffff : $random(seed);
            cmd_scale = f32r(0);
            for (q = 0; q < 16; q = q + 1) cmd_vdata[q*32 +: 32] = f32r(0);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    integer guard;
    task wait_idle;
        begin
            guard = 0;
            @(negedge clk);
            while (a_busy || b_busy) begin
                @(negedge clk); guard = guard + 1;
                if (guard > 400000) begin $display("FAIL: hung"); $finish; end
            end
        end
    endtask

    integer k0, dt0, fl_at;
    initial begin
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        if (!$value$plusargs("cmds=%d", ncmd)) ncmd = 400;
        vseed = 32'h4321 + seed;
        flush = 0; cmd_valid = 0; ard_slot = 0; ard_row = 0; rsp_valid = 0; acc_p = 0; waitc = 0;
        cmd_op = 0; cmd_k = 0; cmd_dt = 0; cmd_bdt = 0; cmd_rowop = 0; cmd_slot = 0; cmd_row = 0; cmd_va = 0; cmd_vb = 0;
        cmd_dp = 0; cmd_bar_en = 0; cmd_sf_en = 0; cmd_bar = 0; cmd_sf_bar = 0; cmd_vdata = 0; cmd_scale = 0; cmd_mask = 0;
        // every byte has bits [6:5] = 01 or 10: finite, normal operands in FP16, BF16, E5M2 and E4M3 alike
        for (j = 0; j < 1024; j = j + 1)
            for (b8 = 0; b8 < 64; b8 = b8 + 1) begin
                rnd = $random(seed);
                vmem[j][b8*8 +: 8] = {rnd[7], rnd[6], ~rnd[6], rnd[4:0]};
            end
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);
        for (j = 0; j < 6; j = j + 1) begin issue(2'd0, j, 0, 0, 0, 0); wait_idle; end
        for (ci = 0; ci < ncmd; ci = ci + 1) begin
            rnd = $random(seed);
            k0  = (rnd[3:0] < 9) ? 0 : (rnd[3:0] < 14) ? 1 : 2;
            dt0 = rnd[5:4];
            // a read of the accumulator file between commands, at a rotating address
            ard_slot = rnd[27:24] % 6; ard_row = rnd[31:28];
            if (rnd[9:6] < 11)      begin issue(2'd1, rnd[11:10], k0, dt0, rnd[12], rnd[14:13]); nmma = nmma + 1; end
            else if (rnd[9:6] < 13)       issue(2'd0, rnd[11:10], 0, 0, 0, 0);
            else                          issue(2'd2, rnd[11:10], 0, 0, 0, 0);
            if (rnd[23:20] == 0) begin
                // a flush a few cycles into the command (the engine is left however it was)
                rnd = $random(seed);
                repeat (rnd[7:0] % 40) @(negedge clk);
                flush = 1'b1; @(negedge clk); flush = 1'b0;
            end
            wait_idle;
            if (ci % 40 == 39) begin
                for (j = 0; j < 6; j = j + 1) begin issue(2'd0, j, 0, 0, 0, 0); wait_idle; end
            end
        end
        repeat (4) @(negedge clk);
        if (ndone < ncmd / 2 || nmma < ncmd / 4) begin $display("FAIL: too few events (%0d done, %0d m.mma)", ndone, nmma); fails = fails + 1; end
        if (fails == 0) $display("MATRIX LOCK PASS MAC_DP=%0d VRF_MODE=%0d ABUF=%0d MAC_F32=%0d seed=%0d: %0d cycles identical, %0d commands (%0d m.mma), %0d done events, %0d vector reads",
                                 MAC_DP, VRF_MODE, ABUF, MAC_F32, seed, ncyc, ncmd, nmma, ndone, nreq);
        else            $display("MATRIX LOCK FAIL: %0d cycles differ", fails);
        $finish;
    end
endmodule
