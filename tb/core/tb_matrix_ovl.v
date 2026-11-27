//==============================================================================
// tb_matrix_ovl -- tnca_matrix OVL=1 (R42, tnca_core MTX_PIPE) against OVL=0
//
// The overlapped sequencer is scheduling only, so the reference is the
// engine it replaces. Both get the same random command list -- m.mma and
// m.mma.dp of every K and format on the same and on different slots, row
// ops and m.zero between them, random contexts -- each from a driver that
// issues the way tnca_core does (a command when the engine is free; an
// m.mma also when pend_ok), against a vector file that answers after a
// random wait. They run at their own speed. Compared, cycle-independently:
//   * the accumulator file, read back through the a.vread port whenever
//     both sides are idle at the end of a segment;
//   * the order and content of the done and source-free events;
//   * done_keep against the m.mma commands still outstanding on the slot.
// Some segments are cut by a flush at a random cycle; the slots are then
// zeroed and the comparison resumes.
//
//   +seed=N  +segs=N   VRF_MODE 0 valid tied high (behavioural file),
//                               1 banked file, 2 banked + registered response
//==============================================================================
`timescale 1ns/1ps
`include "tnca_defs.vh"

module tb_mtx_side #(
    parameter OVL = 0, parameter MAC_DP = 1, parameter VRF_MODE = 1, parameter SIDE = 0
) ();
    localparam VW = 512;
    wire clk   = tb_matrix_ovl.clk;
    wire rst_n = tb_matrix_ovl.rst_n;

    reg          flush;
    reg          cmd_valid;
    reg  [1:0]   cmd_op, cmd_k, cmd_dt, cmd_bdt, cmd_rowop, cmd_tag;
    reg  [3:0]   cmd_slot, cmd_row;
    reg  [7:0]   cmd_va, cmd_vb;
    reg          cmd_dp, cmd_bar_en, cmd_sf_en;
    reg  [4:0]   cmd_bar, cmd_sf_bar;
    reg  [VW-1:0] cmd_vdata;
    reg  [31:0]  cmd_scale;
    reg  [15:0]  cmd_mask;
    wire         busy, pend_ok, slot_live, done_keep, pend_freed, pend_v, ret_v;
    wire [1:0]   vrf_tag, cur_tag, pend_tag, ret_tag;
    wire [7:0]   vrf_addr;
    wire         vrf_req;
    reg  [VW-1:0] vrf_data;
    reg          vrf_valid;
    reg  [3:0]   ard_slot, ard_row;
    wire [VW-1:0] ard_data;
    wire         done, done_bar_en, src_free;
    wire [3:0]   done_slot;
    wire [4:0]   done_bar, src_free_bar;

    tnca_matrix #(
        .FMA_LAT(2), .FMA_IMPL(1), .MAC_DP(MAC_DP), .MAC_CUTS(5'd29), .ABUF(1),
        .OVL(OVL), .VRF_LAT(VRF_MODE), .TAG_W(2)
    ) u (
        .clk(clk), .rst_n(rst_n), .flush(flush),
        .cmd_valid(cmd_valid), .cmd_op(cmd_op), .cmd_slot(cmd_slot),
        .cmd_va(cmd_va), .cmd_vb(cmd_vb), .cmd_k(cmd_k), .cmd_dt(cmd_dt),
        .cmd_dp(cmd_dp), .cmd_bdt(cmd_bdt),
        .cmd_bar_en(cmd_bar_en), .cmd_bar(cmd_bar),
        .cmd_sf_en(cmd_sf_en), .cmd_sf_bar(cmd_sf_bar),
        .cmd_rowop(cmd_rowop), .cmd_row(cmd_row), .cmd_vdata(cmd_vdata),
        .cmd_scale(cmd_scale), .cmd_mask(cmd_mask), .cmd_tag(cmd_tag),
        .busy(busy), .pend_ok(pend_ok), .cmd_slot_live(slot_live),
        .done_keep(done_keep), .pend_freed(pend_freed),
        .vrf_tag(vrf_tag), .cur_tag(cur_tag), .pend_v(pend_v), .pend_tag(pend_tag),
        .ret_v(ret_v), .ret_tag(ret_tag),
        .vrf_addr(vrf_addr), .vrf_data(vrf_data), .vrf_req(vrf_req), .vrf_valid(vrf_valid),
        .ard_slot(ard_slot), .ard_row(ard_row), .ard_data(ard_data),
        .done(done), .done_slot(done_slot), .done_bar_en(done_bar_en), .done_bar(done_bar),
        .src_free(src_free), .src_free_bar(src_free_bar)
    );

    //--------------------------------------------------------------------------
    // The vector file. The reference engine reads in the context of the one
    // command it runs (tnca_core's mtx_rctx); the overlapped one says whose
    // read it is.
    //--------------------------------------------------------------------------
    reg  [1:0] own_tag;
    wire [1:0] rd_tag = (OVL != 0) ? vrf_tag : own_tag;
    wire [VW-1:0] rd_now = tb_matrix_ovl.vmem[{rd_tag, vrf_addr}];
    reg  [VW-1:0] rsp_data, q_data;
    reg        rsp_valid, q_valid, acc_p;
    integer    waitc, vseed;
    wire rq = vrf_req && !rsp_valid && !((VRF_MODE == 2) && q_valid);
    always @(*) begin
        if (VRF_MODE == 0)      begin vrf_valid = 1'b1;      vrf_data = rd_now;   end
        else if (VRF_MODE == 1) begin vrf_valid = rsp_valid; vrf_data = rsp_data; end
        else                    begin vrf_valid = q_valid;   vrf_data = q_data;   end
    end
    always @(posedge clk) begin
        rsp_valid <= 1'b0;
        q_valid   <= rst_n && !flush && rsp_valid;
        if (rsp_valid) q_data <= rsp_data;
        if (!rst_n) begin
            waitc <= 0; acc_p <= 1'b0;
        end else if (rq) begin
            // a bank that is busy with another reader accepts later
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

    //--------------------------------------------------------------------------
    // The driver: tnca_core's issue rule.
    //--------------------------------------------------------------------------
    integer ci, seg_end, gapc, fl_at, fl_cnt, ncyc, i;
    reg     fl_arm, fl_run, seg_go;
    integer cnt [0:15];
    integer ev_n;
    reg [15:0] ev [0:65535];
    integer t_first, t_last;       // first issue and last done of the segment
    wire is_mma = (tb_matrix_ovl.c_op[ci] == 2'd1);
    wire can = seg_go && (ci < seg_end) && !cmd_valid && (gapc == 0)
               && (!busy || (is_mma && pend_ok));
    wire side_idle = seg_go && (ci >= seg_end) && !busy && !cmd_valid && !fl_run && !flush;

    always @(posedge clk) begin
        ncyc <= ncyc + 1;
        flush <= 1'b0;
        cmd_valid <= 1'b0;
        if (!rst_n) begin
            gapc <= 0; fl_run <= 1'b0; ev_n <= 0; own_tag <= 2'd0;
            for (i = 0; i < 16; i = i + 1) cnt[i] <= 0;
        end else begin
            if (gapc != 0) gapc <= gapc - 1;
            if (can) begin
                cmd_valid  <= 1'b1;
                cmd_op     <= tb_matrix_ovl.c_op[ci];
                cmd_slot   <= tb_matrix_ovl.c_slot[ci];
                cmd_va     <= tb_matrix_ovl.c_va[ci];
                cmd_vb     <= tb_matrix_ovl.c_vb[ci];
                cmd_k      <= tb_matrix_ovl.c_k[ci];
                cmd_dt     <= tb_matrix_ovl.c_dt[ci];
                cmd_dp     <= tb_matrix_ovl.c_dp[ci];
                cmd_bdt    <= tb_matrix_ovl.c_bdt[ci];
                cmd_bar_en <= (tb_matrix_ovl.c_op[ci] == 2'd1);
                cmd_bar    <= tb_matrix_ovl.c_bar[ci];
                cmd_sf_en  <= (tb_matrix_ovl.c_op[ci] == 2'd1) && tb_matrix_ovl.c_sf[ci];
                cmd_sf_bar <= tb_matrix_ovl.c_sfbar[ci];
                cmd_rowop  <= tb_matrix_ovl.c_rowop[ci];
                cmd_row    <= tb_matrix_ovl.c_row[ci];
                cmd_vdata  <= tb_matrix_ovl.c_vdata[ci];
                cmd_scale  <= tb_matrix_ovl.c_scale[ci];
                cmd_mask   <= tb_matrix_ovl.c_mask[ci];
                cmd_tag    <= tb_matrix_ovl.c_tag[ci];
                gapc       <= tb_matrix_ovl.c_gap[ci + 1];
                if (fl_arm && ci == fl_at) begin fl_run <= 1'b1; end
                ci <= ci + 1;
            end
            if (cmd_valid) begin
                if (t_first < 0) t_first <= ncyc;
                own_tag <= cmd_tag;
                if (cmd_op == 2'd1) cnt[cmd_slot] <= cnt[cmd_slot] + 1;
                if (busy && !(OVL != 0)) begin
                    $display("FAIL side %0d: driver issued into a busy reference engine", SIDE);
                    tb_matrix_ovl.fails = tb_matrix_ovl.fails + 1;
                end
            end
            if (fl_run) begin
                if (fl_cnt == 0) begin
                    flush <= 1'b1; fl_run <= 1'b0; fl_arm <= 1'b0; ci <= seg_end; gapc <= 0;
                    cmd_valid <= 1'b0;
                    for (i = 0; i < 16; i = i + 1) cnt[i] <= 0;
                end else fl_cnt <= fl_cnt - 1;
            end
            if (src_free && !flush) begin
                ev[ev_n] <= {1'b0, 4'd0, 1'b1, src_free_bar, 5'd0}; ev_n <= ev_n + 1;
            end
            if (done && !flush) begin
                t_last <= ncyc;
                ev[ev_n + ((src_free) ? 1 : 0)] <= {1'b1, done_slot, done_bar_en,
                                                    done_bar_en ? done_bar : 5'd0, 5'd0};
                ev_n <= ev_n + 1 + ((src_free) ? 1 : 0);
                if (done_bar_en) begin
                    cnt[done_slot] <= cnt[done_slot] - 1
                        + ((cmd_valid && cmd_op == 2'd1 && cmd_slot == done_slot) ? 1 : 0);
                    // done_keep: another accepted m.mma still writes the slot.
                    // The command presented this very cycle is the core's to see.
                    if ((OVL != 0) && (done_keep !== (cnt[done_slot] > 1))) begin
                        $display("FAIL side %0d @%0d: done slot %0d keep %b, outstanding %0d",
                                 SIDE, ncyc, done_slot, done_keep, cnt[done_slot]);
                        tb_matrix_ovl.fails = tb_matrix_ovl.fails + 1;
                    end
                end
                if ((OVL == 0 || !done_bar_en) && done_keep !== 1'b0) begin
                    $display("FAIL side %0d: done_keep set where it cannot be", SIDE);
                    tb_matrix_ovl.fails = tb_matrix_ovl.fails + 1;
                end
            end
        end
    end

    initial begin
        flush = 0; cmd_valid = 0; ci = 0; seg_end = 0; gapc = 0; fl_arm = 0; fl_run = 0;
        fl_at = 0; fl_cnt = 0; ncyc = 0; seg_go = 0; ev_n = 0; ard_slot = 0; ard_row = 0;
        t_first = -1; t_last = 0; vseed = 32'h1234 + SIDE; rsp_valid = 0; q_valid = 0; acc_p = 0;
        waitc = 0; own_tag = 0;
    end
endmodule

module tb_matrix_ovl;
    parameter MAC_DP   = 1;
    parameter VRF_MODE = 1;
    localparam VW = 512;
    localparam NC = 8192;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    integer fails = 0;
    reg [VW-1:0] vmem [0:1023];

    reg [1:0]    c_op [0:NC], c_k [0:NC], c_dt [0:NC], c_bdt [0:NC], c_rowop [0:NC], c_tag [0:NC];
    reg [3:0]    c_slot [0:NC], c_row [0:NC];
    reg [7:0]    c_va [0:NC], c_vb [0:NC];
    reg          c_dp [0:NC], c_sf [0:NC];
    reg [4:0]    c_bar [0:NC], c_sfbar [0:NC];
    reg [VW-1:0] c_vdata [0:NC];
    reg [31:0]   c_scale [0:NC];
    reg [15:0]   c_mask [0:NC];
    integer      c_gap [0:NC];

    tb_mtx_side #(.OVL(0), .MAC_DP(MAC_DP), .VRF_MODE(VRF_MODE), .SIDE(0)) s0 ();
    tb_mtx_side #(.OVL(1), .MAC_DP(MAC_DP), .VRF_MODE(VRF_MODE), .SIDE(1)) s1 ();

    integer seed, nseg, nc, sg, n, j, b, w, r, sl, a0, a1, tot0, tot1, mmas;
    reg [31:0] rnd;

    // a float32 of moderate size
    function [31:0] f32r(input integer dummy);
        reg [31:0] x;
        begin x = $random(seed); f32r = {x[31], 4'b0111, x[26:0]}; end
    endfunction

    task add_cmd(input [1:0] op, input [3:0] slot, input [1:0] k, input [1:0] dt,
                 input dp, input [1:0] bdt, input integer gap);
        integer q;
        begin
            c_op[nc] = op; c_slot[nc] = slot; c_k[nc] = k; c_dt[nc] = dt;
            c_dp[nc] = (MAC_DP != 0) && dp; c_bdt[nc] = bdt;
            rnd = $random(seed); c_va[nc] = rnd[7:0] % 8'd220;
            rnd = $random(seed); c_vb[nc] = rnd[7:0] % 8'd220;
            rnd = $random(seed);
            c_bar[nc] = rnd[4:0]; c_sf[nc] = rnd[5]; c_sfbar[nc] = rnd[10:6];
            c_rowop[nc] = rnd[12:11] % 3; c_row[nc] = rnd[16:13]; c_tag[nc] = rnd[18:17];
            c_mask[nc] = rnd[19] ? 16'hffff : $random(seed);
            c_scale[nc] = f32r(0);
            for (q = 0; q < 16; q = q + 1) c_vdata[nc][q*32 +: 32] = f32r(0);
            c_gap[nc] = gap;
            nc = nc + 1;
        end
    endtask

    task run_seg(input integer first, input integer last, input integer flush_at,
                 input integer flush_cnt, input integer check, input integer verbose);
        integer guard, e;
        begin
            @(negedge clk);
            s0.ci = first; s0.seg_end = last; s0.gapc = c_gap[first];
            s1.ci = first; s1.seg_end = last; s1.gapc = c_gap[first];
            s0.fl_arm = (flush_at >= 0); s0.fl_at = flush_at; s0.fl_cnt = flush_cnt;
            s1.fl_arm = (flush_at >= 0); s1.fl_at = flush_at; s1.fl_cnt = flush_cnt;
            s0.ev_n = 0; s1.ev_n = 0; s0.t_first = -1; s1.t_first = -1;
            s0.seg_go = 1; s1.seg_go = 1;
            a0 = s0.ncyc; a1 = s1.ncyc;
            guard = 0;
            while (!(s0.side_idle && s1.side_idle)) begin
                @(negedge clk);
                guard = guard + 1;
                if (guard > 400000) begin
                    $display("FAIL: segment %0d..%0d hung (idle %b %b, ci %0d %0d, st %0d %0d)",
                             first, last, s0.side_idle, s1.side_idle, s0.ci, s1.ci, s0.u.st, s1.u.st);
                    $finish;
                end
            end
            repeat (2) @(negedge clk);
            s0.seg_go = 0; s1.seg_go = 0;
            if (verbose)
                $display("  cycles first issue -> last done: OVL=0 %0d   OVL=1 %0d",
                         s0.t_last - s0.t_first + 1, s1.t_last - s1.t_first + 1);
            if (check) begin
                tot0 = tot0 + (s0.t_last - s0.t_first + 1);
                tot1 = tot1 + (s1.t_last - s1.t_first + 1);
                if (s0.ev_n !== s1.ev_n) begin
                    $display("FAIL: segment %0d..%0d: %0d events against %0d", first, last, s0.ev_n, s1.ev_n);
                    fails = fails + 1;
                end else for (e = 0; e < s0.ev_n; e = e + 1)
                    if (s0.ev[e] !== s1.ev[e]) begin
                        $display("FAIL: segment %0d..%0d event %0d: %h against %h", first, last, e, s0.ev[e], s1.ev[e]);
                        fails = fails + 1;
                    end
            end
            // the accumulator file, slots 0..5 (the list uses 0..3)
            for (sl = 0; sl < 6; sl = sl + 1)
                for (r = 0; r < 16; r = r + 1) begin
                    s0.ard_slot = sl; s0.ard_row = r; s1.ard_slot = sl; s1.ard_row = r;
                    #1;
                    if (check && (s0.ard_data !== s1.ard_data)) begin
                        $display("FAIL: after segment %0d..%0d accumulator %0d row %0d differs", first, last, sl, r);
                        $display("   OVL=0 %h", s0.ard_data);
                        $display("   OVL=1 %h", s1.ard_data);
                        fails = fails + 1;
                    end
                    if (^s1.ard_data === 1'bx) begin
                        $display("FAIL: accumulator %0d row %0d has X", sl, r); fails = fails + 1;
                    end
                end
            if (fails > 20) begin $display("FAIL: too many errors"); $finish; end
        end
    endtask

    integer first, fl, k0, dt0;
    initial begin
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        if (!$value$plusargs("segs=%d", nseg)) nseg = 60;
        // Every byte has bits [6:5] = 01 or 10, so no format's exponent is
        // all ones or all zeros: finite, normal operands in FP16, BF16,
        // E5M2 and E4M3 alike.
        for (j = 0; j < 1024; j = j + 1)
            for (b = 0; b < 64; b = b + 1) begin
                rnd = $random(seed);
                vmem[j][b*8 +: 8] = {rnd[7], rnd[6], ~rnd[6], rnd[4:0]};
            end
        nc = 0; tot0 = 0; tot1 = 0; mmas = 0;
        c_gap[0] = 0;
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        // ---- directed: zero the slots, then single commands and trains ----
        first = nc; for (j = 0; j < 6; j = j + 1) add_cmd(2'd0, j, 0, 0, 0, 0, 0);
        run_seg(first, nc, -1, 0, 1, 0);
        $display("one m.mma.async k16 f16:");
        first = nc; add_cmd(2'd1, 4'd0, `TNCA_MMA_K16, `TNCA_MMA_F16, 1'b0, 2'd0, 0);
        run_seg(first, nc, -1, 0, 1, 1);
        if (MAC_DP != 0) begin
            $display("one m.mma.dp.async k16 f16 x f16:");
            first = nc; add_cmd(2'd1, 4'd0, `TNCA_MMA_K16, `TNCA_MMA_F16, 1'b1, `TNCA_MMA_F16, 0);
            run_seg(first, nc, -1, 0, 1, 1);
            $display("one m.mma.dp.async k32 f16 x f16:");
            first = nc; add_cmd(2'd1, 4'd0, `TNCA_MMA_K32, `TNCA_MMA_F16, 1'b1, `TNCA_MMA_F16, 0);
            run_seg(first, nc, -1, 0, 1, 1);
            $display("one m.mma.dp.async k16 e4m3 x e4m3:");
            first = nc; add_cmd(2'd1, 4'd1, `TNCA_MMA_K16, `TNCA_MMA_E4M3, 1'b1, `TNCA_MMA_E4M3, 0);
            run_seg(first, nc, -1, 0, 1, 1);
            $display("eight m.mma.dp.async k16 f16 x f16 back to back, one slot:");
            first = nc; for (j = 0; j < 8; j = j + 1) add_cmd(2'd1, 4'd2, `TNCA_MMA_K16, `TNCA_MMA_F16, 1'b1, `TNCA_MMA_F16, 0);
            run_seg(first, nc, -1, 0, 1, 1);
        end
        $display("eight m.mma.async k16 f16 back to back, one slot:");
        first = nc; for (j = 0; j < 8; j = j + 1) add_cmd(2'd1, 4'd3, `TNCA_MMA_K16, `TNCA_MMA_F16, 1'b0, 2'd0, 0);
        run_seg(first, nc, -1, 0, 1, 1);
        $display("eight m.mma.async k16 f16 back to back, two slots:");
        first = nc; for (j = 0; j < 8; j = j + 1) add_cmd(2'd1, j[0] ? 4'd1 : 4'd3, `TNCA_MMA_K16, `TNCA_MMA_F16, 1'b0, 2'd0, 0);
        run_seg(first, nc, -1, 0, 1, 1);
        tot0 = 0; tot1 = 0;

        // ---- random segments ----
        for (sg = 0; sg < nseg; sg = sg + 1) begin
            first = nc;
            rnd = $random(seed); n = 1 + rnd[3:0] % 12;
            rnd = $random(seed); fl = (rnd[2:0] == 0);
            for (j = 0; j < n; j = j + 1) begin
                rnd = $random(seed);
                // mostly m.mma; K64 seldom (long); gaps seldom
                k0  = (rnd[3:0] < 9) ? 0 : (rnd[3:0] < 14) ? 1 : 2;
                dt0 = rnd[5:4];
                if (rnd[9:6] < 11) begin
                    add_cmd(2'd1, rnd[11:10], k0, dt0, rnd[12], rnd[14:13],
                            rnd[15] ? rnd[24:16] % 330 : 0);
                    mmas = mmas + 1;
                end else if (rnd[9:6] < 13)
                    add_cmd(2'd0, rnd[11:10], 0, 0, 0, 0, (rnd[17:15] == 0) ? rnd[20:18] : 0);
                else
                    add_cmd(2'd2, rnd[11:10], 0, 0, 0, 0, (rnd[17:15] == 0) ? rnd[20:18] : 0);
            end
            if (fl) begin
                rnd = $random(seed);
                run_seg(first, nc, first + rnd[7:0] % n, rnd[17:8] % 700, 0, 0);
                first = nc; for (j = 0; j < 6; j = j + 1) add_cmd(2'd0, j, 0, 0, 0, 0, 0);
                run_seg(first, nc, -1, 0, 0, 0);
                // the file is zero on both sides from here: compare it
                first = nc; add_cmd(2'd0, 4'd0, 0, 0, 0, 0, 0);
                run_seg(first, nc, -1, 0, 1, 0);
            end else
                run_seg(first, nc, -1, 0, 1, 0);
            if (nc > NC - 40) sg = nseg;
        end
        $display("random: %0d commands (%0d m.mma), busy cycles OVL=0 %0d, OVL=1 %0d (%0d.%0d %%)",
                 nc, mmas, tot0, tot1, tot1 * 100 / tot0, (tot1 * 1000 / tot0) % 10);
        if (fails == 0) $display("PASS");
        else            $display("FAIL: %0d", fails);
        $finish;
    end
endmodule
