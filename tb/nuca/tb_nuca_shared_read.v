//==============================================================================
// tb_nuca_shared_read -- lockstep equivalence of SHARED_DATA_READ (R41 G2)
//
// Two tnca_nuca_slice instances with the run-1 image's slice parameters, one
// with SHARED_DATA_READ=0 (the delivered build) and one with SHARED_MODE (1 or 2). Both receive
// the same inputs every cycle; the reference instance drives the DRAM model
// and the stimulus. Every output of the two instances is compared on every
// cycle, so any difference in a value read from the line array -- a hit, a
// write-back line, an atomic's old word, a woken target's data -- shows up as
// a mismatch the cycle it happens.
//
// Stimulus: random loads, stores (random byte enables) and every atomic
// opcode over a footprint four times the slice capacity, so lookups hit, miss,
// merge, evict dirty lines and wake chained targets; random response
// back-pressure; an out-of-order, stalling DRAM model; and, between drained
// phases, ownership maintenance over random line ranges.
//
//   +seed=N  +cycles=N  (defaults 1, 200000)
//==============================================================================
`timescale 1ns/1ps
`include "tnca_defs.vh"

module tb_nuca_shared_read;
    parameter CACHE_DATA_PIPELINE = 1;
    parameter SETS = 32, SET_W = 5, WAYS = 8, WAY_W = 3;
    parameter POLICY = 0;
    parameter SHARED_MODE = 1;     // the second instance's SHARED_DATA_READ (1 or 2)

    localparam MSHR_N = 8, MSHR_W = 3, TARGET_N = 16, TARGET_W = 4;
    localparam RQ_N = 8, RQ_W = 3, SRC_W = 4, TAG_W = 5;
    localparam N_LINES = SETS * WAYS;
    localparam FOOT = N_LINES * 4;              // footprint, in lines

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ---- shared inputs --------------------------------------------------
    reg              req_valid;
    reg  [63:0]      req_addr;
    reg              req_we, req_amo;
    reg  [7:0]       req_be;
    reg  [63:0]      req_wdata, req_cmp;
    reg  [3:0]       req_aop;
    reg  [SRC_W-1:0] req_src;
    reg  [TAG_W-1:0] req_tag;
    reg              rsp_ready;
    reg              maint_valid;
    reg  [63:0]      maint_base, maint_size;
    reg              tl_ready;
    reg  [15:0]      dco_slow;

    wire             m_req_ready, m_rsp_valid;
    wire [MSHR_W-1:0] m_rsp_tag;
    wire [511:0]     m_rsp_rdata;

    wire [75:0] dco_cfg = {16'd2, 16'd8, 16'd64, 4'd0, 6'd0, 6'd0, 4'd0, POLICY[7:0]};

    // ---- outputs of both instances, as one vector each ------------------
    localparam OUT_W = 1+1+1+1 + 1+SRC_W+TAG_W+64 + 1+64+1+512+MSHR_W
                     + 1+12+64+SRC_W+12 + 1+64+1 + 32*10+4;   // ten 32-bit counters and the gear
    wire [OUT_W-1:0] out_ref, out_new;

    genvar d;
    generate for (d = 0; d < 2; d = d + 1) begin : g_dut
        wire maint_ready, maint_done, maint_busy, req_ready;
        wire rsp_valid; wire [SRC_W-1:0] rsp_src; wire [TAG_W-1:0] rsp_tag;
        wire [63:0] rsp_rdata;
        wire mem_req_valid, mem_req_we; wire [63:0] mem_req_addr;
        wire [511:0] mem_req_wdata; wire [MSHR_W-1:0] mem_req_tag;
        wire tmu_wr_valid; wire [11:0] tmu_wr_off, tmu_rd_off;
        wire [63:0] tmu_wr_data; wire [SRC_W-1:0] tmu_wr_src;
        wire tmu_tl_valid, tmu_tl_drop; wire [63:0] tmu_tl_addr;
        wire [31:0] p_byp, p_ev, p_dev, p_hit, p_miss, p_merge, p_mf, p_tf, p_wb, p_hpb;
        wire [3:0] p_gear;
        tnca_nuca_slice #(
            .REGISTER_VICTIM_OWNER(1), .REGISTER_VICTIM_SELECT(1),
            .CACHE_DATA_PIPELINE(CACHE_DATA_PIPELINE), .SHARED_DATA_READ(d ? SHARED_MODE : 0),
            .REGISTER_VICTIM_RANK(1), .MAINTENANCE(1), .DCO_REGISTERED_RANGES(1),
            .FULL_LINE_TAG(1),
            .ADDR_W(64), .LINE_BYTES(64), .LINE_BITS(512),
            .WORD_BITS(64), .WORD_BYTES(8),
            .SETS(SETS), .SET_W(SET_W), .WAYS(WAYS), .WAY_W(WAY_W),
            .MSHR_N(MSHR_N), .MSHR_W(MSHR_W), .TARGET_N(TARGET_N), .TARGET_W(TARGET_W),
            .RQ_N(RQ_N), .RQ_W(RQ_W), .SRC_W(SRC_W), .TAG_W(TAG_W), .DCO_T(2)
        ) dut (
            .clk(clk), .rst_n(rst_n),
            .maint_valid(maint_valid), .maint_ready(maint_ready),
            .maint_base(maint_base), .maint_size(maint_size),
            .maint_done(maint_done), .maint_busy(maint_busy),
            .req_ready(req_ready), .req_valid(req_valid), .req_addr(req_addr),
            .req_we(req_we), .req_be(req_be), .req_wdata(req_wdata),
            .req_src(req_src), .req_tag(req_tag),
            .req_amo(req_amo), .req_aop(req_aop), .req_cmp(req_cmp),
            .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_src(rsp_src),
            .rsp_tag(rsp_tag), .rsp_rdata(rsp_rdata),
            .mem_req_valid(mem_req_valid), .mem_req_ready(m_req_ready),
            .mem_req_addr(mem_req_addr), .mem_req_we(mem_req_we),
            .mem_req_wdata(mem_req_wdata), .mem_req_tag(mem_req_tag),
            .mem_rsp_valid(m_rsp_valid), .mem_rsp_tag(m_rsp_tag),
            .mem_rsp_rdata(m_rsp_rdata),
            .dco_cfg(dco_cfg), .dco_owner_epoch(1'b0),
            .dco_t_valid(2'b00), .dco_t_base(128'd0), .dco_t_size(128'd0),
            .dco_t_end(130'd0), .dco_t_bypass(2'b00), .dco_t_lsb(12'd0),
            .dco_dead_valid(16'd0), .dco_dead_id(256'd0), .dco_slow(dco_slow),
            .tmu_wr_valid(tmu_wr_valid), .tmu_wr_ready(1'b1), .tmu_wr_off(tmu_wr_off),
            .tmu_wr_data(tmu_wr_data), .tmu_wr_src(tmu_wr_src), .tmu_rd_off(tmu_rd_off),
            .tmu_rd_data(64'd0),
            .tmu_tl_valid(tmu_tl_valid), .tmu_tl_ready(tl_ready),
            .tmu_tl_addr(tmu_tl_addr), .tmu_tl_drop(tmu_tl_drop),
            .perf_bypass(p_byp), .perf_evict(p_ev), .perf_dead_evict(p_dev),
            .perf_gear(p_gear), .perf_hit(p_hit), .perf_miss(p_miss),
            .perf_merge(p_merge), .perf_mshr_full(p_mf), .perf_target_full(p_tf),
            .perf_wb(p_wb), .perf_hit_past_block(p_hpb)
        );
        wire [OUT_W-1:0] o = {maint_ready, maint_done, maint_busy, req_ready,
            rsp_valid, rsp_src, rsp_tag, rsp_rdata,
            mem_req_valid, mem_req_addr, mem_req_we, mem_req_wdata, mem_req_tag,
            tmu_wr_valid, tmu_wr_off, tmu_wr_data, tmu_wr_src, tmu_rd_off,
            tmu_tl_valid, tmu_tl_addr, tmu_tl_drop,
            p_byp, p_ev, p_dev, p_hit, p_miss, p_merge, p_mf, p_tf, p_wb, p_hpb, p_gear};
    end endgenerate
    assign out_ref = g_dut[0].o;
    assign out_new = g_dut[1].o;

    // The reference instance drives the memory. While the outputs agree, the
    // second instance sees exactly the same responses.
    tnca_dram_model #(
        .ADDR_W(64), .LINE_BYTES(64), .LINE_BITS(512), .LINES(FOOT * 4),
        .TAG_W(MSHR_W), .SRC_W(1), .LATENCY(40), .RANDOM_STALL(1), .REORDER(1)
    ) u_dram (
        .clk(clk), .rst_n(rst_n),
        .req_valid(g_dut[0].mem_req_valid), .req_ready(m_req_ready),
        .req_addr(g_dut[0].mem_req_addr), .req_we(g_dut[0].mem_req_we),
        .req_wdata(g_dut[0].mem_req_wdata), .req_tag(g_dut[0].mem_req_tag),
        .req_src(1'b0),
        .rsp_valid(m_rsp_valid), .rsp_tag(m_rsp_tag), .rsp_src(),
        .rsp_rdata(m_rsp_rdata)
    );

    // ---- comparison, every cycle ----------------------------------------
    integer cyc = 0, mismatches = 0;
    always @(negedge clk) if (rst_n) begin
        cyc = cyc + 1;
        if (out_ref !== out_new) begin
            mismatches = mismatches + 1;
            if (mismatches <= 5)
                $display("FAIL cycle %0d: outputs differ (xor %h)", cyc, out_ref ^ out_new);
        end
    end

    // ---- coverage counters ------------------------------------------------
    integer n_issue = 0, n_rsp = 0, n_amo = 0, n_st = 0, n_maint = 0;
    integer n_look_rd = 0, n_wbrd = 0, n_wake = 0, n_mread = 0;
    always @(posedge clk) if (rst_n) begin
        if (req_valid && g_dut[0].req_ready) begin
            n_issue = n_issue + 1;
            if (req_amo) n_amo = n_amo + 1; else if (req_we) n_st = n_st + 1;
        end
        if (g_dut[0].rsp_valid && rsp_ready) n_rsp = n_rsp + 1;
        if (g_dut[1].dut.est == 4'd1)  n_look_rd = n_look_rd + 1;
        if (g_dut[1].dut.est == 4'd4)  n_wbrd    = n_wbrd + 1;
        if (g_dut[1].dut.est == 4'd8)  n_wake    = n_wake + 1;
        if (g_dut[1].dut.est == 4'd11) n_mread   = n_mread + 1;
    end

    // ---- stimulus -----------------------------------------------------------
    // Handshakes as the slices saw them at the edge.
    reg req_fire = 0, maint_fire = 0;
    always @(posedge clk) begin
        req_fire   <= req_valid && g_dut[0].req_ready;
        maint_fire <= maint_valid && g_dut[0].maint_ready;
    end
    integer seed, seed0, cycles, phase_left, r;
    reg draining;
    task new_request;
        begin
            r = $random(seed);
            req_addr  = ((($random(seed) & (FOOT-1))) << 6)
                      | (($random(seed) & 7) << 3);
            req_src   = $random(seed);
            req_tag   = $random(seed);
            req_wdata = {$random(seed), $random(seed)};
            req_cmp   = ($random(seed) & 1) ? req_wdata : {$random(seed), $random(seed)};
            req_be    = $random(seed);
            if (req_be == 0) req_be = 8'hff;
            req_amo   = ((r & 7) == 0);
            req_we    = !req_amo && ((r & 7) < 4);
            req_aop   = $random(seed) % 12;
        end
    endtask

    initial begin
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        seed0 = seed;
        if (!$value$plusargs("cycles=%d", cycles)) cycles = 200000;
        req_valid = 0; rsp_ready = 0; maint_valid = 0; maint_base = 0; maint_size = 0;
        tl_ready = 0; dco_slow = 0; draining = 0; phase_left = 4000;
        new_request;
        repeat (5) @(posedge clk);
        rst_n = 1;
        while (cyc < cycles) begin
            @(posedge clk); #1;
            rsp_ready = ($random(seed) & 3) != 0;
            tl_ready  = $random(seed);
            if (maint_fire) maint_valid = 0;          // accepted at the edge just passed
            if (req_fire)   req_valid = 0;
            if (!req_valid && !draining && (($random(seed) & 3) != 0)) begin
                new_request;
                req_valid = 1;
            end
            phase_left = phase_left - 1;
            if (phase_left == 0 && !draining) draining = 1;
            if (draining && !req_valid && n_issue == n_rsp && !g_dut[0].maint_busy
                && !maint_valid) begin
                // quiescent: run an ownership maintenance over a random range
                maint_base  = ($random(seed) & (FOOT-1)) << 6;
                maint_size  = ((($random(seed) & 255) + 1) << 6) - ($random(seed) & 63);
                maint_valid = 1;
                n_maint = n_maint + 1;
                draining = 0;
                phase_left = 2000 + ($random(seed) & 4095);
            end
        end
        $display("seed=%0d mode=%0d pipeline=%0d policy=%0d cycles=%0d issued=%0d responses=%0d stores=%0d atomics=%0d maint=%0d",
                 seed0, SHARED_MODE, CACHE_DATA_PIPELINE, POLICY, cyc, n_issue, n_rsp, n_st, n_amo, n_maint);
        $display("engine states with an array read: look=%0d wbrd=%0d wake=%0d mread=%0d; hits=%0d misses=%0d writebacks=%0d",
                 n_look_rd, n_wbrd, n_wake, n_mread, g_dut[0].p_hit, g_dut[0].p_miss, g_dut[0].p_wb);
        if (mismatches == 0 && n_rsp > 1000 && n_amo > 100 && n_maint > 2 && g_dut[0].p_wb > 100)
            $display("PASS tb_nuca_shared_read");
        else
            $display("FAIL tb_nuca_shared_read: mismatches=%0d", mismatches);
        $finish;
    end
endmodule
