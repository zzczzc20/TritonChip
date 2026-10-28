//==============================================================================
// tnca_home_pipe -- a pipelined home slice for line-wide requests (R41 G4)
//
// The same cache as tnca_nuca_slice -- one level, one home per line, the same
// set/tag decomposition, the same ports -- but organised to accept one request
// per cycle instead of running one engine state per cycle. It is used only
// where a request word is a whole line (WORD_BYTES == LINE_BYTES, the wide
// HYBRID_LSU tile) and only when the tile asks for it (HOME_LINE_PIPE).
//
// PIPELINE
//
//   arbiter -> S1: registered tag read of the set -> S2: decide and act
//   memory response -> MSHR line buffer -> fill unit (one target per cycle)
//
// S2 does everything a request needs in one cycle when it hits or when a miss
// needs no memory read: a read hit answers from the data array, a write hit
// merges its byte enables, an atomic reads, computes and writes, and a store
// that writes a whole line (LINE_WRITE_NOFILL) installs the line over a clean
// or written-back victim. Any other miss allocates an MSHR and a target,
// claims the victim way, and queues the victim's writeback; the fill unit
// installs the returned line after applying every target in arrival order.
//
// ORDER (the slice's sec 14.1 rule, unchanged): a request stays live in the
// request buffer until it completes or joins an MSHR, and the arbiter never
// starts a request while an older live request targets the same line. A
// request whose set S2 changed while its tags were being read is replayed.
// A read miss is not issued to memory while a writeback of the same line is
// still queued, so memory always returns the newest data.
//
// SCOPE: the replacement is invalid-first then round-robin (DCO policy 0).
// The DCO policy inputs are not used, and no tile-last-word notices are sent:
// the DCO policies move only where a line lives, never what it holds, and they
// remain available with HOME_LINE_PIPE = 0. TMU register accesses, atomics and the
// maintenance handoff behave exactly as in tnca_nuca_slice.
//==============================================================================
`include "tnca_defs.vh"
`default_nettype none

module tnca_home_pipe #(
    parameter ADDR_W     = 64,
    parameter LINE_BYTES = 64,
    parameter LINE_BITS  = 512,
    parameter WORD_BITS  = 512,
    parameter WORD_BYTES = 64,
    parameter LINE_WRITE_NOFILL = 0,
    parameter SETS       = 32,
    parameter SET_W      = 5,
    parameter WAYS       = 8,
    parameter WAY_W      = 3,
    parameter MSHR_N     = 8,
    parameter MSHR_W     = 3,
    parameter TARGET_N   = 16,
    parameter TARGET_W   = 4,
    parameter RQ_N       = 8,
    parameter RQ_W       = 3,
    parameter SRC_W      = 4,
    parameter TAG_W      = 5,
    parameter SLICE_ID   = 0,
    parameter HOME_W     = 2,
    parameter FULL_LINE_TAG = 0,
    parameter DCO_T      = 8,
    parameter DCO_DEAD   = 16,
    parameter DCO_ID_W   = 16,
    parameter MAINTENANCE = 0,
    parameter WBQ_N      = 4,
    parameter [63:0] TMU_BASE = 64'h0000_0000_7FFF_0000
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     maint_valid,
    output wire                     maint_ready,
    input  wire [ADDR_W-1:0]        maint_base, maint_size,
    output reg                      maint_done,
    output wire                     maint_busy,
    output wire                     req_ready,
    input  wire                     req_valid,
    input  wire [ADDR_W-1:0]        req_addr,
    input  wire                     req_we,
    input  wire [WORD_BYTES-1:0]    req_be,
    input  wire [WORD_BITS-1:0]     req_wdata,
    input  wire [SRC_W-1:0]         req_src,
    input  wire [TAG_W-1:0]         req_tag,
    input  wire                     req_amo,
    input  wire [3:0]               req_aop,
    input  wire [WORD_BITS-1:0]     req_cmp,
    output wire                     rsp_valid,
    input  wire                     rsp_ready,
    output wire [SRC_W-1:0]         rsp_src,
    output wire [TAG_W-1:0]         rsp_tag,
    output wire [WORD_BITS-1:0]     rsp_rdata,
    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [ADDR_W-1:0]        mem_req_addr,
    output wire                     mem_req_we,
    output wire [LINE_BITS-1:0]     mem_req_wdata,
    output wire [MSHR_W-1:0]        mem_req_tag,
    input  wire                     mem_rsp_valid,
    input  wire [MSHR_W-1:0]        mem_rsp_tag,
    input  wire [LINE_BITS-1:0]     mem_rsp_rdata,
    input  wire [75:0]              dco_cfg,
    input  wire                     dco_owner_epoch,
    input  wire [DCO_T-1:0]         dco_t_valid,
    input  wire [DCO_T*64-1:0]      dco_t_base,
    input  wire [DCO_T*64-1:0]      dco_t_size,
    input  wire [DCO_T*65-1:0]      dco_t_end,
    input  wire [DCO_T-1:0]         dco_t_bypass,
    input  wire [DCO_T*6-1:0]       dco_t_lsb,
    input  wire [DCO_DEAD-1:0]      dco_dead_valid,
    input  wire [DCO_DEAD*DCO_ID_W-1:0] dco_dead_id,
    input  wire [15:0]              dco_slow,
    output wire                     tmu_wr_valid,
    input  wire                     tmu_wr_ready,
    output wire [11:0]              tmu_wr_off,
    output wire [63:0]              tmu_wr_data,
    output wire [SRC_W-1:0]         tmu_wr_src,
    output wire [11:0]              tmu_rd_off,
    input  wire [63:0]              tmu_rd_data,
    output wire                     tmu_tl_valid,
    input  wire                     tmu_tl_ready,
    output wire [63:0]              tmu_tl_addr,
    output reg                      tmu_tl_drop,
    output reg  [31:0]              perf_bypass,
    output reg  [31:0]              perf_evict,
    output reg  [31:0]              perf_dead_evict,
    output reg  [3:0]               perf_gear,
    output reg  [31:0]              perf_hit,
    output reg  [31:0]              perf_miss,
    output reg  [31:0]              perf_merge,
    output reg  [31:0]              perf_mshr_full,
    output reg  [31:0]              perf_target_full,
    output reg  [31:0]              perf_wb,
    output reg  [31:0]              perf_hit_past_block
);
    localparam OFF_W     = 6;
    localparam HOME_SKIP = FULL_LINE_TAG ? 0 : HOME_W;
    localparam TAGB      = ADDR_W - OFF_W - HOME_SKIP - SET_W;
    localparam LINES     = SETS * WAYS;
    localparam LINE_W    = SET_W + WAY_W;
    localparam WBQ_W     = WBQ_N > 1 ? $clog2(WBQ_N) : 1;
    localparam RSPQ_N    = 4;

    integer i, j, k;

`ifndef SYNTHESIS
    initial if (WORD_BYTES != LINE_BYTES || WORD_BITS != LINE_BITS || LINE_BYTES != 64)
        $fatal(1, "tnca_home_pipe serves line-wide requests only (WORD_BYTES == LINE_BYTES == 64)");
`endif

`include "tnca_amo.vh"

    function [WORD_BITS-1:0] cache_amo_new;
        input [3:0] op;
        input [WORD_BYTES-1:0] be;
        input [WORD_BITS-1:0] old_word, write_word, compare_word;
        integer group;
        begin
            cache_amo_new = 0;
            for (group = 0; group < WORD_BYTES/8; group = group + 1)
                if (be[group*8 +: 8] != 0)
                    cache_amo_new[group*64 +: 64] = amo_new(op, be[group*8 +: 8], old_word[group*64 +: 64],
                                                            write_word[group*64 +: 64], compare_word[63:0]);
        end
    endfunction

    function [LINE_BITS-1:0] merge;
        input [LINE_BITS-1:0] line;
        input [LINE_BITS-1:0] data;
        input [LINE_BYTES-1:0] be;
        integer b;
        begin
            merge = line;
            for (b = 0; b < LINE_BYTES; b = b + 1)
                if (be[b]) merge[b*8 +: 8] = data[b*8 +: 8];
        end
    endfunction

    `define HP_SET(ad)  (ad[OFF_W + HOME_SKIP +: SET_W])
    `define HP_TAG(ad)  (ad[ADDR_W-1 -: TAGB])
    `define HP_LINE(ad) (ad[ADDR_W-1:OFF_W])

    function [ADDR_W-1:0] line_addr;
        input [TAGB-1:0]  tg;
        input [SET_W-1:0] st;
        reg   [HOME_W-1:0] home;
        begin
            home = SLICE_ID[HOME_W-1:0] ^ st[HOME_W-1:0];
            if (FULL_LINE_TAG) line_addr = {tg, st, {OFF_W{1'b0}}};
            else line_addr = {tg, st, home, {OFF_W{1'b0}}};
        end
    endfunction

    //==========================================================================
    // Arrays: data (one read, one write port), tags per way (registered read)
    //==========================================================================
    (* ram_style = "distributed" *) reg [LINE_BITS-1:0] data_mem [0:LINES-1];
    reg [LINES-1:0]  v_bits, d_bits, cl_bits;   // valid, dirty, claimed by an MSHR
    reg [WAY_W-1:0]  victim_ptr [0:SETS-1];

    //==========================================================================
    // Request buffer and arbiter (as tnca_nuca_slice; `busy` = in the pipe)
    //==========================================================================
    reg [RQ_N-1:0]       rq_v, rq_blocked, rq_busy;
    reg [ADDR_W-1:0]     rq_addr  [0:RQ_N-1];
    reg                  rq_we    [0:RQ_N-1];
    reg                  rq_amo   [0:RQ_N-1];
    reg [3:0]            rq_aop   [0:RQ_N-1];
    reg [WORD_BITS-1:0]  rq_cmp   [0:RQ_N-1];
    reg [WORD_BYTES-1:0] rq_be    [0:RQ_N-1];
    reg [WORD_BITS-1:0]  rq_wdata [0:RQ_N-1];
    reg [SRC_W-1:0]      rq_src   [0:RQ_N-1];
    reg [TAG_W-1:0]      rq_tag   [0:RQ_N-1];
    reg [RQ_N-1:0]       older    [0:RQ_N-1];

    reg [RQ_W-1:0] rq_free_idx; reg rq_has_free;
    always @(*) begin
        rq_free_idx = 0; rq_has_free = 1'b0;
        for (i = RQ_N-1; i >= 0; i = i - 1)
            if (!rq_v[i]) begin rq_free_idx = i[RQ_W-1:0]; rq_has_free = 1'b1; end
    end

    reg [RQ_W-1:0] sel_idx; reg sel_valid; reg shadowed; reg [RQ_N-1:0] elig;
    always @(*) begin
        for (i = 0; i < RQ_N; i = i + 1) begin
            shadowed = 1'b0;
            for (j = 0; j < RQ_N; j = j + 1)
                if ((j != i) && rq_v[j] && older[j][i] &&
                    (rq_addr[j][ADDR_W-1:OFF_W] == rq_addr[i][ADDR_W-1:OFF_W]))
                    shadowed = 1'b1;
            elig[i] = rq_v[i] && !rq_blocked[i] && !rq_busy[i] && !shadowed;
        end
        sel_idx = 0; sel_valid = 1'b0;
        for (i = 0; i < RQ_N; i = i + 1)
            if (elig[i]) begin
                shadowed = 1'b0;
                for (j = 0; j < RQ_N; j = j + 1)
                    if ((j != i) && elig[j] && older[j][i]) shadowed = 1'b1;
                if (!shadowed) begin sel_idx = i[RQ_W-1:0]; sel_valid = 1'b1; end
            end
    end
    reg any_blocked;
    always @(*) begin
        any_blocked = 1'b0;
        for (i = 0; i < RQ_N; i = i + 1) if (rq_v[i] && rq_blocked[i]) any_blocked = 1'b1;
    end

    //==========================================================================
    // MSHRs, targets, writeback queue, response queue
    //==========================================================================
    localparam MS_FREE = 2'd0, MS_REQ = 2'd1, MS_WAIT = 2'd2, MS_FILL = 2'd3;
    reg [1:0]          ms_state [0:MSHR_N-1];
    reg [ADDR_W-1:0]   ms_addr  [0:MSHR_N-1];
    reg [SET_W-1:0]    ms_set   [0:MSHR_N-1];
    reg [WAY_W-1:0]    ms_way   [0:MSHR_N-1];
    reg [TARGET_W-1:0] ms_head  [0:MSHR_N-1];
    reg [TARGET_W-1:0] ms_tail  [0:MSHR_N-1];
    (* ram_style = "distributed" *) reg [LINE_BITS-1:0] ms_line [0:MSHR_N-1];

    reg [TARGET_N-1:0]   tg_v;
    reg [TARGET_W-1:0]   tg_next  [0:TARGET_N-1];
    reg [SRC_W-1:0]      tg_src   [0:TARGET_N-1];
    reg [TAG_W-1:0]      tg_tag   [0:TARGET_N-1];
    reg                  tg_we    [0:TARGET_N-1];
    reg                  tg_amo   [0:TARGET_N-1];
    reg [3:0]            tg_aop   [0:TARGET_N-1];
    reg [63:0]           tg_cmp   [0:TARGET_N-1];
    reg [WORD_BYTES-1:0] tg_be    [0:TARGET_N-1];
    (* ram_style = "distributed" *) reg [WORD_BITS-1:0] tg_wdata [0:TARGET_N-1];

    reg [MSHR_W-1:0] ms_free_idx; reg ms_has_free;
    reg [TARGET_W-1:0] tg_free_idx; reg tg_has_free;
    reg ms_all_free;
    always @(*) begin
        ms_free_idx = 0; ms_has_free = 1'b0; ms_all_free = 1'b1;
        for (i = MSHR_N-1; i >= 0; i = i - 1) begin
            if (ms_state[i] == MS_FREE) begin ms_free_idx = i[MSHR_W-1:0]; ms_has_free = 1'b1; end
            else ms_all_free = 1'b0;
        end
        tg_free_idx = 0; tg_has_free = 1'b0;
        for (i = TARGET_N-1; i >= 0; i = i - 1)
            if (!tg_v[i]) begin tg_free_idx = i[TARGET_W-1:0]; tg_has_free = 1'b1; end
    end

    reg [WBQ_N-1:0]      wbq_v;
    reg [ADDR_W-1:0]     wbq_addr [0:WBQ_N-1];
    reg [LINE_BITS-1:0]  wbq_data [0:WBQ_N-1];
    reg [WBQ_W-1:0]      wbq_rd, wbq_wr;
    reg [WBQ_W:0]        wbq_cnt;
    wire wbq_full = (wbq_cnt == WBQ_N);

    reg [SRC_W-1:0]     rsq_src  [0:RSPQ_N-1];
    reg [TAG_W-1:0]     rsq_tag  [0:RSPQ_N-1];
    reg [WORD_BITS-1:0] rsq_data [0:RSPQ_N-1];
    reg [1:0]           rsq_rd, rsq_wr;
    reg [2:0]           rsq_cnt;
    assign rsp_valid = rsq_cnt != 3'd0;
    assign rsp_src   = rsq_src[rsq_rd];
    assign rsp_tag   = rsq_tag[rsq_rd];
    assign rsp_rdata = rsq_data[rsq_rd];
    wire rsq_full = (rsq_cnt >= RSPQ_N);
    wire rsq_pop  = rsp_valid && rsp_ready;

    //==========================================================================
    // S1: the selected request and its set's tags (registered LUTRAM read)
    //==========================================================================
    reg              s1_v, s1_stale;
    reg [RQ_W-1:0]   s1_rq;
    reg [ADDR_W-1:0] s1_addr;
    wire [SET_W-1:0] s1_set = `HP_SET(s1_addr);
    wire [TAGB-1:0]  lookup_tag [0:WAYS-1];

    // S2 writes a set's tag when it claims a way; the read that S1 holds of
    // the same set is then stale and its request is replayed.
    // (written on the same edge as S2 acts; see the S2 section)
    wire             tag_we;
    wire [SET_W-1:0] tag_wset;
    wire [WAY_W-1:0] tag_wway;
    wire [TAGB-1:0]  tag_wdata;

    localparam M_IDLE = 2'd0, M_SCAN = 2'd1, M_WAIT = 2'd2, M_DRAIN = 2'd3;
    reg [1:0]        m_state;
    reg [LINE_W-1:0] m_index;
    reg [ADDR_W-1:0] m_first;
    reg [ADDR_W:0]   m_end;
    wire [SET_W-1:0] m_set = m_index / WAYS;
    wire [WAY_W-1:0] m_way = m_index % WAYS;
    wire pipe_stall;              // S2 holds: S1 and the arbiter hold too
    wire maint_hold;              // a maintenance request is pending: start nothing
    wire take = sel_valid && !pipe_stall && !maint_hold;
    wire [SET_W-1:0] read_set = `HP_SET(rq_addr[sel_idx]);
    genvar gw;
    generate for (gw = 0; gw < WAYS; gw = gw + 1) begin : g_tag
        (* ram_style = "distributed" *) reg [TAGB-1:0] bank [0:SETS-1];
        reg [TAGB-1:0] read_tag;
        always @(posedge clk) begin
            if (tag_we && tag_wway == gw) bank[tag_wset] <= tag_wdata;
            if (take) read_tag <= bank[read_set];
            else if (m_state == M_WAIT) read_tag <= bank[m_set];   // maintenance scan
        end
        assign lookup_tag[gw] = read_tag;
    end endgenerate

    //==========================================================================
    // S2 decision (combinational over S1's registered request)
    //==========================================================================
    wire [ADDR_W-1:0]     s2_addr  = s1_addr;
    wire [SET_W-1:0]      s2_set   = s1_set;
    wire [TAGB-1:0]       s2_tagb  = `HP_TAG(s1_addr);
    wire                  s2_we    = rq_we[s1_rq];
    wire                  s2_amo   = rq_amo[s1_rq];
    wire [3:0]            s2_aop   = rq_aop[s1_rq];
    wire [WORD_BYTES-1:0] s2_be    = rq_be[s1_rq];
    wire [WORD_BITS-1:0]  s2_wdata = rq_wdata[s1_rq];
    wire [WORD_BITS-1:0]  s2_cmp   = rq_cmp[s1_rq];
    wire [SRC_W-1:0]      s2_src   = rq_src[s1_rq];
    wire [TAG_W-1:0]      s2_tag   = rq_tag[s1_rq];
    wire s2_mmio = (s1_addr[ADDR_W-1:12] == TMU_BASE[ADDR_W-1:12]);
    wire s2_full_store = LINE_WRITE_NOFILL && s2_we && !s2_amo && (&s2_be);

    reg s2_hit; reg [WAY_W-1:0] s2_hit_way;
    reg s2_vic_ok; reg [WAY_W-1:0] s2_vic_way;
    reg s2_ms; reg [MSHR_W-1:0] s2_ms_id;
    reg [WAY_W-1:0] rr;
    always @(*) begin
        s2_hit = 1'b0; s2_hit_way = 0;
        for (k = 0; k < WAYS; k = k + 1)
            if (v_bits[s2_set*WAYS + k] && lookup_tag[k] == s2_tagb) begin
                s2_hit = 1'b1; s2_hit_way = k[WAY_W-1:0];
            end
        // victim: an invalid unclaimed way first, else round-robin over the
        // unclaimed ways from the set's pointer
        s2_vic_ok = 1'b0; s2_vic_way = 0;
        for (k = WAYS-1; k >= 0; k = k - 1)
            if (!v_bits[s2_set*WAYS + k] && !cl_bits[s2_set*WAYS + k]) begin
                s2_vic_ok = 1'b1; s2_vic_way = k[WAY_W-1:0];
            end
        if (!s2_vic_ok)
            for (k = WAYS-1; k >= 0; k = k - 1) begin
                rr = victim_ptr[s2_set] + k[WAY_W-1:0];
                if (!cl_bits[s2_set*WAYS + rr]) begin s2_vic_ok = 1'b1; s2_vic_way = rr; end
            end
        s2_ms = 1'b0; s2_ms_id = 0;
        for (k = 0; k < MSHR_N; k = k + 1)
            if (ms_state[k] != MS_FREE && ms_addr[k][ADDR_W-1:OFF_W] == s1_addr[ADDR_W-1:OFF_W]) begin
                s2_ms = 1'b1; s2_ms_id = k[MSHR_W-1:0];
            end
    end
    wire [LINE_W-1:0] s2_hit_index = s2_set*WAYS + s2_hit_way;
    wire [LINE_W-1:0] s2_vic_index = s2_set*WAYS + s2_vic_way;
    wire [LINE_W-1:0] s2_rd_index  = s2_hit ? s2_hit_index : s2_vic_index;
    wire [LINE_BITS-1:0] s2_line   = data_mem[s2_rd_index];
    wire s2_vic_dirty = v_bits[s2_vic_index] && d_bits[s2_vic_index];
    wire [ADDR_W-1:0] s2_vic_addr = line_addr(lookup_tag[s2_vic_way], s2_set);
    wire [WORD_BITS-1:0] s2_amo_res = cache_amo_new(s2_aop, s2_be, s2_line, s2_wdata, s2_cmp);

    //==========================================================================
    // Fill unit: one target of one filled MSHR per cycle
    //==========================================================================
    reg              fu_v;
    reg [MSHR_W-1:0] fu_ms;
    reg [TARGET_W-1:0] fu_t;
    reg [LINE_BITS-1:0] fu_line;
    reg              fu_dirty;
    reg              fu_first;
    reg fill_found; reg [MSHR_W-1:0] fill_pick;
    always @(*) begin
        fill_found = 1'b0; fill_pick = 0;
        for (k = MSHR_N-1; k >= 0; k = k - 1)
            if (ms_state[k] == MS_FILL) begin fill_found = 1'b1; fill_pick = k[MSHR_W-1:0]; end
    end
    // the line the current target sees: the returned line, or the running merge
    wire [LINE_BITS-1:0] fu_cur  = fu_first ? ms_line[fu_ms] : fu_line;
    wire [TARGET_W-1:0]  fu_tc   = fu_t;
    wire fu_last = (fu_tc == ms_tail[fu_ms]);
    wire [WORD_BITS-1:0] fu_amo_res = cache_amo_new(tg_aop[fu_tc], tg_be[fu_tc], fu_cur,
                                                    tg_wdata[fu_tc], {448'd0, tg_cmp[fu_tc]});
    wire [LINE_BITS-1:0] fu_next =
        tg_amo[fu_tc] ? ((tg_aop[fu_tc] != `TNCA_AOP_LD) ? merge(fu_cur, fu_amo_res, tg_be[fu_tc]) : fu_cur) :
        tg_we[fu_tc]  ? merge(fu_cur, tg_wdata[fu_tc], tg_be[fu_tc]) : fu_cur;
    wire fu_writes = (tg_amo[fu_tc] && tg_aop[fu_tc] != `TNCA_AOP_LD) || (tg_we[fu_tc] && !tg_amo[fu_tc]);
    // A step pushes one response; the last also writes the data array.
    wire fu_step = fu_v && !rsq_full;

    //==========================================================================
    // Memory requests: queued writebacks first, then MSHR reads that no queued
    // writeback of the same line precedes
    //==========================================================================
    reg rd_found; reg [MSHR_W-1:0] rd_pick; reg rd_blocked_by_wb;
    always @(*) begin
        rd_found = 1'b0; rd_pick = 0;
        for (k = MSHR_N-1; k >= 0; k = k - 1)
            if (ms_state[k] == MS_REQ) begin
                rd_blocked_by_wb = 1'b0;
                for (j = 0; j < WBQ_N; j = j + 1)
                    if (wbq_v[j] && wbq_addr[j][ADDR_W-1:OFF_W] == ms_addr[k][ADDR_W-1:OFF_W])
                        rd_blocked_by_wb = 1'b1;
                if (!rd_blocked_by_wb) begin rd_found = 1'b1; rd_pick = k[MSHR_W-1:0]; end
            end
    end
    // A presented request that memory did not take stays presented, unchanged
    // (the outlet samples it later): the choice is locked until accepted.
    reg ml_v, ml_wb; reg [MSHR_W-1:0] ml_ms;
    wire wb_issue = ml_v ? ml_wb : (wbq_cnt != 0);
    wire [MSHR_W-1:0] rd_issue = ml_v ? ml_ms : rd_pick;
    assign mem_req_valid = ml_v || wbq_cnt != 0 || rd_found;
    assign mem_req_we    = wb_issue;
    assign mem_req_addr  = wb_issue ? wbq_addr[wbq_rd] : ms_addr[rd_issue];
    assign mem_req_wdata = wb_issue ? wbq_data[wbq_rd] : {LINE_BITS{1'b0}};
    assign mem_req_tag   = wb_issue ? {MSHR_W{1'b0}} : rd_issue;
    wire mem_req_fire = mem_req_valid && mem_req_ready;

    //==========================================================================
    // TMU registers (a load reads, a store waits for the TMU to take it)
    //==========================================================================
    wire [8:0] tmu_word_shift = ((s1_addr & (WORD_BYTES-1)) >> 3) * 64;
    assign tmu_wr_off  = s1_addr[11:0];
    assign tmu_rd_off  = s1_addr[11:0];
    assign tmu_wr_data = s2_wdata >> tmu_word_shift;
    assign tmu_wr_src  = s2_src;
    assign tmu_wr_valid = s1_v && !s1_stale && s2_mmio && s2_we && !s2_amo;
    assign tmu_tl_valid = 1'b0;
    assign tmu_tl_addr  = 64'd0;

    //==========================================================================
    // What S2 does this cycle
    //==========================================================================
    // Resources S2 needs (transient: hold the pipe) or lacks (structural:
    // block the request so younger ones go past it, sec 9.3).
    wire s2_live   = s1_v && !s1_stale;
    wire fu_write_now = fu_step && fu_last;        // the fill unit owns the write port
    wire s2_push_ok   = !fu_step && !rsq_full;     // and the response push
    reg  s2_act, s2_hold, s2_block, s2_replay;
    reg  s2_kind_hit, s2_kind_merge, s2_kind_install, s2_kind_alloc, s2_kind_mmio;
    always @(*) begin
        s2_act = 1'b0; s2_hold = 1'b0; s2_block = 1'b0; s2_replay = 1'b0;
        s2_kind_hit = 1'b0; s2_kind_merge = 1'b0; s2_kind_install = 1'b0; s2_kind_alloc = 1'b0;
        s2_kind_mmio = 1'b0;
        if (s1_v && s1_stale) s2_replay = 1'b1;
        else if (s2_live) begin
            if (s2_mmio) begin
                s2_kind_mmio = 1'b1;
                if (s2_push_ok && (!s2_we || s2_amo || tmu_wr_ready)) s2_act = 1'b1; else s2_hold = 1'b1;
            end else if (s2_ms) begin
                s2_kind_merge = 1'b1;
                // the fill unit is closing this MSHR's list: wait, then hit
                if (fu_v && fu_ms == s2_ms_id) s2_hold = 1'b1;
                else if (tg_has_free) s2_act = 1'b1; else s2_block = 1'b1;
            end else if (s2_hit) begin
                s2_kind_hit = 1'b1;
                if (s2_push_ok && !fu_write_now) s2_act = 1'b1; else s2_hold = 1'b1;
            end else if (!s2_vic_ok) begin
                s2_block = 1'b1;
            end else if (s2_full_store) begin
                s2_kind_install = 1'b1;
                if (s2_push_ok && !fu_write_now && !(s2_vic_dirty && wbq_full)) s2_act = 1'b1; else s2_hold = 1'b1;
            end else begin
                s2_kind_alloc = 1'b1;
                if (!ms_has_free || !tg_has_free) s2_block = 1'b1;
                else if (s2_vic_dirty && wbq_full) s2_hold = 1'b1;
                else s2_act = 1'b1;
            end
        end
    end
    assign pipe_stall = s2_hold;
    assign tag_we    = s2_act && (s2_kind_install || s2_kind_alloc);
    assign tag_wset  = s2_set;
    assign tag_wway  = s2_vic_way;
    assign tag_wdata = s2_tagb;

    //==========================================================================
    // Maintenance (as tnca_nuca_slice: drain, then write back and invalidate
    // every line that overlaps the range)
    //==========================================================================
    wire maint_requested = (MAINTENANCE != 0) && maint_valid;
    assign maint_hold = maint_requested || m_state != M_IDLE;
    wire idle = rq_v == 0 && !s1_v && rsq_cnt == 0 && tg_v == 0 && ms_all_free && !fu_v &&
                wbq_cnt == 0 && !mem_rsp_valid;
    assign maint_ready = (MAINTENANCE != 0) && rst_n && m_state == M_IDLE && idle;
    assign maint_busy  = (m_state != M_IDLE);
    assign req_ready   = rq_has_free && !maint_busy && !maint_requested;
    wire [ADDR_W-1:0] m_address = line_addr(lookup_tag[m_way], m_set);
    wire m_overlap = ({1'b0, m_address} < m_end) && ({1'b0, m_address} + LINE_BYTES > {1'b0, m_first});

    //==========================================================================
    // Sequential
    //==========================================================================
    reg push; reg [SRC_W-1:0] push_src; reg [TAG_W-1:0] push_tag; reg [WORD_BITS-1:0] push_data;
    reg dw_en; reg [LINE_W-1:0] dw_index; reg [LINE_BITS-1:0] dw_line;
    reg wbq_push; reg [ADDR_W-1:0] wbq_push_addr; reg [LINE_BITS-1:0] wbq_push_data;
    reg free_resources;

    always @(posedge clk)
        if (mem_rsp_valid) ms_line[mem_rsp_tag] <= mem_rsp_rdata;

    always @(posedge clk) begin
        if (!rst_n) begin
            rq_v <= 0; rq_blocked <= 0; rq_busy <= 0;
            for (i = 0; i < RQ_N; i = i + 1) older[i] <= 0;
            v_bits <= 0; d_bits <= 0; cl_bits <= 0;
            for (i = 0; i < SETS; i = i + 1) victim_ptr[i] <= 0;
            for (i = 0; i < MSHR_N; i = i + 1) ms_state[i] <= MS_FREE;
            tg_v <= 0; wbq_v <= 0; wbq_rd <= 0; wbq_wr <= 0; wbq_cnt <= 0;
            rsq_rd <= 0; rsq_wr <= 0; rsq_cnt <= 0;
            s1_v <= 0; s1_stale <= 0; fu_v <= 0; ml_v <= 0;
            m_state <= M_IDLE; maint_done <= 0;
            tmu_tl_drop <= 0; perf_bypass <= 0; perf_evict <= 0; perf_dead_evict <= 0; perf_gear <= 0;
            perf_hit <= 0; perf_miss <= 0; perf_merge <= 0; perf_mshr_full <= 0; perf_target_full <= 0;
            perf_wb <= 0; perf_hit_past_block <= 0;
        end else begin
            push = 0; push_src = 0; push_tag = 0; push_data = 0;
            dw_en = 0; dw_index = 0; dw_line = 0;
            wbq_push = 0; wbq_push_addr = 0; wbq_push_data = 0;
            free_resources = 0;
            maint_done <= 1'b0;

            // ---- accept a request ----
            if (req_valid && req_ready) begin
                rq_v[rq_free_idx] <= 1'b1; rq_blocked[rq_free_idx] <= 1'b0; rq_busy[rq_free_idx] <= 1'b0;
                rq_addr[rq_free_idx] <= req_addr; rq_we[rq_free_idx] <= req_we;
                rq_amo[rq_free_idx] <= req_amo; rq_aop[rq_free_idx] <= req_aop;
                rq_cmp[rq_free_idx] <= req_cmp; rq_be[rq_free_idx] <= req_be;
                rq_wdata[rq_free_idx] <= req_wdata; rq_src[rq_free_idx] <= req_src;
                rq_tag[rq_free_idx] <= req_tag;
                older[rq_free_idx] <= 0;
                for (i = 0; i < RQ_N; i = i + 1)
                    if (i != rq_free_idx) older[i][rq_free_idx] <= rq_v[i];
            end

            // ---- memory response ----
            if (mem_rsp_valid) ms_state[mem_rsp_tag] <= MS_FILL;

            // ---- memory request ----
            if (mem_req_fire) begin
                ml_v <= 1'b0;
                if (wb_issue) begin
                    wbq_v[wbq_rd] <= 1'b0;
                    wbq_rd <= (wbq_rd == WBQ_N-1) ? 0 : wbq_rd + 1'b1;
                    perf_wb <= perf_wb + 1;
                end else ms_state[rd_issue] <= MS_WAIT;
            end else if (mem_req_valid) begin
                ml_v <= 1'b1; ml_wb <= wb_issue; ml_ms <= rd_issue;
            end

            // ---- fill unit ----
            if (!fu_v && fill_found) begin
                fu_v <= 1'b1; fu_ms <= fill_pick; fu_t <= ms_head[fill_pick];
                fu_first <= 1'b1; fu_dirty <= 1'b0;
            end else if (fu_step) begin
                push = 1; push_src = tg_src[fu_tc]; push_tag = tg_tag[fu_tc];
                push_data = tg_we[fu_tc] && !tg_amo[fu_tc] ? {WORD_BITS{1'b0}} : fu_cur;
                tg_v[fu_tc] <= 1'b0;
                fu_line <= fu_next; fu_first <= 1'b0;
                fu_dirty <= fu_dirty || fu_writes;
                if (fu_last) begin
                    dw_en = 1; dw_index = ms_set[fu_ms]*WAYS + ms_way[fu_ms]; dw_line = fu_next;
                    v_bits[dw_index] <= 1'b1; cl_bits[dw_index] <= 1'b0;
                    d_bits[dw_index] <= fu_dirty || fu_writes;
                    ms_state[fu_ms] <= MS_FREE;
                    fu_v <= 1'b0;
                    free_resources = 1;
                end else fu_t <= tg_next[fu_tc];
            end

            // ---- S2 ----
            if (s2_replay) begin
                rq_busy[s1_rq] <= 1'b0;
            end else if (s2_block) begin
                rq_busy[s1_rq] <= 1'b0; rq_blocked[s1_rq] <= 1'b1;
                if (s2_kind_alloc && !ms_has_free) perf_mshr_full <= perf_mshr_full + 1;
                if (!tg_has_free) perf_target_full <= perf_target_full + 1;
            end else if (s2_act) begin
                rq_v[s1_rq] <= 1'b0; rq_busy[s1_rq] <= 1'b0;
                if (s2_kind_mmio) begin
                    push = 1; push_src = s2_src; push_tag = s2_tag;
                    push_data = s2_we ? {WORD_BITS{1'b0}} : ({448'd0, tmu_rd_data} << tmu_word_shift);
                end else if (s2_kind_merge) begin
                    perf_merge <= perf_merge + 1;
                    tg_v[tg_free_idx] <= 1'b1; tg_src[tg_free_idx] <= s2_src; tg_tag[tg_free_idx] <= s2_tag;
                    tg_we[tg_free_idx] <= s2_we; tg_amo[tg_free_idx] <= s2_amo; tg_aop[tg_free_idx] <= s2_aop;
                    tg_cmp[tg_free_idx] <= s2_cmp[63:0]; tg_be[tg_free_idx] <= s2_be;
                    tg_wdata[tg_free_idx] <= s2_wdata;
                    tg_next[ms_tail[s2_ms_id]] <= tg_free_idx;
                    ms_tail[s2_ms_id] <= tg_free_idx;
                end else if (s2_kind_hit) begin
                    perf_hit <= perf_hit + 1;
                    push = 1; push_src = s2_src; push_tag = s2_tag;
                    if (s2_amo) begin
                        push_data = s2_line;
                        if (s2_aop != `TNCA_AOP_LD) begin
                            dw_en = 1; dw_index = s2_hit_index; dw_line = merge(s2_line, s2_amo_res, s2_be);
                            d_bits[s2_hit_index] <= 1'b1;
                        end
                    end else if (s2_we) begin
                        push_data = 0;
                        dw_en = 1; dw_index = s2_hit_index; dw_line = merge(s2_line, s2_wdata, s2_be);
                        d_bits[s2_hit_index] <= 1'b1;
                    end else push_data = s2_line;
                end else begin
                    // install (full-line store) or allocate (everything else)
                    perf_miss <= perf_miss + 1;
                    if (v_bits[s2_vic_index]) perf_evict <= perf_evict + 1;
                    if (s2_vic_dirty) begin
                        wbq_push = 1; wbq_push_addr = s2_vic_addr; wbq_push_data = s2_line;
                    end
                    victim_ptr[s2_set] <= s2_vic_way + 1'b1;
                    if (s2_kind_install) begin
                        dw_en = 1; dw_index = s2_vic_index; dw_line = s2_wdata;
                        v_bits[s2_vic_index] <= 1'b1; d_bits[s2_vic_index] <= 1'b1;
                        push = 1; push_src = s2_src; push_tag = s2_tag; push_data = 0;
                    end else begin
                        v_bits[s2_vic_index] <= 1'b0; d_bits[s2_vic_index] <= 1'b0;
                        cl_bits[s2_vic_index] <= 1'b1;
                        ms_state[ms_free_idx] <= MS_REQ;
                        ms_addr[ms_free_idx] <= {s1_addr[ADDR_W-1:OFF_W], {OFF_W{1'b0}}};
                        ms_set[ms_free_idx] <= s2_set; ms_way[ms_free_idx] <= s2_vic_way;
                        ms_head[ms_free_idx] <= tg_free_idx; ms_tail[ms_free_idx] <= tg_free_idx;
                        tg_v[tg_free_idx] <= 1'b1; tg_src[tg_free_idx] <= s2_src; tg_tag[tg_free_idx] <= s2_tag;
                        tg_we[tg_free_idx] <= s2_we; tg_amo[tg_free_idx] <= s2_amo; tg_aop[tg_free_idx] <= s2_aop;
                        tg_cmp[tg_free_idx] <= s2_cmp[63:0]; tg_be[tg_free_idx] <= s2_be;
                        tg_wdata[tg_free_idx] <= s2_wdata;
                    end
                end
            end

            // ---- S1: load, hold, or empty ----
            if (!s2_hold) begin
                s1_v <= take;
                if (take) begin
                    s1_rq <= sel_idx; s1_addr <= rq_addr[sel_idx]; rq_busy[sel_idx] <= 1'b1;
                    if (any_blocked) perf_hit_past_block <= perf_hit_past_block + 1;
                    // a tag S2 writes this cycle in the set S1 reads is not in the read
                    s1_stale <= s2_act && (s2_kind_install || s2_kind_alloc) &&
                                (`HP_SET(rq_addr[sel_idx]) == s2_set);
                end else s1_stale <= 1'b0;
            end

            // ---- resources freed: blocked requests try again ----
            if (free_resources) rq_blocked <= 0;

            // ---- maintenance ----
            case (m_state)
            M_IDLE: if (maint_requested && maint_ready) begin
                m_first <= maint_base; m_end <= {1'b0, maint_base} + {1'b0, maint_size};
                m_index <= 0;
                if (maint_size == 0) maint_done <= 1'b1;
                else m_state <= M_WAIT;           // read the first tag
            end
            M_WAIT: m_state <= M_SCAN;
            M_SCAN: begin
                if (v_bits[m_index] && m_overlap && d_bits[m_index] && wbq_full) begin
                    // wait for room
                end else begin
                    if (v_bits[m_index] && m_overlap) begin
                        if (d_bits[m_index]) begin
                            wbq_push = 1; wbq_push_addr = m_address; wbq_push_data = data_mem[m_index];
                        end
                        v_bits[m_index] <= 1'b0; d_bits[m_index] <= 1'b0;
                    end
                    if (m_index == LINES-1) m_state <= M_DRAIN;
                    else begin m_index <= m_index + 1'b1; m_state <= M_WAIT; end
                end
            end
            M_DRAIN: if (wbq_cnt == 0 && !(mem_req_fire && wb_issue)) begin
                // completion means every writeback was accepted
                maint_done <= 1'b1; m_state <= M_IDLE;
            end
            endcase

            // ---- writeback queue ----
            if (wbq_push) begin
                wbq_v[wbq_wr] <= 1'b1; wbq_addr[wbq_wr] <= wbq_push_addr; wbq_data[wbq_wr] <= wbq_push_data;
                wbq_wr <= (wbq_wr == WBQ_N-1) ? 0 : wbq_wr + 1'b1;
            end
            case ({wbq_push, mem_req_fire && wb_issue})
            2'b10: wbq_cnt <= wbq_cnt + 1'b1;
            2'b01: wbq_cnt <= wbq_cnt - 1'b1;
            default: ;
            endcase

            // ---- data array write port (fill unit or S2, never both) ----
            if (dw_en) data_mem[dw_index] <= dw_line;

            // ---- response queue ----
            if (push) begin
                rsq_src[rsq_wr] <= push_src; rsq_tag[rsq_wr] <= push_tag; rsq_data[rsq_wr] <= push_data;
                rsq_wr <= rsq_wr + 1'b1;
            end
            if (rsq_pop) rsq_rd <= rsq_rd + 1'b1;
            case ({push, rsq_pop})
            2'b10: rsq_cnt <= rsq_cnt + 1'b1;
            2'b01: rsq_cnt <= rsq_cnt - 1'b1;
            default: ;
            endcase
        end
    end

    // Observation names shared with tnca_nuca_slice (testbench probes only):
    // est 0 idle, 1 held, 2 a hit, 3 a miss allocated or installed, 7 a fill
    // step, 15 anything else (merge, MMIO, replay, block).
    wire [3:0] est = !s1_v ? (fu_step ? 4'd7 : 4'd0) : s2_hold ? 4'd1 :
                     (s2_act && s2_kind_hit) ? 4'd2 :
                     (s2_act && (s2_kind_alloc || s2_kind_install)) ? 4'd3 : 4'd15;
    wire fill_fast_take = 1'b0, hit_chain_take = 1'b0;
    wire use_background = rd_found;

    wire _unused = &{1'b0, dco_cfg, dco_owner_epoch, dco_t_valid, dco_t_base, dco_t_size, dco_t_end,
                     dco_t_bypass, dco_t_lsb, dco_dead_valid, dco_dead_id, dco_slow, tmu_tl_ready, 1'b0};
endmodule
`default_nettype wire
