//==============================================================================
// tnca_axi_bridge_lb -- rtl/fpga/tnca_axi_bridge.v with reads by the line
//
// doc/r40_tasks_and_acceptance.md R40-04: the `burst` mechanism, with its own
// switch (tnca_fpga_top AXI_LINE: 0 builds tnca_axi_bridge, this file's
// LINE_BEATS otherwise) so that burst=0/1 is an A/B on one design.
//
// The port, the routing, the on-chip RAM below LOCAL_LIMIT, out-of-range
// answers, errors, the reset quiet window and the counters are
// tnca_axi_bridge's, and its header is the argument for each. What changes is
// how a read of the AXI window is served.
//
// READS
//   The window is read in lines of LINE_BEATS 8-byte words, aligned, each by
//   one AXI INCR burst (ARLEN = LINE_BEATS-1, ARID = the line's slot), into a
//   fully associative buffer of LB_N lines. A read that finds its line there
//   is answered from it; one whose line is on its way waits for it; one that
//   misses allocates a slot and asks for the line. Either way the request is
//   accepted into one of WAIT_N waiter slots and answered when its line is in,
//   in any order -- which the port allows. A request is refused (ready low)
//   only when no waiter slot or no line slot is free.
//
//   PF further lines past the one asked for are fetched as well, so a stream
//   finds its next line already there or on its way. A prefetch is never
//   waited on by anything but a later read, and never fetched past the end of
//   the window.
//
// ORDER (the port promises acceptance order; AXI does not)
//   * A write updates the buffered copy of its line, if there is one, and goes
//     to AXI as tnca_axi_bridge's do -- one 8-byte beat, one ID, in order --
//     so a later read of the buffer sees it. A write is refused while its line
//     is being filled (the fill would overwrite it) or while a waiter on its
//     line has not been answered (the waiter was accepted first and must not
//     see it).
//   * A line is not asked for while a write to it is in flight (the table of
//     tnca_axi_bridge, holding writes only here): AXI may let the read pass.
//   * An atomic waits until nothing of ours is in flight and no waiter is
//     unanswered, drops its line from the buffer, and is the read-modify-write
//     of tnca_axi_bridge (its read on ARID = LB_N).
//
// WHAT THE BUFFER DOES NOT SEE
//   Memory written behind its back -- the host's DMA into the window
//   (python/tnca/backends/u50.py) -- is made visible by `inv`, which
//   tnca_mmio raises whenever it serves a command: every slot is dropped, and
//   a line still on its way is dropped when it arrives. The host's DMA is
//   always followed by a command (UPLOAD, then LAUNCH), and is never
//   concurrent with a kernel. The host's register window reaches memory
//   through this port and needs nothing.
//==============================================================================
`default_nettype none

module tnca_axi_bridge_lb #(
    parameter TAG_W       = 5,
    parameter OUT_N       = 8,          // writes in flight
    parameter LOCAL_LIMIT = 64'h20000,
    parameter AXI_ID_W    = TAG_W,
    parameter QUIET       = 1024,
    parameter LINE_BEATS  = 8,          // 8-byte beats per line (a power of two, >= 2)
    parameter LB_N        = 8,          // lines in the buffer
    parameter WAIT_N      = 8,          // reads accepted and not yet answered
    parameter PF          = 1,          // lines fetched ahead of the one asked for
    parameter LOCAL_PIPE  = 0,          // queue local RAM requests as well
    parameter RAM_CACHE   = 0           // synchronous dual-port line storage
) (
    input  wire              clk,
    input  wire              rst_n,

    input  wire              req_valid,
    output wire              req_ready,
    input  wire [63:0]       req_addr,
    input  wire              req_we,
    input  wire [7:0]        req_be,
    input  wire [63:0]       req_wdata,
    input  wire [TAG_W-1:0]  req_tag,
    input  wire              req_amo,
    input  wire [3:0]        req_aop,
    input  wire [63:0]       req_cmp,
    output reg               rsp_valid,
    output reg  [TAG_W-1:0]  rsp_tag,
    output reg  [63:0]       rsp_rdata,

    output wire              l_req_valid,
    input  wire              l_req_ready,
    output wire [63:0]       l_req_addr,
    output wire              l_req_we,
    output wire [7:0]        l_req_be,
    output wire [63:0]       l_req_wdata,
    output wire [TAG_W-1:0]  l_req_tag,
    output wire              l_req_amo,
    output wire [3:0]        l_req_aop,
    output wire [63:0]       l_req_cmp,
    input  wire              l_rsp_valid,
    input  wire [TAG_W-1:0]  l_rsp_tag,
    input  wire [63:0]       l_rsp_rdata,

    input  wire [63:0]       axi_base,
    input  wire [63:0]       axi_size,
    input  wire              inv,
    output reg               axi_err,
    output reg               axi_stray,
    output reg  [63:0]       cnt_rd,
    output reg  [63:0]       cnt_wr,
    output reg  [63:0]       cnt_rd_lat,
    output reg  [63:0]       cnt_rbeat,

    output wire              m_awvalid,
    input  wire              m_awready,
    output wire [63:0]       m_awaddr,
    output wire [AXI_ID_W-1:0] m_awid,
    output wire [7:0]        m_awlen,
    output wire [2:0]        m_awsize,
    output wire [1:0]        m_awburst,
    output wire              m_wvalid,
    input  wire              m_wready,
    output wire [63:0]       m_wdata,
    output wire [7:0]        m_wstrb,
    output wire              m_wlast,
    input  wire              m_bvalid,
    output wire              m_bready,
    input  wire [AXI_ID_W-1:0] m_bid,
    input  wire [1:0]        m_bresp,
    output reg               m_arvalid,
    input  wire              m_arready,
    output reg  [63:0]       m_araddr,
    output reg  [AXI_ID_W-1:0] m_arid,
    output reg  [7:0]        m_arlen,
    output wire [2:0]        m_arsize,
    output wire [1:0]        m_arburst,
    input  wire              m_rvalid,
    output wire              m_rready,
    input  wire [63:0]       m_rdata,
    input  wire [AXI_ID_W-1:0] m_rid,
    input  wire [1:0]        m_rresp,
    input  wire              m_rlast
);
`include "tnca_amo.vh"
    localparam LW    = $clog2(LINE_BEATS);
    localparam LB_W  = (LB_N > 1) ? $clog2(LB_N) : 1;
    localparam WT_W  = (WAIT_N > 1) ? $clog2(WAIT_N) : 1;
    localparam CNT_W = $clog2(OUT_N + 1);
    localparam [AXI_ID_W-1:0] AMO_ID = LB_N;

    assign m_awid    = {AXI_ID_W{1'b0}};
    assign m_awlen   = 8'd0;
    assign m_awsize  = 3'd3;
    assign m_awburst = 2'b01;
    assign m_wlast   = 1'b1;
    assign m_arsize  = 3'd3;
    assign m_arburst = 2'b01;

    //--------------------------------------------------------------------------
    // The window's requests wait in a two-deep FIFO
    //
    // So that whether the core may hand over a request for the window depends
    // on a registered count and nothing else: the lookup below -- a request's
    // line against every buffered one -- runs on the FIFO's head, a cycle
    // later, instead of between the core's address arithmetic and its accept.
    // That path was 9.1 ns post-synthesis at 100 MHz without it. The on-chip
    // RAM's requests pass straight through as before; the two memories have
    // no order between them to keep.
    //--------------------------------------------------------------------------
    wire        req_local = req_addr < LOCAL_LIMIT;
    reg [63:0]       f_addr  [0:1];
    reg              f_we    [0:1];
    reg [7:0]        f_be    [0:1];
    reg [63:0]       f_wdata [0:1];
    reg [TAG_W-1:0]  f_tag   [0:1];
    reg              f_amo   [0:1];
    reg [3:0]        f_aop   [0:1];
    reg [63:0]       f_cmp   [0:1];
    reg [1:0]        f_n;
    reg              f_h, f_t;
    wire             h_v     = f_n != 2'd0;
    wire [63:0]      h_addr  = f_addr[f_h];
    wire             h_we    = f_we[f_h];
    wire [7:0]       h_be    = f_be[f_h];
    wire [63:0]      h_wdata = f_wdata[f_h];
    wire [TAG_W-1:0] h_tag   = f_tag[f_h];
    wire             h_amo   = f_amo[f_h];
    wire [3:0]       h_aop   = f_aop[f_h];
    wire [63:0]      h_cmp   = f_cmp[f_h];
    wire            h_local = LOCAL_PIPE && h_addr < LOCAL_LIMIT;

    //--------------------------------------------------------------------------
    // The window request at the FIFO's head
    //--------------------------------------------------------------------------
    wire [63:0] off      = h_addr - LOCAL_LIMIT;
    wire        in_range = off < axi_size;
    wire [63:0] word     = {h_addr[63:3], 3'b000};
    wire [63:0] axi_addr = axi_base + {off[63:3], 3'b000};
    wire [63:0] rline    = off >> (3 + LW);           // line index within the window
    wire [LW-1:0] rbeat  = off[3 + LW - 1:3];

    //--------------------------------------------------------------------------
    // State
    //--------------------------------------------------------------------------
    // lines
    reg              lb_v    [0:LB_N-1];   // slot holds (or is getting) lb_line
    reg              lb_fill [0:LB_N-1];   // its data is on its way
    reg              lb_ask  [0:LB_N-1];   // its AR has not been issued yet
    reg              lb_dem  [0:LB_N-1];   // asked for by a read (not a prefetch)
    reg              lb_kill [0:LB_N-1];   // invalidated while on its way
    reg [63:0]       lb_line [0:LB_N-1];
    reg [LW:0]       lb_cnt  [0:LB_N-1];
    reg [63:0]       lb_data [0:LB_N*LINE_BEATS-1];
    wire             cache_rsp_valid;
    wire [TAG_W-1:0]  cache_rsp_tag;
    wire [63:0]       cache_rsp_data;
    reg              wr_go;
    reg [WT_W-1:0]    wr_i;
    reg [LB_W-1:0]   lb_rr;

    // waiters
    reg              w_v   [0:WAIT_N-1];
    reg [TAG_W-1:0]  w_tag [0:WAIT_N-1];
    reg [LB_W-1:0]   w_ent [0:WAIT_N-1];
    reg [LW-1:0]     w_bt  [0:WAIT_N-1];

    // writes in flight (one ID, in order): word address and tag
    reg              t_v   [0:OUT_N-1];
    reg [63:0]       t_a   [0:OUT_N-1];
    reg [TAG_W-1:0]  t_tag [0:OUT_N-1];
    reg [CNT_W-1:0]  t_n;
    reg [TAG_W-1:0]  wq [0:OUT_N-1];
    reg [$clog2(OUT_N)-1:0] wq_head, wq_tail;

    reg              null_v;
    reg [TAG_W-1:0]  null_tag;
    reg              amo_rsp_v;
    reg [TAG_W-1:0]  amo_rsp_tag;
    reg [63:0]       amo_rsp_data;
    wire cache_port_free = !cache_rsp_valid ||
                           (!l_rsp_valid && !null_v && !amo_rsp_v);

    localparam A_IDLE = 3'd0, A_DRAIN = 3'd1, A_AR = 3'd2, A_R = 3'd3,
               A_W = 3'd4, A_B = 3'd5;
    reg [2:0]        a_st;
    reg [63:0]       a_addr, a_line;
    reg [7:0]        a_be;
    reg [63:0]       a_wdata, a_cmp, a_old;
    reg [3:0]        a_op;
    reg [TAG_W-1:0]  a_tag;
    wire amo_own = (a_st == A_AR) || (a_st == A_R) || (a_st == A_W) || (a_st == A_B);

    localparam Q_W = $clog2(QUIET + 1);
    reg [Q_W-1:0] quiet;
    wire quiet_now = quiet != 0;
    reg [LB_W+1:0] rd_fly;

    // Writes leave through two small FIFOs, one per channel, so a write can be
    // accepted every cycle while the last one's AW or W is still being taken
    // (tnca_axi_bridge accepts one only once both are gone: one per two
    // cycles at best). AXI keeps AW order and W order; that is all one ID
    // needs.
    localparam WQ = 4;
    reg [63:0] awq_a [0:WQ-1];
    reg [63:0] wq_d  [0:WQ-1];
    reg [7:0]  wq_s  [0:WQ-1];
    reg [$clog2(WQ):0]   awq_n, wqq_n;
    reg [$clog2(WQ)-1:0] awq_h, awq_t, wqq_h, wqq_t;
    assign m_awvalid = awq_n != 0;
    assign m_awaddr  = awq_a[awq_h];
    assign m_wvalid  = wqq_n != 0;
    assign m_wdata   = wq_d[wqq_h];
    assign m_wstrb   = wq_s[wqq_h];
    wire w_room = (awq_n < WQ) && (wqq_n < WQ);
    wire w_hold = m_awvalid | m_wvalid;

    //--------------------------------------------------------------------------
    // Lookups
    //--------------------------------------------------------------------------
    // Per slot: a waiter is on it (it may not be reused), and a write to its
    // line is in flight (its line may not be asked for). Vectors computed here
    // rather than functions called below: a function's reads of these arrays
    // are not in an always @(*)'s sensitivity.
    reg [LB_N-1:0] wref, lwb;
    integer e, j;
    reg [63:0] t_off;
    always @(*) begin
        for (e = 0; e < LB_N; e = e + 1) begin
            wref[e] = 1'b0;
            lwb[e]  = 1'b0;
            for (j = 0; j < WAIT_N; j = j + 1)
                if (w_v[j] && w_ent[j] == e) wref[e] = 1'b1;
            for (j = 0; j < OUT_N; j = j + 1) begin
                t_off = t_a[j] - LOCAL_LIMIT;
                if (t_v[j] && (t_off >> (3 + LW)) == lb_line[e]) lwb[e] = 1'b1;
            end
        end
    end

    integer k;
    reg              l_hit, l_pend, have_slot, have_wait, w_line_ref, t_word_hz, any_fill, any_wait;
    reg [LB_W-1:0]   l_ent, slot_i;
    reg [WT_W-1:0]   wait_i;
    reg              rr_found;
    always @(*) begin
        l_hit = 1'b0; l_pend = 1'b0; l_ent = 0;
        have_slot = 1'b0; slot_i = 0; rr_found = 1'b0;
        have_wait = 1'b0; wait_i = 0;
        w_line_ref = 1'b0; t_word_hz = 1'b0; any_fill = 1'b0; any_wait = 1'b0;
        for (k = 0; k < LB_N; k = k + 1) begin
            if (lb_v[k] && lb_line[k] == rline) begin
                l_ent = k;
                if (lb_fill[k]) l_pend = 1'b1; else l_hit = 1'b1;
            end
            if (lb_fill[k]) any_fill = 1'b1;
        end
        // a slot to (re)use: an empty one, else the first from lb_rr that is
        // neither on its way nor waited on
        for (k = 0; k < LB_N; k = k + 1)
            if (!lb_v[k] && !have_slot) begin have_slot = 1'b1; slot_i = k; end
        for (k = 0; k < 2 * LB_N; k = k + 1)
            if (!have_slot && k >= lb_rr && k < lb_rr + LB_N &&
                !lb_fill[k % LB_N] && !wref[k % LB_N]) begin
                have_slot = 1'b1; slot_i = k % LB_N;
            end
        for (k = 0; k < WAIT_N; k = k + 1) begin
            if (!w_v[k] && !have_wait) begin have_wait = 1'b1; wait_i = k; end
            if (w_v[k]) any_wait = 1'b1;
        end
        for (k = 0; k < OUT_N; k = k + 1)
            if (t_v[k] && t_a[k] == word) t_word_hz = 1'b1;
        // A waiter reads a buffered line through its slot, so "a waiter on this
        // line" is a waiter on the slot the line is in: an index compare, not
        // a line-address compare per waiter.
        w_line_ref = (l_hit || l_pend) && wref[l_ent];
    end

    // The next AR: a demand fill first, else a prefetch, whose line has no
    // write in flight.
    reg              ar_go;
    reg [LB_W-1:0]   ar_e;
    always @(*) begin
        ar_go = 1'b0; ar_e = 0;
        for (k = LB_N - 1; k >= 0; k = k - 1)
            if (lb_ask[k] && !lb_dem[k] && !lwb[k]) begin ar_go = 1'b1; ar_e = k; end
        for (k = LB_N - 1; k >= 0; k = k - 1)
            if (lb_ask[k] && lb_dem[k] && !lwb[k]) begin ar_go = 1'b1; ar_e = k; end
    end

    // The slot a prefetch takes: an empty one, else the first from lb_rr that
    // is not on its way, not waited on, not the line just read and not the
    // slot a miss is taking this cycle.
    reg              pf_ok;
    reg [LB_W-1:0]   pf_slot;
    always @(*) begin
        pf_ok = 1'b0; pf_slot = 0;
        for (k = 0; k < LB_N; k = k + 1)
            if (!pf_ok && !lb_v[k] && !(have_slot && !l_hit && !l_pend && k == slot_i)) begin
                pf_ok = 1'b1; pf_slot = k;
            end
        for (k = 0; k < 2 * LB_N; k = k + 1)
            if (!pf_ok && k > lb_rr && k <= lb_rr + LB_N && !lb_fill[k % LB_N] && !wref[k % LB_N] &&
                !((l_hit || l_pend) && (k % LB_N) == l_ent) &&
                !(!l_hit && !l_pend && (k % LB_N) == slot_i)) begin
                pf_ok = 1'b1; pf_slot = k % LB_N;
            end
    end

    // Prefetch: the first of the PF lines past the one a read touched that is
    // neither buffered nor past the window's end.
    reg              pf_go;
    reg [63:0]       pf_line;
    reg              pf_have;
    integer p;
    always @(*) begin
        pf_go = 1'b0; pf_line = 64'd0;
        for (p = PF; p >= 1; p = p - 1) begin
            pf_have = 1'b0;
            for (k = 0; k < LB_N; k = k + 1)
                if (lb_v[k] && lb_line[k] == rline + p) pf_have = 1'b1;
            if (!pf_have && ((rline + p + 1) << (3 + LW)) <= axi_size) begin
                pf_go = 1'b1; pf_line = rline + p;
            end
        end
    end

    //--------------------------------------------------------------------------
    // Acceptance
    //--------------------------------------------------------------------------
    wire axi_path_free = (a_st == A_IDLE) && !amo_rsp_v;
    // a read: a waiter slot, and its line buffered, on its way, or a slot for it
    wire can_read  = axi_path_free && have_wait && (l_hit || l_pend || have_slot);
    // a write: not into a line on its way or one a waiter still reads
    wire can_write = axi_path_free && !l_pend && !w_line_ref && !t_word_hz &&
                     (t_n < OUT_N) && w_room &&
                     (!RAM_CACHE || !l_hit || (cache_port_free && !wr_go));
    wire can_amo   = axi_path_free;
    wire can_null  = !null_v;

    assign req_ready = quiet_now ? 1'b0
                     : LOCAL_PIPE ? (f_n != 2'd2)
                     : req_local ? l_req_ready
                     : (f_n != 2'd2);
    wire f_push  = req_valid && (LOCAL_PIPE || !req_local) && !quiet_now && (f_n != 2'd2);
    // the head is taken
    wire acc     = h_v && !h_local && (!in_range ? can_null
                           : h_amo ? can_amo
                           : h_we ? can_write : can_read);
    wire acc_axi = acc && in_range;
    wire acc_rd  = acc_axi && !h_we && !h_amo;
    wire acc_wr  = acc_axi && h_we && !h_amo;

    // In the pipelined mode only registered head fields reach BRAM.
    // Keep the complete request stable until the local ready handshake.
    assign l_req_valid = !quiet_now && (LOCAL_PIPE ? h_v && h_local : req_valid && req_local);
    assign l_req_addr  = LOCAL_PIPE ? h_addr : req_addr;
    assign l_req_we    = LOCAL_PIPE ? h_we : req_we;
    assign l_req_be    = LOCAL_PIPE ? h_be : req_be;
    assign l_req_wdata = LOCAL_PIPE ? h_wdata : req_wdata;
    assign l_req_tag   = LOCAL_PIPE ? h_tag : req_tag;
    assign l_req_amo   = LOCAL_PIPE ? h_amo : req_amo;
    assign l_req_aop   = LOCAL_PIPE ? h_aop : req_aop;
    assign l_req_cmp   = LOCAL_PIPE ? h_cmp : req_cmp;
    wire f_pop = acc || (LOCAL_PIPE && l_req_valid && l_req_ready);

    //--------------------------------------------------------------------------
    // Responses: the RAM first, then out-of-range, an atomic, a read, a write
    //--------------------------------------------------------------------------
    always @(*) begin
        wr_go = 1'b0; wr_i = 0;
        for (k = WAIT_N - 1; k >= 0; k = k - 1)
            if (w_v[k] && !lb_fill[w_ent[k]]) begin wr_go = 1'b1; wr_i = k; end
    end

    wire r_amo   = m_rvalid && m_rid == AMO_ID;
    wire r_fill  = m_rvalid && m_rid < LB_N && lb_fill[m_rid[LB_W-1:0]];
    // this beat completes its line
    wire r_done  = r_fill && (m_rlast || lb_cnt[m_rid[LB_W-1:0]] == LINE_BEATS - 1);
    wire drop_r  = m_rvalid && (quiet_now || !(r_fill || (r_amo && a_st == A_R)));
    wire b_match = (t_n != 0);
    wire drop_b  = m_bvalid && (quiet_now || (!amo_own && !b_match));

    wire take_null = !l_rsp_valid && null_v;
    wire take_amo  = !l_rsp_valid && !null_v && amo_rsp_v;
    wire read_response = RAM_CACHE ? cache_rsp_valid : wr_go;
    wire cache_read = RAM_CACHE && wr_go && cache_port_free;
    wire take_w    = !l_rsp_valid && !null_v && !amo_rsp_v && read_response;
    wire take_b    = !l_rsp_valid && !null_v && !amo_rsp_v && !read_response && m_bvalid && !amo_own
                     && !drop_b;
    assign m_rready = 1'b1;                  // fills land in the buffer; never held
    assign m_bready = drop_b || (amo_own ? (a_st == A_B) : take_b);

    generate if (RAM_CACHE) begin : g_line_ram
        reg pending;
        reg [TAG_W-1:0] tag;
        wire write_hit = acc_wr && l_hit;
        tnca_axi_line_ram #(.WORDS(LB_N*LINE_BEATS), .ADDR_W(LB_W+LW)) u_data (
            .clk(clk), .a_we(r_fill && !drop_r),
            .a_addr({m_rid[LB_W-1:0], lb_cnt[m_rid[LB_W-1:0]][LW-1:0]}),
            .a_wdata(m_rdata), .b_en(cache_read || write_hit),
            .b_we(write_hit ? h_be : 8'd0),
            .b_addr(cache_read ? {w_ent[wr_i], w_bt[wr_i]} : {l_ent, rbeat}),
            .b_wdata(h_wdata), .b_rdata(cache_rsp_data)
        );
        assign cache_rsp_valid = pending;
        assign cache_rsp_tag = tag;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin pending <= 1'b0; tag <= 0; end
            else begin
                if (take_w) pending <= 1'b0;
                if (cache_read) begin pending <= 1'b1; tag <= w_tag[wr_i]; end
            end
        end
    end else begin : g_line_registers
        assign cache_rsp_valid = 1'b0;
        assign cache_rsp_tag = {TAG_W{1'b0}};
        assign cache_rsp_data = 64'd0;
    end endgenerate

    always @(*) begin
        rsp_valid = 1'b0;
        rsp_tag   = {TAG_W{1'b0}};
        rsp_rdata = 64'd0;
        if (l_rsp_valid) begin
            rsp_valid = 1'b1; rsp_tag = l_rsp_tag; rsp_rdata = l_rsp_rdata;
        end else if (take_null) begin
            rsp_valid = 1'b1; rsp_tag = null_tag;
        end else if (take_amo) begin
            rsp_valid = 1'b1; rsp_tag = amo_rsp_tag; rsp_rdata = amo_rsp_data;
        end else if (take_w) begin
            rsp_valid = 1'b1; rsp_tag = RAM_CACHE ? cache_rsp_tag : w_tag[wr_i];
            rsp_rdata = RAM_CACHE ? cache_rsp_data : lb_data[{w_ent[wr_i], w_bt[wr_i]}];
        end else if (take_b) begin
            rsp_valid = 1'b1; rsp_tag = wq[wq_head];
        end
    end

    function [63:0] merge;
        input [63:0] old, data; input [7:0] be;
        integer b;
        begin
            merge = old;
            for (b = 0; b < 8; b = b + 1) if (be[b]) merge[b*8 +: 8] = data[b*8 +: 8];
        end
    endfunction

    //--------------------------------------------------------------------------
    // State
    //--------------------------------------------------------------------------
    integer i;
    reg [LB_W-1:0] re;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < LB_N; i = i + 1) begin
                lb_v[i] <= 1'b0; lb_fill[i] <= 1'b0; lb_ask[i] <= 1'b0; lb_dem[i] <= 1'b0;
                lb_kill[i] <= 1'b0; lb_cnt[i] <= 0;
            end
            for (i = 0; i < WAIT_N; i = i + 1) w_v[i] <= 1'b0;
            for (i = 0; i < OUT_N; i = i + 1) t_v[i] <= 1'b0;
            lb_rr <= 0; t_n <= 0; wq_head <= 0; wq_tail <= 0;
            null_v <= 1'b0; amo_rsp_v <= 1'b0; a_st <= A_IDLE;
            awq_n <= 0; wqq_n <= 0; awq_h <= 0; awq_t <= 0; wqq_h <= 0; wqq_t <= 0;
            f_n <= 2'd0; f_h <= 1'b0; f_t <= 1'b0;
            m_arvalid <= 1'b0;
            axi_err <= 1'b0; axi_stray <= 1'b0;
            quiet <= QUIET;
            cnt_rd <= 64'd0; cnt_wr <= 64'd0; cnt_rd_lat <= 64'd0; cnt_rbeat <= 64'd0; rd_fly <= 0;
        end else begin
            if (quiet_now) quiet <= quiet - 1'b1;
            if ((drop_r || drop_b) && !quiet_now) axi_stray <= 1'b1;

            // ---- counters ----------------------------------------------------
            if (m_arvalid && m_arready) cnt_rd <= cnt_rd + 64'd1;
            if (m_awvalid && m_awready) cnt_wr <= cnt_wr + 64'd1;
            if (m_rvalid && !drop_r) cnt_rbeat <= cnt_rbeat + 64'd1;
            cnt_rd_lat <= cnt_rd_lat + rd_fly;
            rd_fly <= rd_fly + ((m_arvalid && m_arready) ? 1 : 0)
                             - ((m_rvalid && !drop_r && m_rlast) ? 1 : 0);

            // ---- channels completing ------------------------------------------
            if (m_arvalid && m_arready) m_arvalid <= 1'b0;
            // (the write FIFOs' counts are settled at the end of this block)
            if (take_null) null_v <= 1'b0;
            if (take_amo)  amo_rsp_v <= 1'b0;
            if (RAM_CACHE ? cache_read : take_w) w_v[wr_i] <= 1'b0;

            // ---- a fill beat ------------------------------------------------
            if (r_fill && !drop_r) begin
                re = m_rid[LB_W-1:0];
                if (m_rresp[1]) axi_err <= 1'b1;
                if (!RAM_CACHE) lb_data[{re, lb_cnt[re][LW-1:0]}] <= m_rdata;
                lb_cnt[re] <= lb_cnt[re] + 1'b1;
                if (m_rlast || lb_cnt[re] == LINE_BEATS - 1) begin
                    lb_fill[re] <= 1'b0;
                    if (lb_kill[re]) begin lb_v[re] <= 1'b0; lb_kill[re] <= 1'b0; end
                end
            end

            // ---- a write's B --------------------------------------------------
            if (m_bvalid && m_bready && !drop_b) begin
                if (m_bresp[1]) axi_err <= 1'b1;
                if (!amo_own) begin
                    for (i = 0; i < OUT_N; i = i + 1)
                        if (t_v[i] && t_tag[i] == wq[wq_head]) t_v[i] <= 1'b0;
                    wq_head <= wq_head + 1'b1;
                end
            end

            // ---- the next AR ---------------------------------------------------
            // (an atomic's drain waits for these; from its read on it owns AR)
            if (!m_arvalid && ar_go && !amo_own) begin
                m_arvalid <= 1'b1;
                m_araddr  <= axi_base + (lb_line[ar_e] << (3 + LW));
                m_arid    <= ar_e;
                m_arlen   <= LINE_BEATS - 1;
                lb_ask[ar_e] <= 1'b0;
            end

            // ---- a new request ------------------------------------------------
            if (acc && !in_range) begin
                null_v <= 1'b1; null_tag <= h_tag;
            end
            if (acc_rd) begin
                w_v[wait_i] <= 1'b1; w_tag[wait_i] <= h_tag; w_bt[wait_i] <= rbeat;
                if (l_hit || l_pend) begin
                    w_ent[wait_i] <= l_ent;
                end else begin
                    w_ent[wait_i] <= slot_i;
                    lb_v[slot_i] <= 1'b1; lb_fill[slot_i] <= 1'b1; lb_ask[slot_i] <= 1'b1;
                    lb_dem[slot_i] <= 1'b1; lb_kill[slot_i] <= 1'b0; lb_cnt[slot_i] <= 0;
                    lb_line[slot_i] <= rline;
                    lb_rr <= slot_i + 1'b1;
                end
            end
            // a prefetch, into a slot other than the one a miss just took
            if (acc_rd && pf_go && pf_ok && PF > 0) begin
                lb_v[pf_slot] <= 1'b1; lb_fill[pf_slot] <= 1'b1; lb_ask[pf_slot] <= 1'b1;
                lb_dem[pf_slot] <= 1'b0; lb_kill[pf_slot] <= 1'b0; lb_cnt[pf_slot] <= 0;
                lb_line[pf_slot] <= pf_line;
            end
            if (acc_wr) begin
                t_v[t_free(0)] <= 1'b1; t_a[t_free(0)] <= word; t_tag[t_free(0)] <= h_tag;
                awq_a[awq_t] <= axi_addr;
                wq_d[wqq_t] <= h_wdata; wq_s[wqq_t] <= h_be;
                wq[wq_tail] <= h_tag; wq_tail <= wq_tail + 1'b1;
                if (l_hit && !RAM_CACHE)
                    lb_data[{l_ent, rbeat}] <= merge(lb_data[{l_ent, rbeat}], h_wdata, h_be);
            end
            if (acc_axi && h_amo) begin
                a_st <= A_DRAIN; a_addr <= axi_addr; a_line <= rline;
                a_be <= h_be; a_wdata <= h_wdata; a_cmp <= h_cmp;
                a_op <= h_aop; a_tag <= h_tag;
            end

            // ---- invalidation (a command was served) ---------------------------
            // After the fill above: a line completing on this very cycle is
            // dropped outright, since its kill would come too late.
            if (inv) begin
                for (i = 0; i < LB_N; i = i + 1)
                    if (lb_fill[i] && !(r_done && m_rid[LB_W-1:0] == i)) lb_kill[i] <= 1'b1;
                    else begin lb_v[i] <= 1'b0; lb_kill[i] <= 1'b0; end
            end

            // ---- the atomic -------------------------------------------------
            case (a_st)
            A_DRAIN: begin
                if (!m_arvalid && !w_hold && t_n == 0 && !any_fill && !any_wait) begin
                    for (i = 0; i < LB_N; i = i + 1)
                        if (lb_v[i] && lb_line[i] == a_line) lb_v[i] <= 1'b0;
                    m_arvalid <= 1'b1; m_araddr <= a_addr; m_arid <= AMO_ID; m_arlen <= 8'd0;
                    a_st <= A_AR;
                end
            end
            A_AR: if (m_arvalid && m_arready) a_st <= A_R;
            A_R: if (r_amo && !drop_r) begin
                if (m_rresp[1]) axi_err <= 1'b1;
                a_old <= m_rdata;
                if (a_op == `TNCA_AOP_LD) begin
                    amo_rsp_v <= 1'b1; amo_rsp_tag <= a_tag; amo_rsp_data <= m_rdata;
                    a_st <= A_IDLE;
                end else begin
                    awq_a[awq_t] <= a_addr;
                    wq_s[wqq_t]  <= a_be;
                    wq_d[wqq_t]  <= amo_new(a_op, a_be, m_rdata, a_wdata, a_cmp);
                    a_st <= A_W;
                end
            end
            A_W: if (!w_hold) a_st <= A_B;
            A_B: if (m_bvalid && !drop_b) begin
                amo_rsp_v <= 1'b1; amo_rsp_tag <= a_tag; amo_rsp_data <= a_old;
                a_st <= A_IDLE;
            end
            default: ;
            endcase

            // ---- the write FIFOs: one push (a write, or an atomic's), one pop each
            begin : wfifo
                reg push;
                push = acc_wr || (a_st == A_R && r_amo && !drop_r && a_op != `TNCA_AOP_LD);
                if (push) begin awq_t <= awq_t + 1'b1; wqq_t <= wqq_t + 1'b1; end
                if (m_awvalid && m_awready) awq_h <= awq_h + 1'b1;
                if (m_wvalid && m_wready)   wqq_h <= wqq_h + 1'b1;
                awq_n <= awq_n + (push ? 1 : 0) - ((m_awvalid && m_awready) ? 1 : 0);
                wqq_n <= wqq_n + (push ? 1 : 0) - ((m_wvalid && m_wready) ? 1 : 0);
            end

            // ---- the window's input FIFO ------------------------------------------
            if (f_push) begin
                f_addr[f_t] <= req_addr; f_we[f_t] <= req_we; f_be[f_t] <= req_be;
                f_wdata[f_t] <= req_wdata; f_tag[f_t] <= req_tag; f_amo[f_t] <= req_amo;
                f_aop[f_t] <= req_aop; f_cmp[f_t] <= req_cmp;
                f_t <= ~f_t;
            end
            if (f_pop) f_h <= ~f_h;
            f_n <= f_n + (f_push ? 2'd1 : 2'd0) - (f_pop ? 2'd1 : 2'd0);

            // ---- occupancy of the write table ---------------------------------
            t_n <= t_n + (acc_wr ? 1 : 0)
                       - ((m_bvalid && m_bready && !amo_own && !drop_b) ? 1 : 0);
        end
    end

    // the lowest free write-table entry
    function [$clog2(OUT_N)-1:0] t_free;
        input integer dummy;
        integer j;
        reg found;
        begin
            t_free = 0; found = 1'b0;
            for (j = 0; j < OUT_N; j = j + 1)
                if (!t_v[j] && !found) begin t_free = j; found = 1'b1; end
        end
    endfunction
endmodule
`default_nettype wire
