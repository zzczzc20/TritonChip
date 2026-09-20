//==============================================================================
// tnca_nuca_slice -- one slice of the single-level distributed shared cache
//
// This is the only data storage between the register files and DRAM (arch
// overview sec 1.1). There is no private L1, no replication and no migration:
// every physical line has exactly one home slice, chosen by a hash of its
// address (sec 10.1), so "which slice owns this line" is never ambiguous and
// there is no coherence protocol between slices.
//
// STRUCTURE (sec 8.1)
//
//   request buffer -> arbiter -> tag lookup -> hit: data array -> response
//                                           -> miss: MSHR entry + target
//   memory response ----------> fill -> data array -> wake targets -> responses
//   dirty victim --------------> writeback buffer -> memory
//
// WHAT THE ARBITER IS FOR (sec 9.3)
//
// "MSHR 满时也应尽量继续服务 hit" -- a miss that cannot allocate must not block
// hits queued behind it. The request buffer is therefore not a FIFO: each cycle
// the arbiter picks the oldest entry that is not blocked and has no older entry
// to the same line. A miss that fails to allocate is marked blocked and retried
// when an MSHR retires; hits to other lines keep flowing past it.
//
// The "no older entry to the same line" rule is what serialises conflicting
// accesses to one address at its home slice (sec 14.1). It is a correctness
// property, not a performance heuristic.
//
// WHAT IS DELIBERATELY NOT HERE (Stage 4)
//
// No TensorMap metadata, no dead-block prediction, no coordinated bypass, no
// priority-based replacement, no pressure-adaptive admission. Stage 3's job is
// a baseline that those can be measured against one at a time (sec 21), and a
// baseline with a policy already in it cannot serve that purpose. Replacement
// here is invalid-first then round-robin, which is deliberately uninteresting.
//
// The engine processes one thing per cycle with completion paths ahead of new
// work, so a fill or a writeback always wins over a fresh request: throttling
// must never hold up the paths that release resources (sec 2.4, sec 10.5).
//==============================================================================
`include "tnca_defs.vh"

module tnca_nuca_slice #(
    parameter ADDR_W     = 64,
    parameter LINE_BYTES = 64,
    parameter LINE_BITS  = 512,
    parameter WORD_BITS  = 64,
    parameter WORD_BYTES = 8,
    parameter SETS       = 64,
    parameter SET_W      = 6,        // $clog2(SETS)
    parameter WAYS       = 8,
    parameter WAY_W      = 3,        // $clog2(WAYS)
    parameter MSHR_N     = 8,
    parameter MSHR_W     = 3,
    parameter TARGET_N   = 16,
    parameter TARGET_W   = 4,
    parameter RQ_N       = 8,        // request buffer entries
    parameter RQ_W       = 3,
    parameter SRC_W      = 4,
    parameter TAG_W      = 4
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- requester side -----------------------------------------------------
    output wire                     req_ready,
    input  wire                     req_valid,
    input  wire [ADDR_W-1:0]        req_addr,      // word-aligned byte address
    input  wire                     req_we,
    input  wire [WORD_BYTES-1:0]    req_be,
    input  wire [WORD_BITS-1:0]     req_wdata,
    input  wire [SRC_W-1:0]         req_src,
    input  wire [TAG_W-1:0]         req_tag,
    // Atomics execute HERE, at the home slice (ISA sec 9.2, arch sec 12.9),
    // because that is the only place in the machine where every access to a
    // physical line is already serialised. The arbiter's "no older entry to
    // the same line" rule is what makes the read-modify-write indivisible;
    // no lock, no retry, and nothing added to the miss path.
    input  wire                     req_amo,
    input  wire [3:0]               req_aop,
    input  wire [WORD_BITS-1:0]     req_cmp,

    output wire                     rsp_valid,
    input  wire                     rsp_ready,
    output wire [SRC_W-1:0]         rsp_src,
    output wire [TAG_W-1:0]         rsp_tag,
    output wire [WORD_BITS-1:0]     rsp_rdata,

    // ---- memory side (line granular) ----------------------------------------
    // Driven combinationally from the engine state. A registered `valid` that
    // is set and cleared in the same cycle never appears on the wire, and the
    // request is silently lost while the state machine believes it was sent.
    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [ADDR_W-1:0]        mem_req_addr,  // line aligned
    output wire                     mem_req_we,
    output wire [LINE_BITS-1:0]     mem_req_wdata,
    output wire [MSHR_W-1:0]        mem_req_tag,

    input  wire                     mem_rsp_valid,
    input  wire [MSHR_W-1:0]        mem_rsp_tag,
    input  wire [LINE_BITS-1:0]     mem_rsp_rdata,

    // ---- performance counters (sec 20.2) ------------------------------------
    output reg  [31:0]              perf_hit,
    output reg  [31:0]              perf_miss,
    output reg  [31:0]              perf_merge,
    output reg  [31:0]              perf_mshr_full,
    output reg  [31:0]              perf_target_full,
    output reg  [31:0]              perf_wb,
    output reg  [31:0]              perf_hit_past_block
);

    localparam OFF_W     = 6;                       // log2(LINE_BYTES)
    localparam WORDS_PER_LINE = LINE_BYTES / WORD_BYTES;
    localparam WSEL_W   = 3;                        // log2(WORDS_PER_LINE)
    localparam TAGB     = ADDR_W - OFF_W - SET_W;
    localparam LINES    = SETS * WAYS;
    localparam LINE_W   = SET_W + WAY_W;

    integer i, j, w;

    //==========================================================================
    // The atomic ALU (ISA sec 9.2)
    //
    // Atomics execute here, at the home slice, because this is the only place
    // where every access to a physical line is already serialised: the
    // arbiter will not start a second request to a line while one is in
    // flight (sec 14.1). So the read-modify-write needs no lock, no retry and
    // nothing added to the miss path -- it is one more thing the existing
    // order does for free.
    //==========================================================================
`include "tnca_amo.vh"

    //==========================================================================
    // Address decomposition
    //==========================================================================
    `define SLICE_SET(ad)  (ad[OFF_W +: SET_W])
    `define SLICE_TAG(ad)  (ad[ADDR_W-1 -: TAGB])
    `define SLICE_WSEL(ad) (ad[WORD_BYTES==8 ? 3 : 2 +: WSEL_W])

    //==========================================================================
    // Arrays
    //==========================================================================
    reg [LINE_BITS-1:0] data_mem [0:LINES-1];
    reg [TAGB-1:0]      tag_mem  [0:LINES-1];
    reg [LINES-1:0]     v_bits;
    reg [LINES-1:0]     d_bits;
    reg [WAY_W-1:0]     victim_ptr [0:SETS-1];

    //==========================================================================
    // Request buffer -- not a FIFO; see the arbiter note in the header
    //==========================================================================
    reg [RQ_N-1:0]          rq_v;
    reg [RQ_N-1:0]          rq_blocked;
    reg [ADDR_W-1:0]        rq_addr  [0:RQ_N-1];
    reg                     rq_we    [0:RQ_N-1];
    reg                     rq_amo   [0:RQ_N-1];
    reg [3:0]               rq_aop   [0:RQ_N-1];
    reg [WORD_BITS-1:0]     rq_cmp   [0:RQ_N-1];
    reg [WORD_BYTES-1:0]    rq_be    [0:RQ_N-1];
    reg [WORD_BITS-1:0]     rq_wdata [0:RQ_N-1];
    reg [SRC_W-1:0]         rq_src   [0:RQ_N-1];
    reg [TAG_W-1:0]         rq_tag   [0:RQ_N-1];

    // Age as a matrix, not a counter: older[i][j] means entry i arrived before
    // entry j. A counter wraps, and a wrapped counter makes "older than" cyclic
    // -- every entry then shadows another and the buffer deadlocks with nothing
    // blocked and every MSHR free. The matrix is a genuine total order on the
    // live entries and cannot wrap.
    reg [RQ_N-1:0]          older [0:RQ_N-1];

    reg [RQ_W-1:0] rq_free_idx;
    reg            rq_has_free;
    always @(*) begin
        rq_free_idx = {RQ_W{1'b0}};
        rq_has_free = 1'b0;
        for (i = RQ_N-1; i >= 0; i = i - 1)
            if (!rq_v[i]) begin
                rq_free_idx = i[RQ_W-1:0];
                rq_has_free = 1'b1;
            end
    end

    assign req_ready = rq_has_free;

    //==========================================================================
    // Arbiter: oldest entry that is neither blocked nor shadowed by an older
    // entry to the same line.
    //==========================================================================
    reg [RQ_W-1:0] sel_idx;
    reg            sel_valid;
    reg            shadowed;

    reg [RQ_N-1:0] elig;

    always @(*) begin
        // An entry is eligible when it is live, not blocked, and no OLDER live
        // entry targets the same line. That second clause is what serialises
        // conflicting accesses to one address at its home slice (sec 14.1).
        for (i = 0; i < RQ_N; i = i + 1) begin
            shadowed = 1'b0;
            for (j = 0; j < RQ_N; j = j + 1)
                if ((j != i) && rq_v[j] && older[j][i] &&
                    (rq_addr[j][ADDR_W-1:OFF_W] == rq_addr[i][ADDR_W-1:OFF_W]))
                    shadowed = 1'b1;
            elig[i] = rq_v[i] && !rq_blocked[i] && !shadowed;
        end

        // Among the eligible, take the oldest.
        sel_idx   = {RQ_W{1'b0}};
        sel_valid = 1'b0;
        for (i = 0; i < RQ_N; i = i + 1) begin
            if (elig[i]) begin
                shadowed = 1'b0;                 // reused: "an older eligible exists"
                for (j = 0; j < RQ_N; j = j + 1)
                    if ((j != i) && elig[j] && older[j][i]) shadowed = 1'b1;
                if (!shadowed) begin
                    sel_idx   = i[RQ_W-1:0];
                    sel_valid = 1'b1;
                end
            end
        end
    end

    // Did the arbiter skip over a blocked entry to get here? That is sec 9.3
    // working, and it is worth counting.
    reg any_blocked;
    always @(*) begin
        any_blocked = 1'b0;
        for (i = 0; i < RQ_N; i = i + 1)
            if (rq_v[i] && rq_blocked[i]) any_blocked = 1'b1;
    end

    //==========================================================================
    // MSHRs and targets
    //==========================================================================
    localparam MS_FREE = 3'd0,
               MS_WB   = 3'd1,   // victim must be written back first
               MS_REQ  = 3'd2,   // memory read not yet issued
               MS_WAIT = 3'd3,   // memory read outstanding
               MS_FILL = 3'd4,   // line arrived, not yet written
               MS_WAKE = 3'd5;   // servicing targets

    reg [2:0]           ms_state [0:MSHR_N-1];
    reg [ADDR_W-1:0]    ms_addr  [0:MSHR_N-1];   // line-aligned
    reg [WAY_W-1:0]     ms_way   [0:MSHR_N-1];
    reg [SET_W-1:0]     ms_set   [0:MSHR_N-1];
    reg [TARGET_W-1:0]  ms_head  [0:MSHR_N-1];
    reg [TARGET_W-1:0]  ms_tail  [0:MSHR_N-1];
    reg [MSHR_N-1:0]    ms_hasT;
    reg [LINE_BITS-1:0] ms_line  [0:MSHR_N-1];   // filled data, before wake
    reg [ADDR_W-1:0]    ms_wbadr [0:MSHR_N-1];

    reg [TARGET_N-1:0]      tg_v;
    reg [TARGET_W-1:0]      tg_next  [0:TARGET_N-1];
    reg [SRC_W-1:0]         tg_src   [0:TARGET_N-1];
    reg [TAG_W-1:0]         tg_tag   [0:TARGET_N-1];
    reg                     tg_we    [0:TARGET_N-1];
    reg                     tg_amo   [0:TARGET_N-1];
    reg [3:0]               tg_aop   [0:TARGET_N-1];
    reg [WORD_BITS-1:0]     tg_cmp   [0:TARGET_N-1];
    reg [WORD_BYTES-1:0]    tg_be    [0:TARGET_N-1];
    reg [WORD_BITS-1:0]     tg_wdata [0:TARGET_N-1];
    reg [WSEL_W-1:0]        tg_wsel  [0:TARGET_N-1];

    reg [MSHR_W-1:0] ms_free_idx;  reg ms_has_free;
    reg [TARGET_W-1:0] tg_free_idx; reg tg_has_free;
    always @(*) begin
        ms_free_idx = {MSHR_W{1'b0}}; ms_has_free = 1'b0;
        for (i = MSHR_N-1; i >= 0; i = i - 1)
            if (ms_state[i] == MS_FREE) begin
                ms_free_idx = i[MSHR_W-1:0]; ms_has_free = 1'b1;
            end
        tg_free_idx = {TARGET_W{1'b0}}; tg_has_free = 1'b0;
        for (i = TARGET_N-1; i >= 0; i = i - 1)
            if (!tg_v[i]) begin
                tg_free_idx = i[TARGET_W-1:0]; tg_has_free = 1'b1;
            end
    end

    //==========================================================================
    // Response queue -- so the engine never stalls on a backed-up network
    //==========================================================================
    localparam RSPQ_N = 4;
    reg [RSPQ_N-1:0]     rsq_v;
    reg [SRC_W-1:0]      rsq_src  [0:RSPQ_N-1];
    reg [TAG_W-1:0]      rsq_tag  [0:RSPQ_N-1];
    reg [WORD_BITS-1:0]  rsq_data [0:RSPQ_N-1];
    reg [1:0]            rsq_rd, rsq_wr;
    reg [2:0]            rsq_cnt;

    assign rsp_valid = rsq_cnt != 3'd0;
    assign rsp_src   = rsq_src[rsq_rd];
    assign rsp_tag   = rsq_tag[rsq_rd];
    assign rsp_rdata = rsq_data[rsq_rd];

    wire rsq_full = (rsq_cnt >= RSPQ_N);
    wire rsq_pop  = rsp_valid && rsp_ready;

    reg                 push_rsp;
    reg [SRC_W-1:0]     push_src;
    reg [TAG_W-1:0]     push_tag;
    reg [WORD_BITS-1:0] push_data;

    //==========================================================================
    // Engine
    //==========================================================================
    localparam E_SEL   = 4'd0,
               E_LOOK  = 4'd1,
               E_HIT   = 4'd2,
               E_MISS  = 4'd3,
               E_WBRD  = 4'd4,
               E_WBSND = 4'd5,
               E_MREQ  = 4'd6,
               E_FILL  = 4'd7,
               E_WAKE  = 4'd8;

    reg [3:0]            est;
    reg [RQ_W-1:0]       cur_rq;
    reg [ADDR_W-1:0]     cur_addr;
    reg                  cur_we;
    reg                  cur_amo;
    reg [3:0]            cur_aop;
    reg [WORD_BITS-1:0]  cur_cmp;
    reg [WORD_BYTES-1:0] cur_be;
    reg [WORD_BITS-1:0]  cur_wdata;
    reg [SRC_W-1:0]      cur_src;
    reg [TAG_W-1:0]      cur_tag;
    reg [MSHR_W-1:0]     cur_ms;
    reg [LINE_BITS-1:0]  wb_buf;

    wire [SET_W-1:0]  cur_set  = `SLICE_SET(cur_addr);
    wire [TAGB-1:0]   cur_tagb = `SLICE_TAG(cur_addr);
    wire [WSEL_W-1:0] cur_wsel = cur_addr[3 +: WSEL_W];

    // ---- tag lookup (combinational over the ways of one set) ----------------
    reg            hit;
    reg [WAY_W-1:0] hit_way;
    always @(*) begin
        hit = 1'b0; hit_way = {WAY_W{1'b0}};
        for (w = 0; w < WAYS; w = w + 1)
            if (v_bits[cur_set*WAYS + w] && (tag_mem[cur_set*WAYS + w] == cur_tagb)) begin
                hit = 1'b1; hit_way = w[WAY_W-1:0];
            end
    end

    // ---- an MSHR already covering this line? --------------------------------
    reg              ms_match;
    reg [MSHR_W-1:0] ms_match_id;
    always @(*) begin
        ms_match = 1'b0; ms_match_id = {MSHR_W{1'b0}};
        for (i = 0; i < MSHR_N; i = i + 1)
            if ((ms_state[i] != MS_FREE) &&
                (ms_addr[i][ADDR_W-1:OFF_W] == cur_addr[ADDR_W-1:OFF_W])) begin
                ms_match = 1'b1; ms_match_id = i[MSHR_W-1:0];
            end
    end

    // ---- victim selection: invalid first, then round robin, never a way that
    //      an in-flight MSHR has already claimed in this set -------------------
    reg              way_claimed [0:WAYS-1];
    reg [WAY_W-1:0]  vic_way;
    reg              vic_ok;
    reg [WAY_W-1:0]  cand;
    always @(*) begin
        for (w = 0; w < WAYS; w = w + 1) begin
            way_claimed[w] = 1'b0;
            for (i = 0; i < MSHR_N; i = i + 1)
                if ((ms_state[i] != MS_FREE) && (ms_set[i] == cur_set) &&
                    (ms_way[i] == w[WAY_W-1:0]))
                    way_claimed[w] = 1'b1;
        end
        vic_way = {WAY_W{1'b0}}; vic_ok = 1'b0;
        // invalid ways first
        for (w = WAYS-1; w >= 0; w = w - 1)
            if (!v_bits[cur_set*WAYS + w] && !way_claimed[w]) begin
                vic_way = w[WAY_W-1:0]; vic_ok = 1'b1;
            end
        if (!vic_ok) begin
            for (w = WAYS-1; w >= 0; w = w - 1) begin
                cand = victim_ptr[cur_set] + w[WAY_W-1:0];
                if (!way_claimed[cand]) begin
                    vic_way = cand; vic_ok = 1'b1;
                end
            end
        end
    end

    // ---- pending work elsewhere ---------------------------------------------
    reg              ms_pick;       // some MSHR wants the engine
    reg [MSHR_W-1:0] ms_pick_id;
    reg [2:0]        ms_pick_st;
    always @(*) begin
        ms_pick = 1'b0; ms_pick_id = {MSHR_W{1'b0}}; ms_pick_st = MS_FREE;
        // Completion first: FILL and WAKE release resources.
        for (i = MSHR_N-1; i >= 0; i = i - 1)
            if (ms_state[i] == MS_WAKE) begin
                ms_pick = 1'b1; ms_pick_id = i[MSHR_W-1:0]; ms_pick_st = MS_WAKE;
            end
        if (!ms_pick)
            for (i = MSHR_N-1; i >= 0; i = i - 1)
                if (ms_state[i] == MS_FILL) begin
                    ms_pick = 1'b1; ms_pick_id = i[MSHR_W-1:0]; ms_pick_st = MS_FILL;
                end
        if (!ms_pick)
            for (i = MSHR_N-1; i >= 0; i = i - 1)
                if (ms_state[i] == MS_WB) begin
                    ms_pick = 1'b1; ms_pick_id = i[MSHR_W-1:0]; ms_pick_st = MS_WB;
                end
        if (!ms_pick)
            for (i = MSHR_N-1; i >= 0; i = i - 1)
                if (ms_state[i] == MS_REQ) begin
                    ms_pick = 1'b1; ms_pick_id = i[MSHR_W-1:0]; ms_pick_st = MS_REQ;
                end
    end

    // ---- memory request, straight off the engine state ----------------------
    assign mem_req_valid = (est == E_MREQ) || (est == E_WBSND);
    assign mem_req_we    = (est == E_WBSND);
    assign mem_req_addr  = (est == E_WBSND) ? ms_wbadr[cur_ms] : ms_addr[cur_ms];
    assign mem_req_wdata = wb_buf;
    assign mem_req_tag   = cur_ms;

    wire mem_req_fire = mem_req_valid && mem_req_ready;

    // ---- byte merge helper ---------------------------------------------------
    reg [LINE_BITS-1:0] merged;
    reg [WORD_BITS-1:0] amo_word, amo_res;
    reg [WORD_BITS-1:0] wordv;

    //==========================================================================
    // Sequential
    //==========================================================================
    reg [TARGET_W-1:0] t_cur;
    reg [WAY_W-1:0]    fill_way;
    reg [SET_W-1:0]    fill_set;

    always @(posedge clk) begin
        if (!rst_n) begin
            est           <= E_SEL;
            rq_v          <= {RQ_N{1'b0}};
            rq_blocked    <= {RQ_N{1'b0}};
            for (i = 0; i < RQ_N; i = i + 1) older[i] <= {RQ_N{1'b0}};
            v_bits        <= {LINES{1'b0}};
            d_bits        <= {LINES{1'b0}};
            tg_v          <= {TARGET_N{1'b0}};
            rsq_v         <= {RSPQ_N{1'b0}};
            rsq_rd        <= 2'd0;
            rsq_wr        <= 2'd0;
            rsq_cnt       <= 3'd0;
            perf_hit <= 0; perf_miss <= 0; perf_merge <= 0;
            perf_mshr_full <= 0; perf_target_full <= 0; perf_wb <= 0;
            perf_hit_past_block <= 0;
            for (i = 0; i < MSHR_N; i = i + 1) ms_state[i] <= MS_FREE;
            for (i = 0; i < SETS; i = i + 1)   victim_ptr[i] <= {WAY_W{1'b0}};
        end else begin
            push_rsp = 1'b0;
            push_src = {SRC_W{1'b0}};
            push_tag = {TAG_W{1'b0}};
            push_data = {WORD_BITS{1'b0}};

            //------------------------------------------------------------------
            // accept a new request
            //------------------------------------------------------------------
            if (req_valid && req_ready) begin
                rq_v[rq_free_idx]       <= 1'b1;
                rq_blocked[rq_free_idx] <= 1'b0;
                rq_addr[rq_free_idx]    <= req_addr;
                rq_we[rq_free_idx]      <= req_we;
                rq_amo[rq_free_idx]     <= req_amo;
                rq_aop[rq_free_idx]     <= req_aop;
                rq_cmp[rq_free_idx]     <= req_cmp;
                rq_be[rq_free_idx]      <= req_be;
                rq_wdata[rq_free_idx]   <= req_wdata;
                rq_src[rq_free_idx]     <= req_src;
                rq_tag[rq_free_idx]     <= req_tag;
                // the new entry is younger than every live entry, older than none
                older[rq_free_idx] <= {RQ_N{1'b0}};
                for (i = 0; i < RQ_N; i = i + 1)
                    if (i != rq_free_idx) older[i][rq_free_idx] <= rq_v[i];
            end

            //------------------------------------------------------------------
            // memory response -> mark the MSHR ready to fill
            //------------------------------------------------------------------
            if (mem_rsp_valid) begin
                ms_line[mem_rsp_tag]  <= mem_rsp_rdata;
                ms_state[mem_rsp_tag] <= MS_FILL;
            end

            //------------------------------------------------------------------
            // engine
            //------------------------------------------------------------------
            case (est)

            //---------------------------------------------------------------
            E_SEL: begin
                if (ms_pick) begin
                    cur_ms <= ms_pick_id;
                    case (ms_pick_st)
                    MS_WAKE: est <= E_WAKE;
                    MS_FILL: est <= E_FILL;
                    MS_WB:   est <= E_WBRD;
                    default: est <= E_MREQ;
                    endcase
                end else if (sel_valid && !rsq_full) begin
                    cur_rq    <= sel_idx;
                    cur_addr  <= rq_addr[sel_idx];
                    cur_we    <= rq_we[sel_idx];
                    cur_amo   <= rq_amo[sel_idx];
                    cur_aop   <= rq_aop[sel_idx];
                    cur_cmp   <= rq_cmp[sel_idx];
                    cur_be    <= rq_be[sel_idx];
                    cur_wdata <= rq_wdata[sel_idx];
                    cur_src   <= rq_src[sel_idx];
                    cur_tag   <= rq_tag[sel_idx];
                    if (any_blocked) perf_hit_past_block <= perf_hit_past_block + 1;
                    est <= E_LOOK;
                end
            end

            //---------------------------------------------------------------
            E_LOOK: begin
                if (hit && !ms_match) est <= E_HIT;
                else                  est <= E_MISS;
            end

            //---------------------------------------------------------------
            E_HIT: begin
                perf_hit <= perf_hit + 1;
                if (cur_amo) begin
                    // Read, compute and write in one indivisible step. Nothing
                    // else can touch this line in between: the arbiter will not
                    // start a second request to the same line while this one is
                    // in flight, which is the serialisation sec 14.1 asks for.
                    merged   = data_mem[cur_set*WAYS + hit_way];
                    amo_word = merged[cur_wsel*WORD_BITS +: WORD_BITS];
                    amo_res  = amo_new(cur_aop, cur_be, amo_word, cur_wdata, cur_cmp);
                    if (cur_aop != `TNCA_AOP_LD) begin
                        for (i = 0; i < WORD_BYTES; i = i + 1)
                            if (cur_be[i])
                                merged[(cur_wsel*WORD_BYTES + i)*8 +: 8] =
                                    amo_res[i*8 +: 8];
                        data_mem[cur_set*WAYS + hit_way] <= merged;
                        d_bits[cur_set*WAYS + hit_way]   <= 1'b1;
                    end
                    push_rsp  = 1'b1;
                    push_src  = cur_src;
                    push_tag  = cur_tag;
                    // The whole old word, not the operand right-aligned: the
                    // requester extracts its bytes from a response exactly as
                    // it would from a load, and an atomic must not need a
                    // second rule for that.
                    push_data = amo_word;
                end else if (cur_we) begin
                    merged = data_mem[cur_set*WAYS + hit_way];
                    for (i = 0; i < WORD_BYTES; i = i + 1)
                        if (cur_be[i])
                            merged[(cur_wsel*WORD_BYTES + i)*8 +: 8] =
                                cur_wdata[i*8 +: 8];
                    data_mem[cur_set*WAYS + hit_way] <= merged;
                    d_bits[cur_set*WAYS + hit_way]   <= 1'b1;
                    push_rsp  = 1'b1;
                    push_src  = cur_src;
                    push_tag  = cur_tag;
                    push_data = {WORD_BITS{1'b0}};
                end else begin
                    push_rsp  = 1'b1;
                    push_src  = cur_src;
                    push_tag  = cur_tag;
                    push_data = data_mem[cur_set*WAYS + hit_way]
                                [cur_wsel*WORD_BITS +: WORD_BITS];
                end
                rq_v[cur_rq] <= 1'b0;
                est <= E_SEL;
            end

            //---------------------------------------------------------------
            E_MISS: begin
                if (ms_match) begin
                    // merge into the existing miss
                    if (tg_has_free) begin
                        perf_merge <= perf_merge + 1;
                        tg_v[tg_free_idx]     <= 1'b1;
                        tg_src[tg_free_idx]   <= cur_src;
                        tg_tag[tg_free_idx]   <= cur_tag;
                        tg_we[tg_free_idx]    <= cur_we;
                        tg_amo[tg_free_idx]   <= cur_amo;
                        tg_aop[tg_free_idx]   <= cur_aop;
                        tg_cmp[tg_free_idx]   <= cur_cmp;
                        tg_be[tg_free_idx]    <= cur_be;
                        tg_wdata[tg_free_idx] <= cur_wdata;
                        tg_wsel[tg_free_idx]  <= cur_wsel;
                        if (ms_hasT[ms_match_id])
                            tg_next[ms_tail[ms_match_id]] <= tg_free_idx;
                        else
                            ms_head[ms_match_id] <= tg_free_idx;
                        ms_tail[ms_match_id] <= tg_free_idx;
                        ms_hasT[ms_match_id] <= 1'b1;
                        rq_v[cur_rq] <= 1'b0;
                    end else begin
                        perf_target_full <= perf_target_full + 1;
                        rq_blocked[cur_rq] <= 1'b1;
                    end
                end else if (ms_has_free && tg_has_free && vic_ok) begin
                    perf_miss <= perf_miss + 1;
                    ms_addr[ms_free_idx] <= {cur_addr[ADDR_W-1:OFF_W], {OFF_W{1'b0}}};
                    ms_way[ms_free_idx]  <= vic_way;
                    ms_set[ms_free_idx]  <= cur_set;
                    ms_head[ms_free_idx] <= tg_free_idx;
                    ms_tail[ms_free_idx] <= tg_free_idx;
                    ms_hasT[ms_free_idx] <= 1'b1;
                    // A dirty victim must leave before the fill overwrites it.
                    if (v_bits[cur_set*WAYS + vic_way] && d_bits[cur_set*WAYS + vic_way]) begin
                        ms_state[ms_free_idx] <= MS_WB;
                        ms_wbadr[ms_free_idx] <= {tag_mem[cur_set*WAYS + vic_way],
                                                  cur_set, {OFF_W{1'b0}}};
                    end else begin
                        ms_state[ms_free_idx] <= MS_REQ;
                    end
                    // the way now belongs to this miss
                    v_bits[cur_set*WAYS + vic_way] <= 1'b0;
                    victim_ptr[cur_set] <= vic_way + 1'b1;

                    tg_v[tg_free_idx]     <= 1'b1;
                    tg_next[tg_free_idx]  <= tg_free_idx;
                    tg_src[tg_free_idx]   <= cur_src;
                    tg_tag[tg_free_idx]   <= cur_tag;
                    tg_we[tg_free_idx]    <= cur_we;
                    tg_be[tg_free_idx]    <= cur_be;
                    tg_wdata[tg_free_idx] <= cur_wdata;
                    tg_wsel[tg_free_idx]  <= cur_wsel;
                    rq_v[cur_rq] <= 1'b0;
                end else begin
                    // No entry, no target, or every way in this set already
                    // claimed. Block this request; hits to other lines keep
                    // going past it (sec 9.3).
                    if (!ms_has_free) perf_mshr_full   <= perf_mshr_full + 1;
                    if (!tg_has_free) perf_target_full <= perf_target_full + 1;
                    rq_blocked[cur_rq] <= 1'b1;
                end
                est <= E_SEL;
            end

            //---------------------------------------------------------------
            E_WBRD: begin
                wb_buf <= data_mem[ms_set[cur_ms]*WAYS + ms_way[cur_ms]];
                est <= E_WBSND;
            end

            E_WBSND: begin
                if (mem_req_fire) begin
                    perf_wb  <= perf_wb + 1;
                    d_bits[ms_set[cur_ms]*WAYS + ms_way[cur_ms]] <= 1'b0;
                    ms_state[cur_ms] <= MS_REQ;
                    est <= E_SEL;
                end
            end

            //---------------------------------------------------------------
            E_MREQ: begin
                if (mem_req_fire) begin
                    ms_state[cur_ms] <= MS_WAIT;
                    est <= E_SEL;
                end
            end

            //---------------------------------------------------------------
            E_FILL: begin
                fill_set = ms_set[cur_ms];
                fill_way = ms_way[cur_ms];
                data_mem[fill_set*WAYS + fill_way] <= ms_line[cur_ms];
                tag_mem[fill_set*WAYS + fill_way]  <= ms_addr[cur_ms][ADDR_W-1 -: TAGB];
                v_bits[fill_set*WAYS + fill_way]   <= 1'b1;
                d_bits[fill_set*WAYS + fill_way]   <= 1'b0;
                ms_state[cur_ms] <= MS_WAKE;
                est <= E_SEL;
            end

            //---------------------------------------------------------------
            // Targets are served in allocation order, which is what gives
            // same-address requests a defined order at the home slice.
            E_WAKE: begin
                if (!rsq_full) begin
                    t_cur    = ms_head[cur_ms];
                    fill_set = ms_set[cur_ms];
                    fill_way = ms_way[cur_ms];
                    if (tg_amo[t_cur]) begin
                        merged   = data_mem[fill_set*WAYS + fill_way];
                        amo_word = merged[tg_wsel[t_cur]*WORD_BITS +: WORD_BITS];
                        amo_res  = amo_new(tg_aop[t_cur], tg_be[t_cur], amo_word,
                                           tg_wdata[t_cur], tg_cmp[t_cur]);
                        if (tg_aop[t_cur] != `TNCA_AOP_LD) begin
                            for (i = 0; i < WORD_BYTES; i = i + 1)
                                if (tg_be[t_cur][i])
                                    merged[(tg_wsel[t_cur]*WORD_BYTES + i)*8 +: 8] =
                                        amo_res[i*8 +: 8];
                            data_mem[fill_set*WAYS + fill_way] <= merged;
                            d_bits[fill_set*WAYS + fill_way]   <= 1'b1;
                        end
                        push_data = amo_word;          // the whole old word
                    end else if (tg_we[t_cur]) begin
                        merged = data_mem[fill_set*WAYS + fill_way];
                        for (i = 0; i < WORD_BYTES; i = i + 1)
                            if (tg_be[t_cur][i])
                                merged[(tg_wsel[t_cur]*WORD_BYTES + i)*8 +: 8] =
                                    tg_wdata[t_cur][i*8 +: 8];
                        data_mem[fill_set*WAYS + fill_way] <= merged;
                        d_bits[fill_set*WAYS + fill_way]   <= 1'b1;
                        push_data = {WORD_BITS{1'b0}};
                    end else begin
                        push_data = data_mem[fill_set*WAYS + fill_way]
                                    [tg_wsel[t_cur]*WORD_BITS +: WORD_BITS];
                    end
                    push_rsp = 1'b1;
                    push_src = tg_src[t_cur];
                    push_tag = tg_tag[t_cur];

                    tg_v[t_cur] <= 1'b0;
                    if (t_cur == ms_tail[cur_ms]) begin
                        // last target: the miss is done, resources released
                        ms_state[cur_ms] <= MS_FREE;
                        ms_hasT[cur_ms]  <= 1'b0;
                        rq_blocked <= {RQ_N{1'b0}};   // retry anything that gave up
                    end else begin
                        ms_head[cur_ms] <= tg_next[t_cur];
                    end
                    est <= E_SEL;
                end
            end

            default: est <= E_SEL;
            endcase

            //------------------------------------------------------------------
            // response queue
            //------------------------------------------------------------------
            if (push_rsp) begin
                rsq_src[rsq_wr]  <= push_src;
                rsq_tag[rsq_wr]  <= push_tag;
                rsq_data[rsq_wr] <= push_data;
                rsq_wr           <= rsq_wr + 1'b1;
            end
            if (rsq_pop) rsq_rd <= rsq_rd + 1'b1;
            case ({push_rsp, rsq_pop})
            2'b10: rsq_cnt <= rsq_cnt + 1'b1;
            2'b01: rsq_cnt <= rsq_cnt - 1'b1;
            default: ;
            endcase
        end
    end

endmodule
