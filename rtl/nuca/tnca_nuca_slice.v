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
// DCO (doc/DCO.pdf, doc/dco.md)
//
// Stage 3 built this slice as a baseline with no policy in it -- invalid-first
// then round-robin -- so that cache policies could be measured against it one
// at a time (sec 21). DCO's are now here, each behind a bit of the TMU's
// POLICY register (rtl/nuca/tnca_tmu.v), and POLICY 0 is still that baseline:
//
//   LRU and anti-thrashing replacement   victim = min {not dead, priority, age}
//   dead-block prediction                tiles the TMU has counted to nAcc
//   dynamic bypass                       priority < B_GEAR, gear set by evictions
//   gqa_bypass                           ... only for the slower core of a pair
//   tensor bypass                        tensors registered with nAcc = 1
//
// None of them changes what a load returns; they decide where a line lives.
// What is still not here: pressure-adaptive admission and MSHR-aware
// arbitration.
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
    parameter TAG_W      = 4,
    // Which slice this is, and how many line-address bits choose a home.
    // The core NI homes a line at line[1:0] ^ line[3:2] (rtl/noc/tnca_core_ni.v),
    // so every line this slice sees has those two bits fixed by its id.
    parameter SLICE_ID   = 0,
    parameter HOME_W     = 2,
    // Generic home hashes cannot recover omitted address bits from slice ID.
    parameter FULL_LINE_TAG = 0,
    // DCO (rtl/nuca/tnca_tmu.v): tensor-table entries, dead-FIFO entries,
    // tile-identifier width, and the address window of the TMU's registers.
    parameter DCO_T      = 8,
    parameter DCO_DEAD   = 16,
    parameter MAINTENANCE = 0,
    parameter DCO_ID_W   = 16,
    parameter [63:0] TMU_BASE = 64'h0000_0000_7FFF_0000
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Quiescent DMA ownership handoff. Write back and invalidate every
    // overlapping line, including partial-line ranges. Completion means
    // writebacks were accepted; the platform must also drain memory links
    // and physical write responses before allowing host DMA.
    input  wire                     maint_valid,
    output wire                     maint_ready,
    input  wire [ADDR_W-1:0]         maint_base, maint_size,
    output reg                      maint_done,
    output wire                     maint_busy,

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

    // ---- DCO: policy and metadata, from the TMU ------------------------------
    input  wire [75:0]              dco_cfg,        // see `DCO_*` slices below
    input  wire [DCO_T-1:0]         dco_t_valid,
    input  wire [DCO_T*64-1:0]      dco_t_base,
    input  wire [DCO_T*64-1:0]      dco_t_size,
    input  wire [DCO_T-1:0]         dco_t_bypass,
    input  wire [DCO_T*6-1:0]       dco_t_lsb,      // per-tensor tile granule (0: D_LSB)
    input  wire [DCO_DEAD-1:0]      dco_dead_valid,
    input  wire [DCO_DEAD*DCO_ID_W-1:0] dco_dead_id,
    input  wire [15:0]              dco_slow,       // per requester: the slower of its pair
    // ---- DCO: to the TMU -----------------------------------------------------
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

    // ---- performance counters (sec 20.2) ------------------------------------
    output reg  [31:0]              perf_bypass,      // DCO: misses served without a way
    output reg  [31:0]              perf_evict,       // valid lines evicted
    output reg  [31:0]              perf_dead_evict,  // ... of which predicted dead
    output reg  [3:0]               perf_gear,        // the bypass gear now
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
    // The set index skips the home bits: they are the same for every line
    // that reaches this slice, so indexing with them -- as this slice did
    // until R38 -- left only SETS/4 of its sets reachable and three quarters
    // of the array unused. The tag is everything above the set.
    localparam HOME_SKIP = FULL_LINE_TAG ? 0 : HOME_W;
    localparam TAGB     = ADDR_W - OFF_W - HOME_SKIP - SET_W;
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
    `define SLICE_SET(ad)  (ad[OFF_W + HOME_SKIP +: SET_W])
    `define SLICE_TAG(ad)  (ad[ADDR_W-1 -: TAGB])
    `define SLICE_WSEL(ad) (ad[WORD_BYTES==8 ? 3 : 2 +: WSEL_W])

    // A cached line's byte address from its tag and set: the home bits are
    // line[1:0] = SLICE_ID ^ line[3:2], and line[3:2] are the set's low bits.
    function [ADDR_W-1:0] line_addr;
        input [TAGB-1:0]  tg;
        input [SET_W-1:0] st;
        reg   [HOME_W-1:0] home;
        begin
            home = SLICE_ID[HOME_W-1:0] ^ st[HOME_W-1:0];
            if(FULL_LINE_TAG) line_addr = {tg, st, {OFF_W{1'b0}}};
            else line_addr = {tg, st, home, {OFF_W{1'b0}}};
        end
    endfunction

    //==========================================================================
    // DCO configuration (doc/DCO.pdf; rtl/nuca/tnca_tmu.v holds the registers)
    //==========================================================================
    wire [7:0]  pol        = dco_cfg[7:0];
    wire [1:0]  pol_repl   = pol[1:0];      // 0 round-robin, 1 LRU, 2 anti-thrash
    wire        pol_dbp    = pol[2];        // dead-block prediction
    wire        pol_dyn    = pol[3];        // dynamic bypass
    wire        pol_gqa    = pol[4];        // ... only for the slower core of a pair
    wire        pol_tbyp   = pol[5];        // bypass tensors registered with nAcc = 1
    wire        pol_static = pol[6];        // the gear is GEAR_INIT, not adapted
    wire [3:0]  b_bits     = dco_cfg[11:8];
    wire [5:0]  d_lsb      = dco_cfg[17:12];  // the default tile granule
    wire [3:0]  gear_init  = dco_cfg[27:24];
    wire [15:0] g_window   = dco_cfg[43:28];
    wire [15:0] g_ub       = dco_cfg[59:44];
    wire [15:0] g_lb       = dco_cfg[75:60];
    // The anti-thrashing priority is tag[B_BITS-1:0], the tag's lowest bits:
    // uniform across a tensor, so keeping the high-priority half keeps half of
    // any working set, whatever its size. At most 3 bits, as in the paper, so
    // the gear runs 0..8 (sec IV-D, Fig. 7).
    wire [3:0]  bmask      = (b_bits >= 4'd3) ? 4'h7 : ((4'd1 << b_bits) - 4'd1);
    wire [3:0]  gear_max   = bmask + 4'd1;

    // The tensor an address belongs to: inside [base, base+size), or with a
    // size of 0, the registered tensor with the highest base not above it (the
    // TMU's rule, tnca_tmu.v). Returns {hit, bypass, index[3:0], tile lsb[5:0]};
    // a tile lsb of 0 means "the default granule", D_LSB.
    function [11:0] owner;
        input [ADDR_W-1:0]      a;
        input [DCO_T-1:0]       tv;
        input [DCO_T*64-1:0]    tbase;
        input [DCO_T*64-1:0]    tsize;
        input [DCO_T-1:0]       tbyp;
        input [DCO_T*6-1:0]     tlsb;
        integer k;
        reg     hit, byp;
        reg [63:0] best, b, z;
        reg [3:0]  idx;
        reg [5:0]  lsb;
        begin
            hit = 1'b0; best = 64'd0; byp = 1'b0; idx = 4'd0; lsb = 6'd0;
            for (k = 0; k < DCO_T; k = k + 1) begin
                b = tbase[k*64 +: 64];
                z = tsize[k*64 +: 64];
                if (tv[k] && (a >= b) && ((z == 64'd0) || (a - b < z)) &&
                    (!hit || b > best)) begin
                    hit = 1'b1; best = b; byp = tbyp[k]; idx = k[3:0];
                    lsb = tlsb[k*6 +: 6];
                end
            end
            owner = {hit, byp, idx, lsb};
        end
    endfunction

    // A tile's identifier, as the TMU forms it: the owning tensor's index
    // above the tile's number within the address space, in that tensor's own
    // granule (DCO's per-tensor tile size). Tiles of different tensors never
    // share an identifier, whatever their sizes.
    function [DCO_ID_W-1:0] tile_id;
        input [ADDR_W-1:0] a;
        input [11:0]       own;
        input [5:0]        dflt;
        reg   [5:0]        lsb;
        reg   [63:0]       sh;
        begin
            lsb = (own[5:0] != 6'd0) ? own[5:0] : dflt;
            sh  = a >> lsb;
            tile_id = {own[8:6], sh[DCO_ID_W-4:0]};
        end
    endfunction

    function dead_tile;
        input [DCO_ID_W-1:0]           id;
        input [DCO_DEAD-1:0]           dv;
        input [DCO_DEAD*DCO_ID_W-1:0]  did;
        integer k;
        begin
            dead_tile = 1'b0;
            for (k = 0; k < DCO_DEAD; k = k + 1)
                if (dv[k] && did[k*DCO_ID_W +: DCO_ID_W] == id) dead_tile = 1'b1;
        end
    endfunction

    //==========================================================================
    // Arrays
    //==========================================================================
    reg [LINE_BITS-1:0] data_mem [0:LINES-1];
    reg [TAGB-1:0]      tag_mem  [0:LINES-1];
    reg [LINES-1:0]     v_bits;
    reg [LINES-1:0]     d_bits;
    reg [WAY_W-1:0]     victim_ptr [0:SETS-1];
    // LRU, as ages: 0 is the most recently used way of its set, WAYS-1 the
    // least. A permutation per set at all times.
    reg [WAY_W-1:0]     age        [0:LINES-1];

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

    wire maint_requested = (MAINTENANCE != 0) && maint_valid;
    assign req_ready = rq_has_free && !maint_busy && !maint_requested;

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
    // A bypassed miss (DCO): the line is fetched for its targets and never
    // allocated, so the MSHR claims no way and the fill writes no array.
    reg [MSHR_N-1:0]    ms_byp;
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
               E_WAKE  = 4'd8,
               E_MMIO  = 4'd9,   // a TMU register, never cached
               E_MSCAN = 4'd10,
               E_MREAD = 4'd11,
               E_MSEND = 4'd12;

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
    reg [LINE_W-1:0] maint_index;
    reg [ADDR_W-1:0] maint_first, maint_line;
    reg [ADDR_W:0] maint_end;
    reg maint_ms_free;
    integer maint_i;
    always @* begin
        maint_ms_free = 1'b1;
        for (maint_i=0; maint_i<MSHR_N; maint_i=maint_i+1)
            if (ms_state[maint_i] != MS_FREE) maint_ms_free = 1'b0;
    end
    assign maint_busy = (est==E_MSCAN || est==E_MREAD || est==E_MSEND);
    assign maint_ready = (MAINTENANCE != 0) && rst_n && est==E_SEL &&
                         rq_v==0 && rsq_cnt==0 && tg_v==0 && maint_ms_free && !mem_rsp_valid;
    wire [SET_W-1:0] maint_set = maint_index / WAYS;
    wire [ADDR_W-1:0] maint_address = line_addr(tag_mem[maint_index], maint_set);
    wire maint_overlap = ({1'b0,maint_address} < maint_end) &&
                         ({1'b0,maint_address}+LINE_BYTES > {1'b0,maint_first});

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

    // ---- victim selection: invalid first, then the policy, never a way that
    //      an in-flight MSHR has already claimed in this set -------------------
    //
    // Round-robin (policy 0) is the baseline this slice started with. The
    // others rank each candidate by a key and take the smallest:
    //
    //   { not dead (DBP), priority (anti-thrash), most recently used (LRU) }
    //
    // so a dead line goes first, then the lowest-priority line, and LRU breaks
    // the tie -- the order DCO's replacement policy names (sec IV-A).
    reg              way_claimed [0:WAYS-1];
    reg [WAY_W-1:0]  vic_way;
    reg              vic_ok;
    reg              vic_dead;
    reg [WAY_W-1:0]  cand;
    reg [7:0]        vkey, vbest;
    reg              vdead;
    reg [TAGB-1:0]   vtag;
    reg [3:0]        vprio;
    reg [11:0]       vown;
    integer          vidx, vw, vi;
    // Its own loop variables: a loop index shared with another always block
    // puts each block in the other's sensitivity list, and the simulator
    // then re-runs both on every write to it. Only E_MISS reads the result,
    // so the search runs only there -- the same logic, without evaluating
    // sixteen dead-tile compares per way on every cycle of a simulation.
    always @(*) begin
        vic_way = {WAY_W{1'b0}}; vic_ok = 1'b0; vic_dead = 1'b0; vbest = 8'hFF;
        for (vw = 0; vw < WAYS; vw = vw + 1) way_claimed[vw] = 1'b0;
        if (est == E_MISS) begin
            for (vw = 0; vw < WAYS; vw = vw + 1)
                for (vi = 0; vi < MSHR_N; vi = vi + 1)
                    if ((ms_state[vi] != MS_FREE) && !ms_byp[vi] && (ms_set[vi] == cur_set) &&
                        (ms_way[vi] == vw[WAY_W-1:0]))
                        way_claimed[vw] = 1'b1;
            // invalid ways first
            for (vw = WAYS-1; vw >= 0; vw = vw - 1)
                if (!v_bits[cur_set*WAYS + vw] && !way_claimed[vw]) begin
                    vic_way = vw[WAY_W-1:0]; vic_ok = 1'b1;
                end
            if (!vic_ok && pol_repl == 2'd0) begin
                for (vw = WAYS-1; vw >= 0; vw = vw - 1) begin
                    cand = victim_ptr[cur_set] + vw[WAY_W-1:0];
                    if (!way_claimed[cand]) begin
                        vic_way = cand; vic_ok = 1'b1;
                    end
                end
            end else if (!vic_ok) begin
                for (vw = 0; vw < WAYS; vw = vw + 1) begin
                    vidx  = cur_set*WAYS + vw;
                    vtag  = tag_mem[vidx];
                    vown  = owner(line_addr(vtag, cur_set), dco_t_valid, dco_t_base,
                                  dco_t_size, dco_t_bypass, dco_t_lsb);
                    vdead = pol_dbp && vown[11] &&
                            (vown[10] ||
                             dead_tile(tile_id(line_addr(vtag, cur_set), vown, d_lsb),
                                       dco_dead_valid, dco_dead_id));
                    // Anti-thrashing ranks only lines of registered tensors,
                    // and only in the older half of the set's LRU stack: a
                    // line just filled stays until the cores sharing it have
                    // read it (DCO's "deprioritised data stay a while"), and
                    // data no tensor owns -- scratch, TensorMaps -- is ranked
                    // by recency alone. With four ways and no protection, a
                    // low-tier line was the next victim the moment it landed,
                    // and every core sharing it fetched it again.
                    vprio = (pol_repl != 2'd2)                      ? 4'd0 :
                            (!vown[11] || age[vidx] < (WAYS / 2))   ? bmask :
                                                                      (vtag[3:0] & bmask);
                    vkey  = {!vdead, vprio, ~age[vidx]};
                    if (!way_claimed[vw] && (!vic_ok || vkey < vbest)) begin
                        vic_way = vw[WAY_W-1:0]; vic_ok = 1'b1; vbest = vkey; vic_dead = vdead;
                    end
                end
            end
        end
    end

    // ---- the bypass decision for the request in hand (DCO sec IV-D) ----------
    // A read miss is not allocated when its tensor was registered with
    // nAcc = 1 (it will not be read again), or when its priority is below the
    // gear the eviction rate has set -- only for the slower core of its pair
    // under gqa_bypass, so shared data the faster core fetches stays cached.
    reg  [3:0]  gear;
    wire [3:0]  gear_eff  = pol_static ? gear_init : gear;
    wire [3:0]  cur_prio  = cur_tagb[3:0] & bmask;
    wire [11:0] cur_own   = owner(cur_addr, dco_t_valid, dco_t_base, dco_t_size,
                                  dco_t_bypass, dco_t_lsb);
    wire        cur_tbyp  = pol_tbyp && cur_own[11] && cur_own[10];
    wire        cur_dbyp  = pol_dyn && cur_own[11] &&
                            ({1'b0, cur_prio} < {1'b0, gear_eff}) &&
                            (!pol_gqa || dco_slow[cur_src]);
    wire        cur_byp   = !cur_we && !cur_amo && (cur_tbyp || cur_dbyp);

    // ---- the TMU register window ---------------------------------------------
    wire        cur_mmio  = (cur_addr[ADDR_W-1:12] == TMU_BASE[ADDR_W-1:12]);
    assign tmu_wr_valid = (est == E_MMIO) && cur_we && !cur_amo;
    assign tmu_wr_off   = cur_addr[11:0];
    assign tmu_wr_data  = cur_wdata;
    assign tmu_wr_src   = cur_src;
    assign tmu_rd_off   = cur_addr[11:0];

    // ---- tile-last-word notices (DCO sec IV-B) ---------------------------------
    // A tile's use is counted when its last 8-byte word is read. One notice is
    // held; a second one while it waits is dropped and counted -- a hint late or
    // lost moves when a tile is declared dead, never what a load returns.
    reg         tl_v;
    reg  [63:0] tl_a;
    assign tmu_tl_valid = tl_v;
    assign tmu_tl_addr  = tl_a;
    // Only a registered tensor's tiles are counted, each in its own granule.
    wire [5:0]  cur_lsb  = (cur_own[5:0] != 6'd0) ? cur_own[5:0] : d_lsb;
    wire [63:0] tlw_mask = ((64'd1 << cur_lsb) - 64'd1) & ~64'd7;
    wire        cur_tlw  = pol_dbp && cur_own[11] && !cur_we && !cur_amo &&
                           (cur_lsb >= 6'd3) && ((cur_addr & tlw_mask) == tlw_mask);

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
    assign mem_req_valid = (est == E_MREQ) || (est == E_WBSND) || (est == E_MSEND);
    assign mem_req_we    = (est == E_WBSND) || (est == E_MSEND);
    assign mem_req_addr  = (est == E_MSEND) ? maint_line :
                          (est == E_WBSND) ? ms_wbadr[cur_ms] : ms_addr[cur_ms];
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

    // DCO bookkeeping
    reg         gear_seed;
    reg [15:0]  ev_cnt, win_cnt;
    reg         tl_load;
    reg [63:0]  tl_naddr;

    // Make `tw` the most recently used way of set `ts`: every way younger than
    // it ages by one, so the ages stay a permutation.
    task touch;
        input [SET_W-1:0] ts;
        input [WAY_W-1:0] tw;
        integer q;
        begin
            for (q = 0; q < WAYS; q = q + 1)
                if (age[ts*WAYS + q] < age[ts*WAYS + tw])
                    age[ts*WAYS + q] <= age[ts*WAYS + q] + 1'b1;
            age[ts*WAYS + tw] <= {WAY_W{1'b0}};
        end
    endtask

    task tl_note;
        input [ADDR_W-1:0] a;
        begin
            tl_load  = 1'b1;
            tl_naddr = a;
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            est           <= E_SEL;
            maint_done <= 0; maint_index <= 0; maint_first <= 0; maint_end <= 0; maint_line <= 0;
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
            for (i = 0; i < LINES; i = i + 1)  age[i] <= i % WAYS;
            ms_byp <= {MSHR_N{1'b0}};
            perf_bypass <= 0; perf_evict <= 0; perf_dead_evict <= 0;
            tl_v <= 1'b0; tl_a <= 64'd0; tmu_tl_drop <= 1'b0;
            gear <= 4'd0; gear_seed <= 1'b1; ev_cnt <= 16'd0; win_cnt <= 16'd0;
        end else begin
            maint_done <= 0;
            push_rsp = 1'b0;
            tl_load  = 1'b0;
            tl_naddr = 64'd0;
            if (gear_seed) begin
                // the first cycle out of reset: start from the configured gear
                gear <= (gear_init > gear_max) ? gear_max : gear_init;
                gear_seed <= 1'b0;
            end
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
                if (maint_requested && maint_ready) begin
                    maint_first <= maint_base;
                    maint_end <= {1'b0,maint_base}+{1'b0,maint_size};
                    maint_index <= 0;
                    if (maint_size==0) maint_done <= 1;
                    else est <= E_MSCAN;
                end else if (ms_pick) begin
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

            E_MSCAN: begin
                if (v_bits[maint_index] && maint_overlap && d_bits[maint_index]) begin
                    maint_line <= maint_address;
                    est <= E_MREAD;
                end else begin
                    if (v_bits[maint_index] && maint_overlap) begin
                        v_bits[maint_index] <= 0;
                        d_bits[maint_index] <= 0;
                    end
                    if (maint_index==LINES-1) begin maint_done<=1; est<=E_SEL; end
                    else maint_index<=maint_index+1'b1;
                end
            end
            E_MREAD: begin
                wb_buf <= data_mem[maint_index];
                est <= E_MSEND;
            end
            E_MSEND: begin
                if (mem_req_fire) begin
                    perf_wb <= perf_wb+1'b1;
                    v_bits[maint_index] <= 0;
                    d_bits[maint_index] <= 0;
                    if (maint_index==LINES-1) begin maint_done<=1; est<=E_SEL; end
                    else begin maint_index<=maint_index+1'b1; est<=E_MSCAN; end
                end
            end

            //---------------------------------------------------------------
            E_LOOK: begin
                if (cur_mmio)              est <= E_MMIO;
                else if (hit && !ms_match) est <= E_HIT;
                else                       est <= E_MISS;
            end

            //---------------------------------------------------------------
            // A TMU register: a store waits for the TMU to take it, a load
            // reads it straight away. Neither touches the arrays.
            E_MMIO: begin
                if (!cur_we || cur_amo || tmu_wr_ready) begin
                    push_rsp  = 1'b1;
                    push_src  = cur_src;
                    push_tag  = cur_tag;
                    push_data = cur_we ? {WORD_BITS{1'b0}} : tmu_rd_data;
                    rq_v[cur_rq] <= 1'b0;
                    est <= E_SEL;
                end
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
                touch(cur_set, hit_way);
                if (cur_tlw) tl_note(cur_addr);
                rq_v[cur_rq] <= 1'b0;
                est <= E_SEL;
            end

            //---------------------------------------------------------------
            E_MISS: begin
                if (ms_match && ms_byp[ms_match_id] && (cur_we || cur_amo)) begin
                    // A store cannot ride a bypassed miss: there is no way for
                    // it to write. It waits for the miss to retire, then
                    // allocates a line of its own.
                    rq_blocked[cur_rq] <= 1'b1;
                end else if (ms_match) begin
                    // merge into the existing miss
                    if (tg_has_free) begin
                        if (cur_tlw) tl_note(cur_addr);
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
                end else if (ms_has_free && tg_has_free && cur_byp) begin
                    // DCO bypass: fetch the line for this read, claim no way.
                    perf_miss   <= perf_miss + 1;
                    perf_bypass <= perf_bypass + 1;
                    ms_addr[ms_free_idx]  <= {cur_addr[ADDR_W-1:OFF_W], {OFF_W{1'b0}}};
                    ms_way[ms_free_idx]   <= {WAY_W{1'b0}};
                    ms_set[ms_free_idx]   <= cur_set;
                    ms_byp[ms_free_idx]   <= 1'b1;
                    ms_head[ms_free_idx]  <= tg_free_idx;
                    ms_tail[ms_free_idx]  <= tg_free_idx;
                    ms_hasT[ms_free_idx]  <= 1'b1;
                    ms_state[ms_free_idx] <= MS_REQ;
                    tg_v[tg_free_idx]     <= 1'b1;
                    tg_next[tg_free_idx]  <= tg_free_idx;
                    tg_src[tg_free_idx]   <= cur_src;
                    tg_tag[tg_free_idx]   <= cur_tag;
                    tg_we[tg_free_idx]    <= 1'b0;
                    tg_amo[tg_free_idx]   <= 1'b0;
                    tg_be[tg_free_idx]    <= cur_be;
                    tg_wdata[tg_free_idx] <= cur_wdata;
                    tg_wsel[tg_free_idx]  <= cur_wsel;
                    if (cur_tlw) tl_note(cur_addr);
                    rq_v[cur_rq] <= 1'b0;
                end else if (ms_has_free && tg_has_free && vic_ok) begin
                    perf_miss <= perf_miss + 1;
                    ms_byp[ms_free_idx]  <= 1'b0;
                    if (v_bits[cur_set*WAYS + vic_way]) begin
                        perf_evict <= perf_evict + 1;
                        ev_cnt     <= ev_cnt + 1'b1;
                        if (vic_dead) perf_dead_evict <= perf_dead_evict + 1;
                    end
                    if (cur_tlw) tl_note(cur_addr);
                    ms_addr[ms_free_idx] <= {cur_addr[ADDR_W-1:OFF_W], {OFF_W{1'b0}}};
                    ms_way[ms_free_idx]  <= vic_way;
                    ms_set[ms_free_idx]  <= cur_set;
                    ms_head[ms_free_idx] <= tg_free_idx;
                    ms_tail[ms_free_idx] <= tg_free_idx;
                    ms_hasT[ms_free_idx] <= 1'b1;
                    // A dirty victim must leave before the fill overwrites it.
                    if (v_bits[cur_set*WAYS + vic_way] && d_bits[cur_set*WAYS + vic_way]) begin
                        ms_state[ms_free_idx] <= MS_WB;
                        ms_wbadr[ms_free_idx] <= line_addr(tag_mem[cur_set*WAYS + vic_way],
                                                           cur_set);
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
                    tg_amo[tg_free_idx]   <= cur_amo;
                    tg_aop[tg_free_idx]   <= cur_aop;
                    tg_cmp[tg_free_idx]   <= cur_cmp;
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
                if (!ms_byp[cur_ms]) begin
                    data_mem[fill_set*WAYS + fill_way] <= ms_line[cur_ms];
                    tag_mem[fill_set*WAYS + fill_way]  <= ms_addr[cur_ms][ADDR_W-1 -: TAGB];
                    v_bits[fill_set*WAYS + fill_way]   <= 1'b1;
                    d_bits[fill_set*WAYS + fill_way]   <= 1'b0;
                    touch(fill_set, fill_way);
                end
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
                    end else if (ms_byp[cur_ms]) begin
                        push_data = ms_line[cur_ms][tg_wsel[t_cur]*WORD_BITS +: WORD_BITS];
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
                        ms_byp[cur_ms]   <= 1'b0;
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
            // tile-last-word notice buffer (one entry)
            //------------------------------------------------------------------
            tmu_tl_drop <= 1'b0;
            if (tl_v && tmu_tl_ready) tl_v <= 1'b0;
            if (tl_load) begin
                if (!tl_v || tmu_tl_ready) begin
                    tl_v <= 1'b1;
                    tl_a <= tl_naddr;
                end else begin
                    tmu_tl_drop <= 1'b1;
                end
            end

            //------------------------------------------------------------------
            // B_GEAR: evictions in each window of g_window cycles move it
            // (sec IV-D). Above bypass_ub bypass more, below bypass_lb less.
            //------------------------------------------------------------------
            if (!gear_seed) begin
                if (win_cnt + 16'd1 >= g_window) begin
                    win_cnt <= 16'd0;
                    ev_cnt  <= 16'd0;
                    if (ev_cnt > g_ub && gear < gear_max)            gear <= gear + 4'd1;
                    else if (ev_cnt < g_lb && gear != 4'd0)          gear <= gear - 4'd1;
                end else begin
                    win_cnt <= win_cnt + 16'd1;
                end
            end
            perf_gear <= gear_eff;

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
