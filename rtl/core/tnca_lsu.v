//==============================================================================
// tnca_lsu -- load/store unit with a physical-word coalescer and multiple
// outstanding transactions (ISA sec 9.1)
//
// Handles the whole 0x20 family: scalar and vector, contiguous and indexed,
// loads and stores, with arbitrary alignment.
//
// COALESCING
// ----------
// Work is driven by memory words, not by lanes. Each beat picks the lowest
// lane still owing bytes, takes the aligned word containing them, and serves
// every other outstanding lane whose bytes fall entirely inside that same word.
// A 32-bit contiguous vector access therefore costs VLEN/2 beats on a 64-bit
// port instead of VLEN, and a gather with locality collapses the same way; a
// fully scattered gather degrades to one beat per lane, which is the honest
// cost.
//
// A lane that straddles a word boundary is served in two beats: the low bytes
// with the first word, the high bytes with the next.
//
// MEMORY-LEVEL PARALLELISM
// ------------------------
// Beats are issued as fast as the port accepts them, up to NUM_BEATS in flight,
// and responses are matched by tag and may arrive in any order. This is the
// point of the unit: arch overview sec 2.3 is explicit that four role contexts
// are not the problem, four outstanding loads are, and that a transfer must
// expand into many independent line transactions. A single-outstanding LSU
// leaves a cache with nothing to be non-blocking about.
//
// One instruction is in flight at a time -- the beats of a second load would
// need a second lane-address array, and the scoreboard already serialises
// dependent accesses. Overlapping whole instructions is a Stage-4 change and
// belongs with the per-source credit scheme of sec 9.1.
//
// MASKING
// -------
// An inactive lane performs no memory access at all -- not a suppressed one.
// Its destination lane keeps its previous value, which the core expresses by
// clearing that lane's write enable (ISA sec 2.2).
//==============================================================================
`include "tnca_defs.vh"

module tnca_lsu #(
    parameter VLEN      = 16,
    parameter ELEN      = 32,
    parameter RCTX_W    = 1,
    parameter MEM_DW    = 64,
    parameter MEM_BYTES = 8,
    parameter MEM_SHIFT = 3,
    parameter NUM_BEATS = 8,          // outstanding beats
    parameter BEAT_W    = 3           // $clog2(NUM_BEATS)
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- request from the core ----------------------------------------------
    output wire                     req_ready,
    input  wire                     req_valid,
    input  wire                     req_store,
    input  wire                     req_vector,
    input  wire                     req_indexed,
    input  wire [2:0]               req_size,      // log2 bytes
    input  wire                     req_signed,
    input  wire [1:0]               req_hint,
    input  wire [63:0]              req_base,
    input  wire [63:0]              req_offset,
    input  wire [VLEN*ELEN-1:0]     req_index,
    input  wire [63:0]              req_sdata,
    // Atomics (ISA sec 9.2). An atomic is a scalar access that both writes and
    // returns, so it takes the load return path with `amo` set rather than a
    // path of its own: the value it brings back is the memory's OLD word.
    input  wire                     req_amo,
    input  wire [`TNCA_AOP_W-1:0]   req_aop,
    input  wire [63:0]              req_cmp,       // atom.cas compare operand
    input  wire [VLEN*ELEN-1:0]     req_vdata,
    input  wire [VLEN-1:0]          req_mask,
    input  wire [RCTX_W-1:0]        req_rctx,
    input  wire [2:0]               req_dst_class,
    input  wire [7:0]               req_dst_idx,

    // ---- completion to the core ---------------------------------------------
    output reg                      cpl_valid,     // held until cpl_ack
    input  wire                     cpl_ack,
    output reg  [RCTX_W-1:0]        cpl_rctx,
    output reg  [2:0]               cpl_dst_class,
    output reg  [7:0]               cpl_dst_idx,
    output reg  [63:0]              cpl_sdata,
    output reg  [VLEN*ELEN-1:0]     cpl_vdata,
    output reg  [VLEN-1:0]          cpl_lane_en,

    // ---- state, for a wrapper that overlaps two of these (tnca_lsu_pair) ----
    // o_draining: every beat is issued, only responses or the completion
    // are outstanding. o_load: the access is a plain load (not a store, not
    // an atomic), so a later load may overtake it without reordering memory.
    output wire                     o_draining,
    output wire                     o_load,

    // ---- memory port (tagged, responses may return out of order) ------------
    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [63:0]              mem_req_addr,  // MEM_BYTES aligned
    output wire                     mem_req_we,
    output wire [MEM_BYTES-1:0]     mem_req_be,
    output wire [MEM_DW-1:0]        mem_req_wdata,
    output wire [BEAT_W-1:0]        mem_req_tag,
    output wire [1:0]               mem_req_hint,
    output wire                     mem_req_amo,
    output wire [`TNCA_AOP_W-1:0]   mem_req_aop,
    output wire [63:0]              mem_req_cmp,

    input  wire                     mem_rsp_valid,
    input  wire [BEAT_W-1:0]        mem_rsp_tag,
    input  wire [MEM_DW-1:0]        mem_rsp_rdata
);

    localparam S_IDLE  = 3'd0,
               S_SETUP = 3'd1,
               S_RUN   = 3'd2,
               S_DRAIN = 3'd3,
               S_DONE  = 3'd4;

    reg [2:0] state;

    // ---- latched request -----------------------------------------------------
    reg              r_store, r_vector, r_indexed, r_signed;
    reg              r_amo;
    reg [`TNCA_AOP_W-1:0] r_aop;
    reg [63:0]       r_cmp;
    // An atomic writes bytes even though it is not a store, so byte enables
    // and write data are formed for both.
    wire             wr_bytes = r_store | r_amo;
    reg [2:0]        r_size;
    reg [1:0]        r_hint;
    reg [RCTX_W-1:0] r_rctx;
    reg [2:0]        r_dst_class;
    reg [7:0]        r_dst_idx;

    reg [63:0]       lane_addr [0:VLEN-1];
    // Staging is 64 bits wide, not ELEN: a scalar 64-bit access is lane 0 and
    // must not be truncated. Vector lanes use only the low ELEN bits.
    reg [63:0]       lane_wdata[0:VLEN-1];
    reg [63:0]       lane_rdata[0:VLEN-1];

    reg [VLEN-1:0]   pending;     // lanes with bytes not yet issued
    reg [VLEN-1:0]   partial;     // low half issued, high half still owed
    reg [VLEN-1:0]   active;      // lanes unmasked at request time

    reg [4:0]        nbytes;      // bytes per lane, 1..8

    // ---- beat table ----------------------------------------------------------
    reg [NUM_BEATS-1:0] bt_valid;
    reg [VLEN-1:0]      bt_lanes [0:NUM_BEATS-1];
    reg [VLEN-1:0]      bt_low   [0:NUM_BEATS-1];   // carried a straddler's low half
    reg [VLEN-1:0]      bt_high  [0:NUM_BEATS-1];   // carried a straddler's high half

    integer i, b;

    assign req_ready = (state == S_IDLE);
    assign o_draining = (state == S_DRAIN) || (state == S_DONE);
    assign o_load     = !r_store && !r_amo;

    //--------------------------------------------------------------------------
    // Beat planning: which word, which lanes, which byte enables
    //--------------------------------------------------------------------------
    localparam [63:0] WORD_MASK = ~((64'd1 << MEM_SHIFT) - 64'd1);

    reg  [63:0]           plan_word;
    reg  [VLEN-1:0]       plan_lanes, plan_low, plan_high;
    reg  [MEM_BYTES-1:0]  plan_be;
    reg  [MEM_DW-1:0]     plan_wdata;
    reg                   plan_valid;

    reg [63:0] la, w_lo, w_hi;
    reg [MEM_SHIFT-1:0] boff;
    reg [5:0]  lo_cnt, hi_cnt;
    reg        found;
    reg [3:0]  sel;

    // Each lane's beat geometry, derived from the lane arrays and captured in
    // S_SETUP -- the cycle that already separates them from the planner, and
    // after which neither changes until the request completes: the words its
    // first and last bytes fall in, whether those differ (it straddles), and
    // the byte enables and write data of its low and high beats, already
    // shifted into place (timing-006). The planner then only chooses a word
    // and collects the lanes that hit it. With all of this inside the
    // planner, an adder, a byte shift and a sixteen-step conditional
    // accumulation sat between `pending` and the memory's write enables: up
    // to 11 ns post-synthesis, 48 levels.
    reg [63:0]          g_wlo   [0:VLEN-1];
    reg [63:0]          g_whi   [0:VLEN-1];
    reg [MEM_BYTES-1:0] g_lo_be [0:VLEN-1];
    reg [MEM_BYTES-1:0] g_hi_be [0:VLEN-1];
    reg [MEM_DW-1:0]    g_lo_wd [0:VLEN-1];
    reg [MEM_DW-1:0]    g_hi_wd [0:VLEN-1];

    always @(*) begin
        for (i = 0; i < VLEN; i = i + 1) begin
            la       = lane_addr[i];
            w_lo     = la & WORD_MASK;
            w_hi     = (la + {59'd0, nbytes} - 64'd1) & WORD_MASK;
            boff     = la[MEM_SHIFT-1:0];
            g_wlo[i] = w_lo;
            g_whi[i] = w_hi;
            // the high half of a straddling lane
            hi_cnt = {1'd0, nbytes} - (MEM_BYTES - {1'd0, boff});
            // the low half, or the whole of a lane that does not straddle
            lo_cnt = (w_hi != w_lo) ? (MEM_BYTES - {1'd0, boff})
                                    : {1'd0, nbytes};
            g_lo_be[i] = {MEM_BYTES{1'b0}};
            g_hi_be[i] = {MEM_BYTES{1'b0}};
            g_lo_wd[i] = {MEM_DW{1'b0}};
            g_hi_wd[i] = {MEM_DW{1'b0}};
            for (b = 0; b < MEM_BYTES; b = b + 1) begin
                if ((w_hi != w_lo) && (b < hi_cnt)) begin
                    g_hi_be[i][b] = 1'b1;
                    g_hi_wd[i][b*8 +: 8] =
                        lane_wdata[i][(MEM_BYTES - boff + b)*8 +: 8];
                end
                if ((b >= boff) && (b < (boff + lo_cnt))) begin
                    g_lo_be[i][b] = 1'b1;
                    g_lo_wd[i][b*8 +: 8] = lane_wdata[i][(b - boff)*8 +: 8];
                end
            end
        end
    end

    reg [63:0]          ln_wlo   [0:VLEN-1];
    reg [63:0]          ln_whi   [0:VLEN-1];
    reg [VLEN-1:0]      ln_strad;
    reg [MEM_BYTES-1:0] ln_lo_be [0:VLEN-1];
    reg [MEM_BYTES-1:0] ln_hi_be [0:VLEN-1];
    reg [MEM_DW-1:0]    ln_lo_wd [0:VLEN-1];
    reg [MEM_DW-1:0]    ln_hi_wd [0:VLEN-1];

    always @(posedge clk) begin
        if (state == S_SETUP) begin
            for (i = 0; i < VLEN; i = i + 1) begin
                ln_wlo[i]   <= g_wlo[i];
                ln_whi[i]   <= g_whi[i];
                ln_strad[i] <= (g_whi[i] != g_wlo[i]);
                ln_lo_be[i] <= g_lo_be[i];
                ln_hi_be[i] <= g_hi_be[i];
                ln_lo_wd[i] <= g_lo_wd[i];
                ln_hi_wd[i] <= g_hi_wd[i];
            end
        end
    end

    always @(*) begin
        plan_valid = 1'b0;
        plan_word  = 64'd0;
        plan_lanes = {VLEN{1'b0}};
        plan_low   = {VLEN{1'b0}};
        plan_high  = {VLEN{1'b0}};
        plan_be    = {MEM_BYTES{1'b0}};
        plan_wdata = {MEM_DW{1'b0}};
        sel        = 4'd0;
        found      = 1'b0;

        // lowest lane still owing bytes
        for (i = VLEN-1; i >= 0; i = i - 1)
            if (pending[i]) begin
                sel   = i[3:0];
                found = 1'b1;
            end

        if (found) begin
            plan_valid = 1'b1;
            plan_word  = partial[sel] ? ln_whi[sel] : ln_wlo[sel];

            for (i = 0; i < VLEN; i = i + 1) begin
                if (pending[i] && partial[i]) begin
                    // high half of a straddling lane
                    if (ln_whi[i] == plan_word) begin
                        plan_lanes[i] = 1'b1;
                        plan_high[i]  = 1'b1;
                    end
                end else if (pending[i] && (ln_wlo[i] == plan_word)) begin
                    plan_lanes[i] = 1'b1;
                    plan_low[i]   = ln_strad[i];
                end
            end

            // Lanes in one beat cover disjoint bytes of the word (or the same
            // bytes, for a scatter that names one address twice -- which this
            // has always ORed), so the collection is a plain OR over lanes.
            for (i = 0; i < VLEN; i = i + 1) begin
                plan_be = plan_be |
                    ({MEM_BYTES{plan_lanes[i] && wr_bytes}} &
                     (plan_high[i] ? ln_hi_be[i] : ln_lo_be[i]));
                plan_wdata = plan_wdata |
                    ({MEM_DW{plan_lanes[i]}} &
                     (plan_high[i] ? ln_hi_wd[i] : ln_lo_wd[i]));
            end
        end
    end

    //--------------------------------------------------------------------------
    // Free beat-table slot
    //--------------------------------------------------------------------------
    reg [BEAT_W-1:0] free_tag;
    reg              have_slot;
    always @(*) begin
        free_tag  = {BEAT_W{1'b0}};
        have_slot = 1'b0;
        for (i = NUM_BEATS-1; i >= 0; i = i - 1)
            if (!bt_valid[i]) begin
                free_tag  = i[BEAT_W-1:0];
                have_slot = 1'b1;
            end
    end

    wire issue_ok = (state == S_RUN) && plan_valid && have_slot;

    assign mem_req_valid = issue_ok;
    assign mem_req_addr  = plan_word;
    assign mem_req_we    = r_store;
    assign mem_req_be    = wr_bytes ? plan_be : {MEM_BYTES{1'b0}};
    assign mem_req_wdata = plan_wdata;
    assign mem_req_tag   = free_tag;
    assign mem_req_hint  = r_hint;
    assign mem_req_amo   = r_amo;
    assign mem_req_aop   = r_aop;
    assign mem_req_cmp   = r_cmp;

    wire issue_fire = mem_req_valid && mem_req_ready;

    //--------------------------------------------------------------------------
    // Sequencer
    //--------------------------------------------------------------------------
    reg [63:0] eff_base;
    reg [63:0] idx_ext;
    reg [MEM_SHIFT-1:0] cboff;
    reg [5:0]           clo, chi;
    reg [VLEN-1:0]      rsp_lanes, rsp_low, rsp_high;

    wire no_beats = (bt_valid == {NUM_BEATS{1'b0}});

    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            cpl_valid <= 1'b0;
            pending   <= {VLEN{1'b0}};
            partial   <= {VLEN{1'b0}};
            active    <= {VLEN{1'b0}};
            bt_valid  <= {NUM_BEATS{1'b0}};
        end else begin

            //------------------------------------------------------------------
            // Response handling runs in every state: a response must never wait
            // on the issue path (arch overview sec 10.5, rule 2).
            //------------------------------------------------------------------
            if (mem_rsp_valid) begin
                rsp_lanes = bt_lanes[mem_rsp_tag];
                rsp_low   = bt_low[mem_rsp_tag];
                rsp_high  = bt_high[mem_rsp_tag];
                if (!r_store) begin
                    for (i = 0; i < VLEN; i = i + 1) begin
                        if (rsp_lanes[i]) begin
                            cboff = lane_addr[i][MEM_SHIFT-1:0];
                            if (rsp_high[i]) begin
                                chi = {1'd0, nbytes} - (MEM_BYTES - {1'd0, cboff});
                                for (b = 0; b < MEM_BYTES; b = b + 1)
                                    if (b < chi)
                                        lane_rdata[i][(MEM_BYTES - cboff + b)*8 +: 8]
                                            <= mem_rsp_rdata[b*8 +: 8];
                            end else begin
                                clo = rsp_low[i] ? (MEM_BYTES - {1'd0, cboff})
                                                 : {1'd0, nbytes};
                                for (b = 0; b < MEM_BYTES; b = b + 1)
                                    if (b < clo)
                                        lane_rdata[i][b*8 +: 8]
                                            <= mem_rsp_rdata[(cboff + b)*8 +: 8];
                            end
                        end
                    end
                end
                bt_valid[mem_rsp_tag] <= 1'b0;
            end

            case (state)

            //------------------------------------------------------------------
            S_IDLE: begin
                cpl_valid <= 1'b0;
                if (req_valid) begin
                    r_store     <= req_store;
                    r_amo       <= req_amo;
                    r_aop       <= req_aop;
                    r_cmp       <= req_cmp;
                    r_vector    <= req_vector;
                    r_indexed   <= req_indexed;
                    r_signed    <= req_signed;
                    r_size      <= req_size;
                    r_hint      <= req_hint;
                    r_rctx      <= req_rctx;
                    r_dst_class <= req_dst_class;
                    r_dst_idx   <= req_dst_idx;
                    nbytes      <= 5'd1 << req_size;

                    eff_base = req_base + req_offset;

                    for (i = 0; i < VLEN; i = i + 1) begin
                        if (req_indexed) begin
                            idx_ext = {{(64-ELEN){req_index[i*ELEN + ELEN-1]}},
                                       req_index[i*ELEN +: ELEN]};
                            lane_addr[i] <= req_base + (idx_ext << req_size);
                        end else if (req_vector) begin
                            lane_addr[i] <= eff_base + (i << req_size);
                        end else begin
                            lane_addr[i] <= eff_base;
                        end
                        lane_wdata[i] <= req_vector
                                       ? {{(64-ELEN){1'b0}}, req_vdata[i*ELEN +: ELEN]}
                                       : req_sdata;
                        lane_rdata[i] <= 64'd0;
                    end

                    // A scalar access is lane 0 only.
                    active  <= req_vector ? req_mask : {{(VLEN-1){1'b0}}, req_mask[0]};
                    pending <= req_vector ? req_mask : {{(VLEN-1){1'b0}}, req_mask[0]};
                    partial <= {VLEN{1'b0}};

                    state <= S_SETUP;
                end
            end

            //------------------------------------------------------------------
            // One cycle for the lane arrays to land before the planner reads
            // them, which keeps the planner purely combinational.
            S_SETUP: state <= S_RUN;

            //------------------------------------------------------------------
            // Issue beats as fast as the port takes them. Retiring a lane from
            // `pending` at ISSUE time, not at response time, is what lets the
            // next beat be planned in the very next cycle.
            S_RUN: begin
                if (issue_fire) begin
                    bt_valid[free_tag] <= 1'b1;
                    bt_lanes[free_tag] <= plan_lanes;
                    bt_low[free_tag]   <= plan_low;
                    bt_high[free_tag]  <= plan_high;

                    for (i = 0; i < VLEN; i = i + 1) begin
                        if (plan_lanes[i]) begin
                            if (plan_low[i]) begin
                                partial[i] <= 1'b1;      // still owes the high half
                            end else begin
                                pending[i] <= 1'b0;
                                partial[i] <= 1'b0;
                            end
                        end
                    end
                end
                if (!plan_valid && !issue_fire)
                    state <= S_DRAIN;
            end

            //------------------------------------------------------------------
            S_DRAIN: begin
                // every beat issued; wait for the last response
                if (no_beats && !mem_rsp_valid) state <= S_DONE;
            end

            //------------------------------------------------------------------
            S_DONE: begin
                cpl_valid     <= 1'b1;
                cpl_rctx      <= r_rctx;
                cpl_dst_class <= r_store ? `TNCA_DST_NONE : r_dst_class;
                cpl_dst_idx   <= r_dst_idx;
                cpl_lane_en   <= r_store ? {VLEN{1'b0}} : active;
                if (cpl_ack) begin
                    cpl_valid <= 1'b0;
                    state     <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end

    //--------------------------------------------------------------------------
    // Result formatting: sign/zero extension per ISA sec 9.1
    //--------------------------------------------------------------------------
    reg [ELEN-1:0] ext [0:VLEN-1];
    reg [63:0]     raw;
    always @(*) begin
        for (i = 0; i < VLEN; i = i + 1) begin
            raw = lane_rdata[i];
            case (r_size)
            `TNCA_SZ_B: ext[i] = r_signed ? {{(ELEN-8){raw[7]}},  raw[7:0]}
                                          : {{(ELEN-8){1'b0}},    raw[7:0]};
            `TNCA_SZ_H: ext[i] = r_signed ? {{(ELEN-16){raw[15]}}, raw[15:0]}
                                          : {{(ELEN-16){1'b0}},    raw[15:0]};
            default:    ext[i] = raw[ELEN-1:0];
            endcase
        end
    end

    always @(*) begin
        cpl_vdata = {(VLEN*ELEN){1'b0}};
        for (i = 0; i < VLEN; i = i + 1)
            cpl_vdata[i*ELEN +: ELEN] = ext[i];
    end

    // Scalar loads read lane 0's 64-bit staging register directly.
    reg [63:0] s_raw;
    always @(*) begin
        s_raw = lane_rdata[0];
        case (r_size)
        `TNCA_SZ_B: cpl_sdata = r_signed ? {{56{s_raw[7]}},  s_raw[7:0]}
                                         : {56'd0,           s_raw[7:0]};
        `TNCA_SZ_H: cpl_sdata = r_signed ? {{48{s_raw[15]}}, s_raw[15:0]}
                                         : {48'd0,           s_raw[15:0]};
        `TNCA_SZ_W: cpl_sdata = r_signed ? {{32{s_raw[31]}}, s_raw[31:0]}
                                         : {32'd0,           s_raw[31:0]};
        default:    cpl_sdata = s_raw;
        endcase
    end

    wire _unused = &{1'b0, r_vector, 1'b0};

endmodule
