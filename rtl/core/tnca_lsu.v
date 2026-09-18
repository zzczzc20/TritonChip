//==============================================================================
// tnca_lsu -- load/store unit with a physical-word coalescer (ISA sec 9.1)
//
// Handles the whole 0x20 family: scalar and vector, contiguous and indexed,
// loads and stores, with arbitrary alignment.
//
// Coalescing
// ----------
// Work is driven by memory words, not by lanes. Each beat picks the lowest
// lane still owing bytes, takes the MEM_DW-aligned word containing them, and
// serves every other outstanding lane whose bytes fall entirely inside that
// same word. A 32-bit contiguous vector access therefore costs VLEN/2 beats on
// a 64-bit port instead of VLEN, and a gather with locality collapses the same
// way; a fully scattered gather degrades to one beat per lane, which is the
// honest cost.
//
// A lane that straddles a word boundary is served in two beats: the low bytes
// with the first word, the high bytes with the next. It is tracked by the
// `partial` bit rather than by splitting the request, so the coalescer never
// has to reason about more than one word at a time.
//
// Masking (ISA sec 2.2): an inactive lane performs no memory access at all --
// not a suppressed one. Its destination lane keeps its previous value, which
// the core expresses by clearing that lane's write enable.
//
// Stage-1 scope: one instruction in flight. The core stalls a second memory op
// until this one retires. Multiple outstanding transactions arrive with the
// NUCA slice and the MSHR credit scheme in Stage 3.
//==============================================================================
`include "tnca_defs.vh"

module tnca_lsu #(
    parameter VLEN   = 16,
    parameter ELEN   = 32,
    parameter RCTX_W = 1,
    parameter MEM_DW = 64,
    parameter MEM_BYTES = 8,
    parameter MEM_SHIFT = 3          // log2(MEM_BYTES)
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
    input  wire [63:0]              req_base,
    input  wire [63:0]              req_offset,
    input  wire [VLEN*ELEN-1:0]     req_index,
    input  wire [63:0]              req_sdata,
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

    // ---- memory port ---------------------------------------------------------
    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [63:0]              mem_req_addr,  // MEM_BYTES aligned
    output wire                     mem_req_we,
    output wire [MEM_BYTES-1:0]     mem_req_be,
    output wire [MEM_DW-1:0]        mem_req_wdata,
    input  wire                     mem_rsp_valid,
    input  wire [MEM_DW-1:0]        mem_rsp_rdata
);

    localparam S_IDLE = 3'd0,
               S_SETUP= 3'd1,
               S_ISSUE= 3'd2,
               S_WAIT = 3'd3,
               S_DONE = 3'd4;

    reg [2:0] state;

    // ---- latched request -----------------------------------------------------
    reg              r_store, r_vector, r_indexed, r_signed;
    reg [2:0]        r_size;
    reg [RCTX_W-1:0] r_rctx;
    reg [2:0]        r_dst_class;
    reg [7:0]        r_dst_idx;

    reg [63:0]       lane_addr [0:VLEN-1];
    // Staging is 64 bits wide, not ELEN: a scalar 64-bit access is lane 0 and
    // must not be truncated. Vector lanes use only the low ELEN bits.
    reg [63:0]       lane_wdata[0:VLEN-1];
    reg [63:0]       lane_rdata[0:VLEN-1];
    reg [VLEN-1:0]   pending;     // lanes still owing bytes
    reg [VLEN-1:0]   partial;     // low half already served, high half owed
    reg [VLEN-1:0]   active;      // lanes that were unmasked at request time

    reg [4:0]        nbytes;      // bytes per lane, 1..8

    // beat bookkeeping
    reg [VLEN-1:0]   beat_lanes;      // lanes served by the outstanding request
    reg [VLEN-1:0]   beat_is_low;     // of those, which contributed a low half
    reg [63:0]       beat_word;

    integer i;
    integer b;

    assign req_ready = (state == S_IDLE);

    // --------------------------------------------------------------------------
    // Combinational beat planning: which word, which lanes, which byte enables
    // --------------------------------------------------------------------------
    reg  [63:0]           plan_word;
    reg  [VLEN-1:0]       plan_lanes;
    reg  [VLEN-1:0]       plan_low;
    reg  [MEM_BYTES-1:0]  plan_be;
    reg  [MEM_DW-1:0]     plan_wdata;
    reg                   plan_valid;

    localparam [63:0] WORD_MASK = ~((64'd1 << MEM_SHIFT) - 64'd1);

    reg [63:0] la;
    reg [63:0] w_lo, w_hi;
    reg [MEM_SHIFT-1:0] boff;
    reg [5:0]  lo_cnt, hi_cnt;
    reg        found;
    reg [3:0]  sel;

    always @(*) begin
        plan_valid = 1'b0;
        plan_word  = 64'd0;
        plan_lanes = {VLEN{1'b0}};
        plan_low   = {VLEN{1'b0}};
        plan_be    = {MEM_BYTES{1'b0}};
        plan_wdata = {MEM_DW{1'b0}};
        sel        = 4'd0;
        found      = 1'b0;

        // lowest pending lane
        for (i = VLEN-1; i >= 0; i = i - 1)
            if (pending[i]) begin
                sel   = i[3:0];
                found = 1'b1;
            end

        if (found) begin
            plan_valid = 1'b1;
            la   = lane_addr[sel];
            w_lo = la & WORD_MASK;
            w_hi = (la + {59'd0, nbytes} - 64'd1) & WORD_MASK;
            plan_word = partial[sel] ? w_hi : w_lo;

            for (i = 0; i < VLEN; i = i + 1) begin
                if (pending[i]) begin
                    la   = lane_addr[i];
                    w_lo = la & WORD_MASK;
                    w_hi = (la + {59'd0, nbytes} - 64'd1) & WORD_MASK;
                    boff = la[MEM_SHIFT-1:0];

                    if (partial[i]) begin
                        // high half of a straddling lane
                        if (w_hi == plan_word) begin
                            hi_cnt = {1'd0, nbytes} - (MEM_BYTES - {1'd0, boff});
                            plan_lanes[i] = 1'b1;
                            for (b = 0; b < MEM_BYTES; b = b + 1)
                                if (b < hi_cnt) begin
                                    plan_be[b] = plan_be[b] | r_store;
                                    plan_wdata[b*8 +: 8] = plan_wdata[b*8 +: 8] |
                                        lane_wdata[i][(MEM_BYTES - boff + b)*8 +: 8];
                                end
                        end
                    end else if (w_lo == plan_word) begin
                        plan_lanes[i] = 1'b1;
                        lo_cnt = (w_hi != w_lo) ? (MEM_BYTES - {1'd0, boff})
                                                : {1'd0, nbytes};
                        if (w_hi != w_lo) plan_low[i] = 1'b1;
                        for (b = 0; b < MEM_BYTES; b = b + 1)
                            if ((b >= boff) && (b < (boff + lo_cnt))) begin
                                plan_be[b] = plan_be[b] | r_store;
                                plan_wdata[b*8 +: 8] = plan_wdata[b*8 +: 8] |
                                    lane_wdata[i][(b - boff)*8 +: 8];
                            end
                    end
                end
            end
        end
    end

    // --------------------------------------------------------------------------
    // Response capture
    // --------------------------------------------------------------------------
    reg [MEM_SHIFT-1:0] cboff;
    reg [5:0]           clo, chi;

    // --------------------------------------------------------------------------
    // Sequencer
    // --------------------------------------------------------------------------
    reg [63:0] eff_base;
    reg [63:0] idx_ext;

    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            cpl_valid     <= 1'b0;
            pending       <= {VLEN{1'b0}};
            partial       <= {VLEN{1'b0}};
            active        <= {VLEN{1'b0}};
            beat_lanes    <= {VLEN{1'b0}};
            beat_is_low   <= {VLEN{1'b0}};
        end else begin
            case (state)

            // ------------------------------------------------------------------
            S_IDLE: begin
                cpl_valid <= 1'b0;
                if (req_valid) begin
                    r_store     <= req_store;
                    r_vector    <= req_vector;
                    r_indexed   <= req_indexed;
                    r_signed    <= req_signed;
                    r_size      <= req_size;
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

                    // A scalar access is lane 0 only. Scalar 64-bit stores need
                    // the upper word too, which the lane_wdata above cannot
                    // carry; it is handled by the wide path below.
                    active  <= req_vector ? req_mask : {{(VLEN-1){1'b0}}, req_mask[0]};
                    pending <= req_vector ? req_mask : {{(VLEN-1){1'b0}}, req_mask[0]};
                    partial <= {VLEN{1'b0}};

                    state <= S_SETUP;
                end
            end

            // ------------------------------------------------------------------
            // One cycle for the lane arrays to settle before the planner reads
            // them, which keeps the planner purely combinational.
            S_SETUP: begin
                if (pending == {VLEN{1'b0}}) state <= S_DONE;
                else                         state <= S_ISSUE;
            end

            // ------------------------------------------------------------------
            S_ISSUE: begin
                if (!plan_valid) begin
                    state <= S_DONE;
                end else if (mem_req_ready) begin
                    // latch which lanes this beat is responsible for
                    beat_lanes  <= plan_lanes;
                    beat_is_low <= plan_low;
                    beat_word   <= plan_word;
                    state       <= S_WAIT;
                end
            end

            // ------------------------------------------------------------------
            S_WAIT: begin
                if (mem_rsp_valid) begin
                    if (!r_store) begin
                        for (i = 0; i < VLEN; i = i + 1) begin
                            if (beat_lanes[i]) begin
                                cboff = lane_addr[i][MEM_SHIFT-1:0];
                                if (partial[i]) begin
                                    // high half: bytes 0..chi-1 of this word
                                    chi = {1'd0, nbytes} - (MEM_BYTES - {1'd0, cboff});
                                    for (b = 0; b < MEM_BYTES; b = b + 1)
                                        if (b < chi)
                                            lane_rdata[i][(MEM_BYTES - cboff + b)*8 +: 8]
                                                <= mem_rsp_rdata[b*8 +: 8];
                                end else begin
                                    clo = beat_is_low[i] ? (MEM_BYTES - {1'd0, cboff})
                                                         : {1'd0, nbytes};
                                    for (b = 0; b < MEM_BYTES; b = b + 1)
                                        if (b < clo)
                                            lane_rdata[i][b*8 +: 8]
                                                <= mem_rsp_rdata[(cboff + b)*8 +: 8];
                                end
                            end
                        end
                    end
                    // retire or re-queue the lanes served by this beat
                    for (i = 0; i < VLEN; i = i + 1) begin
                        if (beat_lanes[i]) begin
                            if (beat_is_low[i]) begin
                                partial[i] <= 1'b1;     // still owes the high half
                            end else begin
                                pending[i] <= 1'b0;
                                partial[i] <= 1'b0;
                            end
                        end
                    end
                    state <= S_SETUP;
                end
            end

            // ------------------------------------------------------------------
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

    assign mem_req_valid = (state == S_ISSUE) && plan_valid;
    assign mem_req_addr  = plan_word;
    assign mem_req_we    = r_store;
    assign mem_req_be    = r_store ? plan_be : {MEM_BYTES{1'b0}};
    assign mem_req_wdata = plan_wdata;

    // --------------------------------------------------------------------------
    // Result formatting: sign/zero extension per ISA sec 9.1
    // --------------------------------------------------------------------------
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
            default:    ext[i] = raw;
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

endmodule
