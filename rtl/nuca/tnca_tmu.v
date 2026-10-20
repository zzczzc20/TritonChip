//==============================================================================
// tnca_tmu -- the Tensor Management Unit (DCO, arch overview sec 8)
//
// DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive
// Management (Zhou, Lai, Gu, Zhang; doc/DCO.pdf). The NUCA slices decide what
// to keep; this unit tells them what software knows and they cannot see:
//
//   tensor metadata   per registered tensor: base, size, nAcc (how many times
//                     each of its tiles will be read in this kernel, from the
//                     dataflow), and whether it is bypassed outright (nAcc = 1)
//   tile metadata     per live tile: how many times it has been read; when that
//                     reaches nAcc the tile is dead
//   dead tiles        a 16-entry FIFO of dead tile identifiers, which every
//                     slice consults when it picks a victim
//
// A tile is an aligned 2^lsb-byte chunk of a registered tensor, lsb being the
// tensor's own tile size (the paper's per-tensor "tile size"; the TILE
// register's D_LSB when it registered none). Its identifier is the tensor's
// index above the chunk number, so tiles of different tensors never alias.
// A tile is counted once per use when its LAST 8-byte word is read (the
// paper's "tile last line", taken to the word because a request here is a
// word). The policy knobs the paper sets with its `set` instruction are
// registers here, and so is registration: software writes a staging line and
// commits it with an ordinary store (see the register map below). Everything
// this unit produces is a hint. A late, dropped or wrong update moves when a
// tile is declared dead; it never changes a value a load returns.
//
// REGISTER MAP (byte offsets in the TMU window, 8-byte registers)
//
//   0x000  POLICY      [1:0] replacement: 0 round-robin, 1 LRU, 2 anti-thrash
//                      [2] dead-block prediction   [3] dynamic bypass
//                      [4] gqa_bypass (bypass only the slower core of a pair)
//                      [5] bypass tensors registered with nAcc = 1
//                      [6] static gear (GEAR_INIT, no adaptation)
//   0x008  PARAMS      [3:0] B_BITS (at most 3)  [27:24] GEAR_INIT
//   0x010  GEAR        [15:0] window (cycles)  [31:16] bypass_ub
//                      [47:32] bypass_lb (evictions per window)
//   0x018  CLEAR       any write: forget every tensor, live tile and dead tile
//   0x020  STAT_REG    (read) tensors registered since reset
//   0x028  STAT_TLW    (read) tile-last-word updates applied
//   0x030  STAT_DEAD   (read) tiles declared dead
//   0x038  STAT_DROP   (read) updates the slices dropped (unit busy)
//   0x040  TILE        [5:0] D_LSB: the default tile granule, for a tensor
//                      registered without one.
//   0x100 + 64*k       staging (any k in 0..3; the compiler uses 0x100):
//                        +0 BASE  +8 SIZE (bytes; 0 = up to the next tensor)
//                        +16 DESC [31:0] nAcc  [32] bypass  [39:36] operand
//                                 [45:40] tile lsb: tiles are 2^lsb bytes (0: D_LSB)
//                      a write to DESC commits: the entry with the same base is
//                      updated, else a free one is taken, else the oldest.
//
// The TMU keeps one staging set PER REQUESTER (the slice request's source,
// a core), so cores registering at the same time do not mix their fields.
// A registration's three stores stay in order because they go to one line:
// a line has one home slice, and a slice serves the requests to one line in
// arrival order (rtl/nuca/tnca_nuca_slice.v, sec 14.1).
//==============================================================================
module tnca_tmu #(
    parameter N_SLICE  = 4,
    parameter REPLICA_STATE = "false",
    // 0 keeps the original combinational path. 1 registers classification;
    // 2 also registers the arbitrated raw address before classification.
    parameter TLW_PIPE_STAGES = 0,
    parameter REGISTER_WRITES = 0,
    parameter REGISTERED_RANGES = 0,
    parameter SRC_W    = 4,
    parameter N_SRC    = 4,
    parameter TENSORS  = 8,
    parameter T_W      = 3,
    parameter LIVE     = 256,
    parameter LIVE_W   = 8,
    parameter DEAD     = 16,
    parameter DEAD_W   = 4,
    parameter ID_W     = 16           // tile identifier: at most 16 address bits
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // ---- register writes, from whichever slice homes the line ------------
    input  wire [N_SLICE-1:0]           mw_valid,
    output reg  [N_SLICE-1:0]           mw_ready,
    input  wire [N_SLICE*12-1:0]        mw_off,
    input  wire [N_SLICE*64-1:0]        mw_data,
    input  wire [N_SLICE*SRC_W-1:0]     mw_src,
    // ---- register reads (combinational) ----------------------------------
    input  wire [N_SLICE*12-1:0]        mr_off,
    output reg  [N_SLICE*64-1:0]        mr_data,

    // ---- tile-last-word notices from the slices --------------------------
    input  wire [N_SLICE-1:0]           tl_valid,
    output reg  [N_SLICE-1:0]           tl_ready,
    input  wire [N_SLICE*64-1:0]        tl_addr,
    input  wire [N_SLICE-1:0]           tl_drop,      // a notice a slice could not hold

    // ---- what every slice sees -------------------------------------------
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [7:0]                   policy,
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [3:0]                   b_bits,
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [5:0]                   d_lsb,
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [3:0]                   gear_init,
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [15:0]                  gear_window,
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [15:0]                  gear_ub,
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [15:0]                  gear_lb,
    (* DONT_TOUCH = REPLICA_STATE *) output reg                          owner_epoch,
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [TENSORS-1:0]           t_valid,
    output wire [TENSORS*64-1:0]        t_base,
    output wire [TENSORS*64-1:0]        t_size,
    output wire [TENSORS*65-1:0]        t_end,
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [TENSORS-1:0]           t_bypass,
    output wire [TENSORS*6-1:0]         t_lsb,
    (* DONT_TOUCH = REPLICA_STATE *) output reg  [DEAD-1:0]              dead_valid,
    output wire [DEAD*ID_W-1:0]         dead_id
);

    // One set of loop variables per always block: sharing one between
    // blocks makes each sensitive to the others' loops in simulation.
    integer i, k_dr, k_ar, s_ar, s_rd, k_nt, k_rg;

    //==========================================================================
    // Tensor metadata
    //==========================================================================
    (* DONT_TOUCH = REPLICA_STATE *) reg [63:0] tb   [0:TENSORS-1];
    reg [63:0] ts   [0:TENSORS-1];
    // Commit with base/size, so slices never combine different metadata epochs.
    // Bit 64 also represents an unbounded (size-zero) interval.
    (* DONT_TOUCH = REPLICA_STATE *) reg [64:0] te   [0:TENSORS-1];
    (* DONT_TOUCH = REPLICA_STATE *) reg [31:0] tn   [0:TENSORS-1];     // nAcc
    reg [3:0]  top  [0:TENSORS-1];     // operand id (recorded, informational)
    (* DONT_TOUCH = REPLICA_STATE *) reg [5:0]  tls  [0:TENSORS-1];     // tile granule, log2 bytes (0: D_LSB)
    (* DONT_TOUCH = REPLICA_STATE *) reg [T_W-1:0] t_next;              // replacement pointer when full

    genvar g;
    generate
        for (g = 0; g < TENSORS; g = g + 1) begin : g_tout
            assign t_base[g*64 +: 64] = tb[g];
            assign t_size[g*64 +: 64] = ts[g];
            assign t_end[g*65 +: 65]  = te[g];
            assign t_lsb[g*6 +: 6]    = tls[g];
        end
    endgenerate

    // Staging, one line per requester
    (* DONT_TOUCH = REPLICA_STATE *) reg [63:0] st_base [0:N_SRC-1];
    (* DONT_TOUCH = REPLICA_STATE *) reg [63:0] st_size [0:N_SRC-1];

    //==========================================================================
    // Live tiles (direct mapped on the identifier's low bits) and dead tiles
    //==========================================================================
    (* DONT_TOUCH = REPLICA_STATE *) reg [LIVE-1:0]   lv_valid;
    (* DONT_TOUCH = REPLICA_STATE *) reg [ID_W-1:0]   lv_id    [0:LIVE-1];
    (* DONT_TOUCH = REPLICA_STATE *) reg [31:0]       lv_cnt   [0:LIVE-1];

    (* DONT_TOUCH = REPLICA_STATE *) reg [ID_W-1:0]   dq_id    [0:DEAD-1];
    (* DONT_TOUCH = REPLICA_STATE *) reg [DEAD_W-1:0] dq_wr;
    generate
        for (g = 0; g < DEAD; g = g + 1) begin : g_dout
            assign dead_id[g*ID_W +: ID_W] = dq_id[g];
        end
    endgenerate

    (* DONT_TOUCH = REPLICA_STATE *) reg [31:0] stat_reg, stat_tlw, stat_dead, stat_drop;
    localparam DROP_W = $clog2(N_SLICE+1);
    reg [DROP_W-1:0] n_drop;
    always @(*) begin
        n_drop = {DROP_W{1'b0}};
        for (k_dr = 0; k_dr < N_SLICE; k_dr = k_dr + 1) n_drop = n_drop + tl_drop[k_dr];
    end

    //==========================================================================
    // Arbitration: one register write and one tile notice per cycle, rotating
    //==========================================================================
    localparam SELECT_W = N_SLICE > 1 ? $clog2(N_SLICE) : 1;
    (* DONT_TOUCH = REPLICA_STATE *) reg [SELECT_W-1:0] mw_rr, tl_rr;
    (* DONT_TOUCH = REPLICA_STATE *) reg tl_raw_valid_q, tl_class_valid_q;
    (* DONT_TOUCH = REPLICA_STATE *) reg [63:0] tl_raw_addr_q;
    (* DONT_TOUCH = REPLICA_STATE *) reg tl_class_hit_q;
    (* DONT_TOUCH = REPLICA_STATE *) reg [ID_W-1:0] tl_class_id_q;
    (* DONT_TOUCH = REPLICA_STATE *) reg [31:0] tl_class_nacc_q;
    wire tl_pipeline_busy = TLW_PIPE_STAGES != 0 &&
        (tl_class_valid_q || (TLW_PIPE_STAGES == 2 && tl_raw_valid_q));
    // Capture before acknowledgement. Publication and ACK remain one edge.
    (* DONT_TOUCH = REPLICA_STATE *) reg mw_pending_q;
    (* DONT_TOUCH = REPLICA_STATE *) reg [SELECT_W-1:0] mw_port_q;
    (* DONT_TOUCH = REPLICA_STATE *) reg [11:0] mw_off_q;
    (* DONT_TOUCH = REPLICA_STATE *) reg [63:0] mw_data_q;
    (* DONT_TOUCH = REPLICA_STATE *) reg [SRC_W-1:0] mw_src_q;
    reg mw_offer;
    reg [SELECT_W-1:0] mw_offer_port;
    reg       mw_go, tl_go;
    reg [SELECT_W-1:0] mw_sel, tl_sel;
    always @(*) begin
        mw_go = 1'b0; mw_sel = {SELECT_W{1'b0}}; mw_ready = {N_SLICE{1'b0}};
        tl_go = 1'b0; tl_sel = {SELECT_W{1'b0}}; tl_ready = {N_SLICE{1'b0}};
        mw_offer=0;mw_offer_port=0;
        for (k_ar = N_SLICE-1; k_ar >= 0; k_ar = k_ar - 1) begin
            s_ar = (k_ar + mw_rr) % N_SLICE;
            if (mw_valid[s_ar]) begin mw_offer = 1'b1; mw_offer_port = s_ar[SELECT_W-1:0]; end
        end
        for (k_ar = N_SLICE-1; k_ar >= 0; k_ar = k_ar - 1) begin
            s_ar = (k_ar + tl_rr) % N_SLICE;
            if (tl_valid[s_ar]) begin tl_go = 1'b1; tl_sel = s_ar[SELECT_W-1:0]; end
        end
        mw_go = REGISTER_WRITES ? mw_pending_q : mw_offer;
        mw_sel = REGISTER_WRITES ? mw_port_q : mw_offer_port;
        if ((TLW_PIPE_STAGES != 0 || REGISTER_WRITES) && (mw_offer || mw_pending_q)) begin
            // Stop new notices while ANY register write is waiting. Existing
            // accepted notices drain in at most TLW_PIPE_STAGES cycles. Only
            // then acknowledge the write, so CLEAR/DESC/default-granule changes
            // cannot pass an older notice or alter its metadata classification.
            // Tables and owner_epoch still commit on the write's handshake edge.
            tl_go = 1'b0;
            if (tl_pipeline_busy) mw_go = 1'b0;
        end
        if (mw_go) mw_ready[mw_sel] = 1'b1;
        if (tl_go) tl_ready[tl_sel] = 1'b1;
    end

    wire [11:0]      w_off  = REGISTER_WRITES ? mw_off_q : mw_off[mw_sel*12 +: 12];
    wire [63:0]      w_data = REGISTER_WRITES ? mw_data_q : mw_data[mw_sel*64 +: 64];
    wire [SRC_W-1:0] w_srcf = REGISTER_WRITES ? mw_src_q : mw_src[mw_sel*SRC_W +: SRC_W];
    // Staging belongs to the complete requester ID, including cores 4..15.
    wire [SRC_W-1:0] w_src  = w_srcf;
    wire [63:0]      accepted_addr = tl_addr[tl_sel*64 +: 64];
    wire [63:0]      n_addr = TLW_PIPE_STAGES == 2 ? tl_raw_addr_q : accepted_addr;
    wire            classify_go = TLW_PIPE_STAGES == 2 ? tl_raw_valid_q : tl_go;
    wire            update_go = TLW_PIPE_STAGES != 0 ? tl_class_valid_q : tl_go;
    wire            update_hit = TLW_PIPE_STAGES != 0 ? tl_class_hit_q : nt_hit;

    // Register reads
    always @(*) begin
        for (s_rd = 0; s_rd < N_SLICE; s_rd = s_rd + 1)
            case (mr_off[s_rd*12 +: 12])
            12'h000: mr_data[s_rd*64 +: 64] = {56'd0, policy};
            12'h008: mr_data[s_rd*64 +: 64] = {36'd0, gear_init, 20'd0, b_bits};
            12'h010: mr_data[s_rd*64 +: 64] = {16'd0, gear_lb, gear_ub, gear_window};
            12'h020: mr_data[s_rd*64 +: 64] = {32'd0, stat_reg};
            12'h028: mr_data[s_rd*64 +: 64] = {32'd0, stat_tlw};
            12'h030: mr_data[s_rd*64 +: 64] = {32'd0, stat_dead};
            12'h038: mr_data[s_rd*64 +: 64] = {32'd0, stat_drop};
            12'h040: mr_data[s_rd*64 +: 64] = {58'd0, d_lsb};
            default: mr_data[s_rd*64 +: 64] = 64'd0;
            endcase
    end

    //==========================================================================
    // A tile notice: which tensor, which tile, one more read
    //==========================================================================
    // The tensor an address belongs to: inside [base, base+size), or with a
    // size of 0, at or above the base and below the next registered base.
    localparam ADDR_W = 64, DCO_T = TENSORS;
`include "tnca_dco_owner_registered.vh"
    wire [11:0] nt_registered = REGISTERED_RANGES ?
        owner_registered(n_addr,t_valid,t_base,t_end,t_bypass,t_lsb) : 12'd0;
    reg            nt_hit;
    reg [T_W-1:0]  nt_idx;
    reg [63:0]     nt_best;
    always @(*) begin
        nt_hit = 1'b0; nt_idx = {T_W{1'b0}}; nt_best = 64'd0;
        if (REGISTERED_RANGES) begin
            nt_hit = nt_registered[11];
            nt_idx = nt_registered[6 +: T_W];
        end else begin
        for (k_nt = 0; k_nt < TENSORS; k_nt = k_nt + 1)
            if (t_valid[k_nt] && (n_addr >= tb[k_nt]) &&
                ((ts[k_nt] != 64'd0) ? (n_addr - tb[k_nt] < ts[k_nt]) : 1'b1) &&
                (!nt_hit || tb[k_nt] > nt_best)) begin
                nt_hit = 1'b1; nt_idx = k_nt[T_W-1:0]; nt_best = tb[k_nt];
            end
        end
    end
    // tile identifier = {tensor index, chunk number in the tensor's granule};
    // the slices form the same one for a victim (tnca_nuca_slice.v tile_id)
    wire [5:0]        n_lsb   = (tls[nt_idx] != 6'd0) ? tls[nt_idx] : d_lsb;
    wire [63:0]       n_shift = n_addr >> n_lsb;
    wire [3:0]        n_tix   = {{(4-T_W){1'b0}}, nt_idx};
    wire [ID_W-1:0]   classified_id = {n_tix[2:0], n_shift[ID_W-4:0]};
    wire [ID_W-1:0]   n_id = TLW_PIPE_STAGES != 0 ? tl_class_id_q : classified_id;
    wire [LIVE_W-1:0] n_slot  = n_id[LIVE_W-1:0];
    wire [31:0]       n_nacc  = TLW_PIPE_STAGES != 0 ? tl_class_nacc_q : tn[nt_idx];
    wire              n_same  = lv_valid[n_slot] && (lv_id[n_slot] == n_id);
    wire [31:0]       n_cnt   = (n_same ? lv_cnt[n_slot] : 32'd0) + 32'd1;

    // Registration: the entry for this base, else a free one, else the oldest
    reg            rg_same;
    reg [T_W-1:0]  rg_idx;
    reg            rg_free;
    reg [T_W-1:0]  rg_fidx;
    always @(*) begin
        rg_same = 1'b0; rg_idx = {T_W{1'b0}};
        rg_free = 1'b0; rg_fidx = {T_W{1'b0}};
        for (k_rg = TENSORS-1; k_rg >= 0; k_rg = k_rg - 1) begin
            if (t_valid[k_rg] && tb[k_rg] == st_base[w_src]) begin
                rg_same = 1'b1; rg_idx = k_rg[T_W-1:0];
            end
            if (!t_valid[k_rg]) begin rg_free = 1'b1; rg_fidx = k_rg[T_W-1:0]; end
        end
    end
    wire [T_W-1:0] rg_slot = rg_same ? rg_idx : rg_free ? rg_fidx : t_next;

    //==========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            mw_pending_q<=0;mw_port_q<=0;mw_off_q<=0;mw_data_q<=0;mw_src_q<=0;
            policy      <= 8'd0;          // round-robin, nothing on: the baseline
            b_bits      <= 4'd3;
            d_lsb       <= 6'd10;
            gear_init   <= 4'd0;
            gear_window <= 16'd1024;
            gear_ub     <= 16'd48;
            gear_lb     <= 16'd8;
            owner_epoch <= 1'b0;
            t_valid     <= {TENSORS{1'b0}};
            t_bypass    <= {TENSORS{1'b0}};
            t_next      <= {T_W{1'b0}};
            lv_valid    <= {LIVE{1'b0}};
            dead_valid  <= {DEAD{1'b0}};
            dq_wr       <= {DEAD_W{1'b0}};
            tl_raw_valid_q <= 0; tl_class_valid_q <= 0;
            tl_raw_addr_q <= 0; tl_class_hit_q <= 0;
            tl_class_id_q <= 0; tl_class_nacc_q <= 0;
            stat_reg <= 32'd0; stat_tlw <= 32'd0; stat_dead <= 32'd0; stat_drop <= 32'd0;
            mw_rr <= {SELECT_W{1'b0}}; tl_rr <= {SELECT_W{1'b0}};
        end else begin
            if (REGISTER_WRITES) begin
                if (!mw_pending_q && mw_offer) begin
                    mw_pending_q<=1;mw_port_q<=mw_offer_port;
                    mw_off_q<=mw_off[mw_offer_port*12+:12];
                    mw_data_q<=mw_data[mw_offer_port*64+:64];
                    mw_src_q<=mw_src[mw_offer_port*SRC_W+:SRC_W];
                end
                if (mw_go) mw_pending_q<=0;
            end
            stat_drop <= stat_drop + n_drop;
            if (TLW_PIPE_STAGES != 0) begin
                tl_raw_valid_q <= TLW_PIPE_STAGES == 2 && tl_go;
                if (TLW_PIPE_STAGES == 2 && tl_go) tl_raw_addr_q <= accepted_addr;
                tl_class_valid_q <= classify_go;
                if (classify_go) begin
                    tl_class_hit_q <= nt_hit;
                    tl_class_id_q <= classified_id;
                    tl_class_nacc_q <= tn[nt_idx];
                end
            end
            //------------------------------------------------------------------
            // register write
            //------------------------------------------------------------------
            if (mw_go) begin
                mw_rr <= mw_sel + 1'b1;
                if ((w_srcf < N_SRC) && w_off[11:8] == 4'h0) begin
                    case (w_off[7:0])
                    8'h00: policy <= w_data[7:0];
                    8'h08: begin
                        b_bits    <= w_data[3:0];
                        gear_init <= w_data[27:24];
                    end
                    8'h40: d_lsb <= w_data[5:0];
                    8'h10: begin
                        gear_window <= w_data[15:0];
                        gear_ub     <= w_data[31:16];
                        gear_lb     <= w_data[47:32];
                    end
                    8'h18: begin
                        owner_epoch <= ~owner_epoch;
                        t_valid    <= {TENSORS{1'b0}};
                        t_bypass   <= {TENSORS{1'b0}};
                        lv_valid   <= {LIVE{1'b0}};
                        dead_valid <= {DEAD{1'b0}};
                    end
                    default: ;
                    endcase
                end else if ((w_srcf < N_SRC) && w_off[11:8] == 4'h1) begin
                    // staging lines 0x100 + 64*src (the src the store came from)
                    case (w_off[5:0])
                    6'h00: st_base[w_src] <= w_data;
                    6'h08: st_size[w_src] <= w_data;
                    6'h10: begin
                        owner_epoch <= ~owner_epoch;
                        t_valid[rg_slot]  <= 1'b1;
                        tb[rg_slot]       <= st_base[w_src];
                        ts[rg_slot]       <= st_size[w_src];
                        te[rg_slot]       <= (st_size[w_src] == 64'd0) ?
                            {1'b1, 64'd0} :
                            ({1'b0, st_base[w_src]} + {1'b0, st_size[w_src]});
                        tn[rg_slot]       <= w_data[31:0];
                        t_bypass[rg_slot] <= w_data[32];
                        top[rg_slot]      <= w_data[39:36];
                        tls[rg_slot]      <= w_data[45:40];
                        if (!rg_same && !rg_free) t_next <= t_next + 1'b1;
                        stat_reg <= stat_reg + 32'd1;
                    end
                    default: ;
                    endcase
                end
            end

            //------------------------------------------------------------------
            // tile-last-word notice
            //------------------------------------------------------------------
            if (tl_go) tl_rr <= tl_sel + 1'b1;
            if (update_go) begin
                if (update_hit && n_nacc != 32'd0) begin
                    stat_tlw <= stat_tlw + 32'd1;
                    if (n_cnt >= n_nacc) begin
                        // Dead: into the FIFO (the oldest entry goes when it is
                        // full), and out of the live table.
                        dq_id[dq_wr]      <= n_id;
                        dead_valid[dq_wr] <= 1'b1;
                        dq_wr             <= dq_wr + 1'b1;
                        lv_valid[n_slot]  <= 1'b0;
                        stat_dead         <= stat_dead + 32'd1;
                    end else begin
                        lv_valid[n_slot] <= 1'b1;
                        lv_id[n_slot]    <= n_id;
                        lv_cnt[n_slot]   <= n_cnt;
                    end
                end
            end
        end
    end

    generate if (TLW_PIPE_STAGES<0 || TLW_PIPE_STAGES>2) begin:g_bad_tlw_pipeline
        tnca_invalid_tlw_pipeline_depth invalid_configuration();
    end endgenerate
    wire _unused = &{1'b0, top[0], 1'b0};

endmodule
