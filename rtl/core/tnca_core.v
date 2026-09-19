//==============================================================================
// tnca_core -- TNCA Stage-1 core: N Role Contexts over one shared datapath
//
// Pipeline (arch overview sec 4.1: in-order per RCTX, no ROB, no renaming,
// scoreboard):
//
//   F  select an RCTX, read the instruction
//   D  decode, check implementation limits
//   R  scoreboard check (indices only)
//   X  read the register files, execute, write back
//
// The register files are read in X, not in R: the operand latches an R-read
// would need are ~1.7 kbit of flops per stage for a 16x32 vector operand, and
// the arch overview is explicit (sec 5.3) that RF ports, not arithmetic, are
// the scarce resource here. Reads are asynchronous and write-first, so a
// consumer entering X in the cycle after its producer wrote sees the new value
// with no bubble.
//
// Role contexts interleave freely in the pipeline. Program order is maintained
// per RCTX by the scoreboard; there is no ordering between RCTXs except through
// memory and (from Stage 2) barriers.
//
// Control flow is predicted not-taken. A redirect in X squashes the F, D and R
// entries belonging to the same RCTX -- and only those, so other contexts keep
// making progress across a mispredict. Nothing younger than the branch has
// touched the scoreboard yet, because the scoreboard is written at the R->X
// boundary, so a squash needs no undo.
//
// WRITEBACK SCHEDULING
//
// Functional units no longer share one latency. The single-cycle units write
// back from X; tnca_vector_fpu and tnca_mul64 write back FP_LAT / MUL_LAT
// cycles later. A reservation shift register books the writeback cycle at issue
// time, so at most one scheduled result lands per cycle and no result can be
// dropped for want of a port.
//
// Completions that cannot be scheduled -- the LSU, whose latency is a property
// of the memory system, and the FP reduction sequencer -- take the port in any
// cycle the reservation register leaves free. When none is free they stall the
// front end, which drains the reservation within WB_DEPTH cycles, so an
// unscheduled completion waits a bounded time and is never starved.
//
// X therefore never repeats a side effect: a stalled front end inserts a bubble
// into X rather than freezing it.
//==============================================================================
`include "tnca_defs.vh"

module tnca_core #(
    parameter NUM_RCTX       = 2,
    parameter RCTX_W         = 1,
    parameter VLEN           = 16,
    parameter ELEN           = 32,
    parameter LW             = 4,          // log2(VLEN)
    parameter VREGS_PER_RCTX = 64,
    parameter PHYS_VREGS     = 128,
    parameter PHYS_W         = 7,
    parameter CORE_ID        = 0,
    parameter NUM_CORES      = 1,
    parameter HAS_SFU        = 0,          // Stage-1a ships without the SFU
    parameter FP_LAT         = 2,          // tnca_vector_fpu result delay, in cycles
    parameter MUL_LAT        = 3,          // tnca_mul64 result delay, in cycles
    parameter MEM_DW         = 64,
    parameter MEM_BYTES      = 8,
    parameter MEM_SHIFT      = 3,
    parameter LSU_BEATS      = 8,          // outstanding memory beats
    parameter LSU_TAG_W      = 3,
    parameter ACC_N          = 16,         // accumulator slots
    parameter ACC_W          = 4,
    parameter ACC_ROWS       = 16,
    parameter ACC_ROW_W      = 4,
    parameter BAR_N          = 32,         // mbarriers
    parameter BAR_W          = 5
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- launch --------------------------------------------------------------
    input  wire                     launch_valid,
    input  wire [63:0]              launch_pc,
    input  wire [NUM_RCTX-1:0]      launch_role_mask,
    input  wire [31:0]              launch_pid_x,
    input  wire [31:0]              launch_pid_y,
    input  wire [31:0]              launch_pid_z,
    input  wire [31:0]              launch_npid_x,
    input  wire [31:0]              launch_npid_y,
    input  wire [31:0]              launch_npid_z,
    input  wire [63:0]              launch_arg_base,
    output wire                     core_idle,

    // ---- fault reporting -----------------------------------------------------
    output reg                      fault_valid,
    output reg  [7:0]               fault_code,
    output reg  [63:0]              fault_pc,
    output reg  [RCTX_W-1:0]        fault_rctx,
    output reg  [23:0]              fault_detail,   // TRAP_CODE[23:0]: `trap` immediate

    // ---- instruction port (asynchronous read; see NOTE below) ----------------
    output wire [63:0]              imem_addr,
    input  wire [63:0]              imem_data,

    // ---- data port -----------------------------------------------------------
    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [63:0]              mem_req_addr,
    output wire                     mem_req_we,
    output wire [MEM_BYTES-1:0]     mem_req_be,
    output wire [MEM_DW-1:0]        mem_req_wdata,
    output wire [LSU_TAG_W-1:0]     mem_req_tag,
    output wire [1:0]               mem_req_hint,
    input  wire                     mem_rsp_valid,
    input  wire [LSU_TAG_W-1:0]     mem_rsp_tag,
    input  wire [MEM_DW-1:0]        mem_rsp_rdata,

    // ---- performance counters (arch overview sec 20.2) -----------------------
    output reg  [63:0]              perf_cycles,
    output reg  [63:0]              perf_issued,
    output reg  [63:0]              perf_stall_hazard,
    output reg  [63:0]              perf_stall_lsu,
    output reg  [63:0]              perf_stall_wb,
    output reg  [63:0]              perf_squash
);

    // NOTE ON THE INSTRUCTION PORT: Stage 1 models instruction fetch as a
    // single-cycle asynchronous read. Real silicon needs a synchronous SRAM
    // plus a fetch buffer, which adds a stage between F and D. That change is
    // local to the front end and is tracked as Stage-2 work; it does not affect
    // the ISA or the hazard logic.

    localparam VREG_IDX_W = 8;

    integer i;
    genvar  g;

    //==========================================================================
    // RCTX state
    //==========================================================================
    reg [63:0]          pc      [0:NUM_RCTX-1];
    reg [NUM_RCTX-1:0]  running;
    reg [NUM_RCTX-1:0]  faulted;
    reg [63:0]          instret [0:NUM_RCTX-1];
    reg [7:0]           f_code  [0:NUM_RCTX-1];
    reg [63:0]          f_pc    [0:NUM_RCTX-1];
    reg [23:0]          f_detail[0:NUM_RCTX-1];

    reg [31:0] pid_x, pid_y, pid_z, npid_x, npid_y, npid_z;
    reg [63:0] arg_base;
    reg [63:0] clock_ctr;

    //==========================================================================
    // Pipeline registers
    //==========================================================================
    // F -> D
    reg                 d_valid;
    reg [RCTX_W-1:0]    d_rctx;
    reg [63:0]          d_pc;
    reg [63:0]          d_insn;

    // decoded bundle carried D -> R -> X
    `define TNCA_BUNDLE(p) \
        reg                  p``_valid;      \
        reg [RCTX_W-1:0]     p``_rctx;       \
        reg [63:0]           p``_pc;         \
        reg [7:0]            p``_major;      \
        reg [7:0]            p``_func;       \
        reg [9:0]            p``_mod;        \
        reg [2:0]            p``_fu;         \
        reg [2:0]            p``_dstc;       \
        reg [7:0]            p``_dsti;       \
        reg                  p``_mask_dst;   \
        reg                  p``_sa_en;      \
        reg [5:0]            p``_sa_idx;     \
        reg                  p``_sb_en;      \
        reg [5:0]            p``_sb_idx;     \
        reg                  p``_sc_en;      \
        reg [5:0]            p``_sc_idx;     \
        reg                  p``_va_en;      \
        reg [7:0]            p``_va_idx;     \
        reg                  p``_vb_en;      \
        reg [7:0]            p``_vb_idx;     \
        reg                  p``_vc_en;      \
        reg [7:0]            p``_vc_idx;     \
        reg                  p``_ra_en;      \
        reg [7:0]            p``_ra_idx;     \
        reg                  p``_pa_en;      \
        reg [3:0]            p``_pa_idx;     \
        reg                  p``_pb_en;      \
        reg [3:0]            p``_pb_idx;     \
        reg                  p``_pm_en;      \
        reg [3:0]            p``_pm_idx;     \
        reg                  p``_pm_neg;     \
        reg [63:0]           p``_imm;        \
        reg                  p``_is_branch;  \
        reg                  p``_br_cond;    \
        reg                  p``_br_on_zero; \
        reg                  p``_br_ind;     \
        reg                  p``_br_link;    \
        reg [63:0]           p``_br_off;     \
        reg                  p``_is_exit;    \
        reg                  p``_is_trap;    \
        reg                  p``_is_yield;   \
        reg                  p``_mem_en;     \
        reg                  p``_mem_store;  \
        reg                  p``_mem_vector; \
        reg                  p``_mem_indexed;\
        reg [2:0]            p``_mem_size;   \
        reg                  p``_mem_signed; \
        reg [1:0]            p``_mem_hint;   \
        reg                  p``_illegal;

    `TNCA_BUNDLE(r)
    `TNCA_BUNDLE(x)

    //==========================================================================
    // F: RCTX selection and fetch
    //==========================================================================
    reg [RCTX_W-1:0] rr_ptr;
    reg [RCTX_W-1:0] f_sel;
    reg              f_hit;
    reg [RCTX_W-1:0] cand;

    // A context asleep on a barrier stops fetching entirely. Sec 13.4 asks for
    // exactly this rather than a polling loop: a spinning waiter would consume
    // issue slots and memory bandwidth that the producer it is waiting for
    // needs.
    reg [NUM_RCTX-1:0] blocked;
    wire [NUM_RCTX-1:0] fetchable = running & ~faulted & ~blocked;

    always @(*) begin
        f_sel = {RCTX_W{1'b0}};
        f_hit = 1'b0;
        for (i = NUM_RCTX-1; i >= 0; i = i - 1) begin
            cand = (rr_ptr + i[RCTX_W-1:0]) % NUM_RCTX;
            if (fetchable[cand]) begin
                f_sel = cand;
                f_hit = 1'b1;
            end
        end
    end

    assign imem_addr = pc[f_sel];

    //==========================================================================
    // D: decode
    //==========================================================================
    wire        dec_illegal;
    wire [2:0]  dec_fu;
    wire [7:0]  dec_major, dec_func;
    wire [9:0]  dec_mod;
    wire [2:0]  dec_dstc;
    wire [7:0]  dec_dsti;
    wire        dec_sa_en, dec_sb_en, dec_sc_en;
    wire [5:0]  dec_sa_idx, dec_sb_idx, dec_sc_idx;
    wire        dec_va_en, dec_vb_en, dec_vc_en;
    wire [7:0]  dec_va_idx, dec_vb_idx, dec_vc_idx;
    wire        dec_ra_en;
    wire [7:0]  dec_ra_idx;
    wire        dec_pa_en, dec_pb_en;
    wire [3:0]  dec_pa_idx, dec_pb_idx;
    wire        dec_pm_en, dec_pm_neg, dec_mask_dst;
    wire [3:0]  dec_pm_idx;
    wire [63:0] dec_imm;
    wire        dec_is_branch, dec_br_cond, dec_br_on_zero, dec_br_ind, dec_br_link;
    wire [63:0] dec_br_off;
    wire        dec_is_exit, dec_is_trap, dec_is_yield;
    wire        dec_mem_en, dec_mem_store, dec_mem_vector, dec_mem_indexed, dec_mem_signed;
    wire [2:0]  dec_mem_size;
    wire [1:0]  dec_mem_hint;

    tnca_decoder u_dec (
        .insn(d_insn),
        .o_illegal(dec_illegal), .o_fu(dec_fu),
        .o_major(dec_major), .o_func(dec_func), .o_mod(dec_mod),
        .o_dst_class(dec_dstc), .o_dst_idx(dec_dsti),
        .o_sa_en(dec_sa_en), .o_sa_idx(dec_sa_idx),
        .o_sb_en(dec_sb_en), .o_sb_idx(dec_sb_idx),
        .o_sc_en(dec_sc_en), .o_sc_idx(dec_sc_idx),
        .o_va_en(dec_va_en), .o_va_idx(dec_va_idx),
        .o_vb_en(dec_vb_en), .o_vb_idx(dec_vb_idx),
        .o_vc_en(dec_vc_en), .o_vc_idx(dec_vc_idx),
        .o_ra_en(dec_ra_en), .o_ra_idx(dec_ra_idx),
        .o_pa_en(dec_pa_en), .o_pa_idx(dec_pa_idx),
        .o_pb_en(dec_pb_en), .o_pb_idx(dec_pb_idx),
        .o_pm_en(dec_pm_en), .o_pm_idx(dec_pm_idx), .o_pm_neg(dec_pm_neg),
        .o_mask_dst(dec_mask_dst),
        .o_imm(dec_imm),
        .o_is_branch(dec_is_branch), .o_br_cond(dec_br_cond),
        .o_br_on_zero(dec_br_on_zero), .o_br_indirect(dec_br_ind),
        .o_br_link(dec_br_link), .o_br_off(dec_br_off),
        .o_is_exit(dec_is_exit), .o_is_trap(dec_is_trap), .o_is_yield(dec_is_yield),
        .o_mem_en(dec_mem_en), .o_mem_store(dec_mem_store),
        .o_mem_vector(dec_mem_vector), .o_mem_indexed(dec_mem_indexed),
        .o_mem_size(dec_mem_size), .o_mem_signed(dec_mem_signed),
        .o_mem_hint(dec_mem_hint)
    );

    // Implementation limits are checked here, not in the decoder: the decoder
    // implements the ISA, this core implements one configuration of it.
    // A vector name past VREGS_PER_RCTX, or an SFU op on a build without the
    // SFU, is ILLEGAL_INSTRUCTION -- never a silent substitution (ISA sec 11).
    wire vreg_oob = (dec_dstc == `TNCA_DST_V && dec_dsti >= VREGS_PER_RCTX)
                  | (dec_va_en && dec_va_idx >= VREGS_PER_RCTX)
                  | (dec_vb_en && dec_vb_idx >= VREGS_PER_RCTX)
                  | (dec_vc_en && dec_vc_idx >= VREGS_PER_RCTX);
    wire sfu_unsupported = (dec_fu == `TNCA_FU_SFU) && (HAS_SFU == 0);

    // ISA v0.1 defines no writable special register. Letting s.wrspecial retire
    // as a silent no-op would make a program that depends on it look correct
    // here and fail on an implementation that does define one, so it traps.
    wire wrspecial_unsupported = (dec_major == `TNCA_MAJ_SYS) &&
                                 (dec_func  == `TNCA_SYS_WRSPECIAL);

    wire d_illegal = dec_illegal | vreg_oob | sfu_unsupported | wrspecial_unsupported;

    //==========================================================================
    // Register files (read in X)
    //==========================================================================
    wire [63:0]          sa_data, sb_data, sc_data;
    wire [VLEN*ELEN-1:0] va_data, vb_data, vc_data;
    wire [VLEN-1:0]      pa_data, pb_data, pm_raw;

    // Writeback stage.
    //
    // The register file write port is driven by a REGISTER, not combinationally
    // from X. Driving it from X closes a combinational loop through the
    // write-first read bypass whenever an instruction reads and writes the same
    // register (s.movk does exactly that):
    //     ra_data -> ALU -> w_data -> ra_data
    // With a WB register in the path, a consumer that enters X one cycle after
    // its producer still gets the value for free from the same write-first
    // bypass, so the register costs latency only when a dependent instruction
    // is already stalled.
    reg                  nw_en;        // combinational: what to latch this cycle
    reg [RCTX_W-1:0]     nw_rctx;
    reg [2:0]            nw_class;
    reg [7:0]            nw_idx;
    reg [63:0]           nw_sdata;
    reg [VLEN*ELEN-1:0]  nw_vdata;
    reg [VLEN-1:0]       nw_lane_en;
    reg [VLEN-1:0]       nw_pdata;

    reg                  w_en;         // registered: drives the RF write port
    reg [RCTX_W-1:0]     w_rctx;
    reg [2:0]            w_class;
    reg [7:0]            w_idx;
    reg [63:0]           w_sdata;
    reg [VLEN*ELEN-1:0]  w_vdata;
    reg [VLEN-1:0]       w_lane_en;
    reg [VLEN-1:0]       w_pdata;

    tnca_scalar_rf #(.NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W)) u_srf (
        .clk(clk), .rst_n(rst_n),
        .ra_rctx(x_rctx), .ra_idx(x_sa_idx), .ra_data(sa_data),
        .rb_rctx(x_rctx), .rb_idx(x_sb_idx), .rb_data(sb_data),
        .rc_rctx(x_rctx), .rc_idx(x_sc_idx), .rc_data(sc_data),
        .w_en(w_en && (w_class == `TNCA_DST_S)),
        .w_rctx(w_rctx), .w_idx(w_idx[5:0]), .w_data(w_sdata)
    );

    tnca_vector_rf #(
        .PHYS_VREGS(PHYS_VREGS), .VREGS_PER_RCTX(VREGS_PER_RCTX),
        .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W), .PHYS_W(PHYS_W)
    ) u_vrf (
        .clk(clk), .rst_n(rst_n),
        .ra_rctx(x_rctx), .ra_idx(x_va_idx), .ra_data(va_data),
        .rb_rctx(x_rctx), .rb_idx(x_vb_idx), .rb_data(vb_data),
        .rc_rctx(x_rctx), .rc_idx(x_vc_idx), .rc_data(vc_data),
        .rd_rctx(mtx_rctx), .rd_idx(mtx_vrf_addr), .rd_data(mtx_vrf_data),
        .w_en(w_en && (w_class == `TNCA_DST_V)),
        .w_rctx(w_rctx), .w_idx(w_idx), .w_lane_en(w_lane_en), .w_data(w_vdata)
    );

    tnca_pred_rf #(.NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN)) u_prf (
        .clk(clk), .rst_n(rst_n),
        .ra_rctx(x_rctx), .ra_idx(x_pa_idx), .ra_data(pa_data),
        .rb_rctx(x_rctx), .rb_idx(x_pb_idx), .rb_data(pb_data),
        .rm_rctx(x_rctx), .rm_idx(x_pm_idx), .rm_data(pm_raw),
        .w_en(w_en && (w_class == `TNCA_DST_P)),
        .w_rctx(w_rctx), .w_idx(w_idx[3:0]), .w_data(w_pdata)
    );

    wire [VLEN-1:0] lane_mask = x_pm_en ? (x_pm_neg ? ~pm_raw : pm_raw)
                                        : {VLEN{1'b1}};

    //==========================================================================
    // Scoreboard
    //==========================================================================
    reg [NUM_RCTX*64-1:0]             sb_s;
    reg [NUM_RCTX*VREGS_PER_RCTX-1:0] sb_v;
    reg [NUM_RCTX*16-1:0]             sb_p;
    reg [ACC_N-1:0]                   sb_acc;

    // NOTE: these are macros, not Verilog functions. A continuous assignment
    // that calls a function is only re-evaluated when the function's arguments
    // change -- signals the function body reads from module scope (here sb_*
    // and wb_*) do not retrigger it. Written as functions, the hazard wires
    // latched at their first value and the pipeline deadlocked.
    `define SB_S_BUSY(rc, idx) ((idx) != 6'd0 && sb_s[(rc)*64 + (idx)])
    `define SB_V_BUSY(rc, idx) (sb_v[(rc)*VREGS_PER_RCTX + (idx)])
    `define SB_P_BUSY(rc, idx) ((idx) != 4'd0 && sb_p[(rc)*16 + (idx)])

    // Two reasons a busy register is not actually a hazard for an instruction
    // sitting in R this cycle:
    //   WB_HIT  the WB stage writes it now, so the file holds it next cycle;
    //   X_PROD  X produces it now, so it reaches WB next cycle and the file the
    //           cycle after -- exactly when this instruction, entering X next
    //           cycle, reads through the write-first bypass.
    // Together they make a back-to-back dependent pair issue with no bubble.
    `define WB_HIT_S(rc, idx) (w_en && (w_class == `TNCA_DST_S) && \
                               (w_rctx == (rc)) && (w_idx[5:0] == (idx)))
    `define WB_HIT_V(rc, idx) (w_en && (w_class == `TNCA_DST_V) && \
                               (w_rctx == (rc)) && (w_idx == (idx)))
    `define WB_HIT_P(rc, idx) (w_en && (w_class == `TNCA_DST_P) && \
                               (w_rctx == (rc)) && (w_idx[3:0] == (idx)))

    `define X_PROD_S(rc, idx) (x_has_dest && (x_dstc == `TNCA_DST_S) && \
                               (x_rctx == (rc)) && (x_dsti[5:0] == (idx)))
    `define X_PROD_V(rc, idx) (x_has_dest && (x_dstc == `TNCA_DST_V) && \
                               (x_rctx == (rc)) && (x_dsti == (idx)))
    `define X_PROD_P(rc, idx) (x_has_dest && (x_dstc == `TNCA_DST_P) && \
                               (x_rctx == (rc)) && (x_dsti[3:0] == (idx)))

    `define AVAIL_S(rc, idx) (`WB_HIT_S(rc, idx) || `X_PROD_S(rc, idx))
    `define AVAIL_V(rc, idx) (`WB_HIT_V(rc, idx) || `X_PROD_V(rc, idx))
    `define AVAIL_P(rc, idx) (`WB_HIT_P(rc, idx) || `X_PROD_P(rc, idx))

    wire hz_sa = r_sa_en && `SB_S_BUSY(r_rctx, r_sa_idx) && !`AVAIL_S(r_rctx, r_sa_idx);
    wire hz_sb = r_sb_en && `SB_S_BUSY(r_rctx, r_sb_idx) && !`AVAIL_S(r_rctx, r_sb_idx);
    wire hz_sc = r_sc_en && `SB_S_BUSY(r_rctx, r_sc_idx) && !`AVAIL_S(r_rctx, r_sc_idx);
    wire hz_va = r_va_en && `SB_V_BUSY(r_rctx, r_va_idx) && !`AVAIL_V(r_rctx, r_va_idx);
    wire hz_vb = r_vb_en && `SB_V_BUSY(r_rctx, r_vb_idx) && !`AVAIL_V(r_rctx, r_vb_idx);
    wire hz_vc = r_vc_en && `SB_V_BUSY(r_rctx, r_vc_idx) && !`AVAIL_V(r_rctx, r_vc_idx);
    wire hz_pa = r_pa_en && `SB_P_BUSY(r_rctx, r_pa_idx) && !`AVAIL_P(r_rctx, r_pa_idx);
    wire hz_pb = r_pb_en && `SB_P_BUSY(r_rctx, r_pb_idx) && !`AVAIL_P(r_rctx, r_pb_idx);
    wire hz_pm = r_pm_en && `SB_P_BUSY(r_rctx, r_pm_idx) && !`AVAIL_P(r_rctx, r_pm_idx);

    // WAW uses WB_HIT only, never X_PROD.
    //
    // The scoreboard is one bit per register, so it cannot represent two
    // in-flight writers. With the X_PROD bypass a second writer would claim the
    // bit one cycle before the first writer's release cleared it, leaving the
    // register marked free while the second write was still outstanding -- a
    // following reader then sampled the stale value. Waiting for WB_HIT puts
    // the claim in the same cycle as the release, and the claim is written
    // last, so the bit stays set.
    wire hz_d  = ((r_dstc == `TNCA_DST_S) && `SB_S_BUSY(r_rctx, r_dsti[5:0])
                                          && !`WB_HIT_S(r_rctx, r_dsti[5:0]))
               | ((r_dstc == `TNCA_DST_V) && `SB_V_BUSY(r_rctx, r_dsti)
                                          && !`WB_HIT_V(r_rctx, r_dsti))
               | ((r_dstc == `TNCA_DST_P) && `SB_P_BUSY(r_rctx, r_dsti[3:0])
                                          && !`WB_HIT_P(r_rctx, r_dsti[3:0]));

    // Accumulators are core-wide, not per-RCTX (ISA sec 1.2), so one bit per
    // slot. A command that writes one holds it until the engine reports done;
    // that is the only thing that makes "the accumulator may not be read before
    // the m.mma completes" (sec 9C.3) enforceable rather than advisory.
    wire r_acc_dst = (r_dstc == `TNCA_DST_A);
    wire [ACC_W-1:0] r_acc_d = r_dsti[ACC_W-1:0];
    wire r_acc_src = r_valid && (r_major == `TNCA_MAJ_ACC)
                             && (r_func  == `TNCA_A_VREAD);
    wire [ACC_W-1:0] r_acc_s = r_ra_idx[ACC_W-1:0];

    wire acc_free_d = !sb_acc[r_acc_d] || (mtx_done && (mtx_done_slot == r_acc_d));
    wire acc_free_s = !sb_acc[r_acc_s] || (mtx_done && (mtx_done_slot == r_acc_s));

    wire hz_acc = (r_valid && r_acc_dst && !acc_free_d)
                | (r_acc_src && !acc_free_s);

    wire hazard = r_valid && (hz_sa|hz_sb|hz_sc|hz_va|hz_vb|hz_vc|hz_pa|hz_pb|hz_pm|hz_d)
                | hz_acc;

    //==========================================================================
    // Execution units (all read their operands in X)
    //==========================================================================
    // s.movk reads its own destination; the decoder already pointed source port
    // A at field d for it, so port A is the first operand in every case.
    wire [63:0] alu_opa = sa_data;
    reg  [63:0] alu_opb;
    always @(*) begin
        case (x_func)
        `TNCA_S_ADDI, `TNCA_S_SUBI, `TNCA_S_MULI, `TNCA_S_ANDI, `TNCA_S_ORI,
        `TNCA_S_XORI, `TNCA_S_SHLI, `TNCA_S_LSHRI, `TNCA_S_ASHRI, `TNCA_S_CMPI,
        `TNCA_S_MOVZ, `TNCA_S_MOVK, `TNCA_S_MOVN: alu_opb = x_imm;
        default:                                  alu_opb = sb_data;
        endcase
    end

    // s.mul / s.mulh.* run on their own pipelined unit; see tnca_mul64.
    wire [63:0] mul_lo, mul_hi_s, mul_hi_u;
    tnca_mul64 u_mul (
        .clk(clk), .rst_n(rst_n),
        .a(alu_opa), .b(alu_opb),
        .lo(mul_lo), .hi_s(mul_hi_s), .hi_u(mul_hi_u)
    );

    wire [63:0] alu_result;
    tnca_scalar_alu u_salu (
        .func(x_func), .mod(x_mod),
        .opa(alu_opa), .opb(alu_opb), .opc(sc_data),
        .result(alu_result)
    );

    // Vector units take at most one scalar operand; which field it lives in is
    // opcode-determined and the two are never both used.
    wire [63:0] v_sop = x_sb_en ? sb_data : sa_data;

    wire [VLEN*ELEN-1:0] valu_v;
    wire [63:0]          valu_s;
    wire [VLEN-1:0]      valu_p;
    tnca_vector_alu #(.VLEN(VLEN), .ELEN(ELEN)) u_valu (
        .major(x_major), .func(x_func), .mod(x_mod), .mask(lane_mask),
        .va(va_data), .vb(vb_data), .sa(v_sop), .imm(x_imm),
        .pa(pa_data), .pb(pb_data),
        .vres(valu_v), .sres(valu_s), .pres(valu_p)
    );

    wire                 fpu_multicycle;
    wire                 fpu_busy, fpu_done;
    wire [VLEN*ELEN-1:0] fpu_v;
    wire [63:0]          fpu_s;
    wire [VLEN-1:0]      fpu_p;
    wire [63:0]          fpu_red_s;
    wire [VLEN*ELEN-1:0] fpu_red_v;

    wire x_is_fp = x_valid && ((x_major == `TNCA_MAJ_VFP) ||
                               (x_major == `TNCA_MAJ_CVT) ||
                               ((x_major == `TNCA_MAJ_RED) && (x_func >= `TNCA_RD_FADD)));

    tnca_vector_fpu #(.VLEN(VLEN), .ELEN(ELEN), .LW(LW), .FP_LAT(FP_LAT)) u_vfpu (
        .clk(clk), .rst_n(rst_n),
        .major(x_major), .func(x_func), .mod(x_mod), .mask(lane_mask),
        .va(va_data), .vb(vb_data), .vc(vc_data),
        .is_multicycle(fpu_multicycle),
        .start(x_is_fp && fpu_multicycle && !x_kill),
        .busy(fpu_busy), .done(fpu_done),
        .vres(fpu_v), .sres(fpu_s), .pres(fpu_p),
        .red_sres(fpu_red_s), .red_vres(fpu_red_v)
    );

    // Integer reductions run on the integer tree; FP reductions run on the FMA
    // array. Both use major 0x15, so the destination is chosen by func.
    wire red_is_fp = (x_major == `TNCA_MAJ_RED) && (x_func >= `TNCA_RD_FADD);
    wire use_fpu   = (x_major == `TNCA_MAJ_VFP) || (x_major == `TNCA_MAJ_CVT) || red_is_fp;

    // Writeback-path classification. `is_mul` and `is_fpnorm` are scheduled
    // with a delay; reductions and memory ops complete out of band.
    `define TNCA_IS_MUL(maj, fn) \
        (((maj) == `TNCA_MAJ_SALU) && (((fn) == `TNCA_S_MUL)    || \
                                       ((fn) == `TNCA_S_MULI)   || \
                                       ((fn) == `TNCA_S_MULH_S) || \
                                       ((fn) == `TNCA_S_MULH_U)))
    `define TNCA_IS_FPNORM(maj) (((maj) == `TNCA_MAJ_VFP) || ((maj) == `TNCA_MAJ_CVT))
    `define TNCA_IS_FPRED(maj, fn) (((maj) == `TNCA_MAJ_RED) && ((fn) >= `TNCA_RD_FADD))

    wire x_is_mul    = x_valid && `TNCA_IS_MUL(x_major, x_func);
    wire x_is_fpnorm = x_valid && `TNCA_IS_FPNORM(x_major);
    wire x_is_fpred  = x_valid && `TNCA_IS_FPRED(x_major, x_func);

    //==========================================================================
    // Mbarrier and matrix engine
    //
    // Both are asynchronous: a command leaves the execute stage and completes
    // later, signalling the barrier it was armed against. That is the whole
    // reason the barrier exists -- sec 13.1's three completions are only
    // distinguishable if the thing that finishes can say so independently of
    // the instruction that started it.
    //==========================================================================
    wire x_is_bar = x_valid && !x_kill && (x_fu == `TNCA_FU_BAR);
    wire x_is_mtx = x_valid && !x_kill && (x_fu == `TNCA_FU_MATRIX);
    wire x_is_avread = x_is_mtx && (x_major == `TNCA_MAJ_ACC)
                                && (x_func == `TNCA_A_VREAD);

    wire [BAR_W-1:0] tok_bar = sa_data[BAR_W-1:0];

    wire        bar_arm_token_v;
    wire [63:0] bar_arm_token;
    wire        bar_test_done, bar_test_error;
    wire [NUM_RCTX-1:0] bar_wake;
    wire        bar_fault;
    wire [BAR_W-1:0] bar_fault_bar;

    wire        mtx_busy, mtx_done, mtx_done_bar_en;
    wire [ACC_W-1:0] mtx_done_slot;
    wire [BAR_W-1:0] mtx_done_bar;
    wire [7:0]  mtx_vrf_addr;
    wire [VLEN*ELEN-1:0] mtx_vrf_data;
    wire [VLEN*ELEN-1:0] acc_rd_data;

    // The engine serves one command at a time, so the RCTX that owns it is the
    // one whose command is running; captured at issue.
    reg [RCTX_W-1:0] mtx_rctx;

    wire mtx_cmd_valid = x_is_mtx && !x_is_avread;

    wire [1:0] mtx_cmd_op =
        ((x_major == `TNCA_MAJ_MATRIX) && (x_func == `TNCA_M_ZERO)) ? 2'd0 :
        ((x_major == `TNCA_MAJ_ACC)    && (x_func == `TNCA_A_ZERO)) ? 2'd0 :
        ((x_major == `TNCA_MAJ_MATRIX) && (x_func == `TNCA_M_MMA))  ? 2'd1 : 2'd2;

    wire [1:0] mtx_rowop =
        (x_func == `TNCA_A_VWRITE) ? 2'd0 :
        (x_func == `TNCA_A_MULROW) ? 2'd1 : 2'd2;

    wire [31:0] mtx_scale = (x_func == `TNCA_A_MULROW) ? sb_data[31:0]
                                                       : sc_data[31:0];

    // An m.mma signals one operation completion against the barrier named by
    // its token (ISA sec 9C).
    wire mtx_bar_en = (x_major == `TNCA_MAJ_MATRIX) && (x_func == `TNCA_M_MMA);
    wire [BAR_W-1:0] mtx_bar = sc_data[BAR_W-1:0];

    tnca_matrix #(
        .VLEN(VLEN), .ELEN(ELEN), .ACC_N(ACC_N), .ACC_W(ACC_W),
        .ROWS(ACC_ROWS), .ROW_W(ACC_ROW_W), .BAR_W(BAR_W), .FMA_LAT(FP_LAT)
    ) u_mtx (
        .clk(clk), .rst_n(rst_n), .flush(launch_valid),
        .cmd_valid(mtx_cmd_valid), .cmd_op(mtx_cmd_op),
        .cmd_slot(x_dsti[ACC_W-1:0]),
        .cmd_va(x_va_idx), .cmd_vb(x_vb_idx),
        .cmd_k(x_mod[1:0]), .cmd_dt(x_mod[2]),
        .cmd_bar_en(mtx_bar_en), .cmd_bar(mtx_bar),
        .cmd_rowop(mtx_rowop), .cmd_row(sa_data[ACC_ROW_W-1:0]),
        .cmd_vdata(vb_data), .cmd_scale(mtx_scale), .cmd_mask(lane_mask),
        .busy(mtx_busy),
        .vrf_addr(mtx_vrf_addr), .vrf_data(mtx_vrf_data),
        .ard_slot(x_ra_idx[ACC_W-1:0]), .ard_row(sb_data[ACC_ROW_W-1:0]),
        .ard_data(acc_rd_data),
        .done(mtx_done), .done_slot(mtx_done_slot),
        .done_bar_en(mtx_done_bar_en), .done_bar(mtx_done_bar)
    );

    tnca_barrier #(
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .BAR_N(BAR_N), .BAR_W(BAR_W)
    ) u_bar (
        .clk(clk), .rst_n(rst_n), .flush(launch_valid),
        .init_en(x_is_bar && (x_func == `TNCA_MB_INIT)),
        .init_bar(x_dsti[BAR_W-1:0]),
        .arm_en(x_is_bar && (x_func == `TNCA_MB_ARM)),
        .arm_bar(x_ra_idx[BAR_W-1:0]),
        .arm_arrivals(x_imm[15:8]), .arm_operations(x_imm[7:0]),
        .arm_token(bar_arm_token),
        .arrive_en(x_is_bar && (x_func == `TNCA_MB_ARRIVE)),
        .arrive_bar(x_ra_idx[BAR_W-1:0]),
        .opdone_en(mtx_done && mtx_done_bar_en),
        .opdone_bar(mtx_done_bar),
        .inval_en(x_is_bar && (x_func == `TNCA_MB_INVAL)),
        .inval_bar(x_ra_idx[BAR_W-1:0]),
        .test_token(sa_data), .test_done(bar_test_done),
        .test_error(bar_test_error),
        .sleep_en(bar_sleep), .sleep_rctx(x_rctx), .sleep_token(sa_data),
        .wake(bar_wake),
        .fault(bar_fault), .fault_bar(bar_fault_bar)
    );

    // mb.wait sleeps the RCTX rather than spinning (sec 13.4): the PC is left
    // pointing at the instruction, the context stops fetching, and the barrier
    // wakes it. Re-executing re-reads the token from the same scalar register,
    // so nothing has to be remembered across the sleep.
    wire x_is_mbwait = x_is_bar && (x_func == `TNCA_MB_WAIT);
    wire bar_sleep   = x_is_mbwait && !bar_test_done;

    assign bar_arm_token_v = 1'b0;   // token is combinational; no valid needed

    //==========================================================================
    // LSU
    //==========================================================================
    wire                 lsu_req_ready;
    wire                 lsu_cpl_valid;
    reg                  lsu_cpl_ack;
    wire [RCTX_W-1:0]    lsu_cpl_rctx;
    wire [2:0]           lsu_cpl_dstc;
    wire [7:0]           lsu_cpl_dsti;
    wire [63:0]          lsu_cpl_sdata;
    wire [VLEN*ELEN-1:0] lsu_cpl_vdata;
    wire [VLEN-1:0]      lsu_cpl_lane_en;

    wire x_mem_go = x_valid && x_mem_en && !x_kill;

    tnca_lsu #(
        .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W),
        .MEM_DW(MEM_DW), .MEM_BYTES(MEM_BYTES), .MEM_SHIFT(MEM_SHIFT),
        .NUM_BEATS(LSU_BEATS), .BEAT_W(LSU_TAG_W)
    ) u_lsu (
        .clk(clk), .rst_n(rst_n),
        .req_ready(lsu_req_ready),
        .req_valid(x_mem_go),
        .req_store(x_mem_store), .req_vector(x_mem_vector),
        .req_indexed(x_mem_indexed), .req_size(x_mem_size),
        .req_signed(x_mem_signed), .req_hint(x_mem_hint),
        .req_base(sa_data), .req_offset(x_imm),
        .req_index(vb_data),
        .req_sdata(sb_data),
        .req_vdata(x_mem_indexed ? vc_data : vb_data),
        .req_mask(lane_mask),
        .req_rctx(x_rctx), .req_dst_class(x_dstc), .req_dst_idx(x_dsti),
        .cpl_valid(lsu_cpl_valid), .cpl_ack(lsu_cpl_ack),
        .cpl_rctx(lsu_cpl_rctx), .cpl_dst_class(lsu_cpl_dstc),
        .cpl_dst_idx(lsu_cpl_dsti), .cpl_sdata(lsu_cpl_sdata),
        .cpl_vdata(lsu_cpl_vdata), .cpl_lane_en(lsu_cpl_lane_en),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_we(mem_req_we),
        .mem_req_be(mem_req_be), .mem_req_wdata(mem_req_wdata),
        .mem_req_tag(mem_req_tag), .mem_req_hint(mem_req_hint),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_rdata(mem_rsp_rdata)
    );

    //==========================================================================
    // Scheduled-writeback bookkeeping
    //
    // wb_rsv[k] means "a scheduled writeback is already booked for k cycles
    // from now". An instruction in R will be in X next cycle and will produce
    // its result d cycles after that, so it books slot (d+1) in this cycle's
    // frame; after the shift that bit sits at index d, and at index 0 on the
    // cycle the result actually lands.
    //==========================================================================
    localparam WB_DEPTH = 8;

    reg [WB_DEPTH-1:0] wb_rsv;

    // Tag pipelines carry the destination alongside each unit's data pipeline.
    reg                fp_v    [0:FP_LAT-1];
    reg [RCTX_W-1:0]   fp_rctx [0:FP_LAT-1];
    reg [2:0]          fp_dstc [0:FP_LAT-1];
    reg [7:0]          fp_dsti [0:FP_LAT-1];
    reg [VLEN-1:0]     fp_len  [0:FP_LAT-1];

    reg                mu_v    [0:MUL_LAT-1];
    reg [RCTX_W-1:0]   mu_rctx [0:MUL_LAT-1];
    reg [7:0]          mu_dsti [0:MUL_LAT-1];
    reg [7:0]          mu_func [0:MUL_LAT-1];

    wire fp_wb  = fp_v[FP_LAT-1];
    wire mul_wb = mu_v[MUL_LAT-1];

    reg fp_inflight;
    integer fi;
    always @(*) begin
        fp_inflight = 1'b0;
        for (fi = 0; fi < FP_LAT; fi = fi + 1)
            if (fp_v[fi]) fp_inflight = 1'b1;
    end

    reg [63:0] mul_sel;
    always @(*) begin
        case (mu_func[MUL_LAT-1])
        `TNCA_S_MULH_S: mul_sel = mul_hi_s;
        `TNCA_S_MULH_U: mul_sel = mul_hi_u;
        default:        mul_sel = mul_lo;
        endcase
    end

    //==========================================================================
    // Deferred FP-reduction completion (1 deep; only one reduce can be in flight)
    //==========================================================================
    reg                  red_pend;
    reg [RCTX_W-1:0]     red_rctx;
    reg [2:0]            red_dstc;
    reg [7:0]            red_dsti;
    reg [63:0]           red_sdata;
    reg [VLEN*ELEN-1:0]  red_vdata;

    //==========================================================================
    // Special registers
    //==========================================================================
    reg [63:0] spec_rd;
    always @(*) begin
        case (x_mod)
        `TNCA_SPEC_PID_X:     spec_rd = {32'd0, pid_x};
        `TNCA_SPEC_PID_Y:     spec_rd = {32'd0, pid_y};
        `TNCA_SPEC_PID_Z:     spec_rd = {32'd0, pid_z};
        `TNCA_SPEC_NPID_X:    spec_rd = {32'd0, npid_x};
        `TNCA_SPEC_NPID_Y:    spec_rd = {32'd0, npid_y};
        `TNCA_SPEC_NPID_Z:    spec_rd = {32'd0, npid_z};
        `TNCA_SPEC_ROLE_ID:   spec_rd = {{(64-RCTX_W){1'b0}}, x_rctx};
        `TNCA_SPEC_NUM_ROLES: spec_rd = NUM_RCTX;
        `TNCA_SPEC_CORE_ID:   spec_rd = CORE_ID;
        `TNCA_SPEC_NUM_CORES: spec_rd = NUM_CORES;
        `TNCA_SPEC_NUM_LANES: spec_rd = VLEN;
        `TNCA_SPEC_NUM_VREGS: spec_rd = VREGS_PER_RCTX;
        `TNCA_SPEC_MAX_RANK:  spec_rd = 64'd2;    // see the transfer engine
        `TNCA_SPEC_NUM_ACC:   spec_rd = ACC_N;
        `TNCA_SPEC_CLOCK:     spec_rd = clock_ctr;
        `TNCA_SPEC_INSTRET:   spec_rd = instret[x_rctx];
        `TNCA_SPEC_ARG_BASE:  spec_rd = arg_base;
        `TNCA_SPEC_ISA_VER:   spec_rd = `TNCA_ISA_VERSION;
        `TNCA_SPEC_CAP0:      spec_rd = (64'd1 << `TNCA_CAP_VECTOR_FP32)
                                      | (HAS_SFU ? (64'd1 << `TNCA_CAP_SFU) : 64'd0)
                                      | (64'd1 << `TNCA_CAP_GATHER)
                                      | (64'd1 << `TNCA_CAP_MATRIX_F16)
                                      | (64'd1 << `TNCA_CAP_MATRIX_BF16)
                                      | (64'd1 << `TNCA_CAP_MBARRIER);
        `TNCA_SPEC_TRAP_PC:   spec_rd = f_pc[x_rctx];
        `TNCA_SPEC_TRAP_CODE: spec_rd = {32'd0, f_detail[x_rctx], f_code[x_rctx]};
        default:              spec_rd = 64'd0;
        endcase
    end

    //==========================================================================
    // Branch resolution
    //==========================================================================
    wire br_taken = x_br_ind ? 1'b1
                  : x_br_cond ? (x_br_on_zero ? (sa_data == 64'd0) : (sa_data != 64'd0))
                  : 1'b1;

    wire [63:0] br_target = x_br_ind ? (sa_data & ~64'd7) : (x_pc + x_br_off);

    wire x_redirect = x_valid && !x_kill &&
                      ((x_is_branch && br_taken) || x_is_exit || x_is_trap ||
                       x_illegal || bar_sleep);

    //==========================================================================
    // X result mux
    //==========================================================================
    reg [63:0]          x_sres;
    reg [VLEN*ELEN-1:0] x_vres;
    reg [VLEN-1:0]      x_pres;

    always @(*) begin
        x_vres = x_is_avread ? acc_rd_data : (use_fpu ? fpu_v : valu_v);
        x_pres = use_fpu ? fpu_p : valu_p;
        if (x_is_bar && (x_func == `TNCA_MB_ARM))  x_sres = bar_arm_token;
        else if (x_is_bar && (x_func == `TNCA_MB_TEST))
                                                 x_sres = {63'd0, bar_test_done};
        else if (x_br_link)                      x_sres = x_pc + 64'd8;
        else if (x_major == `TNCA_MAJ_SYS)       x_sres = spec_rd;
        else if (x_major == `TNCA_MAJ_SALU)      x_sres = alu_result;
        else if (use_fpu)                        x_sres = fpu_s;
        else                                     x_sres = valu_s;
    end

    // Only these three classes go through the register writeback path.
    // Accumulators, barriers and resource handles are written by their own
    // units and never contend for it.
    wire x_wb_reg = (x_dstc == `TNCA_DST_S) || (x_dstc == `TNCA_DST_V)
                                            || (x_dstc == `TNCA_DST_P);
    wire x_defers_wb = x_mem_en || x_is_fpred || x_is_fpnorm || x_is_mul;
    wire x_has_dest  = x_valid && !x_kill && x_wb_reg && !x_defers_wb;

    //==========================================================================
    // Writeback arbitration
    //==========================================================================
    // Exactly one scheduled result can land per cycle -- the reservation
    // register guarantees it -- so these three are mutually exclusive.
    wire sched_wb  = x_has_dest | fp_wb | mul_wb;

    wire cpl_req   = lsu_cpl_valid | red_pend;
    wire cpl_grant = cpl_req && !sched_wb;
    wire wb_steal  = cpl_req &&  sched_wb;   // drain the schedule, then retry

    always @(*) begin
        nw_en      = 1'b0;
        nw_rctx    = x_rctx;
        nw_class   = `TNCA_DST_NONE;
        nw_idx     = 8'd0;
        nw_sdata   = 64'd0;
        nw_vdata   = {(VLEN*ELEN){1'b0}};
        nw_pdata   = {VLEN{1'b0}};
        nw_lane_en = {VLEN{1'b1}};
        lsu_cpl_ack = 1'b0;

        if (cpl_grant && lsu_cpl_valid) begin
            lsu_cpl_ack = 1'b1;
            nw_en       = (lsu_cpl_dstc != `TNCA_DST_NONE);
            nw_rctx     = lsu_cpl_rctx;
            nw_class    = lsu_cpl_dstc;
            nw_idx      = lsu_cpl_dsti;
            nw_sdata    = lsu_cpl_sdata;
            nw_vdata    = lsu_cpl_vdata;
            nw_lane_en  = lsu_cpl_lane_en;
        end else if (cpl_grant && red_pend) begin
            nw_en      = 1'b1;
            nw_rctx    = red_rctx;
            nw_class   = red_dstc;
            nw_idx     = red_dsti;
            nw_sdata   = red_sdata;
            nw_vdata   = red_vdata;
            nw_lane_en = {VLEN{1'b1}};
        end else if (fp_wb) begin
            nw_en      = 1'b1;
            nw_rctx    = fp_rctx[FP_LAT-1];
            nw_class   = fp_dstc[FP_LAT-1];
            nw_idx     = fp_dsti[FP_LAT-1];
            nw_vdata   = fpu_v;
            nw_pdata   = fpu_p;
            nw_lane_en = fp_len[FP_LAT-1];
        end else if (mul_wb) begin
            nw_en      = 1'b1;
            nw_rctx    = mu_rctx[MUL_LAT-1];
            nw_class   = `TNCA_DST_S;
            nw_idx     = mu_dsti[MUL_LAT-1];
            nw_sdata   = mul_sel;
        end else if (x_has_dest) begin
            nw_en      = 1'b1;
            nw_rctx    = x_rctx;
            nw_class   = x_dstc;
            nw_idx     = x_dsti;
            nw_sdata   = x_sres;
            nw_vdata   = x_vres;
            nw_pdata   = x_pres;
            nw_lane_en = x_mask_dst ? lane_mask : {VLEN{1'b1}};
        end
    end

    //==========================================================================
    // Stall and squash
    //==========================================================================
    wire r_needs_lsu = r_valid && r_mem_en;

    wire r_is_mul    = r_valid && `TNCA_IS_MUL(r_major, r_func);
    wire r_is_fpnorm = r_valid && `TNCA_IS_FPNORM(r_major);
    wire r_is_fpred  = r_valid && `TNCA_IS_FPRED(r_major, r_func);

    // A unit is claimed from X, one cycle after the check in R. Asking "is the
    // unit free now?" is therefore the wrong question -- it must be free *next*
    // cycle, which it is not if the instruction currently in X is taking it.
    // Without the second term the LSU silently dropped a request whose
    // predecessor had been accepted in the very cycle R looked at req_ready,
    // leaving the load's destination marked busy forever.
    wire lsu_free_next = lsu_req_ready && !x_mem_go;

    // The reduction sequencer hijacks the FMA array's operands, so it may only
    // start when the array is empty -- otherwise it corrupts a normal FP
    // operation still in flight. Conversely a normal FP op may not issue into a
    // running reduction.
    wire red_busy_next = fpu_busy || red_pend || fpu_done || x_is_fpred;

    // Same "free NEXT cycle" question as the LSU: the engine is claimed from X,
    // one cycle after R looks.
    wire r_needs_mtx = r_valid && (r_fu == `TNCA_FU_MATRIX);
    wire mtx_free_next = !mtx_busy && !mtx_cmd_valid;
    wire stall_mtx = r_needs_mtx && !mtx_free_next;

    wire stall_lsu = r_needs_lsu && !lsu_free_next;
    wire stall_fpu = (r_is_fpnorm && red_busy_next)
                   | (r_is_fpred  && (red_busy_next || fp_inflight || x_is_fpnorm));

    // ---- writeback reservation ------------------------------------------------
    // Delay from "in X" to "result available", per unit.
    wire [3:0] r_wb_delay = r_is_mul    ? MUL_LAT[3:0] :
                            r_is_fpnorm ? FP_LAT[3:0]  : 4'd0;

    // Memory ops and FP reductions complete out of band and book nothing.
    wire r_wb_reg = (r_dstc == `TNCA_DST_S) || (r_dstc == `TNCA_DST_V)
                                             || (r_dstc == `TNCA_DST_P);
    wire r_unscheduled = r_mem_en | r_is_fpred;
    wire r_books_wb    = r_valid && r_wb_reg && !r_unscheduled;

    wire stall_rsv = r_books_wb && wb_rsv[r_wb_delay + 4'd1];

    wire stall_front = hazard | stall_lsu | stall_fpu | stall_mtx | stall_rsv | wb_steal;

    // A redirect kills only the younger entries of the same RCTX.
    wire             squash_en   = x_redirect;
    wire [RCTX_W-1:0] squash_rctx = x_rctx;
    wire kill_f = squash_en && f_hit  && (f_sel  == squash_rctx);
    wire kill_d = squash_en && d_valid && (d_rctx == squash_rctx);
    wire kill_r = squash_en && r_valid && (r_rctx == squash_rctx);

    // x_kill is never set in this design (X is the oldest in-flight entry), but
    // it is kept as an explicit signal so a future second execute stage has a
    // defined place to hook into.
    wire x_kill = 1'b0;

    //==========================================================================
    // Sequential
    //==========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            d_valid    <= 1'b0;
            r_valid    <= 1'b0;
            x_valid    <= 1'b0;
            running    <= {NUM_RCTX{1'b0}};
            faulted    <= {NUM_RCTX{1'b0}};
            rr_ptr     <= {RCTX_W{1'b0}};
            sb_s       <= {(NUM_RCTX*64){1'b0}};
            sb_v       <= {(NUM_RCTX*VREGS_PER_RCTX){1'b0}};
            sb_p       <= {(NUM_RCTX*16){1'b0}};
            sb_acc     <= {ACC_N{1'b0}};
            blocked    <= {NUM_RCTX{1'b0}};
            mtx_rctx   <= {RCTX_W{1'b0}};
            red_pend   <= 1'b0;
            w_en       <= 1'b0;
            wb_rsv     <= {WB_DEPTH{1'b0}};
            for (i = 0; i < FP_LAT; i = i + 1)  fp_v[i] <= 1'b0;
            for (i = 0; i < MUL_LAT; i = i + 1) mu_v[i] <= 1'b0;
            clock_ctr  <= 64'd0;
            fault_valid<= 1'b0;
            fault_code <= 8'd0;
            fault_pc   <= 64'd0;
            fault_rctx <= {RCTX_W{1'b0}};
            fault_detail<= 24'd0;
            perf_cycles       <= 64'd0;
            perf_issued       <= 64'd0;
            perf_stall_hazard <= 64'd0;
            perf_stall_lsu    <= 64'd0;
            perf_stall_wb     <= 64'd0;
            perf_squash       <= 64'd0;
            for (i = 0; i < NUM_RCTX; i = i + 1) begin
                pc[i]      <= 64'd0;
                instret[i] <= 64'd0;
                f_code[i]  <= 8'd0;
                f_pc[i]    <= 64'd0;
                f_detail[i]<= 24'd0;
            end
        end else begin
            clock_ctr <= clock_ctr + 64'd1;

            //------------------------------------------------------------------
            // Scheduled-writeback pipelines. These advance every cycle,
            // independently of stall_front: they carry work that has already
            // been issued, and throttling must never hold up completion
            // (arch overview sec 2.4).
            //------------------------------------------------------------------
            wb_rsv <= {1'b0, wb_rsv[WB_DEPTH-1:1]}
                    | (( !stall_front && r_valid && !kill_r && r_books_wb )
                       ? ({{(WB_DEPTH-1){1'b0}}, 1'b1} << r_wb_delay)
                       : {WB_DEPTH{1'b0}});

            for (i = FP_LAT-1; i > 0; i = i - 1) begin
                fp_v[i]    <= fp_v[i-1];
                fp_rctx[i] <= fp_rctx[i-1];
                fp_dstc[i] <= fp_dstc[i-1];
                fp_dsti[i] <= fp_dsti[i-1];
                fp_len[i]  <= fp_len[i-1];
            end
            fp_v[0]    <= x_is_fpnorm && !x_kill && (x_dstc != `TNCA_DST_NONE);
            fp_rctx[0] <= x_rctx;
            fp_dstc[0] <= x_dstc;
            fp_dsti[0] <= x_dsti;
            fp_len[0]  <= x_mask_dst ? lane_mask : {VLEN{1'b1}};

            for (i = MUL_LAT-1; i > 0; i = i - 1) begin
                mu_v[i]    <= mu_v[i-1];
                mu_rctx[i] <= mu_rctx[i-1];
                mu_dsti[i] <= mu_dsti[i-1];
                mu_func[i] <= mu_func[i-1];
            end
            mu_v[0]    <= x_is_mul && !x_kill && (x_dstc != `TNCA_DST_NONE);
            mu_rctx[0] <= x_rctx;
            mu_dsti[0] <= x_dsti;
            mu_func[0] <= x_func;

            // latch this cycle's writeback decision; it drives the RF next cycle
            w_en      <= nw_en;
            w_rctx    <= nw_rctx;
            w_class   <= nw_class;
            w_idx     <= nw_idx;
            w_sdata   <= nw_sdata;
            w_vdata   <= nw_vdata;
            w_pdata   <= nw_pdata;
            w_lane_en <= nw_lane_en;

            if (|running) perf_cycles <= perf_cycles + 64'd1;
            if (hazard)    perf_stall_hazard <= perf_stall_hazard + 64'd1;
            if (stall_lsu) perf_stall_lsu    <= perf_stall_lsu + 64'd1;
            if (wb_steal)  perf_stall_wb     <= perf_stall_wb + 64'd1;
            if (squash_en) perf_squash       <= perf_squash + 64'd1;

            //------------------------------------------------------------------
            // launch
            //------------------------------------------------------------------
            if (launch_valid) begin
                for (i = 0; i < NUM_RCTX; i = i + 1) begin
                    if (launch_role_mask[i]) pc[i] <= launch_pc;
                    instret[i] <= 64'd0;
                end
                running  <= launch_role_mask;
                faulted  <= {NUM_RCTX{1'b0}};
                pid_x    <= launch_pid_x;
                pid_y    <= launch_pid_y;
                pid_z    <= launch_pid_z;
                npid_x   <= launch_npid_x;
                npid_y   <= launch_npid_y;
                npid_z   <= launch_npid_z;
                arg_base <= launch_arg_base;
                d_valid  <= 1'b0;
                r_valid  <= 1'b0;
                x_valid  <= 1'b0;
                w_en     <= 1'b0;
                wb_rsv   <= {WB_DEPTH{1'b0}};
                for (i = 0; i < FP_LAT; i = i + 1)  fp_v[i] <= 1'b0;
                for (i = 0; i < MUL_LAT; i = i + 1) mu_v[i] <= 1'b0;
                sb_s     <= {(NUM_RCTX*64){1'b0}};
                sb_v     <= {(NUM_RCTX*VREGS_PER_RCTX){1'b0}};
                sb_p     <= {(NUM_RCTX*16){1'b0}};
                sb_acc   <= {ACC_N{1'b0}};
                blocked  <= {NUM_RCTX{1'b0}};
                fault_valid <= 1'b0;
            end else begin

            //------------------------------------------------------------------
            // F -> D
            //------------------------------------------------------------------
            if (!stall_front) begin
                d_valid <= f_hit && !kill_f;
                d_rctx  <= f_sel;
                d_pc    <= pc[f_sel];
                d_insn  <= imem_data;
                if (f_hit) begin
                    rr_ptr <= (f_sel + 1) % NUM_RCTX;
                    if (!kill_f) pc[f_sel] <= pc[f_sel] + 64'd8;
                end
            end else if (kill_f) begin
                // held entry belonged to the squashed context
                d_valid <= 1'b0;
            end

            //------------------------------------------------------------------
            // D -> R
            //------------------------------------------------------------------
            if (!stall_front) begin
                r_valid      <= d_valid && !kill_d;
                r_rctx       <= d_rctx;
                r_pc         <= d_pc;
                r_major      <= dec_major;
                r_func       <= dec_func;
                r_mod        <= dec_mod;
                r_fu         <= dec_fu;
                r_dstc       <= d_illegal ? `TNCA_DST_NONE : dec_dstc;
                r_dsti       <= dec_dsti;
                r_mask_dst   <= dec_mask_dst;
                r_sa_en      <= dec_sa_en && !d_illegal;  r_sa_idx <= dec_sa_idx;
                r_sb_en      <= dec_sb_en && !d_illegal;  r_sb_idx <= dec_sb_idx;
                r_sc_en      <= dec_sc_en && !d_illegal;  r_sc_idx <= dec_sc_idx;
                r_va_en      <= dec_va_en && !d_illegal;  r_va_idx <= dec_va_idx;
                r_vb_en      <= dec_vb_en && !d_illegal;  r_vb_idx <= dec_vb_idx;
                r_vc_en      <= dec_vc_en && !d_illegal;  r_vc_idx <= dec_vc_idx;
                r_ra_en      <= dec_ra_en && !d_illegal;  r_ra_idx <= dec_ra_idx;
                r_pa_en      <= dec_pa_en && !d_illegal;  r_pa_idx <= dec_pa_idx;
                r_pb_en      <= dec_pb_en && !d_illegal;  r_pb_idx <= dec_pb_idx;
                r_pm_en      <= dec_pm_en && !d_illegal;  r_pm_idx <= dec_pm_idx;
                r_pm_neg     <= dec_pm_neg;
                r_imm        <= dec_imm;
                r_is_branch  <= dec_is_branch  && !d_illegal;
                r_br_cond    <= dec_br_cond;
                r_br_on_zero <= dec_br_on_zero;
                r_br_ind     <= dec_br_ind;
                r_br_link    <= dec_br_link    && !d_illegal;
                r_br_off     <= dec_br_off;
                r_is_exit    <= dec_is_exit    && !d_illegal;
                r_is_trap    <= dec_is_trap    && !d_illegal;
                r_is_yield   <= dec_is_yield;
                r_mem_en     <= dec_mem_en     && !d_illegal;
                r_mem_store  <= dec_mem_store;
                r_mem_vector <= dec_mem_vector;
                r_mem_indexed<= dec_mem_indexed;
                r_mem_size   <= dec_mem_size;
                r_mem_signed <= dec_mem_signed;
                r_mem_hint   <= dec_mem_hint;
                r_illegal    <= d_illegal;
            end else if (kill_r) begin
                r_valid <= 1'b0;
            end

            //------------------------------------------------------------------
            // R -> X
            //------------------------------------------------------------------
            if (!stall_front) begin
                x_valid      <= r_valid && !kill_r;
                x_rctx       <= r_rctx;
                x_pc         <= r_pc;
                x_major      <= r_major;
                x_func       <= r_func;
                x_mod        <= r_mod;
                x_fu         <= r_fu;
                x_dstc       <= r_dstc;
                x_dsti       <= r_dsti;
                x_mask_dst   <= r_mask_dst;
                x_sa_en      <= r_sa_en;  x_sa_idx <= r_sa_idx;
                x_sb_en      <= r_sb_en;  x_sb_idx <= r_sb_idx;
                x_sc_en      <= r_sc_en;  x_sc_idx <= r_sc_idx;
                x_va_en      <= r_va_en;  x_va_idx <= r_va_idx;
                x_vb_en      <= r_vb_en;  x_vb_idx <= r_vb_idx;
                x_vc_en      <= r_vc_en;  x_vc_idx <= r_vc_idx;
                x_ra_en      <= r_ra_en;  x_ra_idx <= r_ra_idx;
                x_pa_en      <= r_pa_en;  x_pa_idx <= r_pa_idx;
                x_pb_en      <= r_pb_en;  x_pb_idx <= r_pb_idx;
                x_pm_en      <= r_pm_en;  x_pm_idx <= r_pm_idx;
                x_pm_neg     <= r_pm_neg;
                x_imm        <= r_imm;
                x_is_branch  <= r_is_branch;
                x_br_cond    <= r_br_cond;
                x_br_on_zero <= r_br_on_zero;
                x_br_ind     <= r_br_ind;
                x_br_link    <= r_br_link;
                x_br_off     <= r_br_off;
                x_is_exit    <= r_is_exit;
                x_is_trap    <= r_is_trap;
                x_is_yield   <= r_is_yield;
                x_mem_en     <= r_mem_en;
                x_mem_store  <= r_mem_store;
                x_mem_vector <= r_mem_vector;
                x_mem_indexed<= r_mem_indexed;
                x_mem_size   <= r_mem_size;
                x_mem_signed <= r_mem_signed;
                x_mem_hint   <= r_mem_hint;
                x_illegal    <= r_illegal;

                if (r_valid && !kill_r) perf_issued <= perf_issued + 64'd1;
            end else begin
                x_valid <= 1'b0;             // bubble, never a repeat
            end

            //------------------------------------------------------------------
            // Scoreboard: release first, then claim, so that an instruction
            // claiming the register a completing one is releasing in the same
            // cycle ends up marked busy rather than free. Both are non-blocking
            // writes to the same bit, so the later statement is the one that
            // takes effect.
            //------------------------------------------------------------------
            if (w_en) begin
                case (w_class)
                `TNCA_DST_S: if (w_idx[5:0] != 6'd0) sb_s[w_rctx*64 + w_idx[5:0]] <= 1'b0;
                `TNCA_DST_V: sb_v[w_rctx*VREGS_PER_RCTX + w_idx] <= 1'b0;
                `TNCA_DST_P: if (w_idx[3:0] != 4'd0) sb_p[w_rctx*16 + w_idx[3:0]] <= 1'b0;
                default: ;
                endcase
            end

            if (mtx_done) sb_acc[mtx_done_slot] <= 1'b0;

            if (!stall_front && r_valid && !kill_r) begin
                case (r_dstc)
                `TNCA_DST_S: if (r_dsti[5:0] != 6'd0) sb_s[r_rctx*64 + r_dsti[5:0]] <= 1'b1;
                `TNCA_DST_V: sb_v[r_rctx*VREGS_PER_RCTX + r_dsti] <= 1'b1;
                `TNCA_DST_P: if (r_dsti[3:0] != 4'd0) sb_p[r_rctx*16 + r_dsti[3:0]] <= 1'b1;
                `TNCA_DST_A: sb_acc[r_dsti[ACC_W-1:0]] <= 1'b1;
                default: ;
                endcase
            end

            //------------------------------------------------------------------
            // matrix engine ownership, and barrier wake-ups
            //------------------------------------------------------------------
            if (mtx_cmd_valid) mtx_rctx <= x_rctx;

            for (i = 0; i < NUM_RCTX; i = i + 1)
                if (bar_wake[i]) blocked[i] <= 1'b0;

            //------------------------------------------------------------------
            // FP reduction completion capture
            //------------------------------------------------------------------
            if (fpu_done) begin
                red_pend  <= 1'b1;
                red_sdata <= fpu_red_s;
                red_vdata <= fpu_red_v;
            end else if (cpl_grant && !lsu_cpl_valid && red_pend) begin
                red_pend <= 1'b0;
            end
            if (x_valid && !x_kill && use_fpu && fpu_multicycle) begin
                red_rctx <= x_rctx;
                red_dstc <= x_dstc;
                red_dsti <= x_dsti;
            end

            //------------------------------------------------------------------
            // A barrier count that underflowed means the program's own arrival
            // or operation count was wrong. Reporting it here turns what would
            // otherwise be a hang thousands of cycles later into a located
            // fault (sec 13.4).
            //------------------------------------------------------------------
            if (bar_fault) begin
                fault_valid  <= 1'b1;
                fault_code   <= `TNCA_TRAP_BARRIER;
                fault_pc     <= x_pc;
                fault_rctx   <= x_rctx;
                fault_detail <= {19'd0, bar_fault_bar};
            end

            //------------------------------------------------------------------
            // retire effects of X
            //------------------------------------------------------------------
            if (x_valid && !x_kill) begin
                instret[x_rctx] <= instret[x_rctx] + 64'd1;

                if (bar_sleep) begin
                    // Leave the PC on the mb.wait and stop fetching. Waking
                    // re-executes it, which re-reads the token from the same
                    // scalar register, so nothing survives the sleep.
                    blocked[x_rctx] <= 1'b1;
                    pc[x_rctx]      <= x_pc;
                end else if (x_illegal) begin
                    running[x_rctx] <= 1'b0;
                    faulted[x_rctx] <= 1'b1;
                    f_code[x_rctx]  <= `TNCA_TRAP_ILLEGAL;
                    f_pc[x_rctx]    <= x_pc;
                    f_detail[x_rctx]<= 24'd0;
                    fault_valid     <= 1'b1;
                    fault_code      <= `TNCA_TRAP_ILLEGAL;
                    fault_pc        <= x_pc;
                    fault_rctx      <= x_rctx;
                    fault_detail    <= 24'd0;
                end else if (x_is_trap) begin
                    running[x_rctx] <= 1'b0;
                    faulted[x_rctx] <= 1'b1;
                    f_code[x_rctx]  <= `TNCA_TRAP_USER;
                    f_pc[x_rctx]    <= x_pc;
                    f_detail[x_rctx]<= x_imm[23:0];
                    fault_valid     <= 1'b1;
                    fault_code      <= `TNCA_TRAP_USER;
                    fault_pc        <= x_pc;
                    fault_rctx      <= x_rctx;
                    fault_detail    <= x_imm[23:0];
                end else if (x_is_exit) begin
                    running[x_rctx] <= 1'b0;
                end else if (x_is_branch && br_taken) begin
                    pc[x_rctx] <= br_target;
                end
            end

            end // !launch_valid
        end
    end

    //==========================================================================
    // Scheduling invariants (simulation only)
    //
    // The reservation register is the only thing standing between three
    // independently-timed units and a dropped result. If it is ever wrong, the
    // symptom is a silently lost writeback -- which looks like a hang three
    // thousand instructions later, not like a scheduling bug. These checks turn
    // that into an immediate, located failure.
    //==========================================================================
`ifndef SYNTHESIS
    // synthesis translate_off
    always @(posedge clk) begin
        if (rst_n && !launch_valid) begin
            // The message starts with FAIL so tools/run_test.sh treats it as a
            // failure: an assertion nobody acts on is decoration.
            if (({2'd0, x_has_dest} + {2'd0, fp_wb} + {2'd0, mul_wb}) > 3'd1)
                $display("FAIL tnca_core assert @%0t: two scheduled writebacks in one cycle (alu=%b fp=%b mul=%b)",
                         $time, x_has_dest, fp_wb, mul_wb);
            if (sched_wb !== wb_rsv[0])
                $display("FAIL tnca_core assert @%0t: reservation disagrees with reality (sched=%b rsv0=%b alu=%b fp=%b mul=%b)",
                         $time, sched_wb, wb_rsv[0], x_has_dest, fp_wb, mul_wb);
        end
    end
    // synthesis translate_on
`endif

    //==========================================================================
    // Idle
    //==========================================================================
    assign core_idle = (running == {NUM_RCTX{1'b0}})
                    && !d_valid && !r_valid && !x_valid
                    && lsu_req_ready && !lsu_cpl_valid
                    && !fpu_busy && !red_pend && !w_en
                    && !mtx_busy
                    && (wb_rsv == {WB_DEPTH{1'b0}});

    // Signals decoded but not consumed by the Stage-1 datapath. They are kept
    // on the bundle because Stage 2 needs them (the FU class for dual issue,
    // the cache hint for the NUCA slice, yield for the fetch arbiter).
    wire _unused = &{1'b0, x_fu, r_fu, x_is_yield,
                     x_va_en, x_vc_en, x_pa_en, x_pb_en, x_sa_en, x_sc_en, 1'b0};

endmodule
