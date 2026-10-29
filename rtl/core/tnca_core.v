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

// An undeclared identifier is an error, not a one-bit wire. Three signals
// in this file were silently disconnected by becoming implicit nets inside
// a generate block, and two of them were one bit wide so nothing warned.
`default_nettype none

module tnca_core #(
    parameter USE_BANKED_VRF = 0,
    // R39 hardware switches. Each is 0 in every profile that predates it,
    // and 0 elaborates exactly the core that existed before it.
    //   VRF_BYPASS  (R39-06) a vector source being written back THIS cycle
    //               is not a hazard on the banked file: the read is issued
    //               in the same cycle and the bank's write-first bypass
    //               returns the new value (tnca_rf_bank RDW_BYPASS).
    //   FE_NOBUBBLE (R39-09) the fetch is not held back for a buffered word
    //               the decoder is taking this cycle, so a stall no longer
    //               costs a second fetch slot (a bubble after every stall).
    parameter VRF_BYPASS     = 0,
    parameter FE_NOBUBBLE    = 0,
    //   VRF_PREFETCH (R39-07) on the banked file the instruction in D reads
    //               its vector sources one cycle early, when R is not using
    //               the read ports, so the bank's one-cycle read overlaps the
    //               instruction ahead instead of stalling issue.
    parameter VRF_PREFETCH   = 0,
    //   RCTX_SWITCH (R39-05) an instruction held in R by a register hazard
    //               gives way when the instruction behind it belongs to
    //               another Role Context: it is dropped before it has done
    //               anything, its context refetches it once a writeback to
    //               that context could have cleared the hazard, and the
    //               other context issues meanwhile. A stalled context no
    //               longer stalls the core. 2 (R39-12): also when it waits
    //               for the matrix unit or the transfer engine to be free,
    //               and it refetches once one goes idle.
    parameter RCTX_SWITCH    = 0,
    //   MTX_ABUF (R39-12) the matrix unit reads an MMA's A fragment into a
    //               buffer first (tnca_matrix ABUF) instead of one register
    //               read per MAC.
    parameter MTX_ABUF       = 0,
    //   LSU_INFLIGHT (R39-08) 2: two load/store units behind the one port
    //               (tnca_lsu_pair), so an access starts while the one ahead
    //               waits for memory; atomics and fences see one access at
    //               a time, as before.
    parameter LSU_INFLIGHT   = 1,
    //   FPU_OVERLAP (R39-10) ordinary vector FP issues during an SFU pass,
    //               and an SFU pass starts with ordinary FP in flight: the
    //               SFU has its own datapath and latches its source, so
    //               only a reduction (which folds through the FMA array)
    //               still excludes them.
    parameter FPU_OVERLAP    = 0,
    // R39-05: per-context resource slices. Not a timing switch -- it is a
    // capability (rctx_slice): a launch with launch_slice[2] set gives each
    // context its own accumulators, barriers, descriptors and windows. An
    // index an instruction names (or a barrier token carries) has the
    // context number ORed into its top bits, so context r of S slices uses
    // slice r of each file -- 16/S accumulators, 32/S barriers, 16/S
    // descriptors and 16/S windows -- and S programs of one kernel run at
    // once without naming each other's resources. S = 2^launch_slice[1:0],
    // or every context (2^RCTX_W) when that field is 0: a kernel that needs
    // more than a quarter of a file runs on two contexts of a four-context
    // core with halves. The compiler keeps every index below the slice
    // size; ORing is then idempotent, so a token mb.arm returns (already
    // sliced) means the same thing when it comes back.
    parameter RSLICE         = 0,
    // R39-12: Role Context pairs (capability rctx_pair). A launch with
    // launch_slice[3] set makes contexts 2k and 2k+1 one program's two roles
    // (Mode A): both get the vector base of context 2k, so they name the same
    // registers -- the windows one fills and the other reads, with each role's
    // own temporaries in a range the compiler keeps apart -- and RSLICE slices
    // by the pair, k, so both name the same accumulators, barriers,
    // descriptors and windows. Scalar and predicate files, PCs and faults stay
    // per context. Without the bit, or with RPAIR = 0, nothing changes.
    parameter RPAIR          = 0,
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
    // ISA sec 8.8 and sec 11: the capability bit may be set only once the
    // datapath has been measured against the published bound. It has been --
    // tb_sfu_xls and tb_sfu_core -- so it is on by default. Building with 0
    // leaves the encodings raising ILLEGAL_INSTRUCTION and removes the slice.
    parameter HAS_SFU        = 1,
    // ISA sec 9.2. Device-scope atomics execute at the home slice, which is
    // why they could not exist before Stage 3. System scope needs a second
    // device to be a scope at all, and there is none.
    parameter HAS_ATOMICS    = 1,
    parameter HAS_SYS_ATOMICS = 0,
    // ISA sec 9E mod[5]: the transfer engine's transposed write. 0 makes a
    // mod[5] transfer ILLEGAL_INSTRUCTION (capability tensor_transpose).
    parameter HAS_TENSOR_XPOSE = 0,
    parameter FP_LAT         = 2,          // tnca_vector_fpu result delay, in cycles
    parameter SFU_LAT        = 6,          // rtl/gen/tnca_sfu_*_xls.v latency
    parameter MUL_LAT        = 3,          // tnca_mul64 result delay, in cycles
    parameter MEM_DW         = 64,
    parameter MEM_BYTES      = 8,
    parameter MEM_SHIFT      = 3,
    parameter LSU_BEATS      = 8,          // outstanding memory beats
    parameter LSU_TAG_W      = 5,          // port tag: 1 owner bit + 4
    parameter CTAG_W         = 4,          // per-client tag
    parameter DESC_N         = 16,
    parameter DESC_W         = 4,
    parameter WIN_N          = 32,
    parameter WIN_W          = 5,
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
    input  wire [3:0]               launch_slice,     // {pair, slice, log2 slices}
    input  wire [31:0]              launch_pid_x,
    input  wire [31:0]              launch_pid_y,
    input  wire [31:0]              launch_pid_z,
    input  wire [31:0]              launch_npid_x,
    input  wire [31:0]              launch_npid_y,
    input  wire [31:0]              launch_npid_z,
    input  wire [63:0]              launch_arg_base,
    // Per-RCTX physical vector-register base. Equal bases alias two contexts so
    // a producer role can hand a tile to a consumer through a shared RF window
    // with no copy; disjoint bases keep them private. The kernel manifest
    // states which (doc/compiler_overview.md sec 13.1).
    input  wire [NUM_RCTX*PHYS_W-1:0] launch_vbase,
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
    output wire                     mem_req_amo,
    output wire [3:0]               mem_req_aop,
    output wire [MEM_DW-1:0]        mem_req_cmp,
    input  wire                     mem_rsp_valid,
    input  wire [LSU_TAG_W-1:0]     mem_rsp_tag,
    input  wire [MEM_DW-1:0]        mem_rsp_rdata,

    // ---- performance counters (arch overview sec 20.2) -----------------------
    output reg  [63:0]              perf_cycles,
    output reg  [63:0]              perf_issued,
    output reg  [63:0]              perf_stall_hazard,
    output reg  [63:0]              perf_stall_lsu,
    output reg  [63:0]              perf_stall_wb,
    output reg  [63:0]              perf_squash,

    // ---- R3-06 COUNT-02 ---------------------------------------------------
    //
    // "主stall互斥计数总和与被阻塞cycle相等；重叠计数另列，不重复累计总体
    //  时间."
    //
    // The three counters above do not satisfy that and cannot be made to.
    // `hazard` and `stall_lsu` are separate terms of one OR and are true
    // together all the time, so their sum exceeds the blocked cycles; and
    // stall_fpu, stall_mtx, stall_txf, stall_fence, stall_arrive and
    // stall_rsv were not counted at all, so it also falls short. Both at
    // once, which is why "add them up and see" never revealed it.
    //
    // So the attribution below is EXCLUSIVE: every blocked cycle is charged
    // to exactly one cause, by a fixed priority, and the nine of them sum to
    // perf_stall_total by construction. The old counters stay, unchanged in
    // meaning, as the overlapping view -- COUNT-02 asks for that one to be
    // listed separately rather than deleted, because "how many cycles did
    // the LSU contribute to" is a real question with a different answer.
    output reg  [63:0]              perf_stall_total,
    output reg  [63:0]              perf_x_hazard,
    output reg  [63:0]              perf_x_lsu,
    output reg  [63:0]              perf_x_fpu,
    output reg  [63:0]              perf_x_mtx,
    output reg  [63:0]              perf_x_txf,
    output reg  [63:0]              perf_x_fence,
    output reg  [63:0]              perf_x_arrive,
    output reg  [63:0]              perf_x_rsv,
    output reg  [63:0]              perf_x_wb,
    output reg  [63:0]              perf_x_vrf,
    output reg  [63:0]              perf_x_vrfwb,
    // Cycles a role context runs in which nothing is stalled and nothing
    // issues: R holds no live instruction -- a fetch bubble, a squash, a
    // redirect. With it, perf_issued + perf_stall_total + perf_x_frontend
    // == perf_cycles exactly (R39-00); without it a sizeable share of the
    // measured cycles had no name.
    output reg  [63:0]              perf_x_frontend
);

    // NOTE ON THE INSTRUCTION PORT: Stage 1 models instruction fetch as a
    // single-cycle asynchronous read. Real silicon needs a synchronous SRAM
    // plus a fetch buffer, which adds a stage between F and D. That change is
    // local to the front end and is tracked as Stage-2 work; it does not affect
    // the ISA or the hazard logic.

    localparam VREG_IDX_W = 8;

    //==========================================================================
    // The vector register file: behavioural, or banked with real ports
    //
    // USE_BANKED_VRF is opt-in and off by default. The behavioural file
    // answers four reads combinationally in the same cycle, which is free in
    // simulation and does not exist in silicon; the banked one has four banks
    // of one read and one write port each, a one-cycle synchronous read, and
    // an operand fetch that stalls when the sources it needs are in the same
    // bank.
    //
    // Both are built from the same sources and the old one stays the default,
    // so it remains the differential reference rather than becoming a thing
    // nobody runs (doc/phase3_tasks_and_acceptance.md section 1).
    //==========================================================================

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

    reg [NUM_RCTX*PHYS_W-1:0] vbase_q;
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
        reg                  p``_amo;        \
        reg [`TNCA_AOP_W-1:0] p``_aop;       \
        reg                  p``_amo_cas;    \
        reg                  p``_is_fence;   \
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

    // RCTX_SWITCH: a context whose instruction gave way in R does not fetch
    // until a writeback to one of its registers (or an accumulator's
    // completion) could have cleared what it was waiting for. Fetching it
    // sooner would only bring the same instruction back to the same stall.
    reg [NUM_RCTX-1:0] parked;

    // Roles this launch started. Latched, not derived from `running`, which
    // drops a context as soon as it exits.
    reg [NUM_RCTX-1:0] launched;
    reg [5:0]          n_active;
    integer na_i;
    always @(*) begin
        n_active = 6'd0;
        for (na_i = 0; na_i < NUM_RCTX; na_i = na_i + 1)
            if (launched[na_i]) n_active = n_active + 6'd1;
    end
    wire [NUM_RCTX-1:0] fetchable = running & ~faulted & ~blocked & ~parked;

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
    // The instruction memory is SYNCHRONOUS (ISA-neutral, but it is what a
    // real SRAM is): the address presented this cycle returns its word on the
    // next. So there is one stage between choosing a PC and holding an
    // instruction, and a one-deep buffer behind it -- a stalled D cannot
    // refuse a word that is already on its way.
    //
    //   f1_*   a fetch is in flight; its word is on imem_data next cycle
    //   fb_*   a word arrived while D was stalled, and is waiting for it
    //
    // No fetch is issued while the buffer is occupied, so the two are never
    // both full and one entry is enough. `avail_*` is whichever of them holds
    // D's next instruction.
    //==========================================================================
    reg              f1_valid;
    reg [RCTX_W-1:0] f1_rctx;
    reg [63:0]       f1_pc;

    reg              fb_valid;
    reg [RCTX_W-1:0] fb_rctx;
    reg [63:0]       fb_pc;
    reg [63:0]       fb_insn;

    wire              avail      = fb_valid | f1_valid;
    wire [RCTX_W-1:0] avail_rctx = fb_valid ? fb_rctx : f1_rctx;
    wire [63:0]       avail_pc   = fb_valid ? fb_pc   : f1_pc;
    wire [63:0]       avail_insn = fb_valid ? fb_insn : imem_data;

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
    wire        dec_amo, dec_amo_cas, dec_is_fence;
    wire [`TNCA_AOP_W-1:0] dec_aop;

    //==========================================================================
    // RSLICE: the launch's resource partition, and the remap of the indices
    // an instruction names. Done as D hands the instruction to R, so every
    // later use -- the accumulator scoreboard, the barrier, descriptor and
    // window files, the matrix and transfer engines -- sees the physical
    // index. Destinations of class A, B and RES name an accumulator, a
    // barrier, or a descriptor or window; `ra` names an accumulator (ACC),
    // a barrier (MBAR) or a descriptor (TXFER).
    //==========================================================================
    reg       slice_q;
    reg [1:0] slice_k;            // log2 of the number of slices
    reg       pair_q;             // RPAIR: the slice is the pair's, rctx >> 1
    wire       d_sl    = (RSLICE != 0) && slice_q;
    wire [7:0] d_rctx8 = {{(8-RCTX_W){1'b0}}, d_rctx} >> ((RPAIR != 0) && pair_q);
    // Index width minus log2(slices): where the context number goes.
    wire [3:0] acc_sh  = ACC_W  - slice_k;
    wire [3:0] bar_sh  = BAR_W  - slice_k;
    wire [3:0] res_sh  = DESC_W - slice_k;
    wire [7:0] dec_dsti_sl =
        !d_sl                          ? dec_dsti :
        (dec_dstc == `TNCA_DST_A)      ? (dec_dsti | (d_rctx8 << acc_sh)) :
        (dec_dstc == `TNCA_DST_B)      ? (dec_dsti | (d_rctx8 << bar_sh)) :
        (dec_dstc == `TNCA_DST_RES)    ? (dec_dsti | (d_rctx8 << res_sh)) :
                                         dec_dsti;
    wire [7:0] dec_ra_sl =
        !d_sl                          ? dec_ra_idx :
        (dec_major == `TNCA_MAJ_ACC)   ? (dec_ra_idx | (d_rctx8 << acc_sh)) :
        (dec_major == `TNCA_MAJ_MBAR)  ? (dec_ra_idx | (d_rctx8 << bar_sh)) :
        (dec_major == `TNCA_MAJ_TXFER) ? (dec_ra_idx | (d_rctx8 << res_sh)) :
                                         dec_ra_idx;

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
        .o_mem_hint(dec_mem_hint),
        .o_amo(dec_amo), .o_aop(dec_aop), .o_amo_cas(dec_amo_cas),
        .o_is_fence(dec_is_fence)
    );

    // VRF_PREFETCH: the vector source fields of the word entering D, decoded
    // from the fetch word and registered with it, so D's early read does not
    // wait for the decoder (the request path into the banks is the core's
    // critical path). Unused, synthesis removes all of it.
    wire        pf_va_en_n, pf_vb_en_n, pf_vc_en_n;
    wire [7:0]  pf_va_idx_n, pf_vb_idx_n, pf_vc_idx_n;
    reg         d_pf_va_en, d_pf_vb_en, d_pf_vc_en;
    reg  [7:0]  d_pf_va_idx, d_pf_vb_idx, d_pf_vc_idx;
    tnca_decoder u_pfdec (
        .insn(avail_insn),
        .o_va_en(pf_va_en_n), .o_va_idx(pf_va_idx_n),
        .o_vb_en(pf_vb_en_n), .o_vb_idx(pf_vb_idx_n),
        .o_vc_en(pf_vc_en_n), .o_vc_idx(pf_vc_idx_n)
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

    // ISA sec 9.2 and sec 12.9: system scope is legal only when
    // CAPABILITY0.SYSTEM_ATOMICS is set. Refusing it is the point -- silently
    // executing a system-scope atomic at device scope would give a program
    // that needs the wider scope a right answer here and a wrong one on a
    // machine with more than one device in it.
    wire atomic_unsupported =
        (dec_major == `TNCA_MAJ_ATOM) &&
        (((HAS_ATOMICS == 0)) ||
         ((dec_mod[6:5] == `TNCA_AMO_SCOPE_SYS) && (HAS_SYS_ATOMICS == 0)));

    wire d_illegal = dec_illegal | vreg_oob | sfu_unsupported
                   | wrspecial_unsupported | atomic_unsupported;

    //==========================================================================
    // Register files (read in X)
    //==========================================================================
    wire [63:0]          sa_data, sb_data, sc_data, sb2_raw;
    wire [63:0]          sb3_raw, sb4_raw, sb5_raw;
    wire [VLEN*ELEN-1:0] va_data, vb_data, vc_data;

    // Declared here rather than beside the units that drive them. A
    // forward-referenced net inside a generate block becomes an implicit
    // one-bit net in the generate scope instead of resolving to the
    // module-level wire, and the only sign is a width warning; the VRF
    // block below reads all of these.
    wire [7:0]           mtx_vrf_addr;
    wire [VLEN*ELEN-1:0] mtx_vrf_data;
    wire                 mtx_vrf_req;
    reg  [RCTX_W-1:0]    mtx_rctx;
    // The transfer engine's window READ port: a t.st.async's source. Held
    // until valid, like the matrix unit's (immediate on the behavioural file,
    // a cycle later through the banked file's arbitration).
    wire                 txf_vr_req;
    wire [RCTX_W-1:0]    txf_vr_rctx;
    wire [7:0]           txf_vr_idx;
    wire                 txf_vr_valid;
    wire [VLEN*ELEN-1:0] txf_vr_data;
    wire                 txf_vw_en;
    wire [RCTX_W-1:0]    txf_vw_rctx;
    wire [7:0]           txf_vw_idx;
    wire [VLEN-1:0]      txf_vw_lane;
    wire [VLEN*ELEN-1:0] txf_vw_data;
    // Declared here and assigned where it is computed. The operand fetch
    // below reads it, and a net first mentioned inside a generate block
    // becomes an implicit ONE-BIT net in that scope rather than resolving
    // to the module-level wire -- which for a 1-bit signal produces no
    // warning at all. That is how mtx_rctx, mtx_vrf_req and this one were
    // each silently disconnected.
    wire                 stall_front;
    wire                 kill_d;          // forward: the D-stage squash (below)
    wire                 pf_haz_w;        // forward: VRF_PREFETCH's hazard (below)
    wire                 stall_other;     // everything but the VRF
    wire                 r_vsrc_haz;      // a vector source's producer is in flight
    wire                 r_of_idle;       // R has no vector read asked for, arriving or held
    // Zero on the reference, which has no ports to contend for.
    reg [31:0] vrf_n_req, vrf_n_accept, vrf_n_conflict, vrf_n_stall,
               vrf_n_serial, vrf_n_parallel, vrf_n_mtx_req,
               vrf_n_mtx_wait, vrf_n_wb_retry;
    wire                 vrf_stall;       // operands not yet collected
    wire                 vrf_wb_freeze;   // a vector writeback was refused
    wire                 txf_vw_ready;    // the engine's window write was taken
    wire                 mtx_vrf_valid_w;
    wire [4:0]           vrf_req_accept;
    wire [4:0]           vrf_rsp_valid;
    wire [5*VLEN*ELEN-1:0] vrf_rsp_data;
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
        // One extra scalar read port, shared by the two instructions that read
        // a consecutive pair: a transfer's coordinates start at Sb (sec 9E)
        // and m.mma.async's second barrier token is at Sc+1 (sec 9C). They are
        // different functional units and never issue in the same cycle, so one
        // port serves both and the index is chosen by the opcode.
        .rd_rctx(x_rctx),
        .rd_idx((x_major == `TNCA_MAJ_MATRIX) ? (x_sc_idx + 6'd1)
                                              : (x_sb_idx + 6'd1)),
        .rd_data(sb2_raw),
        // Sb+2..Sb+4: a rank-3..5 transfer's remaining coordinates.
        .re_rctx(x_rctx),
        .re_idx(x_sb_idx + 6'd2), .re_data(sb3_raw),
        .rf_idx(x_sb_idx + 6'd3), .rf_data(sb4_raw),
        .rg_idx(x_sb_idx + 6'd4), .rg_data(sb5_raw),
        .w_en(w_en && (w_class == `TNCA_DST_S)),
        .w_rctx(w_rctx), .w_idx(w_idx[5:0]), .w_data(w_sdata)
    );

    localparam VW = VLEN * ELEN;

    wire w_is_vec = w_en && (w_class == `TNCA_DST_V);

    generate
    if (USE_BANKED_VRF == 0) begin : g_vrf_behavioural

        tnca_vector_rf #(
            .PHYS_VREGS(PHYS_VREGS), .VREGS_PER_RCTX(VREGS_PER_RCTX),
            .NUM_RCTX(NUM_RCTX),
            .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W), .PHYS_W(PHYS_W)
        ) u_vrf (
            .clk(clk), .rst_n(rst_n), .vbase(vbase_q),
            .ra_rctx(x_rctx), .ra_idx(x_va_idx), .ra_data(va_data),
            .rb_rctx(x_rctx), .rb_idx(x_vb_idx), .rb_data(vb_data),
            .rc_rctx(x_rctx), .rc_idx(x_vc_idx), .rc_data(vc_data),
            .rd_rctx(mtx_rctx), .rd_idx(mtx_vrf_addr),
            .rd_data(mtx_vrf_data),
            .re_rctx(txf_vr_rctx), .re_idx(txf_vr_idx),
            .re_data(txf_vr_data),
            .w_en(w_is_vec),
            .w_rctx(w_rctx), .w_idx(w_idx), .w_lane_en(w_lane_en),
            .w_data(w_vdata),
            .x_en(txf_vw_en), .x_rctx(txf_vw_rctx), .x_idx(txf_vw_idx),
            .x_lane_en(txf_vw_lane), .x_data(txf_vw_data)
        );
        assign vrf_stall     = 1'b0;
        assign vrf_wb_freeze = 1'b0;
        assign r_of_idle     = 1'b1;   // X reads the file itself
        assign txf_vw_ready  = 1'b1;   // separate write ports: never refused
        assign mtx_vrf_valid_w = 1'b1;
        assign txf_vr_valid = 1'b1;
        always @(posedge clk) begin
            vrf_n_req      <= 32'd0; vrf_n_accept   <= 32'd0;
            vrf_n_conflict <= 32'd0; vrf_n_stall    <= 32'd0;
            vrf_n_serial   <= 32'd0; vrf_n_parallel <= 32'd0;
            vrf_n_mtx_req  <= 32'd0; vrf_n_mtx_wait <= 32'd0;
            vrf_n_wb_retry <= 32'd0;
        end

    end else begin : g_vrf_banked

        //----------------------------------------------------------------
        // Operand fetch
        //
        // The reads are issued in R and their data is used in X, which is
        // where a one-cycle file has to be read from if X is to stay a
        // single cycle. It has to stay a single cycle: the writeback
        // reservation is booked when an instruction leaves R against a
        // fixed latency, and holding an instruction in X makes it execute
        // -- and write back -- again. The core says so itself, with
        // "reservation disagrees with reality".
        //
        // Reading in R is a cycle earlier than the behavioural file was
        // read, and that is exactly one cycle too early for the X_PROD
        // case of the hazard interlock; see AVAIL_V below.
        //----------------------------------------------------------------
        // Only once no producer of a vector source is in flight. Fetching
        // earlier reads a register a producer has not written yet, and
        // nothing afterwards can tell that the value is stale. The other
        // reasons to hold the instruction in R (a busy unit, a scalar
        // operand, the writeback reservation) do not make a fetched value
        // stale -- only an older instruction can write the register, and
        // every older writer is in the scoreboard -- so the fetch no longer
        // waits for them (timing-010): it overlaps them, and the whole
        // stall decision is no longer in front of the bank arbiters.
        wire of_enable = r_valid && !r_vsrc_haz;
        wire [2:0] of_need =
            {r_vc_en, r_vb_en, r_va_en} & {3{of_enable}};
        reg  [2:0] of_got;
        reg  [VW-1:0] of_d0, of_d1, of_d2;

        // A response to D's early read (pf_tag) is this instruction's only if
        // it advanced into R (adv_q); otherwise it is dropped and D reads
        // again. R's own requests never share a cycle with D's.
        reg  [2:0] pf_tag;
        reg        adv_q;
        wire [2:0] of_arrive = vrf_rsp_valid[2:0] & ~(pf_tag & {3{~adv_q}});
        wire [2:0] of_have   = of_got | of_arrive;
        wire of_ready = ((of_need & ~of_have) == 3'b000);

        // Ask for what is needed and has not arrived -- counting what is
        // arriving THIS cycle. Using only what had already arrived
        // re-requested a source in the same cycle its response landed, and
        // the duplicate came back after the instruction had gone.
        wire [2:0] of_want = of_need & ~of_have;

        wire [VW-1:0] of_v0 = of_arrive[0] ? vrf_rsp_data[0*VW +: VW] : of_d0;
        wire [VW-1:0] of_v1 = of_arrive[1] ? vrf_rsp_data[1*VW +: VW] : of_d1;
        wire [VW-1:0] of_v2 = of_arrive[2] ? vrf_rsp_data[2*VW +: VW] : of_d2;

        always @(posedge clk) begin
            if (!rst_n) begin
                of_got <= 3'b000;
            end else if (!stall_front) begin
                of_got <= 3'b000;      // the instruction left R
            end else begin
                of_got <= of_have;
                if (of_arrive[0]) of_d0 <= vrf_rsp_data[0*VW +: VW];
                if (of_arrive[1]) of_d1 <= vrf_rsp_data[1*VW +: VW];
                if (of_arrive[2]) of_d2 <= vrf_rsp_data[2*VW +: VW];
            end
        end

        // The operands travel into X with their instruction.
        reg [VW-1:0] x_va_q, x_vb_q, x_vc_q;
        always @(posedge clk) begin
            if (!stall_front) begin
                x_va_q <= of_v0;
                x_vb_q <= of_v1;
                x_vc_q <= of_v2;
            end
        end
        assign va_data = x_va_q;
        assign vb_data = x_vb_q;
        assign vc_data = x_vc_q;

        // The matrix consumes its data in the cycle its response is valid,
        // so it comes straight off port 3. Leaving this undriven left the
        // matrix reading Z, and the Z propagated as far as the writeback
        // reservation shift register, which then disagreed with reality --
        // an assertion firing three stages away from its cause.
        assign mtx_vrf_data = vrf_rsp_data[3*VW +: VW];
        assign txf_vr_valid = vrf_rsp_valid[4];
        assign txf_vr_data  = vrf_rsp_data[4*VW +: VW];

        assign vrf_stall = of_enable && !of_ready;
        // RCTX_SWITCH drops the instruction in R only when none of its
        // vector reads is in flight or held: a response arriving after it
        // has gone would be taken as an operand of whatever replaced it.
        assign r_of_idle = (of_got == 3'b000) && (of_arrive == 3'b000)
                        && (of_want == 3'b000);

`ifndef SYNTHESIS
        // A bounded event trace, off unless asked for.
        //
        // R3V2-COUNT-01 wants the counters checked against an independent
        // short trace rather than against themselves. This prints the
        // EVENTS -- each request offered, each one taken, each response --
        // and tools/r3v2/count_tests.py counts them in Python and compares
        // with the hardware counters below. Section 4.3 caps a diagnostic
        // window at 256 cycles, which is what `+vrftrace=` sets.
        // +hangdump: the matrix port's view of the banked file.
        always @(posedge clk)
            if (hang_at != 0 && clock_ctr >= hang_at && clock_ctr < hang_at + 16)
                $display("hang vrf @%0d rr=%0d rq_v=%b accept=%b rsp=%b p3 bank=%0d row=%0d idx=%0d rctx=%0d bank_accept=%b port_busy=%b pf_go=%b",
                         clock_ctr, u_vrf.rr_ptr, rq_v, vrf_req_accept, vrf_rsp_valid, u_vrf.req_bank[3],
                         u_vrf.req_row[3], mtx_vrf_addr, mtx_rctx, u_vrf.bank_accept,
                         u_vrf.port_busy, pf_go);

        integer trace_n, trace_lim;
        initial begin
            trace_n = 0;
            if (!$value$plusargs("vrftrace=%d", trace_lim)) trace_lim = 0;
        end
        always @(posedge clk) begin
            if (rst_n && trace_lim != 0 && trace_n < trace_lim) begin
                if (|of_want || mtx_vrf_req || |wr_we) begin
                    $display("VRFEV %0d want=%b acc=%b rsp=%b mreq=%b mval=%b we=%b wok=%b",
                             trace_n, of_want, vrf_req_accept[2:0],
                             vrf_rsp_valid[2:0], mtx_vrf_req,
                             // writeback first, as tools/r3v2/count_tests.py reads it
                             mtx_vrf_valid_w, {wr_we[0], wr_we[1]}, {wr_ok[0], wr_ok[1]});
                end
                trace_n = trace_n + 1;
            end
        end

        // R39-00 B1: a vector writeback the file refused is still pending
        // -- w_* holds it and retries -- so no operand read of its register
        // may be issued until it lands. Granted now, the read returns the
        // value from before the write, the consumer latches it in of_d*,
        // and nothing afterwards can tell. The message starts with FAIL so
        // tools/run_test.sh fails the program that reaches it.
        integer b1p;
        always @(posedge clk) begin
            if (rst_n && vrf_wb_freeze)
                for (b1p = 0; b1p < 3; b1p = b1p + 1)
                    if (of_want[b1p] && vrf_req_accept[b1p] && (r_rctx == w_rctx)
                        && (((b1p == 0) ? r_va_idx : (b1p == 1) ? r_vb_idx : r_vc_idx)
                            == w_idx))
                        $display("FAIL tnca_core assert @%0t: operand port %0d read v%0d while its writeback is held (stale value)",
                                 $time, b1p, w_idx);
        end
`endif

        //----------------------------------------------------------------
        // What the file cost, per client
        //
        // R3V2-CORE-03 asks that every read client be shown really
        // stalling, and R3V2-COUNT-01 that the counts be exact. A refused
        // request is counted where it happened rather than inferred from a
        // cycle total, because "the LSU waited" and "something waited
        // while an LSU instruction was in R" are different claims.
        //----------------------------------------------------------------
        always @(posedge clk) begin
            if (!rst_n) begin
                vrf_n_req      <= 32'd0;
                vrf_n_accept   <= 32'd0;
                vrf_n_conflict <= 32'd0;
                vrf_n_stall    <= 32'd0;
                vrf_n_serial   <= 32'd0;
                vrf_n_parallel <= 32'd0;
                vrf_n_mtx_req  <= 32'd0;
                vrf_n_mtx_wait <= 32'd0;
                vrf_n_wb_retry <= 32'd0;
            end else begin
                vrf_n_req    <= vrf_n_req +
                    {29'd0, of_want[0]} + {29'd0, of_want[1]} +
                    {29'd0, of_want[2]};
                vrf_n_accept <= vrf_n_accept +
                    {29'd0, vrf_req_accept[0]} + {29'd0, vrf_req_accept[1]} +
                    {29'd0, vrf_req_accept[2]};
                vrf_n_conflict <= vrf_n_conflict +
                    {29'd0, of_want[0] & ~vrf_req_accept[0]} +
                    {29'd0, of_want[1] & ~vrf_req_accept[1]} +
                    {29'd0, of_want[2] & ~vrf_req_accept[2]};
                if (vrf_stall) vrf_n_stall <= vrf_n_stall + 32'd1;
                // More than one source wanted, and not all of them taken:
                // the instruction's reads are being serialised.
                if ((of_want[0] + of_want[1] + of_want[2]) > 2'd1) begin
                    if ((vrf_req_accept[0] + vrf_req_accept[1]
                         + vrf_req_accept[2]) ==
                        (of_want[0] + of_want[1] + of_want[2]))
                        vrf_n_parallel <= vrf_n_parallel + 32'd1;
                    else
                        vrf_n_serial <= vrf_n_serial + 32'd1;
                end
                if (mtx_vrf_req) vrf_n_mtx_req <= vrf_n_mtx_req + 32'd1;
                if (mtx_vrf_req && !mtx_vrf_valid_w)
                    vrf_n_mtx_wait <= vrf_n_mtx_wait + 32'd1;
                if (vrf_wb_freeze) vrf_n_wb_retry <= vrf_n_wb_retry + 32'd1;
            end
        end


        //----------------------------------------------------------------
        // Requests: three execute sources and the matrix unit
        //----------------------------------------------------------------
        //
        // VRF_PREFETCH: when R asks for nothing and is not waiting on a
        // producer, the three source ports serve D. D's sources must have no
        // older writer still to come: none in the scoreboard (unless it is
        // being written back this cycle and VRF_BYPASS lets the bank's
        // write-first path return it) and none in R, the one older writer
        // the scoreboard does not hold yet.
        // (pf_haz_w is assigned after the scoreboard macros, below.)
        wire [2:0] pf_need = {d_pf_vc_en, d_pf_vb_en, d_pf_va_en};
        wire pf_go = (VRF_PREFETCH != 0) && d_valid && (|pf_need)
                   && (of_want == 3'b000) && !(r_valid && r_vsrc_haz) && !pf_haz_w;
        wire [2:0] rq_src = pf_go ? pf_need : of_want;
        wire [4:0] rq_v = {txf_vr_req && !txf_vr_valid,
                           mtx_vrf_req && !mtx_vrf_valid_w, rq_src};
        wire [RCTX_W-1:0] rq_src_rctx = pf_go ? d_rctx : r_rctx;
        wire [5*RCTX_W-1:0] rq_rctx = {txf_vr_rctx, mtx_rctx, {3{rq_src_rctx}}};
        wire [5*8-1:0] rq_idx = pf_go
            ? {txf_vr_idx, mtx_vrf_addr, d_pf_vc_idx, d_pf_vb_idx, d_pf_va_idx}
            : {txf_vr_idx, mtx_vrf_addr, r_vc_idx, r_vb_idx, r_va_idx};

        always @(posedge clk) begin
            if (!rst_n) begin
                pf_tag <= 3'b000;
                adv_q  <= 1'b0;
            end else begin
                pf_tag <= pf_go ? (vrf_req_accept[2:0] & pf_need) : 3'b000;
                adv_q  <= !stall_front && d_valid && !kill_d;
            end
        end


        //----------------------------------------------------------------
        // Writes: the writeback wins, the transfer engine waits (R39-00 B4)
        //
        // It used to be the other way round: the engine took the bank and
        // the writeback stage froze its w_* registers to retry. But the
        // FP and MUL pipes advance regardless of stalls, so a result of
        // theirs landing in a frozen cycle was dropped -- its register kept
        // the old value and its scoreboard bit never cleared, and the next
        // reader waited forever (tb/core/progs/t39_freeze_drop.s hangs).
        // The engine can wait: it holds a refused write and presents it
        // again (tnca_tensor vw_ready), and does not complete a transfer
        // while one is held. Writer 0, the lower index, has the bank first
        // (tnca_vrf_banked), as that file's own comment always said.
        //----------------------------------------------------------------
        wire [1:0] wr_we = {txf_vw_en, w_is_vec};
        wire [2*RCTX_W-1:0] wr_rctx = {txf_vw_rctx, w_rctx};
        wire [2*8-1:0]      wr_idx  = {txf_vw_idx, w_idx};
        wire [2*VLEN-1:0]   wr_mask = {txf_vw_lane, w_lane_en};
        wire [2*VW-1:0]     wr_data = {txf_vw_data, w_vdata};
        wire [1:0]          wr_ok;

        // Structurally never set now (the writeback is never refused); kept
        // so the counters and the B1 assertion keep their meaning.
        assign vrf_wb_freeze = w_is_vec && !wr_ok[0];
        assign txf_vw_ready  = !txf_vw_en || wr_ok[1];
        assign mtx_vrf_valid_w = vrf_rsp_valid[3];

        tnca_vrf_banked #(
            .PHYS_VREGS(PHYS_VREGS), .VREGS_PER_RCTX(VREGS_PER_RCTX),
            .NUM_RCTX(NUM_RCTX), .VLEN(VLEN), .ELEN(ELEN),
            .RCTX_W(RCTX_W), .PHYS_W(PHYS_W),
            .NUM_BANKS(4), .BANK_W(2), .READ_LATENCY(1),
            .NUM_PORTS(5), .ZERO_INVALID(1), .NUM_WRITERS(2)
        ) u_vrf (
            .clk(clk), .rst_n(rst_n), .clear(launch_valid), .vbase(vbase_q),
            .req_valid(rq_v), .req_rctx(rq_rctx), .req_idx(rq_idx),
            .req_accept(vrf_req_accept),
            .rsp_ready(5'b11111),
            .rsp_valid(vrf_rsp_valid), .rsp_data(vrf_rsp_data),
            .we(wr_we), .w_rctx(wr_rctx), .w_idx(wr_idx),
            .wmask(wr_mask), .wdata(wr_data), .w_accept(wr_ok)
        );

    end
    endgenerate

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
    // Both of those exemptions assume the file is read in X.
    //
    // WB_HIT says the WB stage writes the register now, so the file holds
    // it next cycle -- next cycle being when a consumer released from R
    // reaches X. X_PROD says the same thing a cycle further out, covered
    // by the behavioural file's asynchronous write-first read, which IS
    // the forwarding path.
    //
    // The banked file is read in R, one cycle before either write lands,
    // and has no forwarding network. Measured: the consumer's read was
    // issued in cycle 4 and the producer's write happened in cycle 5, so
    // the read returned the stale value -- and the first assertion in
    // t01_vector passed anyway, because iota's lane 0 is zero and so was
    // the stale value.
    //
    // So on the candidate a busy vector register is simply a hazard until
    // the scoreboard clears it, which is when the value is really in the
    // file. That costs a stall on every producer/consumer pair. It is the
    // price of a synchronous read without forwarding, and this stage
    // reports it rather than building the network to hide it.
    `define AVAIL_V(rc, idx) (!USE_BANKED_VRF ? \
                              (`WB_HIT_V(rc, idx) || `X_PROD_V(rc, idx)) : \
                              (VRF_BYPASS != 0 && `WB_HIT_V(rc, idx)))
    `define AVAIL_P(rc, idx) (`WB_HIT_P(rc, idx) || `X_PROD_P(rc, idx))

    // VRF_PREFETCH: a source of the instruction in D that an older writer
    // has still to write -- in the scoreboard (unless written back this
    // cycle, with VRF_BYPASS), or the destination of the instruction in R,
    // the one older writer the scoreboard does not hold yet.
    `define PF_HAZ(idx) ((`SB_V_BUSY(d_rctx, idx) && \
                          !(VRF_BYPASS != 0 && `WB_HIT_V(d_rctx, idx))) || \
                         (r_valid && r_dstc == `TNCA_DST_V && r_rctx == d_rctx && \
                          r_dsti == (idx)))
    assign pf_haz_w = (d_pf_va_en && `PF_HAZ(d_pf_va_idx))
                    | (d_pf_vb_en && `PF_HAZ(d_pf_vb_idx))
                    | (d_pf_vc_en && `PF_HAZ(d_pf_vc_idx));

    wire hz_sa = r_sa_en && `SB_S_BUSY(r_rctx, r_sa_idx) && !`AVAIL_S(r_rctx, r_sa_idx);
    wire hz_sb = r_sb_en && `SB_S_BUSY(r_rctx, r_sb_idx) && !`AVAIL_S(r_rctx, r_sb_idx);
    wire hz_sc = r_sc_en && `SB_S_BUSY(r_rctx, r_sc_idx) && !`AVAIL_S(r_rctx, r_sc_idx);
    wire hz_va = r_va_en && `SB_V_BUSY(r_rctx, r_va_idx) && !`AVAIL_V(r_rctx, r_va_idx);
    wire hz_vb = r_vb_en && `SB_V_BUSY(r_rctx, r_vb_idx) && !`AVAIL_V(r_rctx, r_vb_idx);
    wire hz_vc = r_vc_en && `SB_V_BUSY(r_rctx, r_vc_idx) && !`AVAIL_V(r_rctx, r_vc_idx);
    wire hz_pa = r_pa_en && `SB_P_BUSY(r_rctx, r_pa_idx) && !`AVAIL_P(r_rctx, r_pa_idx);
    wire hz_pb = r_pb_en && `SB_P_BUSY(r_rctx, r_pb_idx) && !`AVAIL_P(r_rctx, r_pb_idx);
    wire hz_pm = r_pm_en && `SB_P_BUSY(r_rctx, r_pm_idx) && !`AVAIL_P(r_rctx, r_pm_idx);

    // A transfer also reads Sb+1 as its second coordinate, so that register
    // needs the same dependency check as any other source. Leaving it out would
    // let a transfer read a coordinate its producer had not written yet -- and
    // the wrong address is a wrong answer, not a slow one.
    wire r_mma_sf = r_valid && (r_major == `TNCA_MAJ_MATRIX)
                            && (r_func == `TNCA_M_MMA) && r_mod[4];
    wire [5:0] r_sb2_idx = r_mma_sf ? (r_sc_idx + 6'd1) : (r_sb_idx + 6'd1);
    wire r_reads_sb2 = (r_valid && (r_major == `TNCA_MAJ_TXFER)) | r_mma_sf;
    wire hz_sb2 = r_reads_sb2 && `SB_S_BUSY(r_rctx, r_sb2_idx)
                              && !`AVAIL_S(r_rctx, r_sb2_idx);
    // And Sb+2..Sb+4 for a rank-3..5 transfer. The rank is in the descriptor
    // snapshot, which the issue stage cannot see, so every transfer checks
    // all five: a busy register it does not need costs a stall, a busy one it
    // does need and was not checked costs a wrong address.
    wire r_is_txfer = r_valid && (r_major == `TNCA_MAJ_TXFER);
    wire [5:0] r_sb3_idx = r_sb_idx + 6'd2;
    wire [5:0] r_sb4_idx = r_sb_idx + 6'd3;
    wire [5:0] r_sb5_idx = r_sb_idx + 6'd4;
    wire hz_sbx = r_is_txfer &&
          ((`SB_S_BUSY(r_rctx, r_sb3_idx) && !`AVAIL_S(r_rctx, r_sb3_idx)) ||
           (`SB_S_BUSY(r_rctx, r_sb4_idx) && !`AVAIL_S(r_rctx, r_sb4_idx)) ||
           (`SB_S_BUSY(r_rctx, r_sb5_idx) && !`AVAIL_S(r_rctx, r_sb5_idx)));

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

    assign r_vsrc_haz = hz_va | hz_vb | hz_vc;

    wire hazard = r_valid && (hz_sa|hz_sb|hz_sc|hz_va|hz_vb|hz_vc|hz_pa|hz_pb|hz_pm|hz_d)
                | hz_acc | hz_sb2 | hz_sbx;

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

    // A transfer's coordinates come from CONSECUTIVE scalar registers starting
    // at Sb (ISA sec 9E), so the second one needs its own read. Reusing port C
    // is free here because a transfer does not use it for anything else.
    wire [63:0] sb2_data = sb2_raw;

    wire [VLEN*ELEN-1:0] valu_v;
    wire [63:0]          valu_s;
    wire [VLEN-1:0]      valu_p;
    // Decoded in R, registered into X with the opcode (tnca_vector_alu's
    // `use_imm`).
    reg x_valu_imm;

    tnca_vector_alu #(.VLEN(VLEN), .ELEN(ELEN)) u_valu (
        .major(x_major), .func(x_func), .mod(x_mod), .mask(lane_mask),
        .va(va_data), .vb(vb_data), .sa(v_sop), .imm(x_imm),
        .use_imm(x_valu_imm),
        .pa(pa_data), .pb(pb_data),
        .vres(valu_v), .sres(valu_s), .pres(valu_p)
    );

    wire                 fpu_multicycle;
    wire                 fpu_busy, fpu_done, fpu_red_busy;
    wire [VLEN*ELEN-1:0] fpu_v;
    wire [63:0]          fpu_s;
    wire [VLEN-1:0]      fpu_p;
    wire [63:0]          fpu_red_s;
    wire [VLEN*ELEN-1:0] fpu_red_v;

    wire x_is_fp = x_valid && ((x_major == `TNCA_MAJ_VFP) ||
                               (x_major == `TNCA_MAJ_CVT) ||
                               (x_major == `TNCA_MAJ_SFU) ||
                               ((x_major == `TNCA_MAJ_RED) && (x_func >= `TNCA_RD_FADD)));

    // The FMA array's controls, decoded in R and registered into X with the
    // opcode they are decoded from (see tnca_vector_fpu's `fpctl`).
    wire r_vfp = (r_major == `TNCA_MAJ_VFP);
    wire [4:0] r_fpctl = {
        (r_major == `TNCA_MAJ_RED) && (r_func >= `TNCA_RD_FADD),
        r_vfp && (r_func == `TNCA_VF_MUL),
        r_vfp && ((r_func == `TNCA_VF_SUB) || (r_func == `TNCA_VF_FMS)),
        (r_func == `TNCA_VF_MUL),
        (r_func == `TNCA_VF_ADD) || (r_func == `TNCA_VF_SUB)};
    reg  [4:0] x_fpctl;

    tnca_vector_fpu #(.VLEN(VLEN), .ELEN(ELEN), .LW(LW), .FP_LAT(FP_LAT),
                      .SFU_LAT(SFU_LAT), .HAS_SFU(HAS_SFU)) u_vfpu (
        .clk(clk), .rst_n(rst_n),
        .major(x_major), .func(x_func), .mod(x_mod), .mask(lane_mask),
        .va(va_data), .vb(vb_data), .vc(vc_data), .fpctl(x_fpctl),
        .is_multicycle(fpu_multicycle),
        .start(x_is_fp && fpu_multicycle && !x_kill),
        .busy(fpu_busy), .red_busy_o(fpu_red_busy), .done(fpu_done),
        .vres(fpu_v), .sres(fpu_s), .pres(fpu_p),
        .mc_sres(fpu_red_s), .mc_vres(fpu_red_v)
    );

    // Integer reductions run on the integer tree; FP reductions run on the FMA
    // array. Both use major 0x15, so the destination is chosen by func.
    wire red_is_fp = (x_major == `TNCA_MAJ_RED) && (x_func >= `TNCA_RD_FADD);
    wire use_fpu   = (x_major == `TNCA_MAJ_VFP) || (x_major == `TNCA_MAJ_CVT)
                                                 || (x_major == `TNCA_MAJ_SFU)
                                                 || red_is_fp;

    // Writeback-path classification. `is_mul` and `is_fpnorm` are scheduled
    // with a delay; reductions and memory ops complete out of band.
    `define TNCA_IS_MUL(maj, fn) \
        (((maj) == `TNCA_MAJ_SALU) && (((fn) == `TNCA_S_MUL)    || \
                                       ((fn) == `TNCA_S_MULI)   || \
                                       ((fn) == `TNCA_S_MULH_S) || \
                                       ((fn) == `TNCA_S_MULH_U)))
    `define TNCA_IS_FPNORM(maj) (((maj) == `TNCA_MAJ_VFP) || ((maj) == `TNCA_MAJ_CVT))
    // FP work that completes out of band: a reduction fold, or an SFU pass.
    // Both retire through the vector FPU's multicycle completion port, so both
    // defer writeback and both hold the unit against the next FP instruction.
    `define TNCA_IS_FPMC(maj, fn) ((((maj) == `TNCA_MAJ_RED) && ((fn) >= `TNCA_RD_FADD)) \
                                   || ((maj) == `TNCA_MAJ_SFU))

    wire x_is_mul    = x_valid && `TNCA_IS_MUL(x_major, x_func);
    wire x_is_fpnorm = x_valid && `TNCA_IS_FPNORM(x_major);
    wire x_is_fpmc   = x_valid && `TNCA_IS_FPMC(x_major, x_func);

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

    // RSLICE: a barrier a token names is in the issuing context's slice.
    wire [BAR_W-1:0] x_bar_sl = ((RSLICE != 0) && slice_q)
                              ? (({{(BAR_W-RCTX_W){1'b0}}, x_rctx}
                                  >> ((RPAIR != 0) && pair_q)) << bar_sh)
                              : {BAR_W{1'b0}};
    wire [BAR_W-1:0] tok_bar = sa_data[BAR_W-1:0] | x_bar_sl;
    // The token mb.wait and mb.test hand the barrier file, in the slice. A
    // token mb.arm returned is sliced already (ORing is idempotent); one the
    // program built from an epoch it counted itself -- each role of a ring
    // does (R39-12) -- names the barrier as the program does, and without
    // this a context in slice 1 waited on slice 0's barrier.
    wire [63:0]      tok_sl  = {sa_data[63:BAR_W], tok_bar};

    wire        bar_arm_token_v;
    wire [63:0] bar_arm_token;
    wire        bar_test_done, bar_test_error;
    wire [NUM_RCTX-1:0] bar_wake;
    wire [NUM_RCTX-1:0] bar_wake_err;
    wire        bar_fault;
    wire [BAR_W-1:0] bar_fault_bar;

    wire        mtx_busy, mtx_done, mtx_done_bar_en;
    wire        mtx_free;
    wire [BAR_W-1:0] mtx_free_bar;
    wire [ACC_W-1:0] mtx_done_slot;
    wire [BAR_W-1:0] mtx_done_bar;
    // The behavioural file answers combinationally, so the matrix never
    // waits. The banked one drives this from its response valid.
    wire        mtx_vrf_valid = mtx_vrf_valid_w;
    wire [VLEN*ELEN-1:0] acc_rd_data;

    // The engine serves one command at a time, so the RCTX that owns it is the
    // one whose command is running; captured at issue.

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
    wire [BAR_W-1:0] mtx_bar = sc_data[BAR_W-1:0] | x_bar_sl;

    tnca_matrix #(
        .VLEN(VLEN), .ELEN(ELEN), .ACC_N(ACC_N), .ACC_W(ACC_W),
        .ROWS(ACC_ROWS), .ROW_W(ACC_ROW_W), .BAR_W(BAR_W), .FMA_LAT(FP_LAT),
        .ABUF(MTX_ABUF)
    ) u_mtx (
        .clk(clk), .rst_n(rst_n), .flush(launch_valid),
        .cmd_valid(mtx_cmd_valid), .cmd_op(mtx_cmd_op),
        .cmd_slot(x_dsti[ACC_W-1:0]),
        .cmd_va(x_va_idx), .cmd_vb(x_vb_idx),
        .cmd_k(x_mod[1:0]), .cmd_dt(x_mod[3:2]),
        .cmd_bar_en(mtx_bar_en), .cmd_bar(mtx_bar),
        // ISA sec 9C mod[4]: also signal source-free, on the barrier whose
        // token sits in Sc+1.
        .cmd_sf_en((x_major == `TNCA_MAJ_MATRIX) && (x_func == `TNCA_M_MMA)
                   && x_mod[4]),
        .cmd_sf_bar(sb2_raw[BAR_W-1:0] | x_bar_sl),
        .cmd_rowop(mtx_rowop), .cmd_row(sa_data[ACC_ROW_W-1:0]),
        .cmd_vdata(vb_data), .cmd_scale(mtx_scale), .cmd_mask(lane_mask),
        .busy(mtx_busy),
        .vrf_addr(mtx_vrf_addr), .vrf_data(mtx_vrf_data),
        .vrf_req(mtx_vrf_req), .vrf_valid(mtx_vrf_valid),
        .ard_slot(x_ra_idx[ACC_W-1:0]), .ard_row(sb_data[ACC_ROW_W-1:0]),
        .ard_data(acc_rd_data),
        .done(mtx_done), .done_slot(mtx_done_slot),
        .done_bar_en(mtx_done_bar_en), .done_bar(mtx_done_bar),
        .src_free(mtx_free), .src_free_bar(mtx_free_bar)
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
        // Both asynchronous engines signal the same way: one completion per
        // command, whatever it expanded into (sec 13.3).
        // One port per engine. Sharing one dropped whichever completion lost
        // the mux when both finished in the same cycle.
        .mtx_done_en(mtx_done && mtx_done_bar_en), .mtx_done_bar(mtx_done_bar),
        .mtx_free_en(mtx_free), .mtx_free_bar(mtx_free_bar),
        .txf_done_en(txf_done && txf_done_bar_en), .txf_done_bar(txf_done_bar),
        .txf_done_err(txf_done_err),
        .inval_en(x_is_bar && (x_func == `TNCA_MB_INVAL)),
        .inval_bar(x_ra_idx[BAR_W-1:0]),
        .test_token(tok_sl), .test_done(bar_test_done),
        .test_error(bar_test_error),
        .sleep_en(bar_sleep), .sleep_rctx(x_rctx), .sleep_token(tok_sl),
        .wake(bar_wake), .wake_err(bar_wake_err),
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

    // ISA sec 9.2: an atomic must be naturally aligned. The LSU splits an
    // unaligned ordinary access across two words, which is legal and merely
    // slow; for an atomic it is the one thing that cannot be done at all, so
    // the encoding faults rather than quietly performing two separate
    // read-modify-writes.
    wire [63:0] x_amo_addr = sa_data;
    wire x_amo_misaligned = x_amo &&
        ((x_mem_size == `TNCA_SZ_D) ? (x_amo_addr[2:0] != 3'd0)
                                    : (x_amo_addr[1:0] != 2'd0));
    wire x_access_fault = x_valid && !x_kill && x_amo_misaligned;

    wire x_mem_go = x_valid && x_mem_en && !x_kill && !x_access_fault;

    wire                 lm_req_valid, lm_req_ready, lm_req_we;
    wire [63:0]          lm_req_addr;
    wire [MEM_BYTES-1:0] lm_req_be;
    wire [MEM_DW-1:0]    lm_req_wdata;
    wire [CTAG_W-1:0]    lm_req_tag;
    wire [1:0]           lm_req_hint;
    wire                 lm_req_amo;
    wire [3:0]           lm_req_aop;
    wire [63:0]          lm_req_cmp;
    wire                 lm_rsp_valid;
    wire [CTAG_W-1:0]    lm_rsp_tag;
    wire [MEM_DW-1:0]    lm_rsp_rdata;

    // R39-08: with LSU_INFLIGHT = 2 a load or store may be accepted while
    // the other unit drains one (lsu_ready_load); req_ready still means no
    // access is in flight at all, which is what a fence and an arrive wait
    // for.
    wire lsu_ready_load;
    generate
    if (LSU_INFLIGHT >= 2) begin : g_lsu_pair
    tnca_lsu_pair #(
        .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W),
        .MEM_DW(MEM_DW), .MEM_BYTES(MEM_BYTES), .MEM_SHIFT(MEM_SHIFT),
        .NUM_BEATS(LSU_BEATS), .BEAT_W(CTAG_W)
    ) u_lsu (
        .req_ready_load(lsu_ready_load),
        .clk(clk), .rst_n(rst_n),
        .req_ready(lsu_req_ready),
        .req_valid(x_mem_go),
        .req_store(x_mem_store), .req_vector(x_mem_vector),
        .req_indexed(x_mem_indexed), .req_size(x_mem_size),
        .req_signed(x_mem_signed), .req_hint(x_mem_hint),
        .req_base(sa_data), .req_offset(x_imm),
        .req_index(vb_data),
        // atom.cas swaps in Sc and compares against Sb; every other atomic
        // has one operand and takes Sb.
        .req_sdata(x_amo_cas ? sc_data : sb_data),
        .req_amo(x_amo), .req_aop(x_aop), .req_cmp(sb_data),
        .req_vdata(x_mem_indexed ? vc_data : vb_data),
        .req_mask(lane_mask),
        .req_rctx(x_rctx), .req_dst_class(x_dstc), .req_dst_idx(x_dsti),
        .cpl_valid(lsu_cpl_valid), .cpl_ack(lsu_cpl_ack),
        .cpl_rctx(lsu_cpl_rctx), .cpl_dst_class(lsu_cpl_dstc),
        .cpl_dst_idx(lsu_cpl_dsti), .cpl_sdata(lsu_cpl_sdata),
        .cpl_vdata(lsu_cpl_vdata), .cpl_lane_en(lsu_cpl_lane_en),
        .mem_req_valid(lm_req_valid), .mem_req_ready(lm_req_ready),
        .mem_req_addr(lm_req_addr), .mem_req_we(lm_req_we),
        .mem_req_be(lm_req_be), .mem_req_wdata(lm_req_wdata),
        .mem_req_tag(lm_req_tag), .mem_req_hint(lm_req_hint),
        .mem_req_amo(lm_req_amo), .mem_req_aop(lm_req_aop),
        .mem_req_cmp(lm_req_cmp),
        .mem_rsp_valid(lm_rsp_valid), .mem_rsp_tag(lm_rsp_tag),
        .mem_rsp_rdata(lm_rsp_rdata)
    );
    end else begin : g_lsu_one
    assign lsu_ready_load = lsu_req_ready;
    tnca_lsu #(
        .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W),
        .MEM_DW(MEM_DW), .MEM_BYTES(MEM_BYTES), .MEM_SHIFT(MEM_SHIFT),
        .NUM_BEATS(LSU_BEATS), .BEAT_W(CTAG_W)
    ) u_lsu (
        .o_draining(), .o_amo(),
        .clk(clk), .rst_n(rst_n),
        .req_ready(lsu_req_ready),
        .req_valid(x_mem_go),
        .req_store(x_mem_store), .req_vector(x_mem_vector),
        .req_indexed(x_mem_indexed), .req_size(x_mem_size),
        .req_signed(x_mem_signed), .req_hint(x_mem_hint),
        .req_base(sa_data), .req_offset(x_imm),
        .req_index(vb_data),
        // atom.cas swaps in Sc and compares against Sb; every other atomic
        // has one operand and takes Sb.
        .req_sdata(x_amo_cas ? sc_data : sb_data),
        .req_amo(x_amo), .req_aop(x_aop), .req_cmp(sb_data),
        .req_vdata(x_mem_indexed ? vc_data : vb_data),
        .req_mask(lane_mask),
        .req_rctx(x_rctx), .req_dst_class(x_dstc), .req_dst_idx(x_dsti),
        .cpl_valid(lsu_cpl_valid), .cpl_ack(lsu_cpl_ack),
        .cpl_rctx(lsu_cpl_rctx), .cpl_dst_class(lsu_cpl_dstc),
        .cpl_dst_idx(lsu_cpl_dsti), .cpl_sdata(lsu_cpl_sdata),
        .cpl_vdata(lsu_cpl_vdata), .cpl_lane_en(lsu_cpl_lane_en),
        .mem_req_valid(lm_req_valid), .mem_req_ready(lm_req_ready),
        .mem_req_addr(lm_req_addr), .mem_req_we(lm_req_we),
        .mem_req_be(lm_req_be), .mem_req_wdata(lm_req_wdata),
        .mem_req_tag(lm_req_tag), .mem_req_hint(lm_req_hint),
        .mem_req_amo(lm_req_amo), .mem_req_aop(lm_req_aop),
        .mem_req_cmp(lm_req_cmp),
        .mem_rsp_valid(lm_rsp_valid), .mem_rsp_tag(lm_rsp_tag),
        .mem_rsp_rdata(lm_rsp_rdata)
    );
    end
    endgenerate

    tnca_mem_mux #(
        .MEM_DW(MEM_DW), .MEM_BYTES(MEM_BYTES),
        .CTAG_W(CTAG_W), .TAG_W(LSU_TAG_W)
    ) u_mmux (
        .a_req_valid(lm_req_valid), .a_req_ready(lm_req_ready),
        .a_req_addr(lm_req_addr), .a_req_we(lm_req_we),
        .a_req_be(lm_req_be), .a_req_wdata(lm_req_wdata),
        .a_req_tag(lm_req_tag), .a_req_hint(lm_req_hint),
        .a_req_amo(lm_req_amo), .a_req_aop(lm_req_aop), .a_req_cmp(lm_req_cmp),
        .a_rsp_valid(lm_rsp_valid), .a_rsp_tag(lm_rsp_tag),
        .a_rsp_rdata(lm_rsp_rdata),
        .b_req_valid(tm_req_valid), .b_req_ready(tm_req_ready),
        .b_req_addr(tm_req_addr), .b_req_we(tm_req_we),
        .b_req_be(tm_req_be), .b_req_wdata(tm_req_wdata),
        .b_req_tag(tm_req_tag),
        .b_rsp_valid(tm_rsp_valid), .b_rsp_tag(tm_rsp_tag),
        .b_rsp_rdata(tm_rsp_rdata),
        .m_req_valid(mem_req_valid), .m_req_ready(mem_req_ready),
        .m_req_addr(mem_req_addr), .m_req_we(mem_req_we),
        .m_req_be(mem_req_be), .m_req_wdata(mem_req_wdata),
        .m_req_tag(mem_req_tag), .m_req_hint(mem_req_hint),
        .m_req_amo(mem_req_amo), .m_req_aop(mem_req_aop), .m_req_cmp(mem_req_cmp),
        .m_rsp_valid(mem_rsp_valid), .m_rsp_tag(mem_rsp_tag),
        .m_rsp_rdata(mem_rsp_rdata)
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
    // Tensor descriptors, RF windows and the transfer engine
    //==========================================================================
    wire x_is_txf = x_valid && !x_kill && (x_fu == `TNCA_FU_TXFER);

    wire txf_is_cfg = x_is_txf && (x_major == `TNCA_MAJ_TDESC);
    wire txf_is_xfr = x_is_txf && (x_major == `TNCA_MAJ_TXFER)
                               && (x_func != `TNCA_T_PREFETCH);

    // t.prefetch is a hint: completion guarantees nothing about a later hit and
    // confers no ownership (sec 17.2), so the simplest honest implementation of
    // it is to drop it. Dropping a hint must not change any result, and here it
    // demonstrably cannot.
    wire txf_is_pfetch = x_is_txf && (x_major == `TNCA_MAJ_TXFER)
                                  && (x_func == `TNCA_T_PREFETCH);

    reg [1:0] txf_cfg_op;
    always @(*) begin
        case (x_func)
        `TNCA_TD_LOAD:    txf_cfg_op = 2'd0;
        `TNCA_TD_RELEASE: txf_cfg_op = 2'd1;
        `TNCA_W_DEF:      txf_cfg_op = 2'd2;
        default:          txf_cfg_op = 2'd3;
        endcase
    end

    wire        txf_busy, txf_done, txf_done_bar_en, txf_fault;
    wire [7:0]  txf_fault_code;
    wire [RCTX_W-1:0] txf_fault_rctx;
    wire [63:0] txf_fault_pc;
    wire        txf_done_err;
    wire [BAR_W-1:0] txf_done_bar;

    wire                 tm_req_valid, tm_req_ready, tm_req_we;
    wire [63:0]          tm_req_addr;
    wire [MEM_BYTES-1:0] tm_req_be;
    wire [MEM_DW-1:0]    tm_req_wdata;
    wire [CTAG_W-1:0]    tm_req_tag;
    wire                 tm_rsp_valid;
    wire [CTAG_W-1:0]    tm_rsp_tag;
    wire [MEM_DW-1:0]    tm_rsp_rdata;

    tnca_tensor #(
        .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W),
        .DESC_N(DESC_N), .DESC_W(DESC_W), .WIN_N(WIN_N), .WIN_W(WIN_W),
        .BAR_W(BAR_W), .MEM_DW(MEM_DW), .MEM_BYTES(MEM_BYTES),
        .TAG_W(CTAG_W), .NTAG(1 << CTAG_W), .HAS_XPOSE(HAS_TENSOR_XPOSE)
    ) u_txf (
        .clk(clk), .rst_n(rst_n), .flush(launch_valid),
        .cfg_valid(txf_is_cfg), .cfg_op(txf_cfg_op),
        .cfg_desc(x_dsti[DESC_W-1:0]), .cfg_win(x_dsti[WIN_W-1:0]),
        .cfg_addr(sa_data), .cfg_wbase(x_imm[15:8]), .cfg_wlen(x_imm[7:0]),
        .cmd_valid(txf_is_xfr), .cmd_store(x_func == `TNCA_T_ST),
        .cmd_desc(x_ra_idx[DESC_W-1:0]), .cmd_win(x_dsti[WIN_W-1:0]),
        .cmd_c0(sb_data), .cmd_c1(sb2_data),
        .cmd_c2(sb3_raw), .cmd_c3(sb4_raw), .cmd_c4(sb5_raw),
        .cmd_sb(x_sb_idx),
        .cmd_esz(x_mod[2:0]),
        .cmd_xpose(x_mod[5]), .cmd_layout(x_mod[4:3]),
        .cmd_bar_en(1'b1), .cmd_bar(sc_data[BAR_W-1:0] | x_bar_sl),
        .cmd_rctx(x_rctx), .cmd_pc(x_pc), .cmd_active(|lane_mask),
        .busy(txf_busy),
        .done(txf_done), .done_bar_en(txf_done_bar_en), .done_bar(txf_done_bar),
        .fault(txf_fault), .fault_code(txf_fault_code),
        .fault_rctx(txf_fault_rctx), .fault_pc(txf_fault_pc),
        .done_err(txf_done_err),
        .vw_en(txf_vw_en), .vw_rctx(txf_vw_rctx), .vw_idx(txf_vw_idx),
        .vw_lane_en(txf_vw_lane), .vw_data(txf_vw_data), .vw_ready(txf_vw_ready),
        .vr_req(txf_vr_req), .vr_rctx(txf_vr_rctx), .vr_idx(txf_vr_idx),
        .vr_valid(txf_vr_valid), .vr_data(txf_vr_data),
        .mem_req_valid(tm_req_valid), .mem_req_ready(tm_req_ready),
        .mem_req_addr(tm_req_addr), .mem_req_we(tm_req_we),
        .mem_req_be(tm_req_be), .mem_req_wdata(tm_req_wdata),
        .mem_req_tag(tm_req_tag),
        .mem_rsp_valid(tm_rsp_valid), .mem_rsp_tag(tm_rsp_tag),
        .mem_rsp_rdata(tm_rsp_rdata)
    );

    //==========================================================================
    // Deferred multicycle-FP completion (1 deep; a reduction or an SFU pass,
    // and only one of either can be in flight -- see red_busy_next)
    //==========================================================================
    reg                  red_pend;
    reg [RCTX_W-1:0]     red_rctx;
    reg [2:0]            red_dstc;
    reg [7:0]            red_dsti;
    reg [63:0]           red_sdata;
    reg [VLEN*ELEN-1:0]  red_vdata;
    // A reduction broadcasts to every lane; an SFU pass writes only the lanes
    // its predicate selected, exactly as a one-cycle vector op would.
    reg [VLEN-1:0]       red_len;

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
        // ISA sec 5.1: the ACTIVE count for this launch, not the core's
        // architectural NUM_RCTX. Reporting the architectural count was wrong
        // and only looked right because every test launched every context.
        // A program that synchronises across roles -- which is the whole
        // reason the register exists -- would then wait for a context the
        // launch never started, and hang. t24_amo_mesh found this.
        `TNCA_SPEC_NUM_ROLES: spec_rd = {58'd0, n_active};
        `TNCA_SPEC_CORE_ID:   spec_rd = CORE_ID;
        `TNCA_SPEC_NUM_CORES: spec_rd = NUM_CORES;
        `TNCA_SPEC_NUM_LANES: spec_rd = VLEN;
        `TNCA_SPEC_NUM_VREGS: spec_rd = VREGS_PER_RCTX;
        `TNCA_SPEC_MAX_RANK:  spec_rd = 64'd5;    // see the transfer engine
        `TNCA_SPEC_NUM_ACC:   spec_rd = ACC_N;
        `TNCA_SPEC_CLOCK:     spec_rd = clock_ctr;
        `TNCA_SPEC_INSTRET:   spec_rd = instret[x_rctx];
        `TNCA_SPEC_NUM_BAR:   spec_rd = BAR_N;
        `TNCA_SPEC_NUM_WIN:   spec_rd = WIN_N;
        `TNCA_SPEC_NUM_DESC:  spec_rd = DESC_N;
        `TNCA_SPEC_NUM_PHYSV: spec_rd = PHYS_VREGS;
        `TNCA_SPEC_ARG_BASE:  spec_rd = arg_base;
        `TNCA_SPEC_ISA_VER:   spec_rd = `TNCA_ISA_VERSION;
        `TNCA_SPEC_CAP0:      spec_rd = (64'd1 << `TNCA_CAP_VECTOR_FP32)
                                      | (HAS_SFU ? (64'd1 << `TNCA_CAP_SFU) : 64'd0)
                                      | (64'd1 << `TNCA_CAP_GATHER)
                                      | (HAS_ATOMICS ? (64'd1 << `TNCA_CAP_ATOM_DEV) : 64'd0)
                                      | (HAS_SYS_ATOMICS ? (64'd1 << `TNCA_CAP_ATOM_SYS) : 64'd0)
                                      | (64'd1 << `TNCA_CAP_MATRIX_F16)
                                      | (64'd1 << `TNCA_CAP_MATRIX_BF16)
                                      | (64'd1 << `TNCA_CAP_MATRIX_FP8)
                                      // The transfer engine decodes and
                                      // executes TXFER unconditionally, so the
                                      // bit must be set: ISA sec 11 says a
                                      // clear bit means the encoding raises
                                      // ILLEGAL_INSTRUCTION, and it does not.
                                      | (64'd1 << `TNCA_CAP_TENSOR_DMA)
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
                       x_illegal || x_access_fault || bar_sleep);

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
    wire x_defers_wb = x_mem_en || x_is_fpmc || x_is_fpnorm || x_is_mul;
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
            nw_lane_en = red_len;
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
    wire r_is_fpmc   = r_valid && `TNCA_IS_FPMC(r_major, r_func);

    // A unit is claimed from X, one cycle after the check in R. Asking "is the
    // unit free now?" is therefore the wrong question -- it must be free *next*
    // cycle, which it is not if the instruction currently in X is taking it.
    // Without the second term the LSU silently dropped a request whose
    // predecessor had been accepted in the very cycle R looked at req_ready,
    // leaving the load's destination marked busy forever.
    // A load or store needs only lsu_ready_load (LSU_INFLIGHT); an atomic
    // needs the unit idle.
    wire r_plain_load  = r_mem_en && !r_amo;
    wire lsu_free_next = (r_plain_load ? lsu_ready_load : lsu_req_ready) && !x_mem_go;

    // The reduction sequencer hijacks the FMA array's operands, so it may only
    // start when the array is empty -- otherwise it corrupts a normal FP
    // operation still in flight. Conversely a normal FP op may not issue into a
    // running reduction. The SFU does not share the FMA array, but it does
    // share this unit's single completion port, so it obeys the same rule.
    wire red_busy_next = fpu_busy || red_pend || fpu_done || x_is_fpmc;

    // Same "free NEXT cycle" question as the LSU: the engine is claimed from X,
    // one cycle after R looks.
    // ---- mb.arrive release semantics (doc/compiler_overview.md sec 13.2) ----
    //
    // `release` has to mean something checkable. Scalar instructions retire in
    // order, but a vector or FP result is still in flight several cycles after
    // its instruction left the execute stage, and an asynchronous command may
    // be running for hundreds. Letting mb.arrive retire before those land would
    // make "everything before the arrive is visible" false in exactly the cases
    // a consumer cares about.
    //
    // So an arrive drains first: it waits until this RCTX has no outstanding
    // register write and no engine is mid-command. Conservative -- it also
    // waits on an engine running another context's command -- and deliberately
    // so, because the precise version needs per-command ownership that buys
    // nothing at two contexts. Role Context pairs (RPAIR) are where it buys
    // something, and there it is precise (below).
    reg [NUM_RCTX-1:0] rctx_wr_busy;
    integer rb_i, rb_j;
    always @(*) begin
        for (rb_i = 0; rb_i < NUM_RCTX; rb_i = rb_i + 1) begin
            rctx_wr_busy[rb_i] = 1'b0;
            for (rb_j = 1; rb_j < 64; rb_j = rb_j + 1)
                if (sb_s[rb_i*64 + rb_j]) rctx_wr_busy[rb_i] = 1'b1;
            for (rb_j = 0; rb_j < VREGS_PER_RCTX; rb_j = rb_j + 1)
                if (sb_v[rb_i*VREGS_PER_RCTX + rb_j]) rctx_wr_busy[rb_i] = 1'b1;
            for (rb_j = 1; rb_j < 16; rb_j = rb_j + 1)
                if (sb_p[rb_i*16 + rb_j]) rctx_wr_busy[rb_i] = 1'b1;
        end
    end

    wire r_is_arrive = r_valid && (r_major == `TNCA_MAJ_MBAR)
                                && (r_func  == `TNCA_MB_ARRIVE);
    // RPAIR (R39-12): the precise version, for the engines. A consumer
    // releasing a ring stage must not wait for the transfers its producer
    // has started into the next one -- that wait serialised the two roles
    // whose overlap is the point -- so an arrive waits for a matrix or
    // transfer command only if its own context issued it: the one running
    // (each engine runs one command at a time and its issuer is recorded),
    // or one in X this cycle. ISA sec 9A.5 allows exactly this precision.
    reg  [RCTX_W-1:0] txf_rctx;          // who issued the transfer engine's command
    wire arr_mtx = (RPAIR != 0)
                 ? ((mtx_busy && mtx_rctx == r_rctx) || (mtx_cmd_valid && x_rctx == r_rctx))
                 : mtx_busy;
    wire arr_txf = (RPAIR != 0)
                 ? ((txf_busy && txf_rctx == r_rctx)
                    || ((txf_is_cfg || txf_is_xfr) && x_rctx == r_rctx))
                 : txf_busy;
    wire stall_arrive = r_is_arrive && (rctx_wr_busy[r_rctx] || !lsu_req_ready
                                        || arr_mtx || arr_txf);

    wire r_needs_txf = r_valid && (r_fu == `TNCA_FU_TXFER);
    wire txf_free_next = !txf_busy && !txf_is_cfg && !txf_is_xfr;
    wire stall_txf = r_needs_txf && !txf_free_next;

    wire r_needs_mtx = r_valid && (r_fu == `TNCA_FU_MATRIX);
    wire mtx_free_next = !mtx_busy && !mtx_cmd_valid;
    wire stall_mtx = r_needs_mtx && !mtx_free_next;

    wire stall_lsu = r_needs_lsu && !lsu_free_next;

    // ISA sec 9.2 `fence`. Every access this core has issued is complete at its
    // home slice once the LSU is drained: the LSU waits for a response to a
    // store as well as to a load, and the home slice is where the order between
    // cores is established (sec 14.1). So draining the LSU *is* the fence at
    // core and device scope, and there is nothing further to do -- which is
    // worth saying, because a fence that expanded to nothing would look the
    // same from outside and would not be the same thing.
    //
    // Asynchronous transfers are deliberately not covered: `t.*.async`
    // completes against an mbarrier (sec 9E.1 step 8), and a fence that also
    // waited for them would make the barrier redundant and the fence
    // unpredictably long.
    wire stall_fence = r_valid && r_is_fence
                    && (!lsu_req_ready || lsu_cpl_valid || rctx_wr_busy[r_rctx]);
    // FPU_OVERLAP: the FMA array is the only thing an ordinary FP op and a
    // multicycle one actually share, and only a reduction uses it -- so an
    // ordinary op waits for a fold (running, or in X and starting next
    // cycle), and an SFU pass waits only for the completion port and the
    // slice (red_busy_next), not for ordinary ops in flight.
    wire r_is_sfu   = r_valid && (r_major == `TNCA_MAJ_SFU);
    wire x_is_fpred = x_valid && (x_major == `TNCA_MAJ_RED) && (x_func >= `TNCA_RD_FADD);
    wire fold_next  = fpu_red_busy || x_is_fpred;
    wire stall_fpu = (FPU_OVERLAP != 0)
        ? ((r_is_fpnorm && fold_next)
           | (r_is_fpmc && !r_is_sfu && (red_busy_next || fp_inflight || x_is_fpnorm))
           | (r_is_sfu && red_busy_next))
        : ((r_is_fpnorm && red_busy_next)
           | (r_is_fpmc   && (red_busy_next || fp_inflight || x_is_fpnorm)));

    // ---- writeback reservation ------------------------------------------------
    // Delay from "in X" to "result available", per unit.
    wire [3:0] r_wb_delay = r_is_mul    ? MUL_LAT[3:0] :
                            r_is_fpnorm ? FP_LAT[3:0]  : 4'd0;

    // Memory ops and FP reductions complete out of band and book nothing.
    wire r_wb_reg = (r_dstc == `TNCA_DST_S) || (r_dstc == `TNCA_DST_V)
                                             || (r_dstc == `TNCA_DST_P);
    wire r_unscheduled = r_mem_en | r_is_fpmc;
    wire r_books_wb    = r_valid && r_wb_reg && !r_unscheduled;

    wire stall_rsv = r_books_wb && wb_rsv[r_wb_delay + 4'd1];

    // Everything that holds an instruction in R for a reason other than
    // its vector operands. The operand fetch is gated on this: a read
    // issued while the instruction is still hazarded comes back with the
    // value from before the producer wrote, and the instruction then
    // advances holding it. Measured: the consumer read v2 in cycle 43,
    // the producer wrote it in 44, and the assertion failed on the value
    // from 43.
    assign stall_other = hazard | stall_lsu | stall_fpu | stall_mtx
                     | stall_txf | stall_fence
                     | stall_arrive | stall_rsv | wb_steal;

    wire stall_raw = stall_other | vrf_stall | vrf_wb_freeze;

    reg [63:0] perf_rctx_switch;     // cycles in which R's instruction gave way
    // RCTX_SWITCH=2 (R39-12): also give way when R's instruction waits for the
    // matrix unit or the transfer engine, which can be busy for hundreds of
    // cycles with ANOTHER context's command. Held in R, it held every context
    // behind it: a consumer waiting for the other pair's MMA stopped the
    // producers issuing their transfers. The context wakes when an engine
    // goes idle (eng_freed) or as before.
    reg  mtx_busy_q, txf_busy_q;
    wire eng_freed = (RCTX_SWITCH >= 2) && ((mtx_busy_q && !mtx_busy) || (txf_busy_q && !txf_busy));
    always @(posedge clk) begin
        mtx_busy_q <= rst_n && mtx_busy;
        txf_busy_q <= rst_n && txf_busy;
    end
    // A parked context wakes on a writeback to one of its registers (every
    // register write passes through W) or an accumulator's completion.
    reg [NUM_RCTX-1:0] rctx_wake;
    integer wk_i;
    always @(*) begin
        for (wk_i = 0; wk_i < NUM_RCTX; wk_i = wk_i + 1)
            rctx_wake[wk_i] = (w_en && (w_rctx == wk_i[RCTX_W-1:0])
                               && !((w_class == `TNCA_DST_V) && vrf_wb_freeze))
                            || mtx_done || eng_freed;
    end

    // RCTX_SWITCH: the instruction in R waits on a register an older
    // instruction of its context has yet to write, and the one in D is
    // another context's. Drop the first -- it has
    // issued nothing, read nothing (r_of_idle) and claimed nothing -- and let
    // the front end move: its context refetches it from r_pc once woken
    // (rctx_wake, below), and the other context's instruction takes R this
    // cycle. Not while X redirects (one squash a cycle) nor while the
    // writeback is frozen (the front end must hold then whatever R holds).
    //
    // Not for a busy load/store unit: measured on t39_switch, giving way
    // there cost 8 % -- the unit is the bottleneck, and the dropped access
    // reached it three cycles after it was free instead of at once.
    wire rp_go = (RCTX_SWITCH != 0) && r_valid && r_of_idle
               && (hazard || ((RCTX_SWITCH >= 2) && (stall_mtx || stall_txf)))
               && d_valid && (d_rctx != r_rctx)
               && !x_redirect && !vrf_wb_freeze;
    assign stall_front = stall_raw && !rp_go;

    // Exclusive attribution. The order is the order of the OR above, which
    // makes it arbitrary but FIXED -- and fixed is what matters: the same
    // blocked cycle must be charged to the same cause every run, or the
    // counters are not comparable between two of them. The priority is part
    // of the counter contract (spec/r3_counters.json), not an accident of
    // how this expression happens to be written.
    wire x_haz   = hazard;
    wire x_lsu   = !x_haz && stall_lsu;
    wire x_fpu   = !x_haz && !x_lsu && stall_fpu;
    wire x_mtx   = !x_haz && !x_lsu && !x_fpu && stall_mtx;
    wire x_txf   = !x_haz && !x_lsu && !x_fpu && !x_mtx && stall_txf;
    wire x_fence = !x_haz && !x_lsu && !x_fpu && !x_mtx && !x_txf
                   && stall_fence;
    wire x_arr   = !x_haz && !x_lsu && !x_fpu && !x_mtx && !x_txf
                   && !x_fence && stall_arrive;
    wire x_rsv   = !x_haz && !x_lsu && !x_fpu && !x_mtx && !x_txf
                   && !x_fence && !x_arr && stall_rsv;
    wire x_wb    = !x_haz && !x_lsu && !x_fpu && !x_mtx && !x_txf
                   && !x_fence && !x_arr && !x_rsv && wb_steal;
    // The two the banked file adds. They are last in the priority chain
    // and zero on the reference, so its attribution is unchanged -- but
    // they have to be IN the chain: adding a stall source to stall_front
    // and not to the attribution breaks the one property the exclusive
    // group exists to have, which is that it sums to the total.
    wire x_vrf   = !x_haz && !x_lsu && !x_fpu && !x_mtx && !x_txf
                   && !x_fence && !x_arr && !x_rsv && !x_wb && vrf_stall;
    wire x_vrfwb = !x_haz && !x_lsu && !x_fpu && !x_mtx && !x_txf
                   && !x_fence && !x_arr && !x_rsv && !x_wb && !x_vrf
                   && vrf_wb_freeze;

    // A redirect kills only the younger entries of the same RCTX.
    wire             squash_en   = x_redirect;
    wire [RCTX_W-1:0] squash_rctx = x_rctx;
    // A switch (rp_go) kills the same younger entries of R's context; D's
    // is another context's by construction.
    wire kill_f = (squash_en && f_hit  && (f_sel  == squash_rctx))
                | (rp_go && f_hit && (f_sel == r_rctx));
    assign kill_d = squash_en && d_valid && (d_rctx == squash_rctx);
    // The in-flight fetch and the buffered word are younger than X, so a
    // redirect kills them on the same rule.
    wire kill_f1 = (squash_en && f1_valid && (f1_rctx == squash_rctx))
                 | (rp_go && f1_valid && (f1_rctx == r_rctx));
    wire kill_fb = (squash_en && fb_valid && (fb_rctx == squash_rctx))
                 | (rp_go && fb_valid && (fb_rctx == r_rctx));
    wire kill_av = fb_valid ? kill_fb : kill_f1;

    // Issue a fetch unless the buffer is full, or a word is already arriving
    // into a front end that is stalled and would have nowhere to put a second.
    // FE_NOBUBBLE: an unstalled decoder takes the buffered word this cycle,
    // so the buffer is free again for the word a fetch issued now returns.
    wire f_issue = (FE_NOBUBBLE != 0)
                 ? (f_hit && !(stall_front && (fb_valid || f1_valid)))
                 : (f_hit && !fb_valid && !(stall_front && f1_valid));
    wire kill_r = (squash_en && r_valid && (r_rctx == squash_rctx)) | rp_go;

    // x_kill is never set in this design (X is the oldest in-flight entry), but
    // it is kept as an explicit signal so a future second execute stage has a
    // defined place to hook into.
    wire x_kill = 1'b0;

    //==========================================================================
    // Sequential
    //==========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            // The FETCH valids too. They were the only pipeline valids
            // left out of this list, and the omission was invisible for
            // as long as nothing looked at the core before the first
            // launch: `avail` is `fb_valid | f1_valid`, so the first
            // edge after reset released latched X into d_valid, r_valid
            // took X from it, and -- because stall_front depends on
            // r_valid -- the X sustained itself. `core_idle` then read X
            // from reset until a launch flushed the pipeline.
            //
            // A launch does flush, which is why every existing test
            // passed: they all launch before they look. R36-RST-01 asks
            // what the device looks like at reset, and the answer was
            // "undefined, until you use it".
            f1_valid   <= 1'b0;
            fb_valid   <= 1'b0;
            d_valid    <= 1'b0;
            r_valid    <= 1'b0;
            x_valid    <= 1'b0;
            running    <= {NUM_RCTX{1'b0}};
            launched   <= {NUM_RCTX{1'b0}};
            faulted    <= {NUM_RCTX{1'b0}};
            rr_ptr     <= {RCTX_W{1'b0}};
            sb_s       <= {(NUM_RCTX*64){1'b0}};
            sb_v       <= {(NUM_RCTX*VREGS_PER_RCTX){1'b0}};
            sb_p       <= {(NUM_RCTX*16){1'b0}};
            sb_acc     <= {ACC_N{1'b0}};
            blocked    <= {NUM_RCTX{1'b0}};
            parked     <= {NUM_RCTX{1'b0}};
            slice_q    <= 1'b0;
            slice_k    <= 2'd0;
            pair_q     <= 1'b0;
            perf_rctx_switch <= 64'd0;
            mtx_rctx   <= {RCTX_W{1'b0}};
            txf_rctx   <= {RCTX_W{1'b0}};
            red_pend   <= 1'b0;
            red_len    <= {VLEN{1'b1}};
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
            perf_stall_total  <= 64'd0;
            perf_x_hazard     <= 64'd0;
            perf_x_lsu        <= 64'd0;
            perf_x_fpu        <= 64'd0;
            perf_x_mtx        <= 64'd0;
            perf_x_txf        <= 64'd0;
            perf_x_fence      <= 64'd0;
            perf_x_arrive     <= 64'd0;
            perf_x_rsv        <= 64'd0;
            perf_x_wb         <= 64'd0;
            perf_x_vrf        <= 64'd0;
            perf_x_vrfwb      <= 64'd0;
            perf_x_frontend   <= 64'd0;
            perf_stall_wb     <= 64'd0;
            perf_squash       <= 64'd0;
            for (i = 0; i < NUM_RCTX; i = i + 1)
                vbase_q[i*PHYS_W +: PHYS_W] <= i * VREGS_PER_RCTX;
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

            // latch this cycle's writeback decision; it drives the RF next
            // cycle -- unless the file refused the last one, in which case
            // w_* IS the holding register and keeps its payload.
            if (!vrf_wb_freeze) begin
            w_en      <= nw_en;
            w_rctx    <= nw_rctx;
            w_class   <= nw_class;
            w_idx     <= nw_idx;
            w_sdata   <= nw_sdata;
            w_vdata   <= nw_vdata;
            w_pdata   <= nw_pdata;
            w_lane_en <= nw_lane_en;
            end

            if (|running) perf_cycles <= perf_cycles + 64'd1;
            // The overlapping view, unchanged.
            if (hazard)    perf_stall_hazard <= perf_stall_hazard + 64'd1;
            if (stall_lsu) perf_stall_lsu    <= perf_stall_lsu + 64'd1;
            if (wb_steal)  perf_stall_wb     <= perf_stall_wb + 64'd1;
            if (squash_en) perf_squash       <= perf_squash + 64'd1;

            // The exclusive view. Counted only while something is running,
            // on the same condition as perf_cycles, so the sum can be
            // compared against a cycle count taken over the same window.
            // stall_raw, not stall_front: a cycle in which R's instruction
            // gave way (RCTX_SWITCH) issued nothing because of its hazard,
            // and is charged to it like any other hazard cycle.
            if (|running && stall_raw)
                perf_stall_total <= perf_stall_total + 64'd1;
            if (|running && x_haz)   perf_x_hazard <= perf_x_hazard + 64'd1;
            if (|running && x_lsu)   perf_x_lsu    <= perf_x_lsu    + 64'd1;
            if (|running && x_fpu)   perf_x_fpu    <= perf_x_fpu    + 64'd1;
            if (|running && x_mtx)   perf_x_mtx    <= perf_x_mtx    + 64'd1;
            if (|running && x_txf)   perf_x_txf    <= perf_x_txf    + 64'd1;
            if (|running && x_fence) perf_x_fence  <= perf_x_fence  + 64'd1;
            if (|running && x_arr)   perf_x_arrive <= perf_x_arrive + 64'd1;
            if (|running && x_rsv)   perf_x_rsv    <= perf_x_rsv    + 64'd1;
            if (|running && x_wb)    perf_x_wb     <= perf_x_wb     + 64'd1;
            if (|running && x_vrf)   perf_x_vrf    <= perf_x_vrf    + 64'd1;
            if (|running && x_vrfwb) perf_x_vrfwb  <= perf_x_vrfwb  + 64'd1;
            if (|running && !stall_raw && !(r_valid && !kill_r))
                perf_x_frontend <= perf_x_frontend + 64'd1;
            if (|running && rp_go) perf_rctx_switch <= perf_rctx_switch + 64'd1;

            //------------------------------------------------------------------
            // launch
            //------------------------------------------------------------------
            if (launch_valid) begin
                for (i = 0; i < NUM_RCTX; i = i + 1) begin
                    if (launch_role_mask[i]) pc[i] <= launch_pc;
                    instret[i] <= 64'd0;
                end
                running  <= launch_role_mask;
                launched <= launch_role_mask;
                slice_q  <= (RSLICE != 0) && launch_slice[2];
                pair_q   <= (RPAIR != 0) && launch_slice[3];
                slice_k  <= (launch_slice[1:0] != 2'd0) ? launch_slice[1:0]
                                                        : RCTX_W[1:0];
                faulted  <= {NUM_RCTX{1'b0}};
                pid_x    <= launch_pid_x;
                pid_y    <= launch_pid_y;
                pid_z    <= launch_pid_z;
                npid_x   <= launch_npid_x;
                npid_y   <= launch_npid_y;
                npid_z   <= launch_npid_z;
                arg_base <= launch_arg_base;
                vbase_q  <= launch_vbase;
                // RPAIR: context 2k+1 takes context 2k's registers.
                if ((RPAIR != 0) && launch_slice[3])
                    for (i = 1; i < NUM_RCTX; i = i + 2)
                        vbase_q[i*PHYS_W +: PHYS_W] <= launch_vbase[(i-1)*PHYS_W +: PHYS_W];
                d_valid  <= 1'b0;
                r_valid  <= 1'b0;
                x_valid  <= 1'b0;
                w_en     <= 1'b0;
                wb_rsv   <= {WB_DEPTH{1'b0}};
                for (i = 0; i < FP_LAT; i = i + 1)  fp_v[i] <= 1'b0;
                for (i = 0; i < MUL_LAT; i = i + 1) mu_v[i] <= 1'b0;
                f1_valid <= 1'b0;
                fb_valid <= 1'b0;
                sb_s     <= {(NUM_RCTX*64){1'b0}};
                sb_v     <= {(NUM_RCTX*VREGS_PER_RCTX){1'b0}};
                sb_p     <= {(NUM_RCTX*16){1'b0}};
                sb_acc   <= {ACC_N{1'b0}};
                blocked  <= {NUM_RCTX{1'b0}};
                parked   <= {NUM_RCTX{1'b0}};
                // Round robin starts from context 0 at every launch, as it
                // does after reset: with several contexts the interleaving
                // -- and so the cycle count -- would otherwise depend on
                // what ran before (B4 found board and simulation a few
                // cycles apart on some kernels).
                rr_ptr   <= {RCTX_W{1'b0}};
                fault_valid <= 1'b0;
            end else begin

            //------------------------------------------------------------------
            // F: issue an address, and hold what comes back
            //
            // These run whether or not the front end is stalled, because the
            // memory does: a word whose address went out last cycle arrives
            // this one regardless of what D is doing with it.
            //------------------------------------------------------------------
            f1_valid <= f_issue && !kill_f;
            if (f_issue) begin
                f1_rctx <= f_sel;
                f1_pc   <= pc[f_sel];
                rr_ptr  <= (f_sel + 1) % NUM_RCTX;
                if (!kill_f) pc[f_sel] <= pc[f_sel] + 64'd8;
            end

            if (!stall_front) begin
                fb_valid <= 1'b0;                  // D took whatever was there
            end else if (f1_valid && !fb_valid) begin
                fb_valid <= !kill_f1;
                fb_rctx  <= f1_rctx;
                fb_pc    <= f1_pc;
                fb_insn  <= imem_data;
            end else if (kill_fb) begin
                fb_valid <= 1'b0;
            end

            //------------------------------------------------------------------
            // F -> D
            //------------------------------------------------------------------
            if (!stall_front) begin
                d_valid <= avail && !kill_av;
                d_rctx  <= avail_rctx;
                d_pc    <= avail_pc;
                d_insn  <= avail_insn;
                d_pf_va_en <= pf_va_en_n;  d_pf_va_idx <= pf_va_idx_n;
                d_pf_vb_en <= pf_vb_en_n;  d_pf_vb_idx <= pf_vb_idx_n;
                d_pf_vc_en <= pf_vc_en_n;  d_pf_vc_idx <= pf_vc_idx_n;
            end else if (kill_d) begin
                // The entry HELD in D is the one at risk here, so the test is
                // on d_rctx (kill_d), not on which context the fetch stage
                // happens to be selecting (kill_f). With one Role Context
                // f_sel is always the squashed one and the two coincide; with
                // two, f_sel alternates, so half the time a taken branch left
                // a younger instruction of the same context alive in D and it
                // executed after the redirect. That is how a `trap` placed
                // after a loop branch fired while the branch was taken.
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
                r_dsti       <= dec_dsti_sl;
                r_mask_dst   <= dec_mask_dst;
                r_sa_en      <= dec_sa_en && !d_illegal;  r_sa_idx <= dec_sa_idx;
                r_sb_en      <= dec_sb_en && !d_illegal;  r_sb_idx <= dec_sb_idx;
                r_sc_en      <= dec_sc_en && !d_illegal;  r_sc_idx <= dec_sc_idx;
                r_va_en      <= dec_va_en && !d_illegal;  r_va_idx <= dec_va_idx;
                r_vb_en      <= dec_vb_en && !d_illegal;  r_vb_idx <= dec_vb_idx;
                r_vc_en      <= dec_vc_en && !d_illegal;  r_vc_idx <= dec_vc_idx;
                r_ra_en      <= dec_ra_en && !d_illegal;  r_ra_idx <= dec_ra_sl;
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
                r_amo        <= dec_amo;
                r_aop        <= dec_aop;
                r_amo_cas    <= dec_amo_cas;
                r_is_fence   <= dec_is_fence;
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
                x_fpctl      <= r_fpctl;
                x_valu_imm   <= `TNCA_V_USES_IMM(r_func);
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
                x_amo        <= r_amo;
                x_aop        <= r_aop;
                x_amo_cas    <= r_amo_cas;
                x_is_fence   <= r_is_fence;
                x_illegal    <= r_illegal;

                // Async faults may leave R occupied after every context
                // stops. Use the same ROI as cycles/stalls/frontend.
                if (|running && r_valid && !kill_r)
                    perf_issued <= perf_issued + 64'd1;
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
            // A vector writeback the banked file refused this cycle has not
            // happened: w_* holds it for a retry, and the register stays busy
            // until the retry lands (R39-00, B1). Releasing it here let a
            // consumer fetch the old value during a second refusal --
            // t38_wbfreeze computed a wrong answer that way.
            if (w_en) begin
                case (w_class)
                `TNCA_DST_S: if (w_idx[5:0] != 6'd0) sb_s[w_rctx*64 + w_idx[5:0]] <= 1'b0;
                `TNCA_DST_V: if (!vrf_wb_freeze) sb_v[w_rctx*VREGS_PER_RCTX + w_idx] <= 1'b0;
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
            if (txf_is_cfg || txf_is_xfr) txf_rctx <= x_rctx;

            for (i = 0; i < NUM_RCTX; i = i + 1)
                if (bar_wake[i]) begin
                    blocked[i] <= 1'b0;
                    // ISA sec 10: a waiter woken by an error raises
                    // BARRIER_FAULT. Letting it resume sends it to wait on
                    // the next generation of a pipeline whose producer has
                    // already faulted, so the located fault becomes a hang.
                    if (bar_wake_err[i]) begin
                        running[i] <= 1'b0;
                        faulted[i] <= 1'b1;
                        f_code[i]  <= `TNCA_TRAP_BARRIER;
                    end
                end

            //------------------------------------------------------------------
            // Multicycle FP completion capture (reduction or SFU)
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
                red_len  <= (x_major == `TNCA_MAJ_SFU)
                            ? (x_mask_dst ? lane_mask : {VLEN{1'b1}})
                            : {VLEN{1'b1}};
            end

            //------------------------------------------------------------------
            // A barrier count that underflowed means the program's own arrival
            // or operation count was wrong. Reporting it here turns what would
            // otherwise be a hang thousands of cycles later into a located
            // fault (sec 13.4).
            //------------------------------------------------------------------
            //------------------------------------------------------------------
            // An engine fault must STOP the Role Context that issued the
            // command, not merely be recorded. ISA sec 10: a trap stops the
            // raising RCTX and must never turn into a silent continuation.
            // Recording it without clearing `running` let the program run on
            // past a transfer that never happened, and the next fault -- a
            // `trap` several instructions later -- overwrote the code, so a
            // DESCRIPTOR_FAULT surfaced as TRAP_USER.
            //
            // The engine reports which context issued: the fault arrives long
            // after the issuing instruction retired, so whatever sits in the
            // execute stage now is the wrong answer.
            //------------------------------------------------------------------
            if (txf_fault) begin
                running[txf_fault_rctx] <= 1'b0;
                faulted[txf_fault_rctx] <= 1'b1;
                f_code[txf_fault_rctx]  <= txf_fault_code;
                f_detail[txf_fault_rctx]<= 24'd0;
                f_pc[txf_fault_rctx]    <= txf_fault_pc;
                fault_valid  <= 1'b1;
                fault_code   <= txf_fault_code;
                fault_pc     <= txf_fault_pc;
                fault_rctx   <= txf_fault_rctx;
                fault_detail <= 24'd0;
            end else if (bar_fault) begin
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
                end else if (x_access_fault) begin
                    running[x_rctx] <= 1'b0;
                    faulted[x_rctx] <= 1'b1;
                    f_code[x_rctx]  <= `TNCA_TRAP_ACCESS;
                    f_pc[x_rctx]    <= x_pc;
                    f_detail[x_rctx]<= 24'd0;
                    fault_valid     <= 1'b1;
                    fault_code      <= `TNCA_TRAP_ACCESS;
                    fault_pc        <= x_pc;
                    fault_rctx      <= x_rctx;
                    fault_detail    <= 24'd0;
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

            //------------------------------------------------------------------
            // RCTX_SWITCH: wake on a writeback to the context (every register
            // write passes through W) or an accumulator's completion; park
            // the context whose instruction gave way, and point it back at
            // that instruction. Written after the fetch's PC increment, which
            // kill_f already suppressed for this context, and exclusive with
            // X's redirect (rp_go excludes it). A wake in the same cycle
            // wins: the writeback that clears the hazard may be this one.
            //------------------------------------------------------------------
            for (i = 0; i < NUM_RCTX; i = i + 1)
                if (rctx_wake[i]) parked[i] <= 1'b0;
            if (rp_go) begin
                pc[r_rctx] <= r_pc;
                if (!rctx_wake[r_rctx]) parked[r_rctx] <= 1'b1;
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
    // +engtrace: every matrix and transfer command as it issues and as it
    // completes, with its context and cycle -- the engines' occupancy and the
    // gaps between commands, which the stall counters do not show.
    reg engtrace;
    initial engtrace = $test$plusargs("engtrace");
    always @(posedge clk)
        if (engtrace && rst_n && !launch_valid) begin
            if (mtx_cmd_valid)
                $display("ENG %0d mtx-issue rctx%0d op%0d", clock_ctr, x_rctx, mtx_cmd_op);
            if (mtx_done)
                $display("ENG %0d mtx-done", clock_ctr);
            if (txf_is_cfg || txf_is_xfr)
                $display("ENG %0d txf-issue rctx%0d %s", clock_ctr, x_rctx, txf_is_cfg ? "cfg" : "xfr");
            if (txf_done)
                $display("ENG %0d txf-done", clock_ctr);
        end
    // +ctxtrace: when each Role Context exits, and how many instructions it
    // retired -- what shows that one context's stall did not hold the others
    // back (R39-05, t39_isolate).
    reg ctxtrace;
    initial ctxtrace = $test$plusargs("ctxtrace");
    // +hangdump=N: every context's state at cycle N -- what a hang is
    // waiting for, without a waveform.
    reg [63:0] hang_at;
    integer hd_i;
    initial if (!$value$plusargs("hangdump=%d", hang_at)) hang_at = 0;
    always @(posedge clk)
        if (hang_at != 0 && clock_ctr == hang_at) begin
            for (hd_i = 0; hd_i < NUM_RCTX; hd_i = hd_i + 1)
                $display("hang ctx %0d running=%b faulted=%b blocked=%b parked=%b pc=%0h retired=%0d sb_s=%h sb_v=%h",
                         hd_i, running[hd_i], faulted[hd_i], blocked[hd_i], parked[hd_i],
                         pc[hd_i], instret[hd_i], sb_s[hd_i*64 +: 64],
                         sb_v[hd_i*VREGS_PER_RCTX +: VREGS_PER_RCTX]);
            $display("hang r_valid=%b r_rctx=%0d r_pc=%0h d_valid=%b d_rctx=%0d x_valid=%b slice=%b sb_acc=%h lsu_ready=%b txf_busy=%b mtx_busy=%b",
                     r_valid, r_rctx, r_pc, d_valid, d_rctx, x_valid, slice_q, sb_acc,
                     lsu_req_ready, txf_busy, mtx_busy);
            $display("hang mtx st=%0d mtx_rctx=%0d vrf_req=%b vrf_valid=%b hazard=%b stall_mtx=%b stall_lsu=%b vrf_stall=%b stall_front=%b",
                     u_mtx.st, mtx_rctx, mtx_vrf_req, mtx_vrf_valid_w, hazard, stall_mtx,
                     stall_lsu, vrf_stall, stall_front);
        end
    always @(posedge clk)
        if (ctxtrace && rst_n && !launch_valid && x_valid && !x_kill && x_is_exit)
            $display("ctx %0d exit cycle=%0d retired=%0d", x_rctx, clock_ctr,
                     instret[x_rctx] + 64'd1);
    always @(posedge clk) begin
        if (rst_n && !launch_valid) begin
            // The message starts with FAIL so tools/run_test.sh treats it as a
            // failure: an assertion nobody acts on is decoration.
            if (({2'd0, x_has_dest} + {2'd0, fp_wb} + {2'd0, mul_wb}) > 3'd1)
                $display("FAIL tnca_core assert @%0t: two scheduled writebacks in one cycle (alu=%b fp=%b mul=%b)",
                         $time, x_has_dest, fp_wb, mul_wb);
            if (sched_wb !== wb_rsv[0])begin
                // Failure-only diagnostics preserve the healthy execution
                // path: no extra clocked monitor state or MMIO probes.
                $display("FAIL tnca_core assert %m @%0t: reservation disagrees with reality (sched=%b rsv0=%b alu=%b fp=%b mul=%b) cycle=%0d running=%b rsv=%b pc=%h ctx=%h major=%h func=%h dst=%h delay=%h issue(valid,stall,kill,book)=%b hazard=%b lsu=%b fpu=%b mtx=%b arrive=%b rsv_stall=%b steal=%b vrf=%b",
                         $time, sched_wb, wb_rsv[0], x_has_dest, fp_wb, mul_wb,
                         clock_ctr,running,wb_rsv,r_pc,r_rctx,r_major,r_func,r_dstc,r_wb_delay,
                         {r_valid,stall_front,kill_r,r_books_wb},hazard,stall_lsu,stall_fpu,stall_mtx,
                         stall_arrive,stall_rsv,wb_steal,vrf_stall);
                if($test$plusargs("assert_fatal"))$fatal(1,"scheduled writeback reservation mismatch");
            end
            // One fetch buffer entry is only enough because no fetch is issued
            // while it is occupied. If both ever hold a word, an instruction
            // has been dropped on the floor.
            if (f1_valid && fb_valid)
                $display("FAIL tnca_core assert @%0t: a fetch arrived with the buffer already full (pc=%0h and %0h)",
                         $time, f1_pc, fb_pc);
        end
    end
    // synthesis translate_on
`endif

    //==========================================================================
    // Idle
    //==========================================================================
    assign core_idle = (running == {NUM_RCTX{1'b0}})
                    && !f1_valid && !fb_valid
                    && !d_valid && !r_valid && !x_valid
                    && lsu_req_ready && !lsu_cpl_valid
                    && !fpu_busy && !red_pend && !w_en
                    && !mtx_busy && !txf_busy
                    && (wb_rsv == {WB_DEPTH{1'b0}});

    // Signals decoded but not consumed by the Stage-1 datapath. They are kept
    // on the bundle because Stage 2 needs them (the FU class for dual issue,
    // the cache hint for the NUCA slice, yield for the fetch arbiter).
    wire _unused = &{1'b0, x_fu, r_fu, x_is_yield,
                     x_va_en, x_vc_en, x_pa_en, x_pb_en, x_sa_en, x_sc_en, 1'b0};

endmodule

`default_nettype wire
