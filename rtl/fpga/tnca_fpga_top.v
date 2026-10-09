//==============================================================================
// tnca_fpga_top -- the device, on a board, as one module
//
// doc/phase3_6_tasks_and_acceptance.md R36-RTL-02, R36-RTL-03, R36-RST-01 and
// R36-HAND-02.
//
// Three things that were ports on tnca_shell are plugged here: the command and
// completion mailbox becomes rtl/fpga/tnca_mmio.v, the memory and fetch ports
// become rtl/fpga/tnca_fpga_mem.v, and the reset becomes a synchroniser.
// Nothing else is added, and in particular the shell is instantiated rather
// than reimplemented -- R36-RTL-03 asks the simulation binding and the platform
// shell to share one core implementation, and the way that is true here is that
// there is only one implementation and this file calls it.
//
// WHAT THIS FILE IS NOT
//
// It is not a board. There is no pin, no clock source, no MMCM, no AXI and no
// PCIe in it. Those belong to a platform wrapper that sits above this one and
// depends on the board; keeping them out is what lets this same file run under
// iverilog in tb/fpga/tb_tnca_fpga.v, which is the whole point of the
// simulation milestone: the thing that is tested off the board is the thing
// that goes on it, not a description of it.
//
// ADAPTATION RESPONSIBILITIES -- R36-HAND-02
//
// R36-HAND-02 asks for the split to be written down rather than assumed, and
// for a gap in storage or interface semantics not to be dressed up as a
// board-level task. So, item by item, with the item that is a gap marked as
// one:
//
//   THE CORE AND ITS TIMING            owned by rtl/tnca_shell.v, unchanged.
//     Launch sequencing, the device deadline, fault latching, the completion
//     record, host/core arbitration and every cycle count the core reports are
//     the shell's and this file does not touch them. A board port that changed
//     any of them would be measuring a different machine.
//
//   THE COMMAND MEDIUM                 owned here, rtl/fpga/tnca_mmio.v.
//     Two text files become a register file and a doorbell. The protocol --
//     sequence numbers, opcodes, statuses, the response fields -- is the one in
//     spec/r2_transport_v1.json and is not renegotiated.
//
//   THE PAYLOAD MEDIUM                 owned here.
//     up.hex and down.hex become a word-at-a-time window on the same host port
//     the shell already arbitrates. Payload is one word per access against a
//     register; a board with a DMA engine would replace the window and keep the
//     mailbox.
//
//   STORAGE                            owned by rtl/fpga/tnca_fpga_mem.v.
//     Capacity, ports, latency, byte masks, read-during-write and atomics are
//     itemised in that file's header. It is RAM and not a model, and the
//     consequence is written there and repeated because it is the one that
//     bites: **the memory is not initialised.** The models zero themselves in
//     an `initial` block; block RAM does not. Memory a kernel will read must be
//     uploaded by the host first. That is a semantic gap between simulation and
//     the board, it is closed by a host responsibility and not by a board
//     trick, and pretending otherwise is exactly what R36-HAND-02 forbids.
//
//   CLOCK                              NOT owned here.
//     This module takes one clock and uses it for everything. Where it comes
//     from -- a pin, a PLL, an MMCM -- and how fast it runs are the platform's
//     decisions. Nothing here assumes a frequency, and nothing here would
//     change if the frequency did, except the cycle counts the core reports,
//     which are the shell's and are cycles rather than time for that reason.
//
//   RESET                              half owned here.
//     `rst_n_async` is taken as given: asserted asynchronously, held until the
//     board is ready. De-assertion is synchronised to `clk` here, because a
//     reset released near a clock edge is a reset that releases on different
//     edges in different flops. What is NOT owned here is generating it at all
//     -- a board that ties it to a PLL lock must know that a lock can drop and
//     come back, and that this module's synchroniser will not protect a design
//     from a reset that was never asserted long enough to be seen.
//
//   HOST INTERFACE                     NOT owned here.
//     `reg_*` is a flat register port, deliberately not AXI. An AXI-Lite
//     adapter attaches above this module; keeping the port flat means the
//     protocol loop can be driven from a testbench that has no bus model, and
//     means the adapter is the only thing that has to be rewritten if the
//     transport changes from AXI-Lite to something else.
//
//   THE GAP                            recorded, not owned by anyone yet.
//     A kernel that reads a location it never wrote is correct in simulation
//     and undefined here. There is no hardware mechanism that fixes it; the
//     host must not launch a kernel whose inputs it has not written. This is
//     the item R36-HAND-02 exists to make visible, and it is visible here, in
//     the memory's header, and in the backend's contract -- three places,
//     because it is the kind of gap that gets lost in one.
//==============================================================================
`default_nettype none

module tnca_fpga_top #(
    // ---- forwarded to tnca_shell, and through it to tnca_core ---------------
    parameter NUM_RCTX        = 2,
    parameter RCTX_W          = 1,
    parameter VLEN            = 16,
    parameter ELEN            = 32,
    parameter VREGS_PER_RCTX  = 64,
    parameter HAS_SFU         = 1,
    parameter HAS_ATOMICS     = 1,
    parameter HAS_SYS_ATOMICS = 0,
    // The profile this device exists to run. spec/r1_profiles.json's
    // core-vrf-sync-v2 is a single core with a banked VRF and the R39 switches
    // (VRF_BYPASS, FE_NOBUBBLE, VRF_PREFETCH), and the defaults here are that
    // and not the core's defaults, so that a build which forgot to set them
    // produces the machine the board backend reports rather than one that
    // silently differs. core-vrf-sync-v1 is the same with the three at 0.
    parameter USE_BANKED_VRF  = 1,
    parameter VRF_BYPASS      = 1,
    parameter FE_NOBUBBLE     = 1,
    parameter VRF_PREFETCH    = 1,
    // R39-05: off in the v2 image, which was built before it existed.
    parameter RCTX_SWITCH     = 0,
    parameter LSU_INFLIGHT    = 1,
    parameter FPU_OVERLAP     = 0,
    parameter RSLICE          = 0,
    // The transposed tensor-DMA write (ISA sec 9E mod[5], R38 S1-07). Off,
    // as in core-vrf-sync-v1; a board built with it on runs a profile that
    // declares the capability, and none does for the banked VRF yet.
    parameter HAS_TENSOR_XPOSE = 0,
    parameter CORE_ID         = 0,
    parameter NUM_CORES       = 1,
    parameter LSU_TAG_W       = 5,
    parameter HOST_TAG        = 0,

    // ---- memory ------------------------------------------------------------
    // WORDS must be 2**ADDR_W. See rtl/fpga/tnca_fpga_mem.v, which is also
    // where the reason 65536 is not a free choice is written down: it is the
    // size of the device the profile describes, and a board that is smaller
    // is a different machine.
    parameter WORDS           = 65536,
    parameter ADDR_W          = 16,

    // ---- host register port ------------------------------------------------
    parameter ADDR_BITS       = 8
) (
    input  wire                    clk,
    input  wire                    rst_n_async,

    input  wire [ADDR_BITS-1:0]    reg_addr,
    input  wire [63:0]             reg_wdata,
    input  wire                    reg_we,
    input  wire                    reg_re,
    output wire [63:0]             reg_rdata
);

    //==========================================================================
    // Reset
    //
    // Asynchronous assertion, synchronous release. The first flop is the one
    // that may go metastable when the de-assertion arrives near an edge; the
    // second is what the design sees, and by the time it is set the first has
    // resolved. Both are reset by the same asynchronous signal, so a reset
    // asserted at any time takes effect immediately and in one cycle.
    //
    // `rst_n_async` must be held long enough for `clk` to see it. A pulse
    // narrower than a clock period can be missed entirely, and no synchroniser
    // can recover a reset that was never observed -- this module synchronises
    // the release, not the assertion, and saying so is more useful than a
    // comment claiming it handles everything.
    //==========================================================================
    reg [1:0] rst_sync;

    always @(posedge clk or negedge rst_n_async) begin
        if (!rst_n_async)
            rst_sync <= 2'b00;
        else
            rst_sync <= {rst_sync[0], 1'b1};
    end

    wire rst_n = rst_sync[1];

    //==========================================================================
    // The mailbox, on the register port
    //==========================================================================
    wire        cmd_valid;
    wire        cmd_ready;
    wire [3:0]  cmd_op;
    wire [63:0] cmd_pc;
    wire [63:0] cmd_arg_base;
    // The physical vector register index: 7 bits for 2 x 64, 8 for 4 x 64
    // (core-vrf-sync-v4) -- derived, so it cannot disagree with the count.
    localparam PHYS_W = $clog2(NUM_RCTX * VREGS_PER_RCTX);

    wire [NUM_RCTX-1:0] cmd_role_mask;
    wire [2:0]          cmd_slice;
    wire [31:0] cmd_pid_x, cmd_pid_y, cmd_pid_z;
    wire [31:0] cmd_npid_x, cmd_npid_y, cmd_npid_z;
    wire [63:0] cmd_deadline;

    wire        done_valid;
    wire        done_ready;
    wire [3:0]  done_status;
    wire [63:0] done_cycles;
    wire [63:0] done_retired;
    wire [7:0]  done_fault_code;
    wire [63:0] done_fault_pc;
    wire [RCTX_W-1:0] done_fault_rctx;
    wire [23:0] done_fault_detail;

    wire        host_req_valid;
    wire        host_req_ready;
    wire        host_req_we;
    wire [63:0] host_req_addr;
    wire [7:0]  host_req_be;
    wire [63:0] host_req_wdata;
    wire        host_rsp_valid;
    wire [63:0] host_rsp_rdata;
    wire [4:0]  perf_sel;
    wire [63:0] perf_rdata;

    tnca_mmio #(
        .ADDR_W(ADDR_BITS),
        .WORDS(WORDS),
        .NUM_RCTX(NUM_RCTX),
        .RCTX_W(RCTX_W)
    ) u_mmio (
        .clk(clk), .rst_n(rst_n),
        .reg_addr(reg_addr), .reg_wdata(reg_wdata),
        .reg_we(reg_we), .reg_re(reg_re), .reg_rdata(reg_rdata),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_op(cmd_op),
        .cmd_pc(cmd_pc), .cmd_arg_base(cmd_arg_base),
        .cmd_role_mask(cmd_role_mask), .cmd_slice(cmd_slice),
        .cmd_pid_x(cmd_pid_x), .cmd_pid_y(cmd_pid_y), .cmd_pid_z(cmd_pid_z),
        .cmd_npid_x(cmd_npid_x), .cmd_npid_y(cmd_npid_y),
        .cmd_npid_z(cmd_npid_z), .cmd_deadline(cmd_deadline),
        .done_valid(done_valid), .done_ready(done_ready),
        .done_status(done_status), .done_cycles(done_cycles),
        .done_retired(done_retired), .done_fault_code(done_fault_code),
        .done_fault_pc(done_fault_pc), .done_fault_rctx(done_fault_rctx),
        .done_fault_detail(done_fault_detail),
        .host_req_valid(host_req_valid), .host_req_ready(host_req_ready),
        .host_req_we(host_req_we), .host_req_addr(host_req_addr),
        .host_req_be(host_req_be), .host_req_wdata(host_req_wdata),
        .host_rsp_valid(host_rsp_valid), .host_rsp_rdata(host_rsp_rdata),
        .perf_sel(perf_sel), .perf_rdata(perf_rdata)
    );

    //==========================================================================
    // The shell, and through it the core
    //==========================================================================
    wire [63:0] imem_addr;
    wire [63:0] imem_data;

    wire        mem_req_valid;
    wire        mem_req_ready;
    wire [63:0] mem_req_addr;
    wire        mem_req_we;
    wire [7:0]  mem_req_be;
    wire [63:0] mem_req_wdata;
    wire [LSU_TAG_W-1:0] mem_req_tag;
    // The shell reports a cache hint per request. Neither memory model takes
    // one and neither does the RAM below: there is no cache in this
    // configuration for a hint to be about. Left named rather than dropped so
    // that a version with a cache has somewhere to attach.
    wire [1:0]  mem_req_hint;
    wire        mem_req_amo;
    wire [3:0]  mem_req_aop;
    wire [63:0] mem_req_cmp;
    wire        mem_rsp_valid;
    wire [LSU_TAG_W-1:0] mem_rsp_tag;
    wire [63:0] mem_rsp_rdata;

    tnca_shell #(
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN), .ELEN(ELEN),
        .VREGS_PER_RCTX(VREGS_PER_RCTX),
        .PHYS_VREGS(NUM_RCTX * VREGS_PER_RCTX),
        .PHYS_W(PHYS_W), .LSU_TAG_W(LSU_TAG_W),
        .HAS_SFU(HAS_SFU), .HAS_ATOMICS(HAS_ATOMICS),
        .HAS_SYS_ATOMICS(HAS_SYS_ATOMICS),
        .USE_BANKED_VRF(USE_BANKED_VRF),
        .VRF_BYPASS(VRF_BYPASS), .FE_NOBUBBLE(FE_NOBUBBLE),
        .VRF_PREFETCH(VRF_PREFETCH), .RCTX_SWITCH(RCTX_SWITCH),
        .LSU_INFLIGHT(LSU_INFLIGHT), .FPU_OVERLAP(FPU_OVERLAP), .RSLICE(RSLICE),
        .HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE),
        .CORE_ID(CORE_ID), .NUM_CORES(NUM_CORES), .HOST_TAG(HOST_TAG)
    ) u_shell (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_op(cmd_op),
        .cmd_pc(cmd_pc), .cmd_arg_base(cmd_arg_base),
        .cmd_role_mask(cmd_role_mask), .cmd_slice(cmd_slice),
        .cmd_pid_x(cmd_pid_x), .cmd_pid_y(cmd_pid_y), .cmd_pid_z(cmd_pid_z),
        .cmd_npid_x(cmd_npid_x), .cmd_npid_y(cmd_npid_y),
        .cmd_npid_z(cmd_npid_z), .cmd_deadline(cmd_deadline),
        .done_valid(done_valid), .done_ready(done_ready),
        .done_status(done_status), .done_cycles(done_cycles),
        .done_retired(done_retired), .done_fault_code(done_fault_code),
        .done_fault_pc(done_fault_pc), .done_fault_rctx(done_fault_rctx),
        .done_fault_detail(done_fault_detail),
        .host_req_valid(host_req_valid), .host_req_ready(host_req_ready),
        .host_req_we(host_req_we), .host_req_addr(host_req_addr),
        .host_req_be(host_req_be), .host_req_wdata(host_req_wdata),
        .host_rsp_valid(host_rsp_valid), .host_rsp_rdata(host_rsp_rdata),
        // The shell's own status. Nothing in the protocol reads it: the
        // completion record already carries cycles and retired instructions,
        // and `busy` is derivable from whether a completion has arrived.
        // Exposed at this level for a bring-up that wants to watch rather than
        // ask, and left unconnected above.
        .busy(), .core_idle(),
        .perf_cycles(), .perf_issued(), .perf_stall_hazard(),
        .perf_stall_lsu(), .perf_stall_wb(), .perf_squash(),
        // Every counter, by index, for the mailbox's PERF_READ (op 7).
        .perf_sel(perf_sel), .perf_rdata(perf_rdata),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_we(mem_req_we),
        .mem_req_be(mem_req_be), .mem_req_wdata(mem_req_wdata),
        .mem_req_tag(mem_req_tag), .mem_req_hint(mem_req_hint),
        .mem_req_amo(mem_req_amo), .mem_req_aop(mem_req_aop),
        .mem_req_cmp(mem_req_cmp),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_rdata(mem_rsp_rdata)
    );

    //==========================================================================
    // The memory
    //==========================================================================
    tnca_fpga_mem #(
        .WORDS(WORDS), .ADDR_W(ADDR_W), .MEM_DW(64), .MEM_BYTES(8),
        .TAG_W(LSU_TAG_W)
    ) u_mem (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .req_valid(mem_req_valid), .req_ready(mem_req_ready),
        .req_addr(mem_req_addr), .req_we(mem_req_we),
        .req_be(mem_req_be), .req_wdata(mem_req_wdata),
        .req_tag(mem_req_tag),
        .req_amo(mem_req_amo), .req_aop(mem_req_aop), .req_cmp(mem_req_cmp),
        .rsp_valid(mem_rsp_valid), .rsp_tag(mem_rsp_tag),
        .rsp_rdata(mem_rsp_rdata)
    );

    // `mem_req_hint` is named above and read by nothing. It is a property of
    // this configuration rather than of the port: with no cache between the
    // core and the RAM, a hint has no consumer.
    wire unused_hint = |mem_req_hint;

endmodule

`default_nettype wire
