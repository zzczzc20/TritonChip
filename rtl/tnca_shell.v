//==============================================================================
// tnca_shell -- the public hardware top: one core, a command port, a
//               completion port, and a host access port
//
// doc/phase3_6_tasks_and_acceptance.md R36-RTL-02 and R36-RTL-03.
//
// WHY THIS FILE EXISTS
//
// Until now the device's behaviour lived in a testbench. tb/device/tnca_device.v
// pulsed `launch_valid` from an `initial` block, counted cycles in a `while`
// loop, decided SUCCEEDED / FAULT / TIMEOUT in Verilog tasks, and moved host
// data by reaching into the memory model with a hierarchical call
// (`u_mem.load_range(...)`). Every one of those is simulator-only. A platform
// that instantiated the core would get a core and none of the behaviour the
// runtime depends on, and would have to reimplement it -- which is how two
// implementations of the same protocol drift apart.
//
// R36-RTL-02 says it plainly: the timing contract belongs to the public top,
// and function must not depend on testbench hierarchical reads and writes. So
// the sequencing, the deadline, the fault latch, the completion record and the
// host/core arbitration are here, in synthesizable Verilog, behind ports.
//
// WHAT IS AND IS NOT INSIDE
//
//   inside:  launch sequencing, the device-side deadline, the completion
//            record, quiesce, host access arbitration, the status and
//            counter outputs.
//
//   outside: memory and instruction fetch (a platform brings its own; the
//            simulation binding brings tb/common/tnca_mem_model.v), and the
//            transport that carries commands (a file mailbox today, an MMIO
//            doorbell on a board). Both are ports, not assumptions.
//
// There is no $fopen, no $readmemh, no plusarg and no delay in this file.
// tools/r36/rtl_tests.py checks that mechanically rather than trusting this
// comment.
//
// THE COMMAND PROTOCOL
//
// One command at a time, `cmd_valid`/`cmd_ready`, accepted only when the shell
// is idle. A command runs to completion and produces exactly one completion
// record, held until `done_ready`. There is no queue: the runtime's transport
// is a request/response mailbox with a monotonic sequence number and already
// serialises, and a depth-one interface that says so is honest about it.
//
//   CMD_LAUNCH   run a kernel, from idle back to idle
//   CMD_QUIESCE  wait for idle without launching anything
//   CMD_NOP      complete immediately; the liveness check
//
// A completion is one of:
//
//   ST_OK       the core reached idle before the deadline
//   ST_FAULT    the core raised a fault; code, pc and detail come with it
//   ST_TIMEOUT  the deadline expired first. The core is NOT reset by this:
//               the shell reports it and stays busy-clean, so the host can
//               tell "still running" from "went wrong".
//
// HOST ACCESS
//
// `host_*` reads and writes one 64-bit word of device memory through the same
// memory port the core uses. It is accepted ONLY while no kernel is running.
// That is not politeness: R36-IO-02 asks for host/device concurrency to be
// refused by the device rather than by convention, and `host_req_ready` going
// low during a launch is that refusal, in hardware.
//==============================================================================
`default_nettype none

module tnca_shell #(
    parameter NUM_RCTX       = 2,
    parameter RCTX_W         = 1,
    parameter VLEN           = 16,
    parameter ELEN           = 32,
    parameter VREGS_PER_RCTX = 64,
    // R41 G2 resource sizing (defaults: the delivered machine): accumulator slots, mbarriers, the barrier epoch width
    parameter ACC_N = 16, ACC_W = 4, BAR_N = 32, BAR_W = 5, BAR_EPOCH_W = 48, FMA_IMPL = 0, MAC_DP = 0, MAC_CUTS = 30,
    parameter PHYS_VREGS     = NUM_RCTX * VREGS_PER_RCTX,
    parameter PHYS_W         = 7,
    parameter LSU_TAG_W      = 5,
    parameter HAS_SFU        = 1,
    // Forwarded to the core, not because this shell uses them, but
    // because spec/r1_profiles.json names them in its elaboration and a
    // platform that cannot set them cannot reproduce the profile. They
    // were missing, so an integrator got the core's defaults and got
    // this configuration right by luck; another profile would have been
    // silently wrong. R36-CFG-01 asks the export description to agree
    // with the profile field by field, and it now does.
    parameter HAS_ATOMICS    = 1,
    parameter HAS_SYS_ATOMICS = 0,
    parameter USE_BANKED_VRF = 0,
    parameter VRF_BYPASS     = 0,
    parameter FE_NOBUBBLE    = 0,
    parameter VRF_PREFETCH   = 0,
    parameter RCTX_SWITCH    = 0,
    parameter LSU_INFLIGHT   = 1,
    parameter RSLICE         = 0,
    parameter RPAIR          = 0,
    parameter MTX_ABUF       = 0,
    parameter FPU_OVERLAP    = 0,
    parameter HAS_TENSOR_XPOSE = 0,
    parameter CORE_ID        = 0,
    parameter NUM_CORES      = 1,
    // The tag the shell uses for its own host accesses. The core is idle
    // whenever one is issued, so nothing of the core's is in flight and any
    // tag is free; naming it makes the response routing checkable.
    parameter HOST_TAG       = 0
) (
    input  wire                     clk,
    input  wire                     rst_n,

    //--------------------------------------------------------------------------
    // Command port
    //--------------------------------------------------------------------------
    input  wire                     cmd_valid,
    output wire                     cmd_ready,
    input  wire [3:0]               cmd_op,
    input  wire [63:0]              cmd_pc,
    input  wire [63:0]              cmd_arg_base,
    input  wire [NUM_RCTX-1:0]      cmd_role_mask,
    // R39-05: give each started context its own slice of the core's
    // accumulators, barriers, descriptors and windows (RSLICE): bit 31 of
    // the launch's role-mask argument, and bits 30:29 the log2 of the
    // number of slices (0: one per context the core has). R39-12: bit 28,
    // pair contexts 2k and 2k+1 (RPAIR). {pair, slice, log2 slices}, the
    // core's launch_slice.
    input  wire [3:0]               cmd_slice,
    input  wire [31:0]              cmd_pid_x,
    input  wire [31:0]              cmd_pid_y,
    input  wire [31:0]              cmd_pid_z,
    input  wire [31:0]              cmd_npid_x,
    input  wire [31:0]              cmd_npid_y,
    input  wire [31:0]              cmd_npid_z,
    // Cycles this command may take before the shell calls it TIMEOUT. Zero
    // means no deadline, which is a choice the host makes and not a default
    // the hardware invents.
    input  wire [63:0]              cmd_deadline,

    //--------------------------------------------------------------------------
    // Completion port
    //--------------------------------------------------------------------------
    output wire                     done_valid,
    input  wire                     done_ready,
    output wire [3:0]               done_status,
    output wire [63:0]              done_cycles,
    output wire [63:0]              done_retired,
    output wire [7:0]               done_fault_code,
    output wire [63:0]              done_fault_pc,
    output wire [RCTX_W-1:0]        done_fault_rctx,
    output wire [23:0]              done_fault_detail,

    //--------------------------------------------------------------------------
    // Host access to device memory, one 64-bit word at a time
    //--------------------------------------------------------------------------
    input  wire                     host_req_valid,
    output wire                     host_req_ready,
    input  wire                     host_req_we,
    input  wire [63:0]              host_req_addr,
    input  wire [7:0]               host_req_be,
    input  wire [63:0]              host_req_wdata,
    output reg                      host_rsp_valid,
    output reg  [63:0]              host_rsp_rdata,

    //--------------------------------------------------------------------------
    // Status
    //--------------------------------------------------------------------------
    output wire                     busy,
    output wire                     core_idle,
    output wire [63:0]              perf_cycles,
    output wire [63:0]              perf_issued,
    output wire [63:0]              perf_stall_hazard,
    output wire [63:0]              perf_stall_lsu,
    output wire [63:0]              perf_stall_wb,
    output wire [63:0]              perf_squash,
    // Every counter in spec/r38/counters.json, by index. Combinational and
    // read-only: selecting a counter changes nothing the kernel can see,
    // which is what lets a host read them between launches without the
    // reading entering the next launch's cycles.
    input  wire [4:0]               perf_sel,
    output reg  [63:0]              perf_rdata,

    //--------------------------------------------------------------------------
    // Instruction fetch, to whatever holds the program
    //--------------------------------------------------------------------------
    output wire [63:0]              imem_addr,
    input  wire [63:0]              imem_data,

    //--------------------------------------------------------------------------
    // Data memory. The shell arbitrates the core and host onto this one port.
    //--------------------------------------------------------------------------
    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [63:0]              mem_req_addr,
    output wire                     mem_req_we,
    output wire [7:0]               mem_req_be,
    output wire [63:0]              mem_req_wdata,
    output wire [LSU_TAG_W-1:0]     mem_req_tag,
    output wire [1:0]               mem_req_hint,
    output wire                     mem_req_amo,
    output wire [3:0]               mem_req_aop,
    output wire [63:0]              mem_req_cmp,
    input  wire                     mem_rsp_valid,
    input  wire [LSU_TAG_W-1:0]     mem_rsp_tag,
    input  wire [63:0]              mem_rsp_rdata
);

    localparam [3:0] CMD_NOP     = 4'd0;
    localparam [3:0] CMD_LAUNCH  = 4'd1;
    localparam [3:0] CMD_QUIESCE = 4'd2;

    localparam [3:0] ST_OK      = 4'd0;
    localparam [3:0] ST_FAULT   = 4'd1;
    localparam [3:0] ST_TIMEOUT = 4'd2;
    localparam [3:0] ST_BADCMD  = 4'd3;

    localparam [2:0] S_IDLE   = 3'd0;
    localparam [2:0] S_PULSE  = 3'd1;
    localparam [2:0] S_SETTLE = 3'd2;
    localparam [2:0] S_RUN    = 3'd3;
    localparam [2:0] S_DONE   = 3'd4;

    // How long after the launch pulse before `core_idle` is believed.
    //
    // The core clears its pipeline valids and sets `running` on the same edge
    // it sees `launch_valid`, so `core_idle` is already low the cycle after --
    // but only if the role mask selected a context. Sampling it too early once
    // made an empty launch look like a completed one.
    //
    // The value is chosen to reproduce the window the testbench this
    // replaces used. It does not reproduce its ARITHMETIC exactly: the
    // testbench read `core_idle` in the active region of a clock edge,
    // where a read can land on either side of it, and this samples on the
    // edge. Measured across every kernel in the repository the difference
    // is exactly one cycle per launch, in the same direction, and the
    // number this reports is the smaller one.
    //
    // The recorded counts in doc/ and tutorial/ were moved to match. The
    // hardware's own measurement is the one to keep; tuning a state
    // machine to reproduce a testbench's off-by-one would be preserving
    // the wrong thing.
    localparam integer SETTLE_CYCLES = 2;

    reg  [2:0]  state;
    reg  [1:0]  settle;
    reg  [63:0] run_cycles;
    reg  [63:0] deadline_q;
    reg  [3:0]  status_q;
    reg  [63:0] retired_q;
    // perf_issued when the command was accepted. perf_issued is never
    // cleared (spec/r3_counters.json), so the completion reports the
    // difference: instructions retired by THIS command, like done_cycles.
    // Reporting perf_issued itself made the second program of a grid
    // claim the first program's instructions as well.
    reg  [63:0] issued_base;
    reg  [7:0]  fcode_q;
    reg  [63:0] fpc_q;
    reg  [RCTX_W-1:0] frctx_q;
    reg  [23:0] fdetail_q;

    reg         launch_valid;
    reg  [63:0] launch_pc;
    reg  [63:0] launch_arg_base;
    reg  [NUM_RCTX-1:0] launch_role_mask;
    reg  [3:0]          launch_slice;
    reg  [31:0] pid_x, pid_y, pid_z, npid_x, npid_y, npid_z;

    wire        core_fault_valid;
    wire [7:0]  core_fault_code;
    wire [63:0] core_fault_pc;
    wire [RCTX_W-1:0] core_fault_rctx;
    wire [23:0] core_fault_detail;

    // The vector base of each context. Fixed by the register file's geometry,
    // not by anything the host chooses, so it is computed here rather than
    // carried in every command.
    wire [NUM_RCTX*PHYS_W-1:0] launch_vbase;
    genvar gi;
    generate
        for (gi = 0; gi < NUM_RCTX; gi = gi + 1) begin : g_vbase
            assign launch_vbase[gi*PHYS_W +: PHYS_W] =
                gi[PHYS_W-1:0] * VREGS_PER_RCTX[PHYS_W-1:0];
        end
    endgenerate

    //==========================================================================
    // The core
    //==========================================================================
    wire        core_req_valid, core_req_ready, core_req_we;
    wire [63:0] core_req_addr, core_req_wdata, core_req_cmp;
    wire [7:0]  core_req_be;
    wire [LSU_TAG_W-1:0] core_req_tag;
    wire [1:0]  core_req_hint;
    wire        core_req_amo;
    wire [3:0]  core_req_aop;

    // The exclusive stall attribution (spec/r3_counters.json stall_exclusive).
    wire [63:0] perf_stall_total, perf_x_hazard, perf_x_lsu, perf_x_fpu,
                perf_x_mtx, perf_x_txf, perf_x_fence, perf_x_arrive,
                perf_x_rsv, perf_x_wb, perf_x_vrf, perf_x_vrfwb, perf_x_frontend;

    tnca_core #(
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN), .ELEN(ELEN), .LW(4),
        .VREGS_PER_RCTX(VREGS_PER_RCTX), .ACC_N(ACC_N), .ACC_W(ACC_W), .BAR_N(BAR_N), .BAR_W(BAR_W), .BAR_EPOCH_W(BAR_EPOCH_W), .FMA_IMPL(FMA_IMPL), .MAC_DP(MAC_DP), .MAC_CUTS(MAC_CUTS), .PHYS_VREGS(PHYS_VREGS),
        .PHYS_W(PHYS_W), .CORE_ID(CORE_ID), .NUM_CORES(NUM_CORES),
        .HAS_SFU(HAS_SFU), .HAS_ATOMICS(HAS_ATOMICS),
        .HAS_SYS_ATOMICS(HAS_SYS_ATOMICS),
        .USE_BANKED_VRF(USE_BANKED_VRF),
        .VRF_BYPASS(VRF_BYPASS), .FE_NOBUBBLE(FE_NOBUBBLE),
        .VRF_PREFETCH(VRF_PREFETCH), .RCTX_SWITCH(RCTX_SWITCH),
        .LSU_INFLIGHT(LSU_INFLIGHT), .FPU_OVERLAP(FPU_OVERLAP), .RSLICE(RSLICE),
        .RPAIR(RPAIR), .MTX_ABUF(MTX_ABUF),
        .HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .launch_valid(launch_valid), .launch_pc(launch_pc),
        .launch_role_mask(launch_role_mask), .launch_slice(launch_slice),
        .launch_pid_x(pid_x), .launch_pid_y(pid_y), .launch_pid_z(pid_z),
        .launch_npid_x(npid_x), .launch_npid_y(npid_y),
        .launch_npid_z(npid_z),
        .launch_arg_base(launch_arg_base),
        .launch_vbase(launch_vbase),
        .core_idle(core_idle),
        .fault_valid(core_fault_valid), .fault_code(core_fault_code),
        .fault_pc(core_fault_pc), .fault_rctx(core_fault_rctx),
        .fault_detail(core_fault_detail),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .mem_req_valid(core_req_valid), .mem_req_ready(core_req_ready),
        .mem_req_amo(core_req_amo), .mem_req_aop(core_req_aop),
        .mem_req_cmp(core_req_cmp),
        .mem_req_addr(core_req_addr), .mem_req_we(core_req_we),
        .mem_req_be(core_req_be), .mem_req_wdata(core_req_wdata),
        .mem_req_tag(core_req_tag), .mem_req_hint(core_req_hint),
        .mem_rsp_valid(mem_rsp_valid && !host_pending),
        .mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_rdata(mem_rsp_rdata),
        .perf_cycles(perf_cycles), .perf_issued(perf_issued),
        .perf_stall_hazard(perf_stall_hazard),
        .perf_stall_lsu(perf_stall_lsu),
        .perf_stall_wb(perf_stall_wb), .perf_squash(perf_squash),
        .perf_stall_total(perf_stall_total),
        .perf_x_hazard(perf_x_hazard), .perf_x_lsu(perf_x_lsu),
        .perf_x_fpu(perf_x_fpu), .perf_x_mtx(perf_x_mtx),
        .perf_x_txf(perf_x_txf), .perf_x_fence(perf_x_fence),
        .perf_x_arrive(perf_x_arrive), .perf_x_rsv(perf_x_rsv),
        .perf_x_wb(perf_x_wb), .perf_x_vrf(perf_x_vrf),
        .perf_x_vrfwb(perf_x_vrfwb), .perf_x_frontend(perf_x_frontend)
    );

    //==========================================================================
    // Host access, arbitrated against the core
    //
    // The host is only ever served while the shell is idle, so there is no
    // priority question: at most one of the two is asking. `host_req_ready`
    // is the refusal R36-IO-02 wants, and `host_pending` steers the single
    // response back to whoever asked for it.
    //==========================================================================
    reg         host_pending;
    reg         host_issue;
    reg         host_we_q;
    reg  [63:0] host_addr_q;
    reg  [7:0]  host_be_q;
    reg  [63:0] host_wdata_q;

    // A kernel is live from the launch until the shell observes the core
    // idle. This is the shell's own record rather than a reading of
    // `core_idle`, for two reasons. A TIMEOUT completion returns the
    // shell to idle with the core still running, and host access must
    // stay refused through that -- `core_idle` would say the opposite.
    // And a combinational dependence on the core's idle signal put the
    // host port's readiness at the mercy of every term in it.
    reg kernel_live;

    wire host_window = (state == S_IDLE) && !kernel_live && !cmd_valid;

    assign host_req_ready = host_window && !host_pending && !host_issue;

    assign mem_req_valid = host_issue ? 1'b1        : core_req_valid;
    assign mem_req_addr  = host_issue ? host_addr_q : core_req_addr;
    assign mem_req_we    = host_issue ? host_we_q   : core_req_we;
    assign mem_req_be    = host_issue ? host_be_q   : core_req_be;
    assign mem_req_wdata = host_issue ? host_wdata_q : core_req_wdata;
    assign mem_req_tag   = host_issue ? HOST_TAG[LSU_TAG_W-1:0] : core_req_tag;
    assign mem_req_hint  = host_issue ? 2'b00       : core_req_hint;
    assign mem_req_amo   = host_issue ? 1'b0        : core_req_amo;
    assign mem_req_aop   = host_issue ? 4'd0        : core_req_aop;
    assign mem_req_cmp   = host_issue ? 64'd0       : core_req_cmp;

    // The core must not see its request accepted while the host owns the port.
    assign core_req_ready = mem_req_ready && !host_issue;

    //==========================================================================
    // Data-port traffic, the core's side only
    //
    // A beat is one accepted 8-byte request. Host transfers go through the
    // same port but are not the kernel's traffic, so they are not counted:
    // host_issue owns the port on exactly the cycles the core cannot be
    // accepted. Cleared on rst_n only, like every counter.
    //==========================================================================
    reg [63:0] perf_mem_rd_beats, perf_mem_wr_beats;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_mem_rd_beats <= 64'd0;
            perf_mem_wr_beats <= 64'd0;
        end else if (core_req_valid && core_req_ready) begin
            if (core_req_we && !core_req_amo)
                perf_mem_wr_beats <= perf_mem_wr_beats + 64'd1;
            else
                perf_mem_rd_beats <= perf_mem_rd_beats + 64'd1;
        end
    end

`include "tnca_perf_sel.vh"

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            host_pending   <= 1'b0;
            host_issue     <= 1'b0;
            host_rsp_valid <= 1'b0;
            host_rsp_rdata <= 64'd0;
            host_we_q      <= 1'b0;
            host_addr_q    <= 64'd0;
            host_be_q      <= 8'd0;
            host_wdata_q   <= 64'd0;
        end else begin
            host_rsp_valid <= 1'b0;

            if (host_req_valid && host_req_ready) begin
                host_issue   <= 1'b1;
                host_pending <= 1'b1;
                host_we_q    <= host_req_we;
                host_addr_q  <= host_req_addr;
                host_be_q    <= host_req_be;
                host_wdata_q <= host_req_wdata;
            end

            if (host_issue && mem_req_ready)
                host_issue <= 1'b0;

            if (host_pending && mem_rsp_valid) begin
                host_pending   <= 1'b0;
                host_rsp_valid <= 1'b1;
                host_rsp_rdata <= mem_rsp_rdata;
            end
        end
    end

    //==========================================================================
    // The command sequencer
    //==========================================================================
    assign cmd_ready  = (state == S_IDLE) && !host_pending && !host_issue;
    assign busy       = (state != S_IDLE);
    assign done_valid = (state == S_DONE);
    assign done_status       = status_q;
    assign done_cycles       = run_cycles;
    assign done_retired      = retired_q;
    assign done_fault_code   = fcode_q;
    assign done_fault_pc     = fpc_q;
    assign done_fault_rctx   = frctx_q;
    assign done_fault_detail = fdetail_q;

    // The deadline has expired when one was asked for and has been reached.
    wire expired = (deadline_q != 64'd0) && (run_cycles >= deadline_q);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            settle           <= 2'd0;
            run_cycles       <= 64'd0;
            deadline_q       <= 64'd0;
            status_q         <= ST_OK;
            retired_q        <= 64'd0;
            issued_base      <= 64'd0;
            fcode_q          <= 8'd0;
            fpc_q            <= 64'd0;
            frctx_q          <= {RCTX_W{1'b0}};
            fdetail_q        <= 24'd0;
            kernel_live      <= 1'b0;
            launch_valid     <= 1'b0;
            launch_pc        <= 64'd0;
            launch_arg_base  <= 64'd0;
            launch_role_mask <= {NUM_RCTX{1'b0}};
            launch_slice     <= 4'd0;
            pid_x <= 32'd0; pid_y <= 32'd0; pid_z <= 32'd0;
            npid_x <= 32'd1; npid_y <= 32'd1; npid_z <= 32'd1;
        end else begin
            launch_valid <= 1'b0;

            case (state)
            S_IDLE: begin
                if (cmd_valid && cmd_ready) begin
                    run_cycles <= 64'd0;
                    deadline_q <= cmd_deadline;
                    fcode_q    <= 8'd0;
                    fpc_q      <= 64'd0;
                    frctx_q    <= {RCTX_W{1'b0}};
                    fdetail_q  <= 24'd0;
                    retired_q  <= 64'd0;
                    issued_base <= perf_issued;
                    case (cmd_op)
                    CMD_LAUNCH: begin
                        launch_pc        <= cmd_pc;
                        launch_arg_base  <= cmd_arg_base;
                        launch_role_mask <= cmd_role_mask;
                        launch_slice     <= cmd_slice;
                        pid_x  <= cmd_pid_x;
                        pid_y  <= cmd_pid_y;
                        pid_z  <= cmd_pid_z;
                        // A grid extent of zero is not a grid. The runtime
                        // already normalises it; the hardware does not rely
                        // on that, because a zero here would make
                        // num_programs(axis) report zero to a kernel that
                        // is demonstrably running.
                        npid_x <= (cmd_npid_x == 32'd0) ? 32'd1 : cmd_npid_x;
                        npid_y <= (cmd_npid_y == 32'd0) ? 32'd1 : cmd_npid_y;
                        npid_z <= (cmd_npid_z == 32'd0) ? 32'd1 : cmd_npid_z;
                        launch_valid <= 1'b1;
                        kernel_live  <= 1'b1;
                        settle <= SETTLE_CYCLES[1:0];
                        state  <= S_PULSE;
                    end
                    CMD_QUIESCE: begin
                        state <= S_RUN;
                    end
                    CMD_NOP: begin
                        status_q <= ST_OK;
                        state    <= S_DONE;
                    end
                    default: begin
                        status_q <= ST_BADCMD;
                        state    <= S_DONE;
                    end
                    endcase
                end
            end

            S_PULSE: begin
                // The pulse went out last cycle; the core sees it this edge.
                // Nothing is counted yet -- run_cycles measures the run, and
                // the core has not started one.
                state <= S_SETTLE;
            end

            S_SETTLE: begin
                if (settle == 2'd0) state <= S_RUN;
                else                settle <= settle - 2'd1;
            end

            // The counter advances only when the core was found still busy,
            // so `done_cycles` is the number of cycles the kernel was
            // observed running and not the number of cycles the shell spent
            // in its own state machine.
            S_RUN: begin
                if (core_fault_valid) begin
                    status_q  <= ST_FAULT;
                    fcode_q   <= core_fault_code;
                    fpc_q     <= core_fault_pc;
                    frctx_q   <= core_fault_rctx;
                    fdetail_q <= core_fault_detail;
                    retired_q <= perf_issued - issued_base;
                    kernel_live <= 1'b0;
                    state     <= S_DONE;
                end else if (core_idle) begin
                    status_q  <= ST_OK;
                    retired_q <= perf_issued - issued_base;
                    kernel_live <= 1'b0;
                    state     <= S_DONE;
                end else if (expired) begin
                    status_q  <= ST_TIMEOUT;
                    retired_q <= perf_issued - issued_base;
                    state     <= S_DONE;
                end else begin
                    run_cycles <= run_cycles + 64'd1;
                end
            end

            S_DONE: begin
                // A TIMEOUT deliberately does NOT clear kernel_live: the
                // core was still running when the deadline expired, and
                // saying otherwise would open the host port onto memory a
                // kernel is still writing.
                if (done_ready) state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

    // Signals present for completeness of the memory contract and not used by
    // this configuration's arbitration.
    wire _unused = &{1'b0, mem_rsp_tag, 1'b0};

endmodule

`default_nettype wire
