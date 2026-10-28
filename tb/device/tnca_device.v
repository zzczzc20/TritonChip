//==============================================================================
// tnca_device -- the SIMULATION BINDING for the persistent single-core device
//
// What this file used to be, and why it is not that any more.
//
// It used to be the device: it pulsed `launch_valid` from an `initial` block,
// counted cycles in a `while` loop, decided SUCCEEDED / FAULT / TIMEOUT in a
// Verilog task, and moved host data by reaching into the memory model with a
// hierarchical call. All of that was simulator-only, which meant the device's
// behaviour existed nowhere a platform could instantiate it, and a board would
// have had to reimplement it from the protocol document.
//
// doc/phase3_6_tasks_and_acceptance.md R36-RTL-02 refuses that arrangement:
// the public top owns the timing contract, and function must not depend on
// testbench hierarchical reads and writes. So the behaviour moved into
// rtl/tnca_shell.v and what is left here is exactly what a simulation of a
// transport is allowed to be:
//
//     a clock, a reset, a memory, and a translation from the file mailbox to
//     the shell's command and host ports.
//
// Nothing below reaches inside the shell. Uploads and downloads go through
// `host_*` one word at a time, the same port a DMA engine would drive, which
// is why `u_mem.load_range` is gone. `+hex=` still loads an image directly
// into the memory model, because that is the simulation's equivalent of a
// board arriving with its flash already programmed -- it happens before the
// device is released from reset and is declared as a simulation affordance in
// the export manifest rather than smuggled in.
//
// The protocol on the wire is unchanged: spec/r2_transport_v1.json, a
// request/response mailbox with a monotonic sequence number. Swapping the two
// text files for two MMIO registers changes the medium and not the protocol.
//
// Plusargs:
//   +req=<file> +rsp=<file>   the mailbox
//   +up=<file>  +down=<file>  the payload files
//   +hex=<file>               an initial image (optional)
//   +pollcycles=<n>           how often to look at the mailbox
//   +deadline=<n>             cycles a single launch may take before the
//                             device reports TIMEOUT. NOT a host wait: this
//                             is the device's own deadline (R2-08), and it is
//                             now enforced by the shell rather than counted
//                             here.
//   +trace                    one line per served request
//
// Op 7 (PERF_READ, transport 1.1) reads one device counter by index.
//==============================================================================
`timescale 1ns/1ps

module tnca_device;

    // 1 elaborates the R3 v2 candidate's register file. Set with -P, so
    // core-sim and core-vrf-sync-v1 come from the same bench source and a
    // difference between them cannot be a difference in the device.
    parameter USE_BANKED_VRF = 0;
    parameter VRF_BYPASS     = 0;
    parameter FE_NOBUBBLE    = 0;
    parameter VRF_PREFETCH   = 0;
    parameter RCTX_SWITCH    = 0;
    parameter LSU_INFLIGHT   = 1;
    parameter FPU_OVERLAP    = 0;
    parameter RSLICE         = 0;
    parameter RPAIR          = 0;
    parameter MTX_ABUF       = 0;
    // 1 elaborates the transposed tensor-DMA write (profile core-xpose-v1).
    parameter HAS_TENSOR_XPOSE = 0;

    parameter NUM_RCTX       = 2;
    parameter RCTX_W         = 1;
    parameter VLEN           = 16;
    parameter ELEN           = 32;
    parameter VREGS_PER_RCTX = 64;
    parameter ACC_N = 16, ACC_W = 4, BAR_N = 32, BAR_W = 5, BAR_EPOCH_W = 48;  // R41 G2 resource sizing (defaults: the delivered machine)
    parameter PHYS_VREGS     = NUM_RCTX * VREGS_PER_RCTX;
    parameter PHYS_W         = 7;
    parameter WORDS          = 65536;
    parameter MEM_LATENCY    = 1;
    parameter RANDOM_STALL   = 0;
    parameter MEM_REORDER    = 0;
    parameter LSU_TAG_W      = 5;
    parameter HAS_SFU        = 1;
    // R40-04: external memory. Every address at or above LOCAL_LIMIT goes
    // through rtl/fpga/tnca_axi_bridge.v to an AXI memory of AXI_WORDS words
    // (tb/common/tnca_axi_mem_model.v, +axi_* plusargs for its timing), the
    // way the U280 image reaches HBM; the memory model keeps the code and
    // everything below LOCAL_LIMIT. 0: the device as it always was.
    parameter DATA_AXI       = 0;
    parameter LOCAL_LIMIT    = 64'h20000;
    parameter AXI_WORDS      = 1 << 22;
    parameter [63:0] AXI_BASE = 64'h40_0000_0000;
    // The burst mechanism (rtl/fpga/tnca_fpga_top.v AXI_LINE): 0 is one 8-byte
    // beat per access (tnca_axi_bridge), otherwise lines of that many beats
    // (tnca_axi_bridge_lb). TNCA_RTL_PARAMS sets them for an A/B.
    parameter AXI_LINE       = 0;
    parameter AXI_LB_N       = 8;
    parameter AXI_PF         = 1;
    parameter AXI_LOCAL_PIPE = 0;
    parameter AXI_RAM_CACHE  = 0;
    parameter AXI_WAIT_N     = 8;
    parameter AXI_OUT_N      = 8;
    // What the mailbox's range checks measure against.
    localparam [63:0] TOTAL_WORDS = DATA_AXI ? (LOCAL_LIMIT / 8 + AXI_WORDS) : WORDS;

    localparam [3:0] CMD_NOP     = 4'd0;
    localparam [3:0] CMD_LAUNCH  = 4'd1;
    localparam [3:0] CMD_QUIESCE = 4'd2;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    wire [63:0] imem_addr, imem_data;
    wire        mreq_valid, mreq_ready, mreq_we;
    wire [63:0] mreq_addr;
    wire [7:0]  mreq_be;
    wire        mreq_amo;
    wire [3:0]  mreq_aop;
    wire [63:0] mreq_cmp, mreq_wdata;
    wire        mrsp_valid;
    wire [63:0] mrsp_rdata;
    wire [LSU_TAG_W-1:0] mreq_tag, mrsp_tag;
    wire [1:0]  mreq_hint;

    //--------------------------------------------------------------------------
    // The shell's command port
    //--------------------------------------------------------------------------
    reg         cmd_valid = 1'b0;
    reg  [3:0]  cmd_op = CMD_NOP;
    reg  [63:0] cmd_pc = 64'd0;
    reg  [63:0] cmd_arg_base = 64'd0;
    reg  [NUM_RCTX-1:0] cmd_role_mask;
    reg  [3:0]          cmd_slice;
    reg  [31:0] cmd_pid_x = 0, cmd_pid_y = 0, cmd_pid_z = 0;
    reg  [31:0] cmd_npid_x = 1, cmd_npid_y = 1, cmd_npid_z = 1;
    reg  [63:0] cmd_deadline = 64'd0;
    wire        cmd_ready;

    wire        done_valid;
    reg         done_ready = 1'b0;
    wire [3:0]  done_status;
    wire [63:0] done_cycles, done_retired;
    wire [7:0]  done_fault_code;
    wire [63:0] done_fault_pc;
    wire [RCTX_W-1:0] done_fault_rctx;
    wire [23:0] done_fault_detail;

    reg         host_req_valid = 1'b0;
    reg         host_req_we = 1'b0;
    reg  [63:0] host_req_addr = 64'd0;
    reg  [7:0]  host_req_be = 8'hFF;
    reg  [63:0] host_req_wdata = 64'd0;
    wire        host_req_ready;
    wire        host_rsp_valid;
    wire [63:0] host_rsp_rdata;

    wire        busy, core_idle;
    wire [63:0] perf_cycles, perf_issued, perf_stall_hazard,
                perf_stall_lsu, perf_stall_wb, perf_squash;

    // PERF_READ (transport op 7). The index space is spec/r38/counters.json.
`include "tnca_perf_count.vh"
    reg  [4:0]  perf_sel = 5'd0;
    wire [63:0] perf_rdata;

    tnca_shell #(
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN), .ELEN(ELEN),
        .VREGS_PER_RCTX(VREGS_PER_RCTX), .ACC_N(ACC_N), .ACC_W(ACC_W), .BAR_N(BAR_N), .BAR_W(BAR_W), .BAR_EPOCH_W(BAR_EPOCH_W), .PHYS_VREGS(PHYS_VREGS),
        .PHYS_W(PHYS_W), .LSU_TAG_W(LSU_TAG_W), .HAS_SFU(HAS_SFU),
        .USE_BANKED_VRF(USE_BANKED_VRF),
        .VRF_BYPASS(VRF_BYPASS), .FE_NOBUBBLE(FE_NOBUBBLE),
        .VRF_PREFETCH(VRF_PREFETCH), .RCTX_SWITCH(RCTX_SWITCH),
        .LSU_INFLIGHT(LSU_INFLIGHT), .FPU_OVERLAP(FPU_OVERLAP), .RSLICE(RSLICE),
        .RPAIR(RPAIR), .MTX_ABUF(MTX_ABUF),
        .HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_op(cmd_op),
        .cmd_pc(cmd_pc), .cmd_arg_base(cmd_arg_base),
        .cmd_role_mask(cmd_role_mask), .cmd_slice(cmd_slice),
        .cmd_pid_x(cmd_pid_x), .cmd_pid_y(cmd_pid_y), .cmd_pid_z(cmd_pid_z),
        .cmd_npid_x(cmd_npid_x), .cmd_npid_y(cmd_npid_y),
        .cmd_npid_z(cmd_npid_z),
        .cmd_deadline(cmd_deadline),
        .done_valid(done_valid), .done_ready(done_ready),
        .done_status(done_status), .done_cycles(done_cycles),
        .done_retired(done_retired),
        .done_fault_code(done_fault_code), .done_fault_pc(done_fault_pc),
        .done_fault_rctx(done_fault_rctx),
        .done_fault_detail(done_fault_detail),
        .host_req_valid(host_req_valid), .host_req_ready(host_req_ready),
        .host_req_we(host_req_we), .host_req_addr(host_req_addr),
        .host_req_be(host_req_be), .host_req_wdata(host_req_wdata),
        .host_rsp_valid(host_rsp_valid), .host_rsp_rdata(host_rsp_rdata),
        .busy(busy), .core_idle(core_idle),
        .perf_cycles(perf_cycles), .perf_issued(perf_issued),
        .perf_stall_hazard(perf_stall_hazard),
        .perf_stall_lsu(perf_stall_lsu),
        .perf_stall_wb(perf_stall_wb), .perf_squash(perf_squash),
        .perf_sel(perf_sel), .perf_rdata(perf_rdata),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .mem_req_valid(mreq_valid), .mem_req_ready(mreq_ready),
        .mem_req_addr(mreq_addr), .mem_req_we(mreq_we),
        .mem_req_be(mreq_be), .mem_req_wdata(mreq_wdata),
        .mem_req_tag(mreq_tag), .mem_req_hint(mreq_hint),
        .mem_req_amo(mreq_amo), .mem_req_aop(mreq_aop),
        .mem_req_cmp(mreq_cmp),
        .mem_rsp_valid(mrsp_valid), .mem_rsp_tag(mrsp_tag),
        .mem_rsp_rdata(mrsp_rdata)
    );

    // A command was served (UPLOAD, DOWNLOAD, LAUNCH, ...): tnca_mmio's
    // axi_inv, for a bridge that buffers lines (tnca_axi_bridge_lb). Here the
    // host's transfers go through the bridge, so nothing depends on it; it is
    // raised as the board raises it so the two run the same RTL the same way.
    reg axi_inv = 1'b0;
    always @(posedge clk) if (axi_inv) axi_inv <= 1'b0;

    // The memory model's data port: the shell's, or the bridge's local side.
    wire        l_req_valid, l_req_ready, l_req_we, l_req_amo, l_rsp_valid;
    wire [63:0] l_req_addr, l_req_wdata, l_req_cmp, l_rsp_rdata;
    wire [7:0]  l_req_be;
    wire [3:0]  l_req_aop;
    wire [LSU_TAG_W-1:0] l_req_tag, l_rsp_tag;

    generate if (DATA_AXI) begin : g_axi
        wire awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready;
        wire rvalid, rready, rlast, wlast, axi_err, axi_stray;
        wire [63:0] awaddr, araddr, wdata, rdata;
        wire [7:0] wstrb, awlen, arlen;
        wire [2:0] awsize, arsize;
        wire [1:0] awburst, arburst, bresp, rresp;
        wire [LSU_TAG_W-1:0] awid, arid, bid, rid;
        if (AXI_LINE == 0) begin : g_one
        tnca_axi_bridge #(.TAG_W(LSU_TAG_W), .OUT_N(AXI_OUT_N), .LOCAL_LIMIT(LOCAL_LIMIT)) u_bridge (
            .clk(clk), .rst_n(rst_n),
            .req_valid(mreq_valid), .req_ready(mreq_ready), .req_addr(mreq_addr),
            .req_we(mreq_we), .req_be(mreq_be), .req_wdata(mreq_wdata), .req_tag(mreq_tag),
            .req_amo(mreq_amo), .req_aop(mreq_aop), .req_cmp(mreq_cmp),
            .rsp_valid(mrsp_valid), .rsp_tag(mrsp_tag), .rsp_rdata(mrsp_rdata),
            .l_req_valid(l_req_valid), .l_req_ready(l_req_ready), .l_req_addr(l_req_addr),
            .l_req_we(l_req_we), .l_req_be(l_req_be), .l_req_wdata(l_req_wdata),
            .l_req_tag(l_req_tag), .l_req_amo(l_req_amo), .l_req_aop(l_req_aop),
            .l_req_cmp(l_req_cmp),
            .l_rsp_valid(l_rsp_valid), .l_rsp_tag(l_rsp_tag), .l_rsp_rdata(l_rsp_rdata),
            .axi_base(AXI_BASE), .axi_size(64'd8 * AXI_WORDS), .inv(axi_inv), .axi_err(axi_err),
            .axi_stray(axi_stray),
            .cnt_rd(), .cnt_wr(), .cnt_rd_lat(), .cnt_rbeat(),
            .m_awvalid(awvalid), .m_awready(awready), .m_awaddr(awaddr), .m_awid(awid),
            .m_awlen(awlen), .m_awsize(awsize), .m_awburst(awburst),
            .m_wvalid(wvalid), .m_wready(wready), .m_wdata(wdata), .m_wstrb(wstrb),
            .m_wlast(wlast), .m_bvalid(bvalid), .m_bready(bready), .m_bid(bid), .m_bresp(bresp),
            .m_arvalid(arvalid), .m_arready(arready), .m_araddr(araddr), .m_arid(arid),
            .m_arlen(arlen), .m_arsize(arsize), .m_arburst(arburst),
            .m_rvalid(rvalid), .m_rready(rready), .m_rdata(rdata), .m_rid(rid),
            .m_rresp(rresp), .m_rlast(rlast)
        );
        end else begin : g_line
        tnca_axi_bridge_lb #(.TAG_W(LSU_TAG_W), .OUT_N(AXI_OUT_N), .LOCAL_LIMIT(LOCAL_LIMIT),
                .LINE_BEATS(AXI_LINE), .LB_N(AXI_LB_N), .WAIT_N(AXI_WAIT_N), .PF(AXI_PF), .LOCAL_PIPE(AXI_LOCAL_PIPE), .RAM_CACHE(AXI_RAM_CACHE)) u_bridge (
            .clk(clk), .rst_n(rst_n),
            .req_valid(mreq_valid), .req_ready(mreq_ready), .req_addr(mreq_addr),
            .req_we(mreq_we), .req_be(mreq_be), .req_wdata(mreq_wdata), .req_tag(mreq_tag),
            .req_amo(mreq_amo), .req_aop(mreq_aop), .req_cmp(mreq_cmp),
            .rsp_valid(mrsp_valid), .rsp_tag(mrsp_tag), .rsp_rdata(mrsp_rdata),
            .l_req_valid(l_req_valid), .l_req_ready(l_req_ready), .l_req_addr(l_req_addr),
            .l_req_we(l_req_we), .l_req_be(l_req_be), .l_req_wdata(l_req_wdata),
            .l_req_tag(l_req_tag), .l_req_amo(l_req_amo), .l_req_aop(l_req_aop),
            .l_req_cmp(l_req_cmp),
            .l_rsp_valid(l_rsp_valid), .l_rsp_tag(l_rsp_tag), .l_rsp_rdata(l_rsp_rdata),
            .axi_base(AXI_BASE), .axi_size(64'd8 * AXI_WORDS), .inv(axi_inv), .axi_err(axi_err),
            .axi_stray(axi_stray),
            .cnt_rd(), .cnt_wr(), .cnt_rd_lat(), .cnt_rbeat(),
            .m_awvalid(awvalid), .m_awready(awready), .m_awaddr(awaddr), .m_awid(awid),
            .m_awlen(awlen), .m_awsize(awsize), .m_awburst(awburst),
            .m_wvalid(wvalid), .m_wready(wready), .m_wdata(wdata), .m_wstrb(wstrb),
            .m_wlast(wlast), .m_bvalid(bvalid), .m_bready(bready), .m_bid(bid), .m_bresp(bresp),
            .m_arvalid(arvalid), .m_arready(arready), .m_araddr(araddr), .m_arid(arid),
            .m_arlen(arlen), .m_arsize(arsize), .m_arburst(arburst),
            .m_rvalid(rvalid), .m_rready(rready), .m_rdata(rdata), .m_rid(rid),
            .m_rresp(rresp), .m_rlast(rlast)
        );
        end
        tnca_axi_mem_model #(.WORDS(AXI_WORDS), .BASE(AXI_BASE), .ID_W(LSU_TAG_W)) u_axi_mem (
            .clk(clk), .rst_n(rst_n),
            .awvalid(awvalid), .awready(awready), .awaddr(awaddr), .awid(awid),
            .wvalid(wvalid), .wready(wready), .wdata(wdata), .wstrb(wstrb),
            .bvalid(bvalid), .bready(bready), .bid(bid), .bresp(bresp),
            .arvalid(arvalid), .arready(arready), .araddr(araddr), .arid(arid), .arlen(arlen),
            .rvalid(rvalid), .rready(rready), .rdata(rdata), .rid(rid), .rresp(rresp),
            .rlast(rlast)
        );
        // The model answers OKAY to everything, so either flag is a bridge bug.
        reg told = 1'b0;
        always @(posedge clk)
            if ((axi_err || axi_stray) && !told) begin
                $display("DEVICE AXI ERROR err=%0d stray=%0d", axi_err, axi_stray);
                told <= 1'b1;
            end
    end else begin : g_mem
        assign l_req_valid = mreq_valid;   assign mreq_ready = l_req_ready;
        assign l_req_addr  = mreq_addr;    assign l_req_we   = mreq_we;
        assign l_req_be    = mreq_be;      assign l_req_wdata = mreq_wdata;
        assign l_req_tag   = mreq_tag;     assign l_req_amo  = mreq_amo;
        assign l_req_aop   = mreq_aop;     assign l_req_cmp  = mreq_cmp;
        assign mrsp_valid  = l_rsp_valid;  assign mrsp_tag   = l_rsp_tag;
        assign mrsp_rdata  = l_rsp_rdata;
    end endgenerate

    tnca_mem_model #(
        .WORDS(WORDS), .MEM_DW(64), .MEM_BYTES(8), .TAG_W(LSU_TAG_W),
        .LATENCY(MEM_LATENCY), .RANDOM_STALL(RANDOM_STALL),
        .REORDER(MEM_REORDER)
    ) u_mem (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .req_valid(l_req_valid), .req_ready(l_req_ready),
        .req_addr(l_req_addr), .req_we(l_req_we),
        .req_be(l_req_be), .req_wdata(l_req_wdata), .req_tag(l_req_tag),
        .req_amo(l_req_amo), .req_aop(l_req_aop), .req_cmp(l_req_cmp),
        .rsp_valid(l_rsp_valid), .rsp_tag(l_rsp_tag), .rsp_rdata(l_rsp_rdata)
    );

    //==========================================================================
    // The mailbox
    //==========================================================================
    reg [1023:0] req_path, rsp_path, up_path, down_path, hexfile;
    integer pollcycles, deadline;
    reg do_trace;

    integer served;              // the highest sequence acted on
    integer fh, n, i, j, k;
    // 64-bit, not `integer`. An `integer` is 32 bits and signed, so a
    // host address of 0xFFFFFFF0 arrives as -16, and the range check below
    // then reads -16/8 + 1 = -1, which is not greater than WORDS -- an
    // address far outside memory passing the guard that exists to catch it.
    reg signed [63:0] seq, op, a0, a1, a2, a3, a4, a5;
    integer status, r0, r1;
    reg [63:0] word;
    reg settled;

    task write_response;
        input integer rseq;
        input integer rstatus;
        input integer rr0;
        input integer rr1;
        integer f;
        begin
            // Written in one go and closed, so a host that reads a partial
            // line sees nothing rather than half an answer.
            f = $fopen(rsp_path, "w");
            $fwrite(f, "%0d %0d %0d %0d\n", rseq, rstatus, rr0, rr1);
            $fclose(f);
        end
    endtask

    //--------------------------------------------------------------------------
    // One word through the host port.
    //
    // Everything here is handshake: wait for ready, hold valid for the one
    // edge that accepts it, wait for the response. The bench never assumes a
    // latency, so raising MEM_LATENCY or turning on RANDOM_STALL changes the
    // timing and not the outcome -- which is what R36-MEM-02 asks to be true.
    //--------------------------------------------------------------------------
    task host_xfer;
        input        we;
        input [63:0] addr;
        input [63:0] wdata;
        output [63:0] rdata;
        begin
            settled = 1'b0;
            while (host_req_ready !== 1'b1) begin @(posedge clk); #1; end
            host_req_valid = 1'b1;
            host_req_we    = we;
            host_req_addr  = addr;
            host_req_be    = 8'hFF;
            host_req_wdata = wdata;
            @(posedge clk);                 // ready was high: this is the accept
            #1 host_req_valid = 1'b0;
            rdata = 64'd0;
            while (settled !== 1'b1) begin
                if (host_rsp_valid === 1'b1) begin
                    rdata   = host_rsp_rdata;
                    settled = 1'b1;
                end else begin
                    @(posedge clk); #1;
                end
            end
        end
    endtask

    //--------------------------------------------------------------------------
    // One command, from issue to completion. Returns through `status`.
    //--------------------------------------------------------------------------
    task run_cmd;
        input [3:0]  cop;
        input [63:0] pc;
        input [63:0] argb;
        input integer mask;
        input integer px;
        input integer py;
        input integer pz;
        input integer nx;
        input integer ny;
        input integer nz;
        begin
            while (cmd_ready !== 1'b1) begin @(posedge clk); #1; end
            cmd_op        = cop;
            cmd_pc        = pc;
            cmd_arg_base  = argb;
            cmd_role_mask = mask[NUM_RCTX-1:0];
            cmd_slice     = {mask[28], mask[31:29]};   // {pair, slice, log2}
            cmd_pid_x = px;  cmd_pid_y = py;  cmd_pid_z = pz;
            cmd_npid_x = nx; cmd_npid_y = ny; cmd_npid_z = nz;
            cmd_deadline = deadline;
            cmd_valid = 1'b1;
            @(posedge clk);
            #1 cmd_valid = 1'b0;

            settled = 1'b0;
            while (settled !== 1'b1) begin
                if (done_valid === 1'b1) settled = 1'b1;
                else begin @(posedge clk); #1; end
            end

            status = done_status;
            case (done_status)
                4'd1: begin                          // FAULT
                    r0 = {24'd0, done_fault_code};
                    r1 = done_fault_pc[31:0];
                end
                4'd2: begin                          // TIMEOUT
                    r0 = done_cycles[31:0];
                    r1 = 0;
                end
                default: begin
                    r0 = done_cycles[31:0];
                    r1 = done_retired[31:0];
                end
            endcase

            done_ready = 1'b1;
            @(posedge clk);
            #1 done_ready = 1'b0;
        end
    endtask

    initial begin
        if (!$value$plusargs("req=%s", req_path)) begin
            $display("FATAL: pass +req=<file>"); $finish;
        end
        if (!$value$plusargs("rsp=%s", rsp_path)) begin
            $display("FATAL: pass +rsp=<file>"); $finish;
        end
        // Not `void'(...)`: that is SystemVerilog, and this compiles as
        // Verilog-2005 like the rest of the benches.
        if ($value$plusargs("up=%s", up_path)) ;
        if ($value$plusargs("down=%s", down_path)) ;
        if (!$value$plusargs("pollcycles=%d", pollcycles)) pollcycles = 20;
        if (!$value$plusargs("deadline=%d", deadline)) deadline = 2000000;
        do_trace = $test$plusargs("trace");

        if ($value$plusargs("hex=%s", hexfile))
            u_mem.load_hex(hexfile);

        cmd_role_mask = {{(NUM_RCTX-1){1'b0}}, 1'b1};
        cmd_slice     = 4'd0;

        served = 0;
        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // The handshake that proves the device came up. A host that never
        // sees this knows the difference between "slow" and "never started".
        $display("DEVICE READY words=%0d rctx=%0d", TOTAL_WORDS, NUM_RCTX);
        write_response(0, 0, 0, 0);

        forever begin
            repeat (pollcycles) @(posedge clk);
            #1;

            fh = $fopen(req_path, "r");
            if (fh != 0) begin
                n = $fscanf(fh, "%d %d %d %d %d %d %d %d",
                            seq, op, a0, a1, a2, a3, a4, a5);
                $fclose(fh);
                if (n == 8 && seq > served) begin
                    served = seq;
                    if (DATA_AXI) axi_inv <= 1'b1;
                    status = 0; r0 = 0; r1 = 0;
                    case (op)
                        1: begin                       // UPLOAD
                            if (a0 < 0 || a0 % 8 != 0 || a1 <= 0 ||
                                (a0/8 + a1) > TOTAL_WORDS) status = 3;
                            else begin
                                fh = $fopen(up_path, "r");
                                if (fh == 0) status = 3;
                                else begin
                                    for (j = 0; j < a1; j = j + 1) begin
                                        k = $fscanf(fh, "%h", word);
                                        if (k != 1) word = 64'd0;
                                        host_xfer(1'b1, a0 + j*8, word,
                                                  word);
                                    end
                                    $fclose(fh);
                                end
                            end
                        end
                        2: begin                       // DOWNLOAD
                            if (a0 < 0 || a0 % 8 != 0 || a1 <= 0 ||
                                (a0/8 + a1) > TOTAL_WORDS) status = 3;
                            else begin
                                fh = $fopen(down_path, "w");
                                for (j = 0; j < a1; j = j + 1) begin
                                    host_xfer(1'b0, a0 + j*8, 64'd0, word);
                                    $fwrite(fh, "%016x\n", word);
                                end
                                $fclose(fh);
                            end
                        end
                        // a3 = pid_x | pid_y<<16, a4 = pid_z | npid_x<<16,
                        // a5 = npid_y | npid_z<<16. Sixteen bits per axis is
                        // ample: the contract caps a grid at 4096 programs.
                        3: run_cmd(CMD_LAUNCH, a0, a1, a2,
                                   a3 & 32'hFFFF, (a3 >> 16) & 32'hFFFF,
                                   a4 & 32'hFFFF, (a4 >> 16) & 32'hFFFF,
                                   a5 & 32'hFFFF, (a5 >> 16) & 32'hFFFF);
                        // R40 negotiated extension; legacy opcode 3 is unchanged.
                        8: begin r0 = 32; r1 = 16777216; end
                        9: run_cmd(CMD_LAUNCH, a0, a1, a2,
                                   a3[31:0], a3[63:32],
                                   a4[31:0], a4[63:32],
                                   a5[31:0], a5[63:32]);
                        4: run_cmd(CMD_QUIESCE, 64'd0, 64'd0, 0,
                                   0, 0, 0, 1, 1, 1);
                        5: begin                       // SHUTDOWN
                            write_response(seq, 0, 0, 0);
                            $display("DEVICE SHUTDOWN after %0d requests", served);
                            $finish;
                        end
                        6: begin                       // PING
                            r0 = served;
                        end
                        // PERF_READ: r0 = low 32 bits, r1 = high 32 bits of
                        // counter a0. Read only between commands -- the
                        // mailbox is served only when no command is running
                        // -- so a read never lands inside a launch's window.
                        7: begin
                            if (a0 < 0 || a0 >= PERF_NUM_COUNTERS) status = 3;
                            else begin
                                perf_sel = a0[4:0];
                                #1;
                                r0 = perf_rdata[31:0];
                                r1 = perf_rdata[63:32];
                            end
                        end
                        default: status = 3;
                    endcase
                    if (do_trace)
                        $display("[dev] seq=%0d op=%0d a0=%0d a1=%0d -> status=%0d r0=%0d", seq, op, a0, a1, status, r0);
                    write_response(seq, status, r0, r1);
                end
            end
        end
    end

endmodule
