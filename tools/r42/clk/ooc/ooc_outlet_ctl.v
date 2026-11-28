// R42 clk: kernel-slice screen of the outlet control cone that the run-3 census flagged twice (HBM adapter enables, dispatch counters): one HBM outlet
// (home link -> tnca_line_axi adapter, as tnca_multicore_axi g_outlet), the memory_drained / launch / done handshake exactly as tnca_multicore_axi writes it
// (CLKFIX[13] IDLE_SRC selects the registered form, CLKFIX[12] is the adapter's REQ_REG), and the three-core tnca_grid_dispatch with its counters
// (done_cycles, active_cycles, issued, completed). The other two outlets and the shell's all-idle are one registered input each (i_other_idle, i_shell_idle).
// Every other port is registered in the wrapper (inputs i_*, outputs hashed into o_*), so neighbours enter as plain registers as in the tile wrappers.
`default_nettype none
module ooc_outlet_ctl #(
    parameter CLKFIX = 0,
    parameter CORE_N = 3,
    parameter SRC_W = 4,
    parameter TAG_W = 8,
    parameter OUT_N = 32,
    parameter ID_W = 5,
    parameter STAGES = 2
) (
    input wire clk,
    input wire i_rst_n,
    // home link request side
    input wire i_s_req_valid, input wire [63:0] i_s_req_addr, input wire i_s_req_we, input wire [511:0] i_s_req_wdata,
    input wire [SRC_W-1:0] i_s_req_src, input wire [TAG_W-1:0] i_s_req_tag,
    output reg o_s_req_ready,
    output reg [1+512+SRC_W+TAG_W-1:0] o_s_rsp,
    // AXI side of the adapter, one vector each way
    input wire [1+1+1+ID_W+2+1+ID_W+2+1+512+ID_W+2+1-1:0] i_ax,
    output reg [1+64+ID_W+8+3+2+1+512+64+1+1+1+64+ID_W+8+3+2+1-1:0] o_ax,
    // neighbours
    input wire i_other_idle, input wire i_shell_idle, input wire [7:0] i_limit, input wire [63:0] i_base, i_size,
    // launch / done handshake and dispatch
    input wire i_launch_valid, input wire i_abort_valid, input wire [3:0] i_abort_status,
    input wire [63:0] i_launch_id, i_launch_pc, i_launch_arg_base, i_launch_arg_stride,
    input wire [31:0] i_launch_gx, i_launch_gy, i_launch_gz,
    input wire [CORE_N-1:0] i_launch_cores, input wire [1:0] i_launch_group_log2,
    input wire [CORE_N-1:0] i_cmd_ready, i_core_done, i_core_abort, input wire [CORE_N*4-1:0] i_core_status,
    input wire i_done_ready,
    output reg [3:0] o_flags,
    output reg [63:0] o_done_id, o_done_cycles,
    output reg [CORE_N*32-1:0] o_issued, o_completed,
    output reg [CORE_N*64-1:0] o_active_cycles,
    output reg [CORE_N-1:0] o_cmd_valid
);
    // ---- registered inputs ----------------------------------------------------------------------------------------------------------------------
    reg r_rst_n; always @(posedge clk) r_rst_n <= i_rst_n;
    reg r_s_req_valid; reg [63:0] r_s_req_addr; reg r_s_req_we; reg [511:0] r_s_req_wdata; reg [SRC_W-1:0] r_s_req_src; reg [TAG_W-1:0] r_s_req_tag;
    always @(posedge clk) begin r_s_req_valid <= i_s_req_valid; r_s_req_addr <= i_s_req_addr; r_s_req_we <= i_s_req_we; r_s_req_wdata <= i_s_req_wdata;
        r_s_req_src <= i_s_req_src; r_s_req_tag <= i_s_req_tag; end
    reg [$bits(i_ax)-1:0] r_ax; always @(posedge clk) r_ax <= i_ax;
    reg r_other_idle, r_shell_idle; reg [7:0] r_limit; reg [63:0] r_base, r_size;
    always @(posedge clk) begin r_other_idle <= i_other_idle; r_shell_idle <= i_shell_idle; r_limit <= i_limit; r_base <= i_base; r_size <= i_size; end
    reg r_launch_valid, r_abort_valid; reg [3:0] r_abort_status; reg [63:0] r_launch_id, r_launch_pc, r_launch_arg_base, r_launch_arg_stride;
    reg [31:0] r_launch_gx, r_launch_gy, r_launch_gz; reg [CORE_N-1:0] r_launch_cores; reg [1:0] r_launch_group_log2;
    reg [CORE_N-1:0] r_cmd_ready, r_core_done, r_core_abort; reg [CORE_N*4-1:0] r_core_status; reg r_done_ready;
    always @(posedge clk) begin
        r_launch_valid <= i_launch_valid; r_abort_valid <= i_abort_valid; r_abort_status <= i_abort_status; r_launch_id <= i_launch_id; r_launch_pc <= i_launch_pc;
        r_launch_arg_base <= i_launch_arg_base; r_launch_arg_stride <= i_launch_arg_stride; r_launch_gx <= i_launch_gx; r_launch_gy <= i_launch_gy;
        r_launch_gz <= i_launch_gz; r_launch_cores <= i_launch_cores; r_launch_group_log2 <= i_launch_group_log2; r_cmd_ready <= i_cmd_ready;
        r_core_done <= i_core_done; r_core_abort <= i_core_abort; r_core_status <= i_core_status; r_done_ready <= i_done_ready;
    end
    // ---- the outlet: link and adapter, as tnca_multicore_axi g_outlet --------------------------------------------------------------------------
    wire qv, qr, qwe, rv; wire [63:0] qa; wire [511:0] qd, rd; wire [SRC_W-1:0] qs, rs; wire [TAG_W-1:0] qt, rt;
    wire link_idle, axi_idle, adapter_fault, s_req_ready, s_rsp_valid; wire [511:0] s_rsp_rdata; wire [SRC_W-1:0] s_rsp_src; wire [TAG_W-1:0] s_rsp_tag;
    tnca_line_link #(.SRC_W(SRC_W), .TAG_W(TAG_W), .STAGES(STAGES)) link (
        .clk(clk), .rst_n(r_rst_n), .idle(link_idle),
        .s_req_valid(r_s_req_valid), .s_req_ready(s_req_ready), .s_req_addr(r_s_req_addr), .s_req_we(r_s_req_we), .s_req_wdata(r_s_req_wdata),
        .s_req_src(r_s_req_src), .s_req_tag(r_s_req_tag),
        .s_rsp_valid(s_rsp_valid), .s_rsp_rdata(s_rsp_rdata), .s_rsp_src(s_rsp_src), .s_rsp_tag(s_rsp_tag),
        .m_req_valid(qv), .m_req_ready(qr), .m_req_addr(qa), .m_req_we(qwe), .m_req_wdata(qd), .m_req_src(qs), .m_req_tag(qt),
        .m_rsp_valid(rv), .m_rsp_rdata(rd), .m_rsp_src(rs), .m_rsp_tag(rt));
    wire m_awvalid; wire m_awready = r_ax[0]; wire [63:0] m_awaddr; wire [ID_W-1:0] m_awid; wire [7:0] m_awlen; wire [2:0] m_awsize; wire [1:0] m_awburst;
    wire m_wvalid; wire m_wready = r_ax[1]; wire [511:0] m_wdata; wire [63:0] m_wstrb; wire m_wlast;
    wire m_bvalid = r_ax[2]; wire m_bready; wire [ID_W-1:0] m_bid = r_ax[3+:ID_W]; wire [1:0] m_bresp = r_ax[3+ID_W+:2];
    wire m_arvalid; wire m_arready = r_ax[3+ID_W+2]; wire [63:0] m_araddr; wire [ID_W-1:0] m_arid; wire [7:0] m_arlen; wire [2:0] m_arsize; wire [1:0] m_arburst;
    wire m_rvalid = r_ax[4+ID_W+2]; wire m_rready; wire [511:0] m_rdata = r_ax[5+ID_W+2+:512]; wire [ID_W-1:0] m_rid = r_ax[5+ID_W+2+512+:ID_W];
    wire [1:0] m_rresp = r_ax[5+2*ID_W+2+512+:2]; wire m_rlast = r_ax[5+2*ID_W+4+512];
    wire [63:0] rl, wl, rb, wb;
    tnca_line_axi #(.HBM_STRIPED(1), .PIPELINED_ISSUE(1), .LINE_LUTRAM(1), .HAZARD_ADDR_W(35), .SRC_W(SRC_W), .TAG_W(TAG_W), .DATA_W(512), .OUT_N(OUT_N),
        .DYNAMIC_LIMIT(1), .ID_W(ID_W), .MEM_BASE(64'h20000), .REQ_REG((CLKFIX>>12)&1)) adapter (
        .clk(clk), .rst_n(r_rst_n), .admission_limit(r_limit), .req_valid(qv), .req_ready(qr), .req_addr(qa), .req_we(qwe), .req_wdata(qd), .req_src(qs), .req_tag(qt),
        .rsp_valid(rv), .rsp_rdata(rd), .rsp_src(rs), .rsp_tag(rt), .axi_base(r_base), .axi_size(r_size), .fault(adapter_fault), .idle(axi_idle),
        .read_lines(rl), .write_lines(wl), .read_beats(rb), .write_beats(wb),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awid(m_awid), .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bid(m_bid), .m_bresp(m_bresp),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_araddr(m_araddr), .m_arid(m_arid), .m_arlen(m_arlen), .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rdata(m_rdata), .m_rid(m_rid), .m_rresp(m_rresp), .m_rlast(m_rlast));
    // ---- memory_drained and the launch / done handshake: tnca_multicore_axi lines, with the other outlets and the shell as registered inputs ----------
    wire shell_cmd_ready, shell_done_valid;
    wire memory_drained = axi_idle && link_idle && r_other_idle;
    wire idle_src = ((CLKFIX >> 13) & 1) != 0;
    reg md_q, idle_q;
    always @(posedge clk) begin
        if (!r_rst_n) begin md_q <= 0; idle_q <= 0; end
        else begin md_q <= memory_drained; idle_q <= memory_drained && r_shell_idle; end
    end
    wire md_eff = idle_src ? md_q : memory_drained;
    wire cmd_ready_out = shell_cmd_ready && md_eff;
    wire done_valid_out = idle_src ? (shell_done_valid && idle_q) : (shell_done_valid && memory_drained && r_shell_idle);
    wire all_idle_out = idle_src ? idle_q : (r_shell_idle && memory_drained);
    wire done_ready_in = r_done_ready && (idle_src ? idle_q : (memory_drained && r_shell_idle));
    wire [CORE_N-1:0] cmd_valid; wire [63:0] cmd_id, cmd_pc; wire [CORE_N*64-1:0] cmd_arg_base; wire [CORE_N*32-1:0] cmd_x, cmd_y, cmd_z;
    wire [31:0] cmd_nx, cmd_ny, cmd_nz; wire [CORE_N-1:0] cmd_cancel, core_busy; wire [3:0] done_status; wire [63:0] done_id, done_cycles;
    wire [$clog2(CORE_N)-1:0] fault_core; wire [CORE_N*32-1:0] issued, completed; wire [31:0] peak_active; wire [CORE_N*64-1:0] active_cycles;
    tnca_grid_dispatch #(.CORE_N(CORE_N)) disp (
        .clk(clk), .rst_n(r_rst_n), .launch_valid(r_launch_valid && md_eff), .abort_valid(r_abort_valid), .abort_status(r_abort_status),
        .launch_ready(shell_cmd_ready), .launch_id(r_launch_id), .launch_pc(r_launch_pc), .launch_arg_base(r_launch_arg_base), .launch_arg_stride(r_launch_arg_stride),
        .launch_gx(r_launch_gx), .launch_gy(r_launch_gy), .launch_gz(r_launch_gz), .launch_cores(r_launch_cores), .launch_group_log2(r_launch_group_log2),
        .cmd_valid(cmd_valid), .cmd_ready(r_cmd_ready), .cmd_id(cmd_id), .cmd_pc(cmd_pc), .cmd_arg_base(cmd_arg_base), .cmd_x(cmd_x), .cmd_y(cmd_y), .cmd_z(cmd_z),
        .cmd_nx(cmd_nx), .cmd_ny(cmd_ny), .cmd_nz(cmd_nz), .core_done(r_core_done), .core_abort(r_core_abort), .core_status(r_core_status), .cmd_cancel(cmd_cancel),
        .done_valid(shell_done_valid), .done_ready(done_ready_in), .done_status(done_status), .done_id(done_id), .done_cycles(done_cycles), .fault_core(fault_core),
        .core_busy(core_busy), .issued(issued), .completed(completed), .peak_active(peak_active), .active_cycles(active_cycles));
    // ---- registered outputs -----------------------------------------------------------------------------------------------------------------------
    always @(posedge clk) begin
        o_s_req_ready <= s_req_ready; o_s_rsp <= {s_rsp_valid, s_rsp_rdata, s_rsp_src, s_rsp_tag};
        o_ax <= {m_awvalid, m_awaddr, m_awid, m_awlen, m_awsize, m_awburst, m_wvalid, m_wdata, m_wstrb, m_wlast, m_bready, m_arvalid, m_araddr, m_arid, m_arlen, m_arsize, m_arburst};
        o_flags <= {cmd_ready_out, done_valid_out, all_idle_out, adapter_fault};
        o_done_id <= done_id ^ cmd_id ^ cmd_pc ^ {32'd0, cmd_nx} ^ {32'd0, cmd_ny} ^ {32'd0, cmd_nz} ^ {30'd0, fault_core, done_status, peak_active[15:0]} ^ {m_rready, 63'd0} ^ rl ^ wl ^ rb ^ wb;
        o_done_cycles <= done_cycles; o_issued <= issued; o_completed <= completed; o_active_cycles <= active_cycles;
        o_cmd_valid <= cmd_valid ^ cmd_cancel ^ core_busy ^ cmd_x[CORE_N-1:0] ^ cmd_y[CORE_N-1:0] ^ cmd_z[CORE_N-1:0] ^ cmd_arg_base[CORE_N-1:0];
    end
endmodule
