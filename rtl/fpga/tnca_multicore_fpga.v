// Coherent register binding and persistent physical-grid AXI shell.
// One allocated global memory window is visible through all AXI masters.
// Platform supplies AXI-Lite crossing, HBM connectivity, resets and placement.
// DMA capability requires the optional ownership barrier and physical drain.
`default_nettype none
module tnca_multicore_fpga #(
 parameter DCO_REGISTERED_RANGES=0,
    parameter NUCA_REGISTER_VICTIM_SELECT = 0,
    parameter NUCA_CACHE_DATA_PIPELINE = 0,
    parameter NUCA_SHARED_DATA_READ = 0,   // R41 G2: one LUTRAM read port per slice line array
    parameter NUCA_REGISTER_VICTIM_RANK = 0,
    parameter DCO_TLW_PIPE_STAGES = 0,
    parameter DCO_REGISTER_WRITES = 0,
    parameter DCO_LOCAL_REPLICAS = 0,MEMORY_CONTROLS=0,
 parameter NX=3,NY=1,N=NX*NY,CODE_WORDS=16384,CODE_LINK_STAGES=0,MEM_PORTS=1,NOC_LINK_STAGES=0,DISPATCH_STAGES=0,MEM_HOME_STAGES=0,
 parameter SRC_W=(N>4?$clog2(N):2),
 parameter NUM_RCTX=4,RCTX_W=2,PHYS_VREGS=256,PHYS_W=8,
 parameter VREGS_PER_RCTX=64,ACC_N=16,ACC_W=4,BAR_N=32,BAR_W=5,BAR_EPOCH_W=48,PHYS_ADDR_W=64,AUTO_VBASE=1,
 parameter HAS_SFU=1,HAS_ATOMICS=1,HAS_SYS_ATOMICS=0,USE_BANKED_VRF=1,VRF_BYPASS=1,FE_NOBUBBLE=1,
 parameter VRF_PREFETCH=1,RCTX_SWITCH=2,MTX_ABUF=1,MTX_VRF_REG=0,LSU_INFLIGHT=2,
 parameter HYBRID_LSU=0,SFU_LANES=1,HOME_BACKGROUND_READS=0,HOME_LINE_WRITE_NOFILL=0,HOME_LINE_PIPE=0,TENSOR_LINE=0,HOME_MSHR_N=8,LINE_SLOTS=8,
 parameter FPU_OVERLAP=1,RSLICE=1,RPAIR=1,HAS_TENSOR_XPOSE=1,
 parameter SETS=64,SET_W=6,WAYS=8,WAY_W=3,LINK_STAGES=2,AXI_DATA_W=256,AXI_OUT_N=8,AXI_PIPELINED_ISSUE=0,AXI_LINE_LUTRAM=0,HBM_STRIPED=0,
 parameter AXI_ID_W=(AXI_OUT_N>1?$clog2(AXI_OUT_N):1),MEM_BASE=64'h20000,REG_ADDR_W=8,CACHE_MAINTENANCE=0
)(
 input wire clk,rst_n,
 input wire [REG_ADDR_W-1:0] reg_addr,input wire [63:0] reg_wdata,
 input wire reg_we,reg_re,output wire [63:0] reg_rdata,
 output wire [MEM_PORTS-1:0] axi_fault,axi_idle,link_idle,
 output wire [MEM_PORTS*64-1:0] axi_read_lines,axi_write_lines,axi_read_beats,axi_write_beats,
 output wire [MEM_PORTS-1:0] m_awvalid,
 input wire [MEM_PORTS-1:0] m_awready,
 output wire [MEM_PORTS*64-1:0] m_awaddr,
 output wire [MEM_PORTS*AXI_ID_W-1:0] m_awid,
 output wire [MEM_PORTS*8-1:0] m_awlen,
 output wire [MEM_PORTS*3-1:0] m_awsize,
 output wire [MEM_PORTS*2-1:0] m_awburst,
 output wire [MEM_PORTS-1:0] m_wvalid,
 input wire [MEM_PORTS-1:0] m_wready,
 output wire [MEM_PORTS*AXI_DATA_W-1:0] m_wdata,
 output wire [MEM_PORTS*AXI_DATA_W/8-1:0] m_wstrb,
 output wire [MEM_PORTS-1:0] m_wlast,
 input wire [MEM_PORTS-1:0] m_bvalid,
 output wire [MEM_PORTS-1:0] m_bready,
 input wire [MEM_PORTS*AXI_ID_W-1:0] m_bid,
 input wire [MEM_PORTS*2-1:0] m_bresp,
 output wire [MEM_PORTS-1:0] m_arvalid,
 input wire [MEM_PORTS-1:0] m_arready,
 output wire [MEM_PORTS*64-1:0] m_araddr,
 output wire [MEM_PORTS*AXI_ID_W-1:0] m_arid,
 output wire [MEM_PORTS*8-1:0] m_arlen,
 output wire [MEM_PORTS*3-1:0] m_arsize,
 output wire [MEM_PORTS*2-1:0] m_arburst,
 input wire [MEM_PORTS-1:0] m_rvalid,
 output wire [MEM_PORTS-1:0] m_rready,
 input wire [MEM_PORTS*AXI_DATA_W-1:0] m_rdata,
 input wire [MEM_PORTS*AXI_ID_W-1:0] m_rid,
 input wire [MEM_PORTS*2-1:0] m_rresp,
 input wire [MEM_PORTS-1:0] m_rlast
);
wire cmd_valid;
 wire cache_maint_valid,cache_maint_ready,cache_maint_done;
 wire [63:0] cache_maint_base,cache_maint_size;
 wire cmd_ready;
 wire [63:0] cmd_id,cmd_pc,cmd_arg_base,cmd_arg_stride,cmd_deadline;
 wire [31:0] cmd_gx,cmd_gy,cmd_gz;
 wire [1:0] cmd_group_log2;
 wire [N-1:0] cmd_cores;
 wire [NUM_RCTX-1:0] cmd_roles;
 wire [3:0] cmd_slice;
 wire [NUM_RCTX*PHYS_W-1:0] cmd_vbase;
 wire done_valid;
 wire done_ready;
 wire [3:0] done_status;
 wire [63:0] done_id,done_cycles;
 wire [(N>1?$clog2(N):1)-1:0] fault_core;
 wire [N*32-1:0] issued,completed;
 wire [31:0] peak_active;
 wire [N*64-1:0] active_cycles;
 wire host_valid;
 wire host_ready;
 wire host_we;
 wire [63:0] host_addr,host_wdata;
 wire [7:0] host_be;
 wire host_rsp_valid;
 wire [63:0] host_rsp_data;
 wire [3:0] host_rsp_status;
 wire busy;
 wire all_idle;
 wire timed_out;
 wire [N-1:0] fault_valid;
 wire [N*8-1:0] fault_code;
 wire [N*64-1:0] fault_pc;
 wire [N*24-1:0] fault_detail;
 wire [7:0] memory_active_ports,memory_outstanding;
 wire [63:0] base,size;wire [MEM_PORTS*64-1:0] axi_base,axi_size;
 generate for(genvar p=0;p<MEM_PORTS;p=p+1)begin:g_window
  assign axi_base[p*64+:64]=base;assign axi_size[p*64+:64]=size;
 end endgenerate
 assign cmd_vbase=0;
 tnca_multicore_axi #(
  .MEMORY_CONTROLS(MEMORY_CONTROLS),
  .CACHE_MAINTENANCE(CACHE_MAINTENANCE), .DCO_REGISTERED_RANGES(DCO_REGISTERED_RANGES), .NUCA_REGISTER_VICTIM_SELECT(NUCA_REGISTER_VICTIM_SELECT), .NUCA_CACHE_DATA_PIPELINE(NUCA_CACHE_DATA_PIPELINE), .NUCA_SHARED_DATA_READ(NUCA_SHARED_DATA_READ), .NUCA_REGISTER_VICTIM_RANK(NUCA_REGISTER_VICTIM_RANK), .DCO_TLW_PIPE_STAGES(DCO_TLW_PIPE_STAGES), .DCO_REGISTER_WRITES(DCO_REGISTER_WRITES), .DCO_LOCAL_REPLICAS(DCO_LOCAL_REPLICAS),
  .NX(NX),
  .NY(NY),
  .N(N),
  .CODE_WORDS(CODE_WORDS),
  .CODE_LINK_STAGES(CODE_LINK_STAGES),
  .MEM_PORTS(MEM_PORTS),
  .NOC_LINK_STAGES(NOC_LINK_STAGES),
  .DISPATCH_STAGES(DISPATCH_STAGES),
  .MEM_HOME_STAGES(MEM_HOME_STAGES),
  .SRC_W(SRC_W),
  .NUM_RCTX(NUM_RCTX),
  .RCTX_W(RCTX_W),
  .PHYS_VREGS(PHYS_VREGS),
  .PHYS_W(PHYS_W),
  .VREGS_PER_RCTX(VREGS_PER_RCTX), .ACC_N(ACC_N), .ACC_W(ACC_W), .BAR_N(BAR_N), .BAR_W(BAR_W), .BAR_EPOCH_W(BAR_EPOCH_W), .PHYS_ADDR_W(PHYS_ADDR_W),
  .AUTO_VBASE(AUTO_VBASE),
  .HAS_SFU(HAS_SFU),
  .HAS_ATOMICS(HAS_ATOMICS),
  .HAS_SYS_ATOMICS(HAS_SYS_ATOMICS),
  .USE_BANKED_VRF(USE_BANKED_VRF),
  .VRF_BYPASS(VRF_BYPASS),
  .FE_NOBUBBLE(FE_NOBUBBLE),
  .VRF_PREFETCH(VRF_PREFETCH),
  .RCTX_SWITCH(RCTX_SWITCH),
  .MTX_ABUF(MTX_ABUF),
  .MTX_VRF_REG(MTX_VRF_REG),
  .LSU_INFLIGHT(LSU_INFLIGHT),
  .HYBRID_LSU(HYBRID_LSU),.SFU_LANES(SFU_LANES),.HOME_BACKGROUND_READS(HOME_BACKGROUND_READS),.HOME_LINE_WRITE_NOFILL(HOME_LINE_WRITE_NOFILL),.HOME_LINE_PIPE(HOME_LINE_PIPE),.TENSOR_LINE(TENSOR_LINE),.HOME_MSHR_N(HOME_MSHR_N),.LINE_SLOTS(LINE_SLOTS),
  .FPU_OVERLAP(FPU_OVERLAP),
  .RSLICE(RSLICE),
  .RPAIR(RPAIR),
  .HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE),
  .SETS(SETS),
  .SET_W(SET_W),
  .WAYS(WAYS),
  .WAY_W(WAY_W),
  .LINK_STAGES(LINK_STAGES),
  .AXI_DATA_W(AXI_DATA_W),
  .AXI_OUT_N(AXI_OUT_N),
  .AXI_PIPELINED_ISSUE(AXI_PIPELINED_ISSUE),
  .AXI_LINE_LUTRAM(AXI_LINE_LUTRAM),
  .HBM_STRIPED(HBM_STRIPED),
  .AXI_ID_W(AXI_ID_W),
  .MEM_BASE(MEM_BASE)
 ) system(
  .memory_active_ports(memory_active_ports),.memory_outstanding(memory_outstanding),
  .clk(clk),
  .rst_n(rst_n),
  .cache_maint_valid(cache_maint_valid),.cache_maint_base(cache_maint_base),.cache_maint_size(cache_maint_size),
  .cache_maint_ready(cache_maint_ready),.cache_maint_done(cache_maint_done),
  .cmd_valid(cmd_valid),
  .cmd_ready(cmd_ready),
  .cmd_id(cmd_id),
  .cmd_pc(cmd_pc),
  .cmd_arg_base(cmd_arg_base),
  .cmd_arg_stride(cmd_arg_stride),
  .cmd_deadline(cmd_deadline),
  .cmd_gx(cmd_gx),
  .cmd_gy(cmd_gy),
  .cmd_gz(cmd_gz),
  .cmd_group_log2(cmd_group_log2),
  .cmd_cores(cmd_cores),
  .cmd_roles(cmd_roles),
  .cmd_slice(cmd_slice),
  .cmd_vbase(cmd_vbase),
  .done_valid(done_valid),
  .done_ready(done_ready),
  .done_status(done_status),
  .done_id(done_id),
  .done_cycles(done_cycles),
  .fault_core(fault_core),
  .issued(issued),
  .completed(completed),
  .peak_active(peak_active),
  .active_cycles(active_cycles),
  .host_valid(host_valid),
  .host_ready(host_ready),
  .host_we(host_we),
  .host_addr(host_addr),
  .host_wdata(host_wdata),
  .host_be(host_be),
  .host_rsp_valid(host_rsp_valid),
  .host_rsp_data(host_rsp_data),
  .host_rsp_status(host_rsp_status),
  .busy(busy),
  .all_idle(all_idle),
  .timed_out(timed_out),
  .fault_valid(fault_valid),
  .fault_code(fault_code),
  .fault_pc(fault_pc),
  .fault_detail(fault_detail),
  .axi_base(axi_base),
  .axi_size(axi_size),
  .axi_fault(axi_fault),
  .axi_idle(axi_idle),
  .link_idle(link_idle),
  .axi_read_lines(axi_read_lines),
  .axi_write_lines(axi_write_lines),
  .axi_read_beats(axi_read_beats),
  .axi_write_beats(axi_write_beats),
  .m_awvalid(m_awvalid),
  .m_awready(m_awready),
  .m_awaddr(m_awaddr),
  .m_awid(m_awid),
  .m_awlen(m_awlen),
  .m_awsize(m_awsize),
  .m_awburst(m_awburst),
  .m_wvalid(m_wvalid),
  .m_wready(m_wready),
  .m_wdata(m_wdata),
  .m_wstrb(m_wstrb),
  .m_wlast(m_wlast),
  .m_bvalid(m_bvalid),
  .m_bready(m_bready),
  .m_bid(m_bid),
  .m_bresp(m_bresp),
  .m_arvalid(m_arvalid),
  .m_arready(m_arready),
  .m_araddr(m_araddr),
  .m_arid(m_arid),
  .m_arlen(m_arlen),
  .m_arsize(m_arsize),
  .m_arburst(m_arburst),
  .m_rvalid(m_rvalid),
  .m_rready(m_rready),
  .m_rdata(m_rdata),
  .m_rid(m_rid),
  .m_rresp(m_rresp),
  .m_rlast(m_rlast)
 );
 tnca_multicore_mmio #(.HBM_STRIPED(HBM_STRIPED),.MEMORY_CONTROLS(MEMORY_CONTROLS),.AXI_OUT_N(AXI_OUT_N),.CACHE_MAINTENANCE(CACHE_MAINTENANCE),.N(N),.PORTS(MEM_PORTS),.NUM_RCTX(NUM_RCTX),.PHYS_W(PHYS_W),.CODE_WORDS(CODE_WORDS),.MEM_BASE(MEM_BASE),.REG_ADDR_W(REG_ADDR_W)) control(
  .memory_active_ports(memory_active_ports),.memory_outstanding(memory_outstanding),
  .clk(clk),
  .rst_n(rst_n),
  .cache_maint_valid(cache_maint_valid),.cache_maint_base(cache_maint_base),.cache_maint_size(cache_maint_size),
  .cache_maint_ready(cache_maint_ready),.cache_maint_done(cache_maint_done),
  .reg_addr(reg_addr),
  .reg_wdata(reg_wdata),
  .reg_we(reg_we),
  .reg_re(reg_re),
  .reg_rdata(reg_rdata),
  .cmd_valid(cmd_valid),
  .cmd_ready(cmd_ready),
  .cmd_id(cmd_id),
  .cmd_pc(cmd_pc),
  .cmd_arg_base(cmd_arg_base),
  .cmd_arg_stride(cmd_arg_stride),
  .cmd_deadline(cmd_deadline),
  .cmd_gx(cmd_gx),
  .cmd_gy(cmd_gy),
  .cmd_gz(cmd_gz),
  .cmd_group_log2(cmd_group_log2),
  .cmd_cores(cmd_cores),
  .cmd_roles(cmd_roles),
  .cmd_slice(cmd_slice),
  .done_valid(done_valid),
  .done_ready(done_ready),
  .done_status(done_status),
  .done_id(done_id),
  .done_cycles(done_cycles),
  .issued(issued),
  .completed(completed),
  .peak_active(peak_active),
  .active_cycles(active_cycles),
  .host_valid(host_valid),
  .host_ready(host_ready),
  .host_we(host_we),
  .host_addr(host_addr),
  .host_wdata(host_wdata),
  .host_be(host_be),
  .host_rsp_valid(host_rsp_valid),
  .host_rsp_data(host_rsp_data),
  .host_rsp_status(host_rsp_status),
  .all_idle(all_idle),
  .axi_fault(axi_fault),
  .memory_reads(axi_read_lines),
  .memory_writes(axi_write_lines),
  .axi_base(base),
  .axi_size(size)
 );
endmodule
`default_nettype wire
