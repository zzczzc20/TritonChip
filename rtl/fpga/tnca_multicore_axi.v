// Public persistent multicore shell with independent AXI line outlets.
// Platform supplies control registers, clock/reset and placement constraints.
// This module alone is not a deployable board image.
`default_nettype none
module tnca_multicore_axi #(
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
 parameter VREGS_PER_RCTX=64,ACC_N=16,ACC_W=4,BAR_N=32,BAR_W=5,BAR_EPOCH_W=48,FMA_IMPL=0,MAC_DP=0,MAC_CUTS=14,PHYS_ADDR_W=64,AUTO_VBASE=1,
 parameter HAS_SFU=1,HAS_ATOMICS=1,HAS_SYS_ATOMICS=0,USE_BANKED_VRF=1,VRF_BYPASS=1,FE_NOBUBBLE=1,
 parameter VRF_PREFETCH=1,RCTX_SWITCH=2,MTX_ABUF=1,MTX_VRF_REG=0,LSU_INFLIGHT=2,
 parameter HYBRID_LSU=0,SFU_LANES=1,HOME_BACKGROUND_READS=0,HOME_LINE_WRITE_NOFILL=0,HOME_LINE_PIPE=0,HOME_MSHR_N=8,LINE_SLOTS=8,MEM_TAG_W=(HOME_MSHR_N>1?$clog2(HOME_MSHR_N):1),
 parameter FPU_OVERLAP=1,RSLICE=1,RPAIR=1,HAS_TENSOR_XPOSE=1,
 parameter SETS=64,SET_W=6,WAYS=8,WAY_W=3,LINK_STAGES=2,AXI_DATA_W=256,AXI_OUT_N=8,AXI_PIPELINED_ISSUE=0,AXI_LINE_LUTRAM=0,HBM_STRIPED=0,
 parameter AXI_ID_W=(AXI_OUT_N>1?$clog2(AXI_OUT_N):1),MEM_BASE=64'h20000,CACHE_MAINTENANCE=0
)(
 input wire clk,rst_n,
 input wire [7:0] memory_active_ports,memory_outstanding,
 input wire cache_maint_valid,input wire [63:0] cache_maint_base,cache_maint_size,
 output wire cache_maint_ready,cache_maint_done,
 input wire cmd_valid,output wire cmd_ready,
 input wire [63:0] cmd_id,cmd_pc,cmd_arg_base,cmd_arg_stride,cmd_deadline,
 input wire [31:0] cmd_gx,cmd_gy,cmd_gz,
 input wire [1:0] cmd_group_log2,
 input wire [N-1:0] cmd_cores,input wire [NUM_RCTX-1:0] cmd_roles,
 input wire [3:0] cmd_slice,input wire [NUM_RCTX*PHYS_W-1:0] cmd_vbase,
 output wire done_valid,input wire done_ready,
 output wire [3:0] done_status,output wire [63:0] done_id,done_cycles,
 output wire [(N>1?$clog2(N):1)-1:0] fault_core,
 output wire [N*32-1:0] issued,completed,
 output wire [31:0] peak_active,output wire [N*64-1:0] active_cycles,
 input wire host_valid,output wire host_ready,input wire host_we,
 input wire [63:0] host_addr,host_wdata,input wire [7:0] host_be,
 output wire host_rsp_valid,output wire [63:0] host_rsp_data,
 output wire [3:0] host_rsp_status,
 output wire busy,output wire all_idle,output wire timed_out,
 output wire [N-1:0] fault_valid,output wire [N*8-1:0] fault_code,
 output wire [N*64-1:0] fault_pc,output wire [N*24-1:0] fault_detail,

 input wire [MEM_PORTS*64-1:0] axi_base,axi_size,
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
 wire [MEM_PORTS-1:0] memory_protocol_fault;
 wire [MEM_PORTS*64-1:0] memory_reads,memory_writes;
 wire [MEM_PORTS-1:0] ports_req_valid;
 wire [MEM_PORTS-1:0] ports_req_ready;
 wire [MEM_PORTS*64-1:0] ports_req_addr;
 wire [MEM_PORTS-1:0] ports_req_we;
 wire [MEM_PORTS*512-1:0] ports_req_wdata;
 wire [MEM_PORTS*MEM_TAG_W-1:0] ports_req_tag;
 wire [MEM_PORTS*SRC_W-1:0] ports_req_src;
 wire [MEM_PORTS-1:0] ports_rsp_valid;
 wire [MEM_PORTS*MEM_TAG_W-1:0] ports_rsp_tag;
 wire [MEM_PORTS*SRC_W-1:0] ports_rsp_src;
 wire [MEM_PORTS*512-1:0] ports_rsp_rdata;
 wire mem_req_valid;
 wire mem_req_ready;
 wire [63:0] mem_req_addr;
 wire mem_req_we;
 wire [511:0] mem_req_wdata;
 wire [MEM_TAG_W-1:0] mem_req_tag;
 wire [SRC_W-1:0] mem_req_src;
 wire mem_rsp_valid;
 wire [MEM_TAG_W-1:0] mem_rsp_tag;
 wire [SRC_W-1:0] mem_rsp_src;
 wire [511:0] mem_rsp_rdata;
 wire memory_mode_fault;
 wire [MEM_PORTS-1:0] adapter_fault;
 assign axi_fault=adapter_fault | {{(MEM_PORTS-1){1'b0}},memory_mode_fault};
 wire platform_fault=(|axi_fault) || memory_mode_fault;
 wire shell_cmd_ready,shell_done_valid,shell_all_idle,shell_busy;
 wire [3:0] shell_done_status;
 wire memory_drained=(&axi_idle) && (&link_idle);
 assign cmd_ready=shell_cmd_ready && memory_drained;
 assign done_valid=shell_done_valid && memory_drained && shell_all_idle;
 assign done_status=(|axi_fault) ? 4'd1 : shell_done_status;
 assign all_idle=shell_all_idle && memory_drained;
 assign busy=shell_busy || !memory_drained;
 tnca_multicore_shell #(
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
  .VREGS_PER_RCTX(VREGS_PER_RCTX), .ACC_N(ACC_N), .ACC_W(ACC_W), .BAR_N(BAR_N), .BAR_W(BAR_W), .BAR_EPOCH_W(BAR_EPOCH_W), .FMA_IMPL(FMA_IMPL), .MAC_DP(MAC_DP), .MAC_CUTS(MAC_CUTS), .PHYS_ADDR_W(PHYS_ADDR_W),
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
  .HYBRID_LSU(HYBRID_LSU),.SFU_LANES(SFU_LANES),.HOME_BACKGROUND_READS(HOME_BACKGROUND_READS),.HOME_LINE_WRITE_NOFILL(HOME_LINE_WRITE_NOFILL),.HOME_LINE_PIPE(HOME_LINE_PIPE),.HOME_MSHR_N(HOME_MSHR_N),.LINE_SLOTS(LINE_SLOTS),
  .FPU_OVERLAP(FPU_OVERLAP),
  .RSLICE(RSLICE),
  .RPAIR(RPAIR),
  .HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE),
  .SETS(SETS),
  .SET_W(SET_W),
  .WAYS(WAYS),
  .WAY_W(WAY_W),
  .PLATFORM_FAULT(1)) shell(
  .clk(clk),
  .rst_n(rst_n),
  .cache_maint_valid(cache_maint_valid),.cache_maint_base(cache_maint_base),.cache_maint_size(cache_maint_size),
  .cache_maint_memory_drained(memory_drained),.cache_maint_ready(cache_maint_ready),.cache_maint_done(cache_maint_done),
  .platform_fault(platform_fault),
  .cmd_valid(cmd_valid && memory_drained),
  .cmd_ready(shell_cmd_ready),
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
  .done_valid(shell_done_valid),
  .done_ready(done_ready && memory_drained && shell_all_idle),
  .done_status(shell_done_status),
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
  .busy(shell_busy),
  .all_idle(shell_all_idle),
  .timed_out(timed_out),
  .fault_valid(fault_valid),
  .fault_code(fault_code),
  .fault_pc(fault_pc),
  .fault_detail(fault_detail),
  .ports_req_valid(raw_ports_req_valid),
  .ports_req_ready(raw_ports_req_ready),
  .ports_req_addr(raw_ports_req_addr),
  .ports_req_we(raw_ports_req_we),
  .ports_req_wdata(raw_ports_req_wdata),
  .ports_req_tag(raw_ports_req_tag),
  .ports_req_src(raw_ports_req_src),
  .ports_rsp_valid(raw_ports_rsp_valid),
  .ports_rsp_tag(raw_ports_rsp_tag),
  .ports_rsp_src(raw_ports_rsp_src),
  .ports_rsp_rdata(raw_ports_rsp_rdata),
  .memory_protocol_fault(memory_protocol_fault),
  .memory_reads(memory_reads),
  .memory_writes(memory_writes),
  .mem_req_valid(mem_req_valid),
  .mem_req_ready(mem_req_ready),
  .mem_req_addr(mem_req_addr),
  .mem_req_we(mem_req_we),
  .mem_req_wdata(mem_req_wdata),
  .mem_req_tag(mem_req_tag),
  .mem_req_src(mem_req_src),
  .mem_rsp_valid(mem_rsp_valid),
  .mem_rsp_tag(mem_rsp_tag),
  .mem_rsp_src(mem_rsp_src),
  .mem_rsp_rdata(mem_rsp_rdata)
 );
 generate if(MEM_PORTS==1)begin:g_scalar
  assign ports_req_valid=mem_req_valid;assign mem_req_ready=ports_req_ready;
  assign ports_req_addr=mem_req_addr;assign ports_req_we=mem_req_we;
  assign ports_req_wdata=mem_req_wdata;assign ports_req_tag=mem_req_tag;assign ports_req_src=mem_req_src;
  assign mem_rsp_valid=ports_rsp_valid;assign mem_rsp_tag=ports_rsp_tag;
  assign mem_rsp_src=ports_rsp_src;assign mem_rsp_rdata=ports_rsp_rdata;
 end else begin:g_vector
  assign mem_req_ready=0;assign mem_rsp_valid=0;assign mem_rsp_tag=0;assign mem_rsp_src=0;assign mem_rsp_rdata=0;
 end endgenerate
 wire [MEM_PORTS-1:0] raw_ports_req_valid;
 wire [MEM_PORTS-1:0] raw_ports_req_ready;
 wire [MEM_PORTS*64-1:0] raw_ports_req_addr;
 wire [MEM_PORTS-1:0] raw_ports_req_we;
 wire [MEM_PORTS*512-1:0] raw_ports_req_wdata;
 wire [MEM_PORTS*MEM_TAG_W-1:0] raw_ports_req_tag;
 wire [MEM_PORTS*SRC_W-1:0] raw_ports_req_src;
 wire [MEM_PORTS-1:0] raw_ports_rsp_valid;
 wire [MEM_PORTS*MEM_TAG_W-1:0] raw_ports_rsp_tag;
 wire [MEM_PORTS*SRC_W-1:0] raw_ports_rsp_src;
 wire [MEM_PORTS*512-1:0] raw_ports_rsp_rdata;
 generate if(MEM_PORTS>1)begin:g_raw
  assign ports_req_valid=raw_ports_req_valid;assign raw_ports_req_ready=ports_req_ready;
  assign ports_req_addr=raw_ports_req_addr;assign ports_req_we=raw_ports_req_we;
  assign ports_req_wdata=raw_ports_req_wdata;assign ports_req_tag=raw_ports_req_tag;assign ports_req_src=raw_ports_req_src;
 end else begin:g_unused
  assign raw_ports_req_ready=0;
 end endgenerate
 assign raw_ports_rsp_valid=ports_rsp_valid;assign raw_ports_rsp_tag=ports_rsp_tag;
 assign raw_ports_rsp_src=ports_rsp_src;assign raw_ports_rsp_rdata=ports_rsp_rdata;
 wire [MEM_PORTS-1:0] mode_req_valid,mode_req_ready,mode_req_we,mode_rsp_valid;
 wire [MEM_PORTS*64-1:0] mode_req_addr;
 wire [MEM_PORTS*512-1:0] mode_req_wdata,mode_rsp_rdata;
 wire [MEM_PORTS*MEM_TAG_W-1:0] mode_req_tag,mode_rsp_tag;
 wire [MEM_PORTS*SRC_W-1:0] mode_req_src,mode_rsp_src;
 generate if(MEMORY_CONTROLS)begin:g_memory_mode
  tnca_memory_mode #(.PORTS(MEM_PORTS),.N_SRC(N),.SRC_W(SRC_W),.TAG_W(MEM_TAG_W)) mode(
   .clk(clk),.rst_n(rst_n),.single_port(memory_active_ports==1),
   .s_req_valid(ports_req_valid),.s_req_ready(ports_req_ready),.s_req_addr(ports_req_addr),
   .s_req_we(ports_req_we),.s_req_wdata(ports_req_wdata),.s_req_tag(ports_req_tag),.s_req_src(ports_req_src),
   .s_rsp_valid(ports_rsp_valid),.s_rsp_tag(ports_rsp_tag),.s_rsp_src(ports_rsp_src),.s_rsp_rdata(ports_rsp_rdata),
   .m_req_valid(mode_req_valid),.m_req_ready(mode_req_ready),.m_req_addr(mode_req_addr),
   .m_req_we(mode_req_we),.m_req_wdata(mode_req_wdata),.m_req_tag(mode_req_tag),.m_req_src(mode_req_src),
   .m_rsp_valid(mode_rsp_valid),.m_rsp_tag(mode_rsp_tag),.m_rsp_src(mode_rsp_src),.m_rsp_rdata(mode_rsp_rdata),
   .fault(memory_mode_fault));
 end else begin:g_static_memory_mode
  assign mode_req_valid=ports_req_valid;assign ports_req_ready=mode_req_ready;
  assign mode_req_addr=ports_req_addr;assign mode_req_we=ports_req_we;
  assign mode_req_wdata=ports_req_wdata;assign mode_req_tag=ports_req_tag;assign mode_req_src=ports_req_src;
  assign ports_rsp_valid=mode_rsp_valid;assign ports_rsp_tag=mode_rsp_tag;
  assign ports_rsp_src=mode_rsp_src;assign ports_rsp_rdata=mode_rsp_rdata;
  assign memory_mode_fault=1'b0;
 end endgenerate
 generate for(genvar p=0;p<MEM_PORTS;p=p+1)begin:g_outlet
  wire qv,qr,qwe,rv;wire [63:0] qa;wire [511:0] qd,rd;
  wire [SRC_W-1:0] qs,rs;wire [MEM_TAG_W-1:0] qt,rt;
  tnca_line_link #(.SRC_W(SRC_W),.TAG_W(MEM_TAG_W),.STAGES(LINK_STAGES)) link(
   .clk(clk),.rst_n(rst_n),.idle(link_idle[p]),
   .s_req_valid(mode_req_valid[p]),.s_req_ready(mode_req_ready[p]),
   .s_req_addr(mode_req_addr[p*64+:64]),.s_req_we(mode_req_we[p]),
   .s_req_wdata(mode_req_wdata[p*512+:512]),.s_req_src(mode_req_src[p*SRC_W+:SRC_W]),.s_req_tag(mode_req_tag[p*MEM_TAG_W+:MEM_TAG_W]),
   .s_rsp_valid(mode_rsp_valid[p]),.s_rsp_rdata(mode_rsp_rdata[p*512+:512]),.s_rsp_src(mode_rsp_src[p*SRC_W+:SRC_W]),.s_rsp_tag(mode_rsp_tag[p*MEM_TAG_W+:MEM_TAG_W]),
   .m_req_valid(qv),.m_req_ready(qr),.m_req_addr(qa),.m_req_we(qwe),.m_req_wdata(qd),.m_req_src(qs),.m_req_tag(qt),
   .m_rsp_valid(rv),.m_rsp_rdata(rd),.m_rsp_src(rs),.m_rsp_tag(rt));
  tnca_line_axi #(.HBM_STRIPED(HBM_STRIPED),.PIPELINED_ISSUE(AXI_PIPELINED_ISSUE),.LINE_LUTRAM(AXI_LINE_LUTRAM),.HAZARD_ADDR_W(PHYS_ADDR_W),.SRC_W(SRC_W),.TAG_W(MEM_TAG_W),.DATA_W(AXI_DATA_W),.OUT_N(AXI_OUT_N),.DYNAMIC_LIMIT(MEMORY_CONTROLS),.ID_W(AXI_ID_W),.MEM_BASE(MEM_BASE)) adapter(
   .clk(clk),.rst_n(rst_n),.req_valid(qv),.req_ready(qr),.req_addr(qa),.req_we(qwe),.req_wdata(qd),.req_src(qs),.req_tag(qt),
   .rsp_valid(rv),.rsp_rdata(rd),.rsp_src(rs),.rsp_tag(rt),
   .admission_limit(memory_outstanding),.axi_base(axi_base[p*64+:64]),.axi_size(axi_size[p*64+:64]),.fault(adapter_fault[p]),.idle(axi_idle[p]),
   .read_lines(axi_read_lines[p*64+:64]),.write_lines(axi_write_lines[p*64+:64]),.read_beats(axi_read_beats[p*64+:64]),.write_beats(axi_write_beats[p*64+:64]),
   .m_awvalid(m_awvalid[p]),
   .m_awready(m_awready[p]),
   .m_awaddr(m_awaddr[p*(64)+:(64)]),
   .m_awid(m_awid[p*(AXI_ID_W)+:(AXI_ID_W)]),
   .m_awlen(m_awlen[p*(8)+:(8)]),
   .m_awsize(m_awsize[p*(3)+:(3)]),
   .m_awburst(m_awburst[p*(2)+:(2)]),
   .m_wvalid(m_wvalid[p]),
   .m_wready(m_wready[p]),
   .m_wdata(m_wdata[p*(AXI_DATA_W)+:(AXI_DATA_W)]),
   .m_wstrb(m_wstrb[p*(AXI_DATA_W/8)+:(AXI_DATA_W/8)]),
   .m_wlast(m_wlast[p]),
   .m_bvalid(m_bvalid[p]),
   .m_bready(m_bready[p]),
   .m_bid(m_bid[p*(AXI_ID_W)+:(AXI_ID_W)]),
   .m_bresp(m_bresp[p*(2)+:(2)]),
   .m_arvalid(m_arvalid[p]),
   .m_arready(m_arready[p]),
   .m_araddr(m_araddr[p*(64)+:(64)]),
   .m_arid(m_arid[p*(AXI_ID_W)+:(AXI_ID_W)]),
   .m_arlen(m_arlen[p*(8)+:(8)]),
   .m_arsize(m_arsize[p*(3)+:(3)]),
   .m_arburst(m_arburst[p*(2)+:(2)]),
   .m_rvalid(m_rvalid[p]),
   .m_rready(m_rready[p]),
   .m_rdata(m_rdata[p*(AXI_DATA_W)+:(AXI_DATA_W)]),
   .m_rid(m_rid[p*(AXI_ID_W)+:(AXI_ID_W)]),
   .m_rresp(m_rresp[p*(2)+:(2)]),
   .m_rlast(m_rlast[p])
  );
 end endgenerate
endmodule
`default_nettype wire
