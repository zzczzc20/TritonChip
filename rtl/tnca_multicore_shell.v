// Persistent command/host binding for a global-dispatch coherent mesh.
// Private synchronous instruction RAM per core, with broadcast code upload.
// External line memory remains a platform port, not simulator storage.
`default_nettype none
module tnca_multicore_shell #(
 parameter DCO_REGISTERED_RANGES=0,
    parameter NUCA_REGISTER_VICTIM_SELECT = 0,
    parameter NUCA_CACHE_DATA_PIPELINE = 0,
    parameter NUCA_SHARED_DATA_READ = 0,   // R41 G2: one LUTRAM read port per slice line array
    parameter NUCA_REGISTER_VICTIM_RANK = 0,
    parameter DCO_TLW_PIPE_STAGES = 0,
    parameter DCO_REGISTER_WRITES = 0,
    parameter DCO_LOCAL_REPLICAS = 0,
 parameter NX=3,NY=1,N=NX*NY,CODE_WORDS=16384,CODE_LINK_STAGES=0,MEM_PORTS=1,NOC_LINK_STAGES=0,DISPATCH_STAGES=0,MEM_HOME_STAGES=0,
 parameter SRC_W=(N>4?$clog2(N):2),
 parameter NUM_RCTX=4,RCTX_W=2,PHYS_VREGS=256,PHYS_W=8,
 parameter VREGS_PER_RCTX=64,ACC_N=16,ACC_W=4,BAR_N=32,BAR_W=5,BAR_EPOCH_W=48,PHYS_ADDR_W=64,AUTO_VBASE=1,
 parameter HAS_SFU=1,HAS_ATOMICS=1,HAS_SYS_ATOMICS=0,USE_BANKED_VRF=1,VRF_BYPASS=1,FE_NOBUBBLE=1,
 parameter VRF_PREFETCH=1,RCTX_SWITCH=2,MTX_ABUF=1,MTX_VRF_REG=0,LSU_INFLIGHT=2,
 parameter HYBRID_LSU=0,SFU_LANES=1,HOME_BACKGROUND_READS=0,HOME_LINE_WRITE_NOFILL=0,HOME_LINE_PIPE=0,TENSOR_LINE=0,HOME_MSHR_N=8,LINE_SLOTS=8,MEM_TAG_W=(HOME_MSHR_N>1?$clog2(HOME_MSHR_N):1),
 parameter FPU_OVERLAP=1,RSLICE=1,RPAIR=1,HAS_TENSOR_XPOSE=1,
 parameter SETS=64,SET_W=6,WAYS=8,WAY_W=3,PLATFORM_FAULT=0,CACHE_MAINTENANCE=0
)(
 input wire clk,rst_n,input wire platform_fault,
 input wire cache_maint_valid,input wire [63:0] cache_maint_base,cache_maint_size,
 input wire cache_maint_memory_drained,output wire cache_maint_ready,cache_maint_done,
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
 output wire [MEM_PORTS-1:0] ports_req_valid,input wire [MEM_PORTS-1:0] ports_req_ready,
 output wire [MEM_PORTS*64-1:0] ports_req_addr,output wire [MEM_PORTS-1:0] ports_req_we,
 output wire [MEM_PORTS*512-1:0] ports_req_wdata,output wire [MEM_PORTS*MEM_TAG_W-1:0] ports_req_tag,
 output wire [MEM_PORTS*SRC_W-1:0] ports_req_src,
 input wire [MEM_PORTS-1:0] ports_rsp_valid,input wire [MEM_PORTS*MEM_TAG_W-1:0] ports_rsp_tag,
 input wire [MEM_PORTS*SRC_W-1:0] ports_rsp_src,input wire [MEM_PORTS*512-1:0] ports_rsp_rdata,
 output wire [MEM_PORTS-1:0] memory_protocol_fault,
 output wire [MEM_PORTS*64-1:0] memory_reads,memory_writes,
 output wire mem_req_valid,input wire mem_req_ready,output wire [63:0] mem_req_addr,
 output wire mem_req_we,output wire [511:0] mem_req_wdata,
 output wire [MEM_TAG_W-1:0] mem_req_tag,output wire [SRC_W-1:0] mem_req_src,
 input wire mem_rsp_valid,input wire [MEM_TAG_W-1:0] mem_rsp_tag,
 input wire [SRC_W-1:0] mem_rsp_src,input wire [511:0] mem_rsp_rdata
);
 // R41 G4: the core port tag: a client bit, a path bit and the line LSU's
 // slots when they exceed the delivered eight (5 bits on the delivered machine).
 localparam PORT_TAG_W = (HYBRID_LSU && LINE_SLOTS > 8) ? 2+$clog2(LINE_SLOTS) : 5;
 localparam CODE_BYTES=CODE_WORDS*8;
 wire [N*64-1:0] ia, insn;
 wire [NUM_RCTX*PHYS_W-1:0] effective_vbase;
 generate for(genvar r=0;r<NUM_RCTX;r=r+1) begin: g_vbase
  // RPAIR remaps odd contexts to their even partner inside the core.
  // Independent programs always begin in distinct physical VRF ranges.
  assign effective_vbase[r*PHYS_W+:PHYS_W]=AUTO_VBASE ? r*VREGS_PER_RCTX : cmd_vbase[r*PHYS_W+:PHYS_W];
 end endgenerate
 wire grid_ready,grid_done,host_busy,data_ready,data_rsp;
 wire system_maint_ready;
 wire [63:0] data_rdata,grid_id,grid_cycles;
 wire [3:0] grid_status;
 reg invalid_done,bad_host_pending,launch_live;
 wire code_pending,code_busy,tiles_all_idle;
 assign all_idle=tiles_all_idle && !code_busy;
 wire [N-1:0] code_command_ready,code_reply_valid;
 wire [N*64-1:0] code_reply_data;
 wire deadline_abort;
 wire start_grid=cmd_valid && cmd_ready && !bad_pc;
 tnca_grid_deadline u_deadline(.clk(clk),.rst_n(rst_n),.start(start_grid),
  .active(launch_live && !grid_done),.deadline(cmd_deadline),.abort_valid(deadline_abort),.timed_out(timed_out),.elapsed());
 reg [63:0] invalid_id;
 wire [63:0] code_rdata;
 wire bad_host_addr=|host_addr[2:0];
 wire code_selected=host_addr<CODE_BYTES;
 wire code_ready=grid_ready && !cmd_valid && !invalid_done && !code_busy && all_idle && (&code_command_ready);
 wire code_fire=host_valid && host_ready && code_selected && !bad_host_addr;
 wire [2:0] chunk_width=cmd_group_log2===2'd2 ? 3'd4 : (cmd_group_log2===2'd1 ? 3'd2 : 3'd1);
 wire [3:0] chunk_roles=chunk_width << cmd_slice[3];
 wire bad_chunk=(cmd_group_log2===2'd3) ||
   ((cmd_group_log2===2'd1 || cmd_group_log2===2'd2) &&
    (chunk_roles>NUM_RCTX || cmd_roles!=((1<<chunk_roles)-1) ||
     !cmd_slice[2] || cmd_slice[1:0]!=cmd_group_log2));
 wire bad_pc=cmd_pc>=CODE_BYTES || cmd_pc[2:0]!=0 || bad_chunk;
 assign cmd_ready=rst_n && grid_ready && !code_busy && !bad_host_pending && !invalid_done;
 assign cache_maint_ready=system_maint_ready && !code_busy && !bad_host_pending && !invalid_done;
 assign done_valid=invalid_done || grid_done;
 assign done_status=invalid_done ? 4'd3 : grid_status;
 assign done_id=invalid_done ? invalid_id : grid_id;
 assign done_cycles=invalid_done ? 64'd0 : grid_cycles;
 assign host_ready=rst_n && ((bad_host_addr || code_selected) ? (code_ready && !bad_host_pending) : (data_ready && !code_busy && !bad_host_pending && !invalid_done));
 assign host_rsp_valid=code_pending || bad_host_pending || data_rsp;
 assign host_rsp_status=bad_host_pending ? 4'd3 : 4'd0;
 assign host_rsp_data=bad_host_pending ? 64'd0 : (code_pending ? code_rdata : data_rdata);
 assign busy=!rst_n || !grid_ready || code_busy || bad_host_pending || invalid_done || host_busy;
 genvar c;
 generate for(c=0;c<N;c=c+1) begin: g_code
  (* ram_style="block" *) reg [63:0] code[0:CODE_WORDS-1];
  reg [63:0] instruction,host_read;
  wire command_valid,command_we;
  wire [63:0] command_addr,command_data;
  wire [7:0] command_be;
  wire [136:0] payload;
  // Broadcast inputs can make these queues logically identical. Preserve
  // each physical copy so the board can place its last stage at its RAM;
  // merging copies would remove the intended registered SLR boundary.
  (* DONT_TOUCH="yes" *) tnca_elastic_link #(.W(137),.STAGES(CODE_LINK_STAGES)) commands(
   .clk(clk),.rst_n(rst_n),.in_valid(code_fire),.in_ready(code_command_ready[c]),
   .in_data({host_we,host_addr,host_wdata,host_be}),
   .out_valid(command_valid),.out_ready(1'b1),.out_data(payload),.empty());
  assign {command_we,command_addr,command_data,command_be}=payload;
  reg reply_valid;
  always @(posedge clk)begin
   if(!rst_n)reply_valid<=0;else reply_valid<=command_valid;
  end
  // One accepted broadcast is outstanding until every reply is consumed.
  // Thus each reply FIFO is empty when its next RAM reply arrives, so it
  // cannot apply backpressure to the synchronous read port.
  (* DONT_TOUCH="yes" *) tnca_elastic_link #(.W(64),.STAGES(CODE_LINK_STAGES)) replies(
   .clk(clk),.rst_n(rst_n),.in_valid(reply_valid),.in_ready(),.in_data(host_read),
   .out_valid(code_reply_valid[c]),.out_ready(1'b1),.out_data(code_reply_data[c*64+:64]),.empty());
  integer b;
  always @(posedge clk) begin
   if(ia[c*64+:64]<CODE_BYTES)instruction<=code[ia[c*64+:64]>>3];
   else instruction<=64'h0218000000030000; // explicit trap for invalid fetch
   if(command_valid) begin
    if(command_we) begin
     host_read<=0;
     for(b=0;b<8;b=b+1)if(command_be[b])code[command_addr>>3][b*8+:8]<=command_data[b*8+:8];
    end else host_read<=code[command_addr>>3];
   end
  end
  assign insn[c*64+:64]=instruction;
 end endgenerate
 generate if(CODE_LINK_STAGES==0)begin:g_code_direct
  assign code_pending=code_reply_valid[0];assign code_busy=code_pending;
  assign code_rdata=code_reply_data[0+:64];
 end else begin:g_code_pipeline
  reg active;reg [N-1:0] seen;reg [63:0] read_data;
  assign code_pending=active && (&seen);assign code_busy=active;
  assign code_rdata=read_data;
  always @(posedge clk)begin
   if(!rst_n)begin active<=0;seen<=0;read_data<=0;end
   else begin
    if(code_fire)begin active<=1;seen<=0;end
    if(active)begin
     seen<=seen | code_reply_valid;
     if(code_reply_valid[0])read_data<=code_reply_data[0+:64];
     if(code_pending)begin active<=0;seen<=0;end
    end
   end
  end
 end endgenerate
 always @(posedge clk) begin
  if(!rst_n)begin launch_live<=0;invalid_done<=0;invalid_id<=0;bad_host_pending<=0;end
  else begin
   bad_host_pending<=0;
   if(start_grid)launch_live<=1;
   if(grid_done)launch_live<=0;
   if(invalid_done && done_ready)invalid_done<=0;
   if(cmd_valid && cmd_ready && bad_pc)begin invalid_done<=1;invalid_id<=cmd_id;end
   if(host_valid && host_ready && bad_host_addr)bad_host_pending<=1;
  end
 end
 tnca_top #(.CACHE_MAINTENANCE(CACHE_MAINTENANCE), .DCO_REGISTERED_RANGES(DCO_REGISTERED_RANGES), .NUCA_REGISTER_VICTIM_SELECT(NUCA_REGISTER_VICTIM_SELECT), .NUCA_CACHE_DATA_PIPELINE(NUCA_CACHE_DATA_PIPELINE), .NUCA_SHARED_DATA_READ(NUCA_SHARED_DATA_READ), .NUCA_REGISTER_VICTIM_RANK(NUCA_REGISTER_VICTIM_RANK), .DCO_TLW_PIPE_STAGES(DCO_TLW_PIPE_STAGES), .DCO_REGISTER_WRITES(DCO_REGISTER_WRITES), .DCO_LOCAL_REPLICAS(DCO_LOCAL_REPLICAS),.NX(NX),.NY(NY),.MEM_PORTS(MEM_PORTS),.GLOBAL_GRID(1),.HOST_ACCESS(1),.MEM_SRC_W(SRC_W),.NOC_LINK_STAGES(NOC_LINK_STAGES),.DISPATCH_STAGES(DISPATCH_STAGES),.MEM_HOME_STAGES(MEM_HOME_STAGES),
  .NUM_RCTX(NUM_RCTX),.RCTX_W(RCTX_W),.PHYS_VREGS(PHYS_VREGS),.PHYS_W(PHYS_W),
  .VREGS_PER_RCTX(VREGS_PER_RCTX), .ACC_N(ACC_N), .ACC_W(ACC_W), .BAR_N(BAR_N), .BAR_W(BAR_W), .BAR_EPOCH_W(BAR_EPOCH_W), .PHYS_ADDR_W(PHYS_ADDR_W),
  .HAS_SFU(HAS_SFU),.HAS_ATOMICS(HAS_ATOMICS),.HAS_SYS_ATOMICS(HAS_SYS_ATOMICS),.USE_BANKED_VRF(USE_BANKED_VRF),.VRF_BYPASS(VRF_BYPASS),
  .FE_NOBUBBLE(FE_NOBUBBLE),.VRF_PREFETCH(VRF_PREFETCH),.RCTX_SWITCH(RCTX_SWITCH),
  .MTX_ABUF(MTX_ABUF),.MTX_VRF_REG(MTX_VRF_REG),.LSU_INFLIGHT(LSU_INFLIGHT),.HYBRID_LSU(HYBRID_LSU),.SFU_LANES(SFU_LANES),.HOME_BACKGROUND_READS(HOME_BACKGROUND_READS),.HOME_LINE_WRITE_NOFILL(HOME_LINE_WRITE_NOFILL),.HOME_LINE_PIPE(HOME_LINE_PIPE),.TENSOR_LINE(TENSOR_LINE),.LSU_TAG_W(PORT_TAG_W),.NOC_PAYLOAD_W(HYBRID_LSU ? 720+PORT_TAG_W-5 : 216),.MSHR_N(HOME_MSHR_N),.MSHR_W(MEM_TAG_W),.TARGET_N(HOME_MSHR_N>8?2*HOME_MSHR_N:16),.TARGET_W(HOME_MSHR_N>8?MEM_TAG_W+1:4),.RQ_N(HOME_MSHR_N>8?16:8),.RQ_W(HOME_MSHR_N>8?4:3),.FPU_OVERLAP(FPU_OVERLAP),
  .RSLICE(RSLICE),.RPAIR(RPAIR),.HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE),
  .SETS(SETS),.SET_W(SET_W),.WAYS(WAYS),.WAY_W(WAY_W)) u_system (
  .ports_req_valid(ports_req_valid),.ports_req_ready(ports_req_ready),.ports_req_addr(ports_req_addr),
  .ports_req_we(ports_req_we),.ports_req_wdata(ports_req_wdata),.ports_req_tag(ports_req_tag),
  .ports_req_src(ports_req_src),.ports_rsp_valid(ports_rsp_valid),.ports_rsp_tag(ports_rsp_tag),
  .ports_rsp_src(ports_rsp_src),.ports_rsp_rdata(ports_rsp_rdata),.memory_protocol_fault(memory_protocol_fault),
  .memory_reads(memory_reads),.memory_writes(memory_writes),
  .clk(clk),.rst_n(rst_n),.launch_valid(1'b0),.launch_pc(cmd_pc),.launch_arg_base(cmd_arg_base),
  .cache_maint_valid(cache_maint_valid && !code_busy && !bad_host_pending && !invalid_done),.cache_maint_base(cache_maint_base),.cache_maint_size(cache_maint_size),
  .cache_maint_memory_drained(cache_maint_memory_drained),.cache_maint_ready(system_maint_ready),.cache_maint_done(cache_maint_done),
  .grid_abort(deadline_abort || (PLATFORM_FAULT && platform_fault)),.grid_abort_status((PLATFORM_FAULT && platform_fault) ? 4'd1 : 4'd2),.launch_role_mask(cmd_roles),.launch_vbase(effective_vbase),.grid_slice(cmd_slice),
  .grid_valid(cmd_valid && !bad_pc && !code_busy && !bad_host_pending && !invalid_done),.grid_ready(grid_ready),
  .grid_group_log2(cmd_group_log2),.grid_id(cmd_id),.grid_arg_stride(cmd_arg_stride),.grid_x(cmd_gx),.grid_y(cmd_gy),.grid_z(cmd_gz),
  .grid_cores(cmd_cores),.grid_done_valid(grid_done),.grid_done_ready(done_ready && !invalid_done),
  .grid_done_status(grid_status),.grid_done_id(grid_id),.grid_done_cycles(grid_cycles),
  .grid_fault_core(fault_core),.grid_issued(issued),.grid_completed(completed),
  .grid_peak_active(peak_active),.grid_active_cycles(active_cycles),
  .host_valid(host_valid && !code_selected && !bad_host_addr && !code_busy && !bad_host_pending && !invalid_done && (!cmd_valid || host_busy)),.host_we(host_we),
  .host_addr(host_addr),.host_wdata(host_wdata),.host_be(host_be),.host_ready(data_ready),
  .host_rsp_valid(data_rsp),.host_rsp_data(data_rdata),.host_busy(host_busy),.all_idle(tiles_all_idle),
  .fault_valid(fault_valid),.fault_code(fault_code),.fault_pc(fault_pc),.fault_detail(fault_detail),
  .imem_addr(ia),.imem_data(insn),.mem_req_valid(mem_req_valid),.mem_req_ready(mem_req_ready),
  .mem_req_addr(mem_req_addr),.mem_req_we(mem_req_we),.mem_req_wdata(mem_req_wdata),
  .mem_req_tag(mem_req_tag),.mem_req_src(mem_req_src),.mem_rsp_valid(mem_rsp_valid),
  .mem_rsp_tag(mem_rsp_tag),.mem_rsp_src(mem_rsp_src),.mem_rsp_rdata(mem_rsp_rdata));
endmodule
`default_nettype wire
