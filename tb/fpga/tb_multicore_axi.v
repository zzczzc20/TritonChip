`timescale 1ns/1ps
module tb_multicore_axi;
parameter N=3,PORTS=3,STAGES=2,WRITE_DELAY=15,AXI_PIPELINED_ISSUE=0,HYBRID_LSU=0,VECTOR_TEST=0,FULL_VECTOR_TEST=0;
localparam POINTS=FULL_VECTOR_TEST?17:455;
localparam NX=N>4?4:N,SW=N>4?$clog2(N):2;
reg clk=0;always #5 clk=~clk;
reg rst=0,cv=0,ack=0,hv=0,hw=0;
reg [63:0] pc=0,id=11,ha=0,hd=0,deadline=0;
wire timed_out;
wire cr,dv,hr,hrv,busy,idle;
wire [3:0] ds,hs;
wire [63:0] did,dc,hrd;
wire [N*32-1:0] issued,completed;
wire mr,mready,mwe,mrv;
wire [63:0] ma;wire [511:0] mw,md;
wire [2:0] mt,mrt;wire [SW-1:0] ms,mrs;
reg [63:0] image[0:255];reg [1023:0] hexfile;
integer cycles=0,drain_wait=0;
wire [PORTS-1:0] af,ai,li;wire [PORTS*64-1:0] rdlines,wrlines;
wire [PORTS*(1)-1:0] m_awvalid;
wire [PORTS*(1)-1:0] m_awready;
wire [PORTS*(64)-1:0] m_awaddr;
wire [PORTS*(3)-1:0] m_awid;
wire [PORTS*(8)-1:0] m_awlen;
wire [PORTS*(3)-1:0] m_awsize;
wire [PORTS*(2)-1:0] m_awburst;
wire [PORTS*(1)-1:0] m_wvalid;
wire [PORTS*(1)-1:0] m_wready;
wire [PORTS*(256)-1:0] m_wdata;
wire [PORTS*(256/8)-1:0] m_wstrb;
wire [PORTS*(1)-1:0] m_wlast;
wire [PORTS*(1)-1:0] m_bvalid;
wire [PORTS*(1)-1:0] m_bready;
wire [PORTS*(3)-1:0] m_bid;
wire [PORTS*(2)-1:0] m_bresp;
wire [PORTS*(1)-1:0] m_arvalid;
wire [PORTS*(1)-1:0] m_arready;
wire [PORTS*(64)-1:0] m_araddr;
wire [PORTS*(3)-1:0] m_arid;
wire [PORTS*(8)-1:0] m_arlen;
wire [PORTS*(3)-1:0] m_arsize;
wire [PORTS*(2)-1:0] m_arburst;
wire [PORTS*(1)-1:0] m_rvalid;
wire [PORTS*(1)-1:0] m_rready;
wire [PORTS*(256)-1:0] m_rdata;
wire [PORTS*(3)-1:0] m_rid;
wire [PORTS*(2)-1:0] m_rresp;
wire [PORTS*(1)-1:0] m_rlast;

tnca_multicore_axi #(.HYBRID_LSU(HYBRID_LSU),.SFU_LANES(HYBRID_LSU?4:1),.NOC_LINK_STAGES(HYBRID_LSU?2:0),.DISPATCH_STAGES(HYBRID_LSU?2:0),.MEM_HOME_STAGES(HYBRID_LSU?2:0),.NUCA_CACHE_DATA_PIPELINE(HYBRID_LSU),.AXI_PIPELINED_ISSUE(AXI_PIPELINED_ISSUE),.MEM_PORTS(PORTS),.LINK_STAGES(STAGES),.MEM_BASE(0),.NX(NX),.NY(N/NX),.CODE_WORDS(256),.SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1)) dut(
.clk(clk),.rst_n(rst),.cmd_valid(cv),.cmd_ready(cr),.cmd_id(id),.cmd_pc(pc),
.cmd_group_log2(2'd0),.cmd_deadline(deadline),.timed_out(timed_out),.cmd_arg_base(64'h8000),.cmd_arg_stride(64'h100),.cmd_gx(FULL_VECTOR_TEST?32'd17:32'd13),.cmd_gy(FULL_VECTOR_TEST?32'd1:32'd7),.cmd_gz(FULL_VECTOR_TEST?32'd1:32'd5),
.cmd_cores({N{1'b1}}),.cmd_roles(4'b0001),.cmd_slice(4'd0),.cmd_vbase(32'd0),
.done_valid(dv),.done_ready(ack),.done_status(ds),.done_id(did),.done_cycles(dc),
.issued(issued),.completed(completed),.host_valid(hv),.host_ready(hr),.host_we(hw),
.host_addr(ha),.host_wdata(hd),.host_be(8'hff),.host_rsp_valid(hrv),.host_rsp_data(hrd),.host_rsp_status(hs),
.busy(busy),.all_idle(idle),
.axi_base({PORTS{64'h100000000}}),.axi_size({PORTS{64'd131072}}),
.axi_fault(af),.axi_idle(ai),.link_idle(li),.axi_read_lines(rdlines),.axi_write_lines(wrlines),
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
.m_rlast(m_rlast));
generate for(genvar p=0;p<PORTS;p=p+1)begin:g_ram
 tnca_axi_line_model #(.DATA_W(256),.OUT_N(8),.LINES(2048),.LATENCY(25+p*19),.WRITE_DELAY(WRITE_DELAY)) ram(
  .clk(clk),.rst_n(rst),
.m_awvalid(m_awvalid[p*(1)+:(1)]),
.m_awready(m_awready[p*(1)+:(1)]),
.m_awaddr(m_awaddr[p*(64)+:(64)]),
.m_awid(m_awid[p*(3)+:(3)]),
.m_awlen(m_awlen[p*(8)+:(8)]),
.m_awsize(m_awsize[p*(3)+:(3)]),
.m_awburst(m_awburst[p*(2)+:(2)]),
.m_wvalid(m_wvalid[p*(1)+:(1)]),
.m_wready(m_wready[p*(1)+:(1)]),
.m_wdata(m_wdata[p*(256)+:(256)]),
.m_wstrb(m_wstrb[p*(256/8)+:(256/8)]),
.m_wlast(m_wlast[p*(1)+:(1)]),
.m_bvalid(m_bvalid[p*(1)+:(1)]),
.m_bready(m_bready[p*(1)+:(1)]),
.m_bid(m_bid[p*(3)+:(3)]),
.m_bresp(m_bresp[p*(2)+:(2)]),
.m_arvalid(m_arvalid[p*(1)+:(1)]),
.m_arready(m_arready[p*(1)+:(1)]),
.m_araddr(m_araddr[p*(64)+:(64)]),
.m_arid(m_arid[p*(3)+:(3)]),
.m_arlen(m_arlen[p*(8)+:(8)]),
.m_arsize(m_arsize[p*(3)+:(3)]),
.m_arburst(m_arburst[p*(2)+:(2)]),
.m_rvalid(m_rvalid[p*(1)+:(1)]),
.m_rready(m_rready[p*(1)+:(1)]),
.m_rdata(m_rdata[p*(256)+:(256)]),
.m_rid(m_rid[p*(3)+:(3)]),
.m_rresp(m_rresp[p*(2)+:(2)]),
.m_rlast(m_rlast[p*(1)+:(1)]));
end endgenerate
wire [N*32-1:0] chained_hits;
wire [N*32-1:0] fast_accepted,slow_accepted,lsu_retired;
generate for(genvar c=0;c<N;c=c+1)begin:g_lsu_coverage
 reg [31:0] chain_count=0;
 always @(posedge clk)if(!rst)chain_count<=0;else if(dut.shell.u_system.g_tile[c].u_tile.u_slice.hit_chain_take)chain_count<=chain_count+1;
 assign chained_hits[c*32+:32]=chain_count;
 if(HYBRID_LSU)begin:g_hybrid
  integer seen_vectors=0;
  always @(posedge clk)if($test$plusargs("diagnose") && rst && dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.req_valid && dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.req_vector && seen_vectors<10)begin
   seen_vectors=seen_vectors+1;
   $display("VECTOR ROUTE core=%0d base=%h offset=%h size=%0d indexed=%b mask=%h eligible=%b",c,dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.req_base,dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.req_offset,dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.req_size,dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.req_indexed,dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.req_mask,dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.fast_eligible);
  end
  assign fast_accepted[c*32+:32]=dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.perf_fast_accepted;
  assign slow_accepted[c*32+:32]=dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.perf_slow_accepted;
  assign lsu_retired[c*32+:32]=dut.shell.u_system.g_tile[c].u_tile.u_core.g_lsu_hybrid.u_lsu.perf_retired;
 end else begin:g_legacy
  assign fast_accepted[c*32+:32]=0;
  assign slow_accepted[c*32+:32]=0;
  assign lsu_retired[c*32+:32]=0;
 end
end endgenerate
always @(posedge clk)begin
 if(rst && dut.shell_done_valid && !((&ai) && (&li)))drain_wait=drain_wait+1;
 if(dv && !((&ai) && (&li)))$fatal(1,"global completion preceded AXI/link drain");
 cycles=cycles+1;if(cycles>150000)$fatal(1,"shell deadline");end
task word;
 input write_enable;input [63:0] address,value;input [3:0] expected_status;
 begin
  @(negedge clk);hv=1;hw=write_enable;ha=address;hd=value;
  @(posedge clk);while(!hr)@(posedge clk);
  @(negedge clk);hv=0;
  while(!hrv)@(negedge clk);
  if(hs!=expected_status || (!write_enable && hs==0 && hrd!==value))$fatal(1,"host mismatch addr=%h got=%h status=%d",address,hrd,hs);
  @(negedge clk);
 end
endtask
task launch;
 input [63:0] address,launch_id;input [3:0] expected_status;
 integer sum;
 begin
  @(negedge clk);pc=address;id=launch_id;cv=1;
  @(posedge clk);while(!cr)@(posedge clk);
  @(negedge clk);cv=0;hv=1;hw=1;ha=0;hd=0;
  if(hr)$fatal(1,"code overwrite admitted during command");
  @(negedge clk);hv=0;
  while(!dv)@(negedge clk);
  if(ds!=expected_status || did!=launch_id || !idle)$fatal(1,"completion mismatch status=%d expected=%d id=%d idle=%b axi_fault=%b protocol_fault=%b core_fault=%h pc=%h detail=%h",ds,expected_status,did,idle,af,dut.memory_protocol_fault,dut.shell.fault_code,dut.shell.fault_pc,dut.shell.fault_detail);
  if(expected_status==2)begin
   sum=0;for(integer c=0;c<N;c=c+1)begin
    if(issued[c*32+:32]!=completed[c*32+:32])$fatal(1,"timeout returned before drain");
    sum=sum+issued[c*32+:32];
   end
   if(!timed_out || sum==0 || sum>=POINTS)$fatal(1,"timeout did not stop issue");
  end
  if(expected_status==0)begin
   sum=0;for(integer c=0;c<N;c=c+1)begin
    if(issued[c*32+:32]==0 || issued[c*32+:32]!=completed[c*32+:32])$fatal(1,"core not drained");
    sum=sum+issued[c*32+:32];
   end
   if(sum!=POINTS || dc==0)$fatal(1,"grid count mismatch");
  end
  repeat(4)begin @(negedge clk);if(!dv || cr || did!=launch_id)$fatal(1,"completion not held");end
  ack=1;@(negedge clk);ack=0;
  $display("MULTICORE SHELL GRID PASS N=%0d id=%0d status=%0d cycles=%0d",N,launch_id,expected_status,cycles);
 end
endtask
initial begin
 if(!$value$plusargs("hex=%s",hexfile))$fatal(1,"missing image");$readmemh(hexfile,image);
 repeat(5)@(negedge clk);rst=1;
 word(0,1,0,3);
 for(integer k=0;k<256;k=k+1)word(1,k*8,image[k],0);
 for(integer k=0;k<256;k=k+1)word(0,k*8,image[k],0);
 launch(2048,9,3);launch(1,10,3);
 deadline=20;launch(0,13,2);deadline=0;
 launch(0,11,0);
 if($test$plusargs("diagnose"))begin
  $display("MEMORY SAMPLE p0=%h p1=%h p2=%h",g_ram[0].ram.memory[1024],g_ram[0].ram.memory[1025],g_ram[0].ram.memory[1026]);
  for(integer k=0;k<8;k=k+1)word(0,65536+k*8,4096+k,0);
 end
 if(HYBRID_LSU)begin
  for(integer c=0;c<N;c=c+1)begin
   $display("HOME CHAIN COVERAGE core=%0d chained=%0d",c,chained_hits[c*32+:32]);
   $display("HYBRID COUPLED core=%0d fast=%0d slow=%0d retired=%0d",c,fast_accepted[c*32+:32],slow_accepted[c*32+:32],lsu_retired[c*32+:32]);
   if(VECTOR_TEST && fast_accepted[c*32+:32]==0)$fatal(1,"vector grid did not exercise compact LSU");
   if(slow_accepted[c*32+:32]==0 || lsu_retired[c*32+:32]!=fast_accepted[c*32+:32]+slow_accepted[c*32+:32])$fatal(1,"hybrid did not drain both paths");
  end
 end
 if(HYBRID_LSU && VECTOR_TEST && N>=3)begin: check_chained_hits
  integer total_chains;
  total_chains=0;
  for(integer c=0;c<N;c=c+1)total_chains=total_chains+chained_hits[c*32+:32];
  if(total_chains==0)$fatal(1,"coupled grid did not exercise chained hits");
 end
 launch(VECTOR_TEST?512:256,12,0);
 for(integer k=0;k<POINTS;k=k+1)word(0,65536+k*8,4096+k,0);
 if(FULL_VECTOR_TEST)begin
  for(integer k=0;k<POINTS;k=k+1)begin
   for(integer w=0;w<8;w=w+1)begin
    word(0,98304+k*64+w*8,((64'd4096+k)<<32)|(64'd4096+k),0);
    word(0,32768+k*64+w*8,((64'd4096+k)<<32)|(64'd4096+k),0);
   end
  end
  $display("FULL VECTOR ROUNDTRIP PASS programs=%0d checked_words=%0d",POINTS,POINTS*16);
 end
 wait((&ai) && (&li));
 if(af || dut.memory_protocol_fault)$fatal(1,"AXI memory fault");
 for(integer p=0;p<PORTS;p=p+1)begin
  if(rdlines[p*64+:64]==0)$fatal(1,"outlet has no reads");
  $display("AXI OUTLET TRAFFIC port=%0d reads=%0d writes=%0d",p,rdlines[p*64+:64],wrlines[p*64+:64]);
 end
 // Error response propagates from the physical outlet to global dispatch.
 force m_rresp={PORTS{2'b10}};word(0,120000,0,0);release m_rresp;
 if(!( |af))$fatal(1,"physical error not latched");
 launch(0,14,1);
 if(issued || completed)$fatal(1,"new grid issued after physical fault");
 if(WRITE_DELAY>=300 && drain_wait==0)$fatal(1,"delayed B did not exercise global drain");
 $display("MULTICORE AXI PERSISTENT PASS N=%0d ports=%0d stages=%0d write_delay=%0d drain_wait=%0d",N,PORTS,STAGES,WRITE_DELAY,drain_wait);$finish;
end
endmodule
