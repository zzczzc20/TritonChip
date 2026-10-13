`timescale 1ns/1ps
// Executes real tnca_core EXIT pipelines; does not replace Torch/board acceptance.
module tb_grid_cores;
parameter N=3;
parameter MEMORY=0;
localparam MX=N>4?4:N;
localparam SW=N>4?$clog2(N):2;
reg clk=0;always #5 clk=~clk;
reg rst_n=0,valid=0,ack=0;
reg [63:0] launch_pc=0;
wire ready,done;wire [3:0] status;
wire [N*32-1:0] issued,completed;
wire [N-1:0] faults;
wire [N*64-1:0] imem_addr;
reg [N*64-1:0] imem_data;
reg [N-1:0] prev_launch=0;
reg [63:0] program_mem[0:255];
reg [1023:0] hexfile;
wire mr,mready,mwe,mrv;
wire [63:0] ma;
wire [511:0] mw,md;
wire [2:0] mt,mrt;
wire [SW-1:0] ms,mrs;
tnca_dram_model #(.SRC_W(SW),.TAG_W(3),.LINES(2048),.LATENCY(11),.RANDOM_STALL(1),.REORDER(1)) ram (
.clk(clk),.rst_n(rst_n),.req_valid(mr),.req_ready(mready),.req_addr(ma),.req_we(mwe),
.req_wdata(mw),.req_tag(mt),.req_src(ms),.rsp_valid(mrv),.rsp_tag(mrt),.rsp_src(mrs),.rsp_rdata(md));
reg stalled=0;
reg [63:0] stalled_addr;
reg [SW-1:0] stalled_src;
reg [2:0] stalled_tag;
reg stalled_we;
reg [511:0] stalled_data;
always @(posedge clk) if(rst_n) begin
 if(stalled && (!mr || ma!=stalled_addr || ms!=stalled_src || mt!=stalled_tag || mwe!=stalled_we || mw!=stalled_data))$fatal(1,"memory request changed under backpressure");
 stalled<=mr && !mready;stalled_addr<=ma;stalled_src<=ms;
 stalled_tag<=mt;stalled_we<=mwe;stalled_data<=mw;
end
integer seen[0:454];integer total=0,cycles=0;
tnca_top #(.NX(MX),.NY(N/MX),.GLOBAL_GRID(1),.SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1)) dut (
.clk(clk),.rst_n(rst_n),.launch_valid(1'b0),.launch_pc(launch_pc),.launch_role_mask(2'b01),
.launch_arg_base(64'h1000),.launch_vbase(14'd0),.grid_valid(valid),.grid_ready(ready),
.grid_id(64'd17),.grid_arg_stride(64'h100),.grid_x(32'd13),.grid_y(32'd7),.grid_z(32'd5),
.grid_cores({N{1'b1}}),.grid_done_valid(done),.grid_done_ready(ack),.grid_done_status(status),
.grid_issued(issued),.grid_completed(completed),.fault_valid(faults),.imem_addr(imem_addr),
.imem_data(imem_data),.mem_req_valid(mr),.mem_req_ready(mready),.mem_req_addr(ma),
.mem_req_we(mwe),.mem_req_wdata(mw),.mem_req_tag(mt),.mem_req_src(ms),
.mem_rsp_valid(mrv),.mem_rsp_tag(mrt),.mem_rsp_src(mrs),.mem_rsp_rdata(md));
always @(posedge clk)begin
 for(integer t=0;t<N;t=t+1)
  imem_data[t*64+:64]<=MEMORY ? program_mem[(imem_addr[t*64+:64]>>3)%256] : 64'h021f000000000000;
 prev_launch<=dut.tile_launch;
 cycles=cycles+1;
 if(N==8 && MEMORY && cycles%1000==0)begin
  $display("GRID progress cycles=%0d accepted=%0d busy=%h",cycles,total,dut.dispatch_busy);$fflush();
 end
 if(cycles>100000)$fatal(1,"core drain timeout");
end
for(genvar c=0;c<N;c=c+1)begin: monitor
 integer key;
 always @(negedge clk)if(rst_n && prev_launch[c])begin
  key=dut.g_tile[c].u_tile.u_core.pid_x+13*(dut.g_tile[c].u_tile.u_core.pid_y+7*dut.g_tile[c].u_tile.u_core.pid_z);
  if(dut.g_tile[c].u_tile.u_core.arg_base!=64'h1000+c*256)$fatal(1,"actual core private args wrong");
  if(key<0 || key>454 || seen[key])$fatal(1,"actual core duplicate PID");
  seen[key]=1;total=total+1;
  if(dut.g_tile[c].u_tile.u_core.npid_x!=13 || dut.g_tile[c].u_tile.u_core.npid_y!=7 || dut.g_tile[c].u_tile.u_core.npid_z!=5)$fatal(1,"actual core grid wrong");
 end
end
initial begin
 if(MEMORY) begin
  for(integer w=0;w<256;w=w+1)program_mem[w]=64'h021f000000000000;
  if(!$value$plusargs("hex=%s",hexfile))$fatal(1,"missing memory kernel");
  $readmemh(hexfile,program_mem);
 end
 for(integer k=0;k<455;k=k+1)seen[k]=0;
 repeat(5)@(negedge clk);rst_n=1;
 for(integer phase=0;phase<(MEMORY?2:1);phase=phase+1)begin
  for(integer k=0;k<455;k=k+1)seen[k]=0;
  total=0;launch_pc=phase*256;valid=1;
  @(negedge clk);valid=0;
  wait(done);@(negedge clk);
  if(status!=0) $display("DEBUG expected=%h actual=%h address=%h dram0=%h",dut.g_tile[0].u_tile.u_core.u_srf.mem[1],dut.g_tile[0].u_tile.u_core.u_srf.mem[6],dut.g_tile[0].u_tile.u_core.u_srf.mem[5],ram.mem[1024][63:0]);
  if(status!=0 || faults!=0 || total!=455 || !dut.all_idle)$fatal(1,"actual core execution failed status=%0d faults=%h total=%0d pc=%h",status,faults,total,dut.g_tile[0].u_tile.u_core.fault_pc);
  for(integer k=0;k<455;k=k+1)if(!seen[k])$fatal(1,"actual core missing PID");
  for(integer k=0;k<N;k=k+1)if(issued[k*32+:32]==0 || issued[k*32+:32]!=completed[k*32+:32])$fatal(1,"inactive or undrained core");
  if(MEMORY && ram.n_rd==0)$fatal(1,"no real memory traffic");
  $display("GRID REAL CORES PASS N=%0d memory=%0d phase=%0d programs=%0d cycles=%0d",N,MEMORY,phase,total,cycles);
  ack=1;@(negedge clk);ack=0;
 end
 $finish;
end
endmodule
