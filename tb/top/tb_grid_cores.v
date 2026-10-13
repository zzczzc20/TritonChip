`timescale 1ns/1ps
// Executes real tnca_core EXIT pipelines; does not replace Torch/board acceptance.
module tb_grid_cores;
parameter N=3;
reg clk=0;always #5 clk=~clk;
reg rst_n=0,valid=0,ack=0;
wire ready,done;wire [3:0] status;
wire [N*32-1:0] issued,completed;
wire [N-1:0] faults;
wire [N*64-1:0] imem_addr;
reg [N*64-1:0] imem_data;
reg [N-1:0] prev_launch=0;
integer seen[0:454];integer total=0,cycles=0;
tnca_top #(.NX(N),.NY(1),.GLOBAL_GRID(1),.SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1)) dut (
.clk(clk),.rst_n(rst_n),.launch_valid(1'b0),.launch_pc(64'd0),.launch_role_mask(2'b01),
.launch_arg_base(64'h1000),.launch_vbase(14'd0),.grid_valid(valid),.grid_ready(ready),
.grid_id(64'd17),.grid_arg_stride(64'h100),.grid_x(32'd13),.grid_y(32'd7),.grid_z(32'd5),
.grid_cores({N{1'b1}}),.grid_done_valid(done),.grid_done_ready(ack),.grid_done_status(status),
.grid_issued(issued),.grid_completed(completed),.fault_valid(faults),.imem_addr(imem_addr),
.imem_data(imem_data),.mem_req_ready(1'b1),.mem_rsp_valid(1'b0),.mem_rsp_tag(3'd0),
.mem_rsp_src(2'd0),.mem_rsp_rdata(512'd0));
always @(posedge clk)begin
 imem_data<={N{64'h021f000000000000}};
 prev_launch<=dut.tile_launch;
 cycles=cycles+1;
 if(cycles>30000)$fatal(1,"core drain timeout");
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
 for(integer k=0;k<455;k=k+1)seen[k]=0;
 repeat(5)@(negedge clk);rst_n=1;valid=1;
 @(negedge clk);valid=0;
 wait(done);@(negedge clk);
 if(status!=0 || faults!=0 || total!=455 || !dut.all_idle)$fatal(1,"actual core execution failed");
 for(integer k=0;k<455;k=k+1)if(!seen[k])$fatal(1,"actual core missing PID");
 for(integer k=0;k<N;k=k+1)if(issued[k*32+:32]==0 || issued[k*32+:32]!=completed[k*32+:32])$fatal(1,"inactive or undrained core");
 $display("GRID REAL CORES PASS N=%0d programs=%0d cycles=%0d",N,total,cycles);$finish;
end
endmodule
