`timescale 1ns/1ps
module tb_multicore_shell;
parameter N=3;
localparam NX=N>4?4:N,SW=N>4?$clog2(N):2;
reg clk=0;always #5 clk=~clk;
reg rst=0,cv=0,ack=0,hv=0,hw=0;
reg [63:0] pc=0,id=11,ha=0,hd=0;
wire cr,dv,hr,hrv,busy,idle;
wire [3:0] ds,hs;
wire [63:0] did,dc,hrd;
wire [N*32-1:0] issued,completed;
wire mr,mready,mwe,mrv;
wire [63:0] ma;wire [511:0] mw,md;
wire [2:0] mt,mrt;wire [SW-1:0] ms,mrs;
reg [63:0] image[0:255];reg [1023:0] hexfile;
integer cycles=0;
tnca_multicore_shell #(.NX(NX),.NY(N/NX),.CODE_WORDS(256),.SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1)) dut(
.clk(clk),.rst_n(rst),.cmd_valid(cv),.cmd_ready(cr),.cmd_id(id),.cmd_pc(pc),
.cmd_arg_base(64'h8000),.cmd_arg_stride(64'h100),.cmd_gx(32'd13),.cmd_gy(32'd7),.cmd_gz(32'd5),
.cmd_cores({N{1'b1}}),.cmd_roles(4'b0001),.cmd_slice(4'd0),.cmd_vbase(32'd0),
.done_valid(dv),.done_ready(ack),.done_status(ds),.done_id(did),.done_cycles(dc),
.issued(issued),.completed(completed),.host_valid(hv),.host_ready(hr),.host_we(hw),
.host_addr(ha),.host_wdata(hd),.host_be(8'hff),.host_rsp_valid(hrv),.host_rsp_data(hrd),.host_rsp_status(hs),
.busy(busy),.all_idle(idle),.mem_req_valid(mr),.mem_req_ready(mready),.mem_req_addr(ma),
.mem_req_we(mwe),.mem_req_wdata(mw),.mem_req_tag(mt),.mem_req_src(ms),
.mem_rsp_valid(mrv),.mem_rsp_tag(mrt),.mem_rsp_src(mrs),.mem_rsp_rdata(md));
tnca_dram_model #(.SRC_W(SW),.TAG_W(3),.LINES(2048),.LATENCY(11),.RANDOM_STALL(1),.REORDER(1)) ram(
.clk(clk),.rst_n(rst),.req_valid(mr),.req_ready(mready),.req_addr(ma),.req_we(mwe),
.req_wdata(mw),.req_tag(mt),.req_src(ms),.rsp_valid(mrv),.rsp_tag(mrt),.rsp_src(mrs),.rsp_rdata(md));
always @(posedge clk)begin cycles=cycles+1;if(cycles>150000)$fatal(1,"shell deadline");end
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
  if(ds!=expected_status || did!=launch_id || !idle)$fatal(1,"completion mismatch");
  if(expected_status==0)begin
   sum=0;for(integer c=0;c<N;c=c+1)begin
    if(issued[c*32+:32]==0 || issued[c*32+:32]!=completed[c*32+:32])$fatal(1,"core not drained");
    sum=sum+issued[c*32+:32];
   end
   if(sum!=455 || dc==0)$fatal(1,"grid count mismatch");
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
 launch(0,11,0);launch(256,12,0);
 for(integer k=0;k<455;k=k+1)word(0,65536+k*8,4096+k,0);
 $display("MULTICORE PERSISTENT SHELL PASS N=%0d",N);$finish;
end
endmodule
