`timescale 1ns/1ps
module tb_mem_arbiter_r40;
parameter N=3;
localparam W=N>4?$clog2(N):2;
reg clk=0;always #5 clk=~clk;
reg rst_n=0,ready=0,rv=0;
reg [N-1:0] valid=0;
reg [N*64-1:0] addr=0,data=0;
reg [N*3-1:0] tags=0;
wire [N-1:0] sr,srv;
wire mv;wire [63:0] ma,md;
wire [2:0] mt;wire [W-1:0] ms;
reg [W-1:0] rs=0;
wire [N*64-1:0] rd;wire [N*3-1:0] rt;
tnca_mem_arbiter #(.N(N),.N_W(W),.LINE_BITS(64)) dut(
.clk(clk),.rst_n(rst_n),.s_req_valid(valid),.s_req_ready(sr),.s_req_addr(addr),
.s_req_we({N{1'b1}}),.s_req_wdata(data),.s_req_tag(tags),.s_rsp_valid(srv),
.s_rsp_tag(rt),.s_rsp_rdata(rd),.m_req_valid(mv),.m_req_ready(ready),
.m_req_addr(ma),.m_req_we(),.m_req_wdata(md),.m_req_tag(mt),.m_req_src(ms),
.m_rsp_valid(rv),.m_rsp_tag(3'd5),.m_rsp_src(rs),.m_rsp_rdata(64'h12345678));
integer j, selected;
initial begin
 for(j=0;j<N;j=j+1)begin addr[j*64+:64]=4096+j*64;data[j*64+:64]=j+17;tags[j*3+:3]=j%8;end
 repeat(3)@(negedge clk);rst_n=1;valid[N-1]=1;
 @(negedge clk);valid={N{1'b1}};
 repeat(4)begin
  @(negedge clk);
  if(!mv || ms!=N-1 || ma!=4096+(N-1)*64 || md!=N+16 || sr!=0)$fatal(1,"backpressure changed selected request");
 end
 ready=1;
 for(j=N-1;j>=0;j=j-1)begin
  @(posedge clk);
  if(!mv || !valid[ms] || sr!=({{(N-1){1'b0}},1'b1}<<ms))$fatal(1,"request source lost");
  selected=ms;
  @(negedge clk);valid[selected]=0;
 end
 #1;if(mv)$fatal(1,"extra request");
 rv=1;
 for(j=N-1;j>=0;j=j-1)begin
  rs=j;#1;
  if(srv!=({{(N-1){1'b0}},1'b1}<<j) || rt[j*3+:3]!=5 || rd[j*64+:64]!=64'h12345678)$fatal(1,"response source aliased");
  @(negedge clk);
 end
 if(N<(1<<W))begin rs=N;#1;if(srv!=0)$fatal(1,"invalid source aliased");end
 $display("R40 MEMORY ARBITER PASS N=%0d",N);$finish;
end
endmodule
