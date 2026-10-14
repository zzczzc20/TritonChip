`timescale 1ns/1ps
module tb_mem_ports_r40;
parameter N=3,P=3;
localparam W=N>4?$clog2(N):2;
reg clk=0;always #5 clk=~clk;
reg rst_n=0;
reg [N-1:0] valid=0,writes=0;
wire [N-1:0] ready,replies;
reg [N*64-1:0] addr=0,data=0;
reg [N*3-1:0] tags=0;
wire [N*64-1:0] returned;
wire [N*3-1:0] rtags;
wire [P-1:0] mv,mwe,fault;
reg [P-1:0] mr=0,rv=0;
wire [P*64-1:0] ma,md;
wire [P*3-1:0] mt;
wire [P*W-1:0] ms;
reg [P*3-1:0] rt=0;
reg [P*W-1:0] rs=0;
reg [P*64-1:0] rd=0;
tnca_mem_ports #(.N(N),.PORTS(P),.N_W(W),.LINE_BITS(64)) dut(
.clk(clk),.rst_n(rst_n),.s_req_valid(valid),.s_req_ready(ready),.s_req_addr(addr),
.s_req_we(writes),.s_req_wdata(data),.s_req_tag(tags),.s_rsp_valid(replies),
.s_rsp_tag(rtags),.s_rsp_rdata(returned),.m_req_valid(mv),.m_req_ready(mr),
.m_req_addr(ma),.m_req_we(mwe),.m_req_wdata(md),.m_req_tag(mt),.m_req_src(ms),
.m_rsp_valid(rv),.m_rsp_src(rs),.m_rsp_tag(rt),.m_rsp_rdata(rd),.protocol_fault(fault));
integer sent[0:N-1],done[0:N-1],tick=0,i,p,s,total=0,parallel=0,count;
integer pending[0:P-1],delay[0:P-1];
reg [P-1:0] held=0;
reg [P*64-1:0] held_addr,held_data;
reg [P*3-1:0] held_tag;
reg [P*W-1:0] held_source;
always @(negedge clk)if(rst_n)begin
 tick=tick+1;rv=0;mr=0;
 for(p=0;p<P;p=p+1)begin
  if(pending[p]>=0)begin
   delay[p]=delay[p]-1;
   if(delay[p]==0)begin
    s=pending[p];rv[p]=1;rs[p*W+:W]=s;rt[p*3+:3]=done[s]%8;
    rd[p*64+:64]=64'habc000+s*32+done[s];pending[p]=-1;
   end
  end
  mr[p]=(pending[p]<0) && (p!=0 || tick>12) && ((tick+p)%5!=0);
 end
 for(i=0;i<N;i=i+1)begin
  valid[i]=sent[i]<16 && sent[i]==done[i];writes[i]=sent[i]%2;
  addr[i*64+:64]=4096+i*1024+sent[i]*64;
  data[i*64+:64]=64'habc000+i*32+sent[i];tags[i*3+:3]=sent[i]%8;
 end
end
always @(posedge clk)if(rst_n)begin
 count=0;
 for(p=0;p<P;p=p+1)begin
  if(held[p] && (!mv[p] || ma[p*64+:64]!=held_addr[p*64+:64] || md[p*64+:64]!=held_data[p*64+:64] || mt[p*3+:3]!=held_tag[p*3+:3] || ms[p*W+:W]!=held_source[p*W+:W]))$fatal(1,"stalled outlet changed tuple");
  if(mv[p] && mr[p])begin
   s=ms[p*W+:W];count=count+1;
   if(s>=N || s%P!=p || !valid[s] || ma[p*64+:64]!=4096+s*1024+sent[s]*64 || md[p*64+:64]!=64'habc000+s*32+sent[s] || mt[p*3+:3]!=sent[s]%8 || mwe[p]!=(sent[s]%2))$fatal(1,"request tuple/source lost");
   pending[p]=s;delay[p]=2+(tick%7);sent[s]=sent[s]+1;
   if(p>0 && tick<12)parallel=parallel+1;
  end
 end
 if(count>1)parallel=parallel+1;
 held=mv & ~mr;held_addr=ma;held_data=md;held_tag=mt;held_source=ms;
 for(i=0;i<N;i=i+1)if(replies[i])begin
  if(done[i]>=sent[i] || rtags[i*3+:3]!=done[i]%8 || returned[i*64+:64]!=64'habc000+i*32+done[i])$fatal(1,"response alias/duplicate");
  done[i]=done[i]+1;total=total+1;
 end
end
initial begin
 for(i=0;i<N;i=i+1)begin sent[i]=0;done[i]=0;end
 for(p=0;p<P;p=p+1)begin pending[p]=-1;delay[p]=0;end
 repeat(3)@(negedge clk);rst_n=1;
 while(total<N*16 && tick<5000)@(negedge clk);
 if(total!=N*16 || fault!=0)$fatal(1,"lost traffic or starvation");
 if(P>1 && parallel==0)$fatal(1,"outlets serialized behind blocked port");
 // Stop the traffic model, then challenge source ownership directly.
 @(negedge clk);rst_n=0;valid=0;rv=0;held=0;
 @(posedge clk);#1;if(fault!=0)$fatal(1,"reset fault state");
 if(P>1)begin
  // Wrong-controller response is never routed into a home.
  force dut.m_rsp_valid={{(P-1){1'b0}},1'b1}<<1;
  force dut.m_rsp_src=0;
  force dut.rst_n=1'b1;
  #1;if(replies!=0)$fatal(1,"wrong-controller response delivered");
  @(posedge clk);#1;if(!fault[1])$fatal(1,"protocol fault not reported");
  release dut.m_rsp_valid;release dut.m_rsp_src;release dut.rst_n;
 end
 $display("MEMORY PORTS PASS N=%0d P=%0d responses=%0d parallel=%0d",N,P,total,parallel);$finish;
end
endmodule
