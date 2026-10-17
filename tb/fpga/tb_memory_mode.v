`timescale 1ns/1ps
module tb_memory_mode;
 localparam P=3,S=4,T=3;
 reg clk=0,rst_n=0,single=0;always #5 clk=~clk;
 reg [P-1:0] qv=0,qwe=0,mready=0,rv=0;
 wire [P-1:0] qr,mv,mwe,sv;
 reg [P*64-1:0] qa=0;reg [P*512-1:0] qd=0,rd=0;
 reg [P*T-1:0] qt=0,rt=0;reg [P*S-1:0] qs=0,rs=0;
 wire [P*64-1:0] ma;wire [P*512-1:0] md,sd;
 wire [P*T-1:0] mt,st;wire [P*S-1:0] ms,ss;
 wire fault;
 tnca_memory_mode #(.PORTS(P),.N_SRC(8),.SRC_W(S),.TAG_W(T)) dut(
  .clk(clk),.rst_n(rst_n),.single_port(single),
  .s_req_valid(qv),.s_req_ready(qr),.s_req_addr(qa),.s_req_we(qwe),.s_req_wdata(qd),.s_req_tag(qt),.s_req_src(qs),
  .s_rsp_valid(sv),.s_rsp_tag(st),.s_rsp_src(ss),.s_rsp_rdata(sd),
  .m_req_valid(mv),.m_req_ready(mready),.m_req_addr(ma),.m_req_we(mwe),.m_req_wdata(md),.m_req_tag(mt),.m_req_src(ms),
  .m_rsp_valid(rv),.m_rsp_tag(rt),.m_rsp_src(rs),.m_rsp_rdata(rd),.fault(fault));
 integer k,phase,cycle=0,issued=0,completed=0;
 integer sent[0:P-1],received[0:P-1],due[0:P-1],outlet_hits[0:P-1];
 reg pending[0:P-1],stalled[0:P-1];
 reg [S-1:0] saved_src[0:P-1];reg [T-1:0] saved_tag[0:P-1];
 reg [511:0] saved_data[0:P-1];reg [64+1+512+T+S-1:0] old[0:P-1];
 reg [7:0] random_state=8'hb7;
 task advance;
  begin
   @(posedge clk);
   for(k=0;k<P;k=k+1)begin
    if(stalled[k] && (!mv[k] || {ma[k*64+:64],mwe[k],md[k*512+:512],mt[k*T+:T],ms[k*S+:S]}!==old[k]))$fatal(1,"request changed under backpressure");
    stalled[k]=mv[k]&&!mready[k];old[k]={ma[k*64+:64],mwe[k],md[k*512+:512],mt[k*T+:T],ms[k*S+:S]};
    if(qv[k]&&qr[k])begin sent[k]=sent[k]+1;issued=issued+1;end
    if(mv[k]&&mready[k])begin
     if(single&&k!=0)$fatal(1,"disabled outlet has traffic");
     if(ms[k*S+:S]!=3+(ms[k*S+:S]%P))$fatal(1,"original home source was lost");
     if(mwe[k]!==ma[k*64] || md[k*512+:512]!=={8{ma[k*64+:64]}})$fatal(1,"request payload changed");
     pending[k]=1;due[k]=cycle+3+k;saved_src[k]=ms[k*S+:S];saved_tag[k]=mt[k*T+:T];saved_data[k]=md[k*512+:512];outlet_hits[k]=outlet_hits[k]+1;
    end
    if(sv[k])begin
     if(ss[k*S+:S]!==S'(k+3) || st[k*T+:T]!==T'(received[k]) || sd[k*512+:512]!=={8{64'(1000*k+received[k])}})$fatal(1,"response source/tag/data mismatch");
     received[k]=received[k]+1;completed=completed+1;
    end
   end
   @(negedge clk);cycle=cycle+1;
   random_state={random_state[6:0],random_state[7]^random_state[5]^random_state[4]^random_state[3]};
   for(k=0;k<P;k=k+1)begin
    rv[k]=0;
    if(pending[k]&&due[k]<=cycle)begin
     rv[k]=1;rs[k*S+:S]=saved_src[k];rt[k*T+:T]=saved_tag[k];rd[k*512+:512]=saved_data[k];pending[k]=0;
    end
    mready[k]=!pending[k] && !rv[k] && random_state[k];
    qv[k]=(sent[k]<30 && sent[k]==received[k]);
    qa[k*64+:64]=1000*k+sent[k];qd[k*512+:512]={8{64'(1000*k+sent[k])}};
    qwe[k]=qa[k*64];qt[k*T+:T]=T'(sent[k]);qs[k*S+:S]=S'(k+3);
   end
  end
 endtask
 initial begin
  for(k=0;k<P;k=k+1)begin pending[k]=0;stalled[k]=0;sent[k]=0;received[k]=0;outlet_hits[k]=0;end
  repeat(3)@(negedge clk);rst_n=1;
  for(phase=0;phase<3;phase=phase+1)begin
   single=(phase==1);issued=0;completed=0;
   for(k=0;k<P;k=k+1)begin sent[k]=0;received[k]=0;outlet_hits[k]=0;end
   while(completed<90)advance;
   qv=0;rv=0;repeat(2)@(negedge clk);
   if(issued!=90 || fault || outlet_hits[0]==0 || (!single&&(outlet_hits[1]==0||outlet_hits[2]==0)))$fatal(1,"routing liveness or fault");
   $display("MEMORY MODE phase=%0d single=%0d issued=%0d completed=%0d outlet_hits=%0d,%0d,%0d",phase,single,issued,completed,outlet_hits[0],outlet_hits[1],outlet_hits[2]);
  end
  rs[0+:S]=8;rv=1;#1;if(sv)$fatal(1,"invalid source forwarded");
  @(posedge clk);@(negedge clk);rv=0;if(!fault)$fatal(1,"invalid source not faulted");
  rst_n=0;@(negedge clk);rst_n=1;single=1;rs[2*S+:S]=5;rv=4;
  @(posedge clk);@(negedge clk);rv=0;if(!fault)$fatal(1,"disabled outlet response not faulted");
  $display("MEMORY MODE PASS original sources, tags, payloads, backpressure, 3/1/3 routing and faults");$finish;
 end
 initial begin #300000;$fatal(1,"memory mode timeout");end
endmodule
