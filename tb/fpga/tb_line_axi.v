`timescale 1ns/1ps
module tb_line_axi;
 parameter DATA_W=256,OUT_N=8,MODE=0,STAGES=0,WRITE_ACK=1;
 localparam ID_W=(OUT_N>1?$clog2(OUT_N):1),BEATS=512/DATA_W;
 reg clk=0,rst_n=0;always #5 clk=~clk;
 reg req_valid=0,req_we=0;wire req_ready;
 reg [63:0] req_addr=0;reg [511:0] req_wdata=0;
 reg [4:0] req_tag=0;reg [2:0] req_src=0;
 wire rsp_valid,fault,idle;wire [511:0] rsp_rdata;
 wire [4:0] rsp_tag;wire [2:0] rsp_src;
 wire [63:0] read_lines,write_lines,read_beats,write_beats;
 wire m_awvalid,m_wvalid,m_arvalid,m_wlast,m_rready,m_bready;
 wire [63:0] m_awaddr,m_araddr;wire [ID_W-1:0] m_awid,m_arid;
 wire [7:0] m_awlen,m_arlen;wire [2:0] m_awsize,m_arsize;
 wire [1:0] m_awburst,m_arburst;
 wire [DATA_W-1:0] m_wdata;wire [DATA_W/8-1:0] m_wstrb;
 reg m_awready=0,m_wready=0,m_arready=0,m_rvalid=0,m_bvalid=0,m_rlast=0;
 reg [ID_W-1:0] m_rid=0,m_bid=0;reg [1:0] m_rresp=0,m_bresp=0;
 reg [DATA_W-1:0] m_rdata=0;
 wire qv,qr,qwe,rv;wire [63:0] qa;wire [511:0] qd,rd;
 wire [2:0] qs,rs;wire [4:0] qt,rt;wire link_idle;
 tnca_line_link #(.SRC_W(3),.TAG_W(5),.STAGES(STAGES)) link(
 .clk(clk),.rst_n(rst_n),.idle(link_idle),
 .s_req_valid(req_valid),.s_req_ready(req_ready),.s_req_addr(req_addr),.s_req_we(req_we),.s_req_wdata(req_wdata),.s_req_src(req_src),.s_req_tag(req_tag),
 .s_rsp_valid(rsp_valid),.s_rsp_rdata(rsp_rdata),.s_rsp_src(rsp_src),.s_rsp_tag(rsp_tag),
 .m_req_valid(qv),.m_req_ready(qr),.m_req_addr(qa),.m_req_we(qwe),.m_req_wdata(qd),.m_req_src(qs),.m_req_tag(qt),
 .m_rsp_valid(rv),.m_rsp_rdata(rd),.m_rsp_src(rs),.m_rsp_tag(rt));
 tnca_line_axi #(.WRITE_RESPONSES(WRITE_ACK),.SRC_W(3),.TAG_W(5),.OUT_N(OUT_N),.DATA_W(DATA_W)) dut(
  .clk(clk),.rst_n(rst_n),.req_valid(qv),.req_ready(qr),
  .req_addr(qa),.req_we(qwe),.req_wdata(qd),.req_src(qs),.req_tag(qt),
  .rsp_valid(rv),.rsp_rdata(rd),.rsp_src(rs),.rsp_tag(rt),
  .axi_base(64'h100000000),.axi_size(64'd1024),.fault(fault),.idle(idle),
  .read_lines(read_lines),.write_lines(write_lines),.read_beats(read_beats),.write_beats(write_beats),
  .m_awvalid(m_awvalid),.m_awready(m_awready),.m_awaddr(m_awaddr),.m_awid(m_awid),
  .m_awlen(m_awlen),.m_awsize(m_awsize),.m_awburst(m_awburst),
  .m_wvalid(m_wvalid),.m_wready(m_wready),.m_wdata(m_wdata),.m_wstrb(m_wstrb),.m_wlast(m_wlast),
  .m_bvalid(m_bvalid),.m_bready(m_bready),.m_bid(m_bid),.m_bresp(m_bresp),
  .m_arvalid(m_arvalid),.m_arready(m_arready),.m_araddr(m_araddr),.m_arid(m_arid),
  .m_arlen(m_arlen),.m_arsize(m_arsize),.m_arburst(m_arburst),
  .m_rvalid(m_rvalid),.m_rready(m_rready),.m_rdata(m_rdata),.m_rid(m_rid),.m_rresp(m_rresp),.m_rlast(m_rlast));
 reg [511:0] memory[0:15],oracle[0:15],expected[0:255];
 reg seen[0:255],wanted[0:255];
 reg [OUT_N-1:0] reads=0,writes=0;
 reg [511:0] write_data[0:OUT_N-1];
 integer location[0:OUT_N-1],beat[0:OUT_N-1],due[0:OUT_N-1];
 reg aw_have=0;integer aw_slot=0,aw_line=0,wcount=0;
 reg [511:0] incoming=0;
 integer cycle=0,accepted=0,completed=0,peak=0,inflight=0,collisions=0;
 integer i,j,pick,seq,index,seed=123;
 reg [511:0] payload;
 reg ar_stalled=0,aw_stalled=0,w_stalled=0;
 reg [63:0] old_araddr,old_awaddr;reg [ID_W-1:0] old_arid,old_awid;
 reg [DATA_W-1:0] old_wdata;reg old_wlast;
 // AXI slave: independent AW/W acceptance, delayed B, interleaved read IDs.
 always @(posedge clk)begin
  if(rst_n)begin
   cycle=cycle+1;
   if(ar_stalled && (!m_arvalid || m_araddr!==old_araddr || m_arid!==old_arid))$fatal(1,"AR changed under stall");
   if(aw_stalled && (!m_awvalid || m_awaddr!==old_awaddr || m_awid!==old_awid))$fatal(1,"AW changed under stall");
   if(w_stalled && (!m_wvalid || m_wdata!==old_wdata || m_wlast!==old_wlast))$fatal(1,"W changed under stall");
   ar_stalled=m_arvalid&&!m_arready;old_araddr=m_araddr;old_arid=m_arid;
   aw_stalled=m_awvalid&&!m_awready;old_awaddr=m_awaddr;old_awid=m_awid;
   w_stalled=m_wvalid&&!m_wready;old_wdata=m_wdata;old_wlast=m_wlast;
   if(req_valid && req_ready)begin
    seq={req_src,req_tag};index=(req_addr-64'h20000)/64;
    wanted[seq]=!req_we || WRITE_ACK;expected[seq]=req_we?0:oracle[index];
    if(req_we)oracle[index]=req_wdata;
    if(!req_we || WRITE_ACK)accepted=accepted+1;
   end
   if(rsp_valid)begin
    seq={rsp_src,rsp_tag};
    if(!wanted[seq] || seen[seq])$fatal(1,"lost/duplicate source tag %d",seq);
    if(rsp_rdata!==expected[seq])$fatal(1,"line mismatch source tag %d",seq);
    seen[seq]=1;completed=completed+1;
   end
   inflight=accepted-completed;if(inflight>peak)peak=inflight;
   if(m_arvalid && m_arready)begin
    if(m_arlen!=BEATS-1 || m_arsize!=$clog2(DATA_W/8) || m_arburst!=1)$fatal(1,"read geometry");
    if(reads[m_arid] || writes[m_arid])$fatal(1,"ID reused before completion");
    reads[m_arid]=1;location[m_arid]=(m_araddr-64'h100000000)/64;
    beat[m_arid]=0;due[m_arid]=cycle+25+(m_arid*11)%31;
   end
   if(m_awvalid && m_awready)begin
    if(aw_have || m_awlen!=BEATS-1 || m_awsize!=$clog2(DATA_W/8) || m_awburst!=1)$fatal(1,"write geometry/order");
    aw_have=1;aw_slot=m_awid;aw_line=(m_awaddr-64'h100000000)/64;
   end
   if(m_wvalid && m_wready)begin
    if(m_wstrb!=={DATA_W/8{1'b1}} || m_wlast!=(wcount==BEATS-1))$fatal(1,"write beat geometry");
    incoming[wcount*DATA_W+:DATA_W]=m_wdata;wcount=wcount+1;
   end
   if(aw_have && wcount==BEATS)begin
    if(reads[aw_slot] || writes[aw_slot])$fatal(1,"write ID reused");
    writes[aw_slot]=1;location[aw_slot]=aw_line;write_data[aw_slot]=incoming;
    due[aw_slot]=cycle+40+(aw_slot*7)%23;aw_have=0;wcount=0;
   end
   if(m_rvalid && m_rready)begin
    if(m_rlast)reads[m_rid]=0;else beat[m_rid]=beat[m_rid]+1;
   end
   if(m_bvalid && m_bready)begin
    memory[location[m_bid]]=write_data[m_bid];writes[m_bid]=0;
   end
   if(m_rvalid && m_bvalid)collisions=collisions+1;
   if(!m_rvalid || m_rready)begin
    pick=-1;
    for(j=0;j<OUT_N;j=j+1)if(reads[j] && due[j]<=cycle && (pick<0 || ($random(seed)&1)))pick=j;
    m_rvalid<=pick>=0;
    if(pick>=0)begin
     m_rid<=pick;m_rdata<=memory[location[pick]][beat[pick]*DATA_W+:DATA_W];m_rlast<=beat[pick]==BEATS-1;
     m_rresp<=MODE==1?2:0;
    end
   end
   if(!m_bvalid || m_bready)begin
    pick=-1;for(j=OUT_N-1;j>=0;j=j-1)if(writes[j] && due[j]<=cycle)pick=j;
    m_bvalid<=pick>=0;if(pick>=0)begin m_bid<=pick;m_bresp<=MODE==2?2:0;end
   end
   // Initially allow W well before AW, then independent random channel stalls.
   m_awready<=cycle>12 && (($random(seed)&32'h7fffffff)%100)>35;
   m_wready<=(($random(seed)&32'h7fffffff)%100)>40;
   m_arready<=(($random(seed)&32'h7fffffff)%100)>30;
   if(cycle>30000)$fatal(1,"timeout accepted=%d completed=%d",accepted,completed);
  end
 end
 task send;
  input integer number,write,line_index;
  begin
   @(negedge clk);req_valid=1;req_src=number>>5;req_tag=number;
   req_addr=64'h20000+line_index*64;req_we=write;
   for(integer word=0;word<8;word=word+1)req_wdata[word*64+:64]=64'hac00000000000000+number*256+word;
   @(posedge clk);while(!req_ready)@(posedge clk);
   @(negedge clk);req_valid=0;
  end
 endtask
 initial begin
  for(i=0;i<16;i=i+1)begin memory[i]=0;oracle[i]=0;end
  for(i=0;i<256;i=i+1)begin seen[i]=0;wanted[i]=0;expected[i]=0;end
  repeat(4)@(negedge clk);rst_n=1;
  if(MODE==4)begin
   send(1,0,0);wait(m_rvalid);@(negedge clk);force m_rlast=1;
   @(posedge clk);@(negedge clk);release m_rlast;
   repeat(4)@(negedge clk);
   if(!fault || completed!=0 || idle || req_ready)$fatal(1,"malformed burst fabricated drain");
  end else if(MODE==1 || MODE==2)begin
   send(1,MODE==2,0);wait(fault);wait(completed==1);@(negedge clk);
   if(!idle || req_ready)$fatal(1,"fault did not drain/stop admission");
  end else begin
   for(i=0;i<16;i=i+1)send(i,1,i);
   wait((WRITE_ACK ? completed==16 : idle) && link_idle);
   for(i=16;i<192;i=i+1)send(i,(i%5==0),i%16);
   wait(completed==(WRITE_ACK ? 192 : 141) && idle && link_idle);@(negedge clk);
   if(fault || !idle || !link_idle || (OUT_N>1 && peak<2))$fatal(1,"normal drain or concurrency");
   if(read_beats!=read_lines*BEATS || write_beats!=write_lines*BEATS)$fatal(1,"traffic count mismatch");
   if(STAGES==0)begin
   if(MODE==3)begin
    force m_rvalid=1;force m_rid=0;force m_rlast=1;
    @(posedge clk);@(negedge clk);release m_rvalid;release m_rid;release m_rlast;
    if(!fault || !idle || completed!=(WRITE_ACK ? 192 : 141))$fatal(1,"stray response was not dropped");
   end
   // Refuse an invalid/unmapped line without AXI side effects or completion.
   req_valid=1;req_addr=64'h20400;@(posedge clk);@(negedge clk);
   if(!fault || req_ready || !idle)$fatal(1,"invalid address not refused");req_valid=0;
   if(MODE==5)begin
    rst_n=0;repeat(3)@(negedge clk);
    if(fault || !idle || read_lines || write_lines || read_beats || write_beats)$fatal(1,"shared fabric reset");
    rst_n=1;
   end
   end
  end
  $display("LINE AXI PASS width=%0d slots=%0d mode=%0d stages=%0d write_ack=%0d completed=%0d peak=%0d collisions=%0d reads=%0d writes=%0d",DATA_W,OUT_N,MODE,STAGES,WRITE_ACK,completed,peak,collisions,read_lines,write_lines);$finish;
 end
endmodule
