`timescale 1ns/1ps
// tb_home_pipe_special -- tb_nuca_wide_special on tnca_home_pipe (R41 G4): every
// scalar atomic at every position, cold and hot, a merged read/write/read miss,
// and the TMU register page.
module tb_home_pipe_special;
 parameter PIPE=1,HIT_CHAIN=0,FAST_FILL_READ=1,BACKGROUND_READ_REQUESTS=0,LINE_WRITE_NOFILL=0;
 reg clk=0;always #5 clk=~clk;
 reg rst_n=0,qv=0,qwe=0,qamo=0,rr=0;
 reg [63:0] qa=0,qbe=0;
 reg [511:0] qdata=0,qcmp=0;
 reg [3:0] qop=0;
 reg [4:0] qtag=7;
 integer fast_count=0;
 wire qr,rv,mv,mready,mwe,mrv,tw;
 wire [511:0] rd,mwd,mrd;
 wire [63:0] ma,twdata;
 wire [1:0] mt,mrt;
 wire [11:0] troff,twoff;
 wire [63:0] trdata=64'hfedcba9876540000^{52'd0,troff};
 wire [4:0] rtag;
 wire [3:0] rsrc;
 reg [31:0] cycle=0;
 integer mmwrites=0;
 reg [11:0] lastoff;
 reg [63:0] lastdata;
 wire twready=cycle%5!=0;
 tnca_home_pipe #(.LINE_WRITE_NOFILL(LINE_WRITE_NOFILL),.WORD_BITS(512),.WORD_BYTES(64),.FULL_LINE_TAG(1),.HOME_W(2),
  .SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1),.MSHR_N(4),.MSHR_W(2),.TARGET_N(8),.TARGET_W(3),
  .TAG_W(5),.DCO_T(2)) dut(
  .clk(clk),.rst_n(rst_n),.maint_valid(1'b0),.maint_base(64'd0),.maint_size(64'd0),
  .req_valid(qv),.req_ready(qr),.req_addr(qa),.req_we(qwe),.req_be(qbe),.req_wdata(qdata),
  .req_src(4'd2),.req_tag(qtag),.req_amo(qamo),.req_aop(qop),.req_cmp(qcmp),
  .rsp_valid(rv),.rsp_ready(rr),.rsp_rdata(rd),.rsp_tag(rtag),.rsp_src(rsrc),
  .mem_req_valid(mv),.mem_req_ready(mready),.mem_req_addr(ma),.mem_req_we(mwe),.mem_req_wdata(mwd),.mem_req_tag(mt),
  .mem_rsp_valid(mrv),.mem_rsp_tag(mrt),.mem_rsp_rdata(mrd),
  .dco_owner_epoch(1'b0),.dco_cfg(76'd0),.dco_t_valid(2'd0),.dco_t_base(128'd0),.dco_t_size(128'd0),
  .dco_t_end(130'd0),.dco_t_bypass(2'd0),.dco_t_lsb(12'd0),.dco_dead_valid(16'd0),.dco_dead_id(256'd0),.dco_slow(16'd0),
  .tmu_wr_valid(tw),.tmu_wr_ready(twready),.tmu_wr_off(twoff),.tmu_wr_data(twdata),.tmu_rd_off(troff),.tmu_rd_data(trdata),
  .tmu_tl_ready(1'b1));
 tnca_dram_model #(.LINES(2048),.TAG_W(2),.SRC_W(1),.LATENCY(17),.RANDOM_STALL(1),.REORDER(1)) dram(
  .clk(clk),.rst_n(rst_n),.req_valid(mv),.req_ready(mready),.req_addr(ma),.req_we(mwe),.req_wdata(mwd),
  .req_tag(mt),.req_src(1'b0),.rsp_valid(mrv),.rsp_tag(mrt),.rsp_rdata(mrd));
 `include "tnca_amo.vh"
 always @(posedge clk)if(rst_n)begin
  cycle<=cycle+1;
  if(tw && twready)begin mmwrites=mmwrites+1;lastoff=twoff;lastdata=twdata;end
 end
 integer transactions=0;
 task tx;
  input [63:0] addr,be;
  input wr,atomic;
  input [3:0] op;
  input [511:0] data,cmp,wanted;
  integer waitcycles;
  begin
   @(negedge clk);while(!qr)@(negedge clk);
   qa=addr;qbe=be;qwe=wr;qamo=atomic;qop=op;qdata=data;qcmp=cmp;qtag=7;qv=1;
   @(negedge clk);qv=0;
   waitcycles=0;while(!rv && waitcycles<2000)begin @(negedge clk);waitcycles=waitcycles+1;end
   if(!rv || rd!==wanted || rtag!=7 || rsrc!=2)$fatal(1,"wide home special mismatch tx=%0d addr=%h op=%0d got=%h expected=%h",transactions,addr,op,rd,wanted);
   repeat(2)begin @(negedge clk);if(!rv || rd!==wanted)$fatal(1,"response not held");end
   rr=1;@(negedge clk);rr=0;transactions=transactions+1;
  end
 endtask
 task enqueue;
  input [4:0] tag;input wr;input [511:0] data;
  begin
   @(negedge clk);while(!qr)@(negedge clk);
   qa=64'd32768;qbe=64'hff;qwe=wr;qamo=0;qop=0;qdata=data;qcmp=0;qtag=tag;qv=1;
   @(negedge clk);qv=0;
  end
 endtask
 integer kind,g,op,n=0,offset,bytes,merge_before,fast_before;
 reg [63:0] addr,mask64,old,arg,cmp,result,word;
 reg [511:0] gold,packed_arg,compare;
 initial begin
  repeat(4)@(negedge clk);rst_n=1;
  for(kind=0;kind<3;kind=kind+1)for(g=0;g<8;g=g+1)for(op=0;op<12;op=op+1)begin
   addr=(n+1)*64;offset=g*8+(kind==2?4:0);bytes=kind==0?8:4;
   mask64=(kind==0?64'hff:64'hf)<<offset;
   arg=kind==0?64'hffffffff00000001:64'h80000001;
   packed_arg=0;packed_arg[offset*8+:32]=arg[31:0];if(kind==0)packed_arg[offset*8+:64]=arg;
   gold=0;compare=0;
   tx(addr+offset,mask64,0,1,op,packed_arg,compare,gold); // atomic on cold miss/WAKE
   result=amo_alu(op,kind!=0,64'd0,arg,64'd0);
   if(kind==0)gold[offset*8+:64]=result;else gold[offset*8+:32]=result[31:0];
   tx(addr,0,0,0,0,0,0,gold);
   for(integer w=0;w<8;w=w+1)gold[w*64+:64]=64'h1020304050607080^(w*111+n);
   tx(addr,~64'd0,1,0,0,gold,0,0);
   old=gold>>(offset*8);if(kind!=0)old={32'd0,old[31:0]};
   cmp=g%2==0?old:old^64'd1;compare=0;compare[63:0]=cmp;
   tx(addr+offset,mask64,0,1,op,packed_arg,compare,gold); // hot atomic, preserve all other words/bytes
   result=amo_alu(op,kind!=0,old,arg,cmp);
   if(kind==0)gold[offset*8+:64]=result;else gold[offset*8+:32]=result[31:0];
   tx(addr,0,0,0,0,0,0,gold);n=n+1;
  end
  if(FAST_FILL_READ)begin
   merge_before=dut.perf_merge;fast_before=fast_count;
   enqueue(8,0,0);enqueue(9,1,512'h123456789abcdef0);enqueue(10,0,0);
   while(!rv)@(negedge clk);
   if(rtag!=8 || rd!==512'd0)$fatal(1,"merged initial read");
   rr=1;@(negedge clk);
   while(!rv)@(negedge clk);
   if(rtag!=9 || rd!==512'd0)$fatal(1,"merged store completion");
   @(negedge clk);while(!rv)@(negedge clk);
   if(rtag!=10 || rd!==512'h123456789abcdef0)$fatal(1,"merged read after store");
   @(negedge clk);rr=0;
   if(dut.perf_merge<=merge_before)$fatal(1,"the second and third requests did not merge into the miss");
   $display("HOME PIPE MERGED READ-WRITE PASS");
  end
  for(g=0;g<8;g=g+1)begin
   addr=64'h7fff0100+g*8;gold=0;gold[g*64+:64]=64'hfedcba9876540000^(64'h100+g*8);
   tx(addr,0,0,0,0,0,0,gold);
   packed_arg=0;packed_arg[g*64+:64]=64'h1234567800000000+g;
   tx(addr,64'hff<<(g*8),1,0,0,packed_arg,0,0);
   if(mmwrites!=g+1 || lastoff!=12'h100+g*8 || lastdata!==64'h1234567800000000+g)$fatal(1,"MMIO write positioning");
  end
  $display("HOME PIPE SPECIAL PASS nofill=%0d atomic_cases=%0d transactions=%0d MMIO_words=8",LINE_WRITE_NOFILL,n,transactions);$finish;
 end
 initial begin #5000000;$fatal(1,"home pipe special watchdog");end
endmodule
