`timescale 1ns/1ps
module tb_lsu_hybrid_reset;
 reg clk=0;always #5 clk=~clk;
 reg rst_n=0,valid=0,rv=0,ack=0,vector=1,atomic=0;
 reg [2:0] size=2;
 reg [63:0] base=64'h1000;
 reg [3:0] rt=0;
 reg [511:0] rd=0;
 wire ready,idle,mv,cv,mamo;
 wire [3:0] mt;
 wire [63:0] ma,be,mc,scalar;
 wire [511:0] wd;
 tnca_lsu_hybrid #(.RCTX_W(2)) dut(
  .clk(clk),.rst_n(rst_n),.req_valid(valid),.req_ready(idle),.req_ready_load(ready),
  .req_store(1'b0),.req_vector(vector),.req_indexed(1'b0),.req_amo(atomic),.req_size(size),.req_signed(1'b0),
  .req_hint(2'd0),.req_base(base),.req_offset(64'd0),.req_vdata(512'd0),.req_index(512'd0),.req_sdata(64'd1),
  .req_cmp(64'hfedcba9876543210),.req_aop(4'd0),.req_mask(16'hffff),.req_rctx(2'd1),.req_dst_class(3'd1),.req_dst_idx(8'd9),
  .cpl_valid(cv),.cpl_ack(ack),.cpl_sdata(scalar),.mem_req_valid(mv),.mem_req_ready(1'b1),.mem_req_tag(mt),
  .mem_req_addr(ma),.mem_req_be(be),.mem_req_wdata(wd),.mem_req_amo(mamo),.mem_req_cmp(mc),
  .mem_rsp_valid(rv),.mem_rsp_tag(rt),.mem_rsp_rdata(rd));
 task issue;
  begin @(negedge clk);while(!(atomic?idle:ready))@(negedge clk);valid=1;@(negedge clk);valid=0;end
 endtask
 integer i;
 initial begin
  repeat(3)@(negedge clk);rst_n=1;
  for(i=0;i<8;i=i+1)issue;
  repeat(3)@(negedge clk);
  if(idle || ready || cv || dut.fast.count!=8)$fatal(1,"hybrid did not reserve all fast slots");
  rst_n=0;repeat(3)@(negedge clk);
  if(!idle || cv || mv || dut.atomic_pending)$fatal(1,"hybrid full reset failed");
  rst_n=1;vector=0;atomic=1;size=3;base=64'h1038;issue;
  while(!mv)@(negedge clk);
  if(ma!=64'h1038 || be!=64'hff00000000000000 || wd[511:448]!=64'd1 || !mamo || mc!=64'hfedcba9876543210 || mt[3])
   $fatal(1,"narrow atomic word placement/compare changed");
  rt=mt;rd=0;rd[511:448]=64'h123456789abcdef0;
  @(negedge clk);rv=1;@(negedge clk);rv=0;
  while(!cv)@(negedge clk);
  if(scalar!==64'h123456789abcdef0 || idle || ready)$fatal(1,"atomic result/quiescence");
  vector=1;atomic=0;size=2;base=64'h1000;valid=1;
  repeat(6)begin @(negedge clk);if(ready || mv || !cv || scalar!==64'h123456789abcdef0)$fatal(1,"younger work crossed held atomic");end
  valid=0;ack=1;repeat(3)@(negedge clk);
  if(!idle || cv || mv || dut.atomic_pending)$fatal(1,"atomic completion did not release barrier");
  // Narrow scalar MMIO address survives expansion; read data returns in its word position.
  vector=0;size=3;base=64'h7fff0120;ack=0;issue;
  while(!mv)@(negedge clk);
  if(ma!=64'h7fff0120 || mt[3])$fatal(1,"MMIO address changed");
  rt=mt;rd=0;rd[319:256]=64'h89abcdef76543210;
  @(negedge clk);rv=1;@(negedge clk);rv=0;
  while(!cv)@(negedge clk);
  if(scalar!==64'h89abcdef76543210)$fatal(1,"MMIO word extraction");
  rst_n=0;repeat(3)@(negedge clk);
  if(!idle || cv || mv)$fatal(1,"held completion reset failed");
  $display("HYBRID RESET/BARRIER PASS full slots, atomic isolation, MMIO placement, held completion reset");$finish;
 end
 initial begin #50000;$fatal(1,"hybrid reset timeout");end
endmodule
