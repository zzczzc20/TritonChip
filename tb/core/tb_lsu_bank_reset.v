`timescale 1ns/1ps
module tb_lsu_bank_reset;
 parameter UNITS=8;
 reg clk=0; always #5 clk=~clk;
 reg rst_n=0,valid=0,rv=0;
 reg [3:0] rt=0;
 wire idle,ready,mv,cv;
 wire [3:0] mt;
 wire [63:0] result;
 tnca_lsu_bank #(.UNITS(UNITS),.NUM_BEATS(16/UNITS)) dut(
  .clk(clk),.rst_n(rst_n),.req_ready(idle),.req_ready_load(ready),.req_valid(valid),
  .req_store(1'b0),.req_vector(1'b0),.req_indexed(1'b0),.req_size(3'd3),.req_signed(1'b0),
  .req_hint(2'd0),.req_base(64'h1000),.req_offset(64'd0),.req_index(512'd0),.req_sdata(64'd0),
  .req_amo(1'b0),.req_aop(4'd0),.req_cmp(64'd0),.req_vdata(512'd0),.req_mask(16'd1),
  .req_rctx(1'b0),.req_dst_class(3'd1),.req_dst_idx(8'd9),
  .cpl_valid(cv),.cpl_ack(1'b1),.cpl_sdata(result),
  .mem_req_valid(mv),.mem_req_ready(1'b1),.mem_req_tag(mt),
  .mem_rsp_valid(rv),.mem_rsp_tag(rt),.mem_rsp_rdata(64'h0123456789abcdef));
 task issue;
  begin
   @(negedge clk); while(!ready) @(negedge clk);
   valid=1; @(negedge clk); valid=0;
  end
 endtask
 integer i;
 initial begin
  repeat(3) @(negedge clk); rst_n=1;
  for(i=0;i<UNITS;i=i+1) issue;
  repeat(10) @(negedge clk);
  if(idle || ready || cv || dut.count!=UNITS) $fatal(1,"bank did not fill with pending loads");
  rst_n=0; repeat(3) @(negedge clk);
  if(!idle || cv || mv || dut.count!=0) $fatal(1,"reset did not discard in-flight bank");
  rst_n=1; issue;
  while(!mv) @(negedge clk);
  rt=mt; @(negedge clk); rv=1; @(negedge clk); rv=0;
  while(!cv) @(negedge clk);
  if(result!==64'h0123456789abcdef) $fatal(1,"wrong post-reset result");
  repeat(3) @(negedge clk);
  if(!idle || cv || mv) $fatal(1,"post-reset completion did not drain");
  $display("LSU BANK RESET PASS units=%0d full-bank abort and recovery",UNITS); $finish;
 end
 initial begin #50000; $fatal(1,"reset timeout"); end
endmodule
