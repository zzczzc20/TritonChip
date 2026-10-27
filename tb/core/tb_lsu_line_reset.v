`timescale 1ns/1ps
module tb_lsu_line_reset;
 parameter SLOTS=8,TAG_W=3,BAD_RESPONSE=0;
 reg clk=0;always #5 clk=~clk;
 reg rst_n=0,valid=0,rv=0,ack=1;
 reg [TAG_W-1:0] rt=0;
 wire ready,eligible,quiet,mv,cv;
 wire [TAG_W-1:0] mt;
 wire [511:0] result;
 wire [31:0] accepted,completed;
 tnca_lsu_line #(.SLOTS(SLOTS),.TAG_W(TAG_W)) dut(
  .clk(clk),.rst_n(rst_n),.req_valid(valid),.req_eligible(eligible),.req_ready(ready),.quiescent(quiet),
  .req_store(1'b0),.req_vector(1'b1),.req_indexed(1'b0),.req_amo(1'b0),.req_size(3'd2),.req_signed(1'b0),
  .req_hint(2'd0),.req_base(64'h1000),.req_offset(64'd0),.req_vdata(512'd0),.req_mask(16'hffff),
  .req_rctx(2'd0),.req_dst_class(3'd2),.req_dst_idx(8'd9),.cpl_valid(cv),.cpl_ack(ack),.cpl_vdata(result),
  .mem_req_valid(mv),.mem_req_ready(1'b1),.mem_req_tag(mt),.mem_rsp_valid(rv),.mem_rsp_tag(rt),.mem_rsp_rdata(512'd0),
  .perf_accepted(accepted),.perf_completed(completed));
 task issue;
  begin @(negedge clk);while(!ready)@(negedge clk);valid=1;@(negedge clk);valid=0;end
 endtask
 integer i;
 initial begin
  repeat(3)@(negedge clk);rst_n=1;
  if(BAD_RESPONSE) begin rv=1;repeat(3)@(negedge clk);$fatal(1,"missing invalid response guard");end
  for(i=0;i<SLOTS;i=i+1)issue;
  repeat(4)@(negedge clk);
  if(quiet || ready || cv || accepted!=SLOTS)$fatal(1,"did not occupy every response slot");
  rst_n=0;repeat(3)@(negedge clk);
  if(!quiet || cv || mv || accepted!=0 || completed!=0)$fatal(1,"full reset failed");
  rst_n=1;issue;
  while(!mv)@(negedge clk);rt=mt;
  @(negedge clk);rv=1;@(negedge clk);rv=0;
  while(!cv)@(negedge clk);
  if(result!==512'd0)$fatal(1,"post-reset result");
  repeat(3)@(negedge clk);
  if(!quiet || cv || mv || accepted!=1 || completed!=1)$fatal(1,"post-reset drain/counters");
  $display("LINE LSU RESET PASS slots=%0d full abort and recovery",SLOTS);$finish;
 end
 initial begin #10000;$fatal(1,"line reset timeout");end
endmodule
