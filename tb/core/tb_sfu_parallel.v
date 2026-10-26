`timescale 1ns/1ps
module tb_sfu_parallel;
 parameter PAR=4;
 reg clk=0; always #5 clk=~clk;
 reg rst_n=0,start=0;
 reg [7:0] func=0;
 reg [511:0] va=0;
 wire b0,d0,bp,dp;
 wire [511:0] r0,rp;
 tnca_sfu #(.LANES(1)) reference(.clk(clk),.rst_n(rst_n),.start(start),.func(func),.va(va),.busy(b0),.done(d0),.vres(r0));
 tnca_sfu #(.LANES(PAR)) parallel(.clk(clk),.rst_n(rst_n),.start(start),.func(func),.va(va),.busy(bp),.done(dp),.vres(rp));
 reg [511:0] saved;
 integer testno,lane,t,fast,slow,seed=12345;
 reg seen;
 initial begin
  repeat(3) @(negedge clk); rst_n=1;
  for(testno=0;testno<160;testno=testno+1) begin
   @(negedge clk);
   case(testno%4)
    0: func=8'h00;
    1: func=8'h01;
    2: func=8'h02;
    3: func=8'h03;
   endcase
   for(lane=0;lane<16;lane=lane+1) va[lane*32+:32]=$random(seed);
   if(testno<4) begin
    va[31:0]=0; va[63:32]=32'h80000000; va[95:64]=32'h7f800000;
    va[127:96]=32'hff800000; va[159:128]=32'h7fc00001; va[191:160]=32'h00000001;
   end
   start=1; @(negedge clk); start=0;
   // Inputs may change while busy: every slice must have latched the launch.
   va=~va; func=~func;
   t=0; fast=0; slow=0; seen=0;
   while(!d0 && t<40) begin
    @(negedge clk); t=t+1;
    if(dp) begin
     if(seen) $fatal(1,"duplicate completion");
     seen=1; saved=rp; fast=t;
    end
   end
   slow=t;
   if(!d0 || !seen || b0 || bp) $fatal(1,"completion handshake");
   if(saved !== r0) $fatal(1,"SFU bit mismatch PAR=%0d test=%0d",PAR,testno);
   if(slow-fast != 16-((16+PAR-1)/PAR)) $fatal(1,"unexpected latency slow=%0d fast=%0d",slow,fast);
   @(negedge clk); if(d0 || dp) $fatal(1,"completion pulse width");
  end
  // Abort in flight; reset must suppress completion and permit the next launch.
  start=1; repeat(3) @(negedge clk); rst_n=0; start=0;
  repeat(2) @(negedge clk);
  if(b0 || bp || d0 || dp) $fatal(1,"reset did not abort");
  rst_n=1; repeat(30) @(negedge clk);
  if(d0 || dp || b0 || bp) $fatal(1,"stale completion after reset");
  $display("SFU PARALLEL PASS PAR=%0d cases=160 exact bits slow_cycles=%0d fast_cycles=%0d reset",PAR,slow,fast); $finish;
 end
 initial begin #200000; $fatal(1,"timeout"); end
endmodule
