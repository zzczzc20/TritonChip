`timescale 1ns/1ps
module tb_cache_handoff;
 parameter N=3;
 reg clk=0,rst_n=0,valid=0,quiescent=0,drained=0;
 reg [63:0] base=0,size=0;
 wire ready,busy,done;wire [63:0] home_base,home_size;
 wire [N-1:0] offers;reg [N-1:0] accept=0,finished=0;
 integer i,observer,cycle=0,counts[0:N-1];
 always #5 clk=~clk;
 tnca_cache_handoff #(.N(N)) dut(.clk(clk),.rst_n(rst_n),.valid(valid),.ready(ready),
  .quiescent(quiescent),.base(base),.size(size),.home_valid(offers),.home_ready(accept),
  .home_done(finished),.home_base(home_base),.home_size(home_size),
  .memory_drained(drained),.done(done),.busy(busy));
 always @(posedge clk)begin
  cycle=cycle+1;
  if(cycle>1000)$fatal(1,"handoff timeout");
  if(rst_n)for(observer=0;observer<N;observer=observer+1)
   if(offers[observer] && accept[observer])counts[observer]=counts[observer]+1;
 end
 initial begin
  for(i=0;i<N;i=i+1)counts[i]=0;
  repeat(2)@(negedge clk);rst_n=1;valid=1;base=64'h100000009;size=1;
  repeat(3)@(negedge clk);if(ready || busy || |offers)$fatal(1,"admitted before quiescence");
  quiescent=1;@(posedge clk);if(!ready)$fatal(1,"idle handoff refused");
  @(negedge clk);valid=0;base=0;size=0;quiescent=0;
  // Staggered home acceptance and out-of-order completion.
  for(i=0;i<N;i=i+1)begin
   @(negedge clk);accept[i]=1;
   @(negedge clk);accept[i]=0;
  end
  if(home_base!==64'h100000009 || home_size!==1)$fatal(1,"range not snapshotted");
  for(i=N-1;i>=0;i=i-1)begin
   @(negedge clk);finished[i]=1;
   @(negedge clk);finished[i]=0;
  end
  repeat(4)@(negedge clk);
  if(done || !busy || |offers)$fatal(1,"completed before physical drain or duplicate offer");
  for(i=0;i<N;i=i+1)if(counts[i]!=1)$fatal(1,"home accepted more or less than once");
  drained=1;@(negedge clk);
  if(!done || busy)$fatal(1,"drained handoff did not complete");
  @(negedge clk);if(done)$fatal(1,"completion not a pulse");
  // Reset discards an unfinished ownership epoch without a stale grant.
  quiescent=1;valid=1;@(negedge clk);valid=0;
  @(negedge clk);rst_n=0;
  repeat(2)@(negedge clk);rst_n=1;
  repeat(3)@(negedge clk);if(done || busy || |offers)$fatal(1,"stale handoff after reset");
  $display("CACHE HANDOFF PASS homes=%0d",N);$finish;
 end
endmodule
