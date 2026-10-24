// Broadcast a snapshotted ownership range, collect each home exactly once,
// then wait for memory links and physical write responses to drain.
//
// The range registers follow base/size while idle and hold from acceptance
// until done (R41 F10). That is the value the accepting edge captured before,
// and nothing reads them while idle (home_valid is busy-gated, and the homes
// sample on valid && ready), but their 128 enables no longer wait on the
// global quiescence tree that `ready` carries across the SLRs.
`default_nettype none
module tnca_cache_handoff #(parameter N=3)(
 input wire clk,rst_n,
 input wire valid,output wire ready,input wire quiescent,
 input wire [63:0] base,size,
 output wire [N-1:0] home_valid,input wire [N-1:0] home_ready,home_done,
 output reg [63:0] home_base,home_size,
 input wire memory_drained,
 output reg done,output reg busy
);
 reg [N-1:0] issued,completed;
 assign ready=rst_n && !busy && quiescent;
 assign home_valid={N{busy}} & ~issued;
 always @(posedge clk)begin
  if(!rst_n)begin busy<=0;done<=0;issued<=0;completed<=0;home_base<=0;home_size<=0;end
  else begin
   done<=0;
   if(!busy)begin home_base<=base;home_size<=size;end
   if(valid && ready)begin
    busy<=1;issued<=0;completed<=0;
   end else if(busy)begin
    issued<=issued | (home_valid & home_ready);
    completed<=completed | (home_done & issued);
    if((&completed) && memory_drained)begin busy<=0;done<=1;end
   end
  end
 end
 generate if(N<1)begin:g_bad tnca_invalid_cache_home_count invalid_configuration();end endgenerate
endmodule
`default_nettype wire
