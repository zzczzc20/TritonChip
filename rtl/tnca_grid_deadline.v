// A deadline requests stop-and-drain. It never fabricates a completion or
// resets an accepted core/transaction. Timeout stays visible until next start.
`default_nettype none
module tnca_grid_deadline(
 input wire clk,rst_n,start,active,
 input wire [63:0] deadline,
 output wire abort_valid,output reg timed_out,
 output reg [63:0] elapsed
);
 reg [63:0] limit;
 assign abort_valid=active && !timed_out && limit!=0 && elapsed>=limit;
 always @(posedge clk)begin
  if(!rst_n)begin limit<=0;elapsed<=0;timed_out<=0;end
  else if(start)begin limit<=deadline;elapsed<=0;timed_out<=0;end
  else begin
   if(active)elapsed<=elapsed+1'b1;
   if(abort_valid)timed_out<=1;
  end
 end
endmodule
`default_nettype wire
