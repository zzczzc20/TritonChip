// Registered-output FIFO for wide stream data. The memory read is synchronous
// so Vivado can use block RAM. An empty queue bypasses into a holding register.
// The memory holds DEPTH entries, plus one output entry; callers may reserve
// only DEPTH entries to keep the credit calculation conservative.
`default_nettype none
module tnca_stream_fifo #(
 parameter W=512,DEPTH=128,AW=$clog2(DEPTH)
)(
 input wire clk,rst_n,
 input wire in_valid,output wire in_ready,input wire [W-1:0] in_data,
 output reg out_valid, input wire out_ready,output wire [W-1:0] out_data
);
 (* ram_style="block" *) reg [W-1:0] memory[0:DEPTH-1];
 reg [AW-1:0] rp,wp;
 reg [AW:0] count;
 reg [W-1:0] memory_data,bypass_data;
 reg output_bypass;
 wire load_output=!out_valid || out_ready;
 wire read_memory=load_output && count!=0;
 wire bypass=load_output && count==0 && in_valid && in_ready;
 wire write_memory=in_valid && in_ready && !bypass;
 assign in_ready=count<DEPTH;
 assign out_data=output_bypass ? bypass_data : memory_data;
 always @(posedge clk)begin
  if(write_memory)memory[wp]<=in_data;
  if(read_memory)memory_data<=memory[rp];
  if(bypass)bypass_data<=in_data;
  if(!rst_n)begin rp<=0;wp<=0;count<=0;out_valid<=0;output_bypass<=0;end
  else begin
   if(load_output)begin out_valid<=read_memory || bypass;output_bypass<=bypass;end
   if(write_memory)wp<=wp==DEPTH-1 ? 0 : wp+1'b1;
   if(read_memory)rp<=rp==DEPTH-1 ? 0 : rp+1'b1;
   case({write_memory,read_memory})
    2'b10:count<=count+1'b1;
    2'b01:count<=count-1'b1;
    default:;
   endcase
  end
 end
endmodule
`default_nettype wire
