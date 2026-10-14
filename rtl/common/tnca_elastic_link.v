// Each stage has two registered payload slots. Ready derives only from local
// occupancy; no combinational ready path spans a physical boundary. Separate
// instances for virtual networks preserve their independent buffering.
module tnca_elastic_link #(
 parameter W=32,STAGES=2
)(
 input wire clk,rst_n,
 input wire in_valid,output wire in_ready,input wire [W-1:0] in_data,
 output wire out_valid,input wire out_ready,output wire [W-1:0] out_data,
 output wire empty
);
 wire [STAGES:0] valid,ready;
 wire [(STAGES+1)*W-1:0] data;
 wire [STAGES:0] occupied;
 assign valid[0]=in_valid;assign in_ready=ready[0];assign data[0+:W]=in_data;
 assign out_valid=valid[STAGES];assign ready[STAGES]=out_ready;
 assign out_data=data[STAGES*W+:W];assign occupied[STAGES]=0;
 assign empty=!(|occupied);
 generate for(genvar s=0;s<STAGES;s=s+1)begin:g_stage
  wire [1:0] count;
  tnca_fifo #(.W(W),.DEPTH(2),.AW(1)) queue(
   .clk(clk),.rst_n(rst_n),.in_valid(valid[s]),.in_ready(ready[s]),.in_data(data[s*W+:W]),
   .out_valid(valid[s+1]),.out_ready(ready[s+1]),.out_data(data[(s+1)*W+:W]),.count(count));
  assign occupied[s]=|count;
 end
 if(STAGES<0)begin:g_bad_stages
  tnca_invalid_elastic_link_stage_count invalid_configuration();
 end endgenerate
endmodule
