// Two-entry elastic request FIFOs cut both payload and ready paths at every
// stage. Responses are one line/cycle with no backpressure, as at the home.
// Platform placement assigns these stages to physical boundaries; this module
// carries no SLR assumptions. idle includes buffered requests and replies.
`default_nettype none
module tnca_line_link #(
 parameter SRC_W=2,TAG_W=3,STAGES=2
)(
 input wire clk,rst_n,
 input wire s_req_valid,output wire s_req_ready,input wire [63:0] s_req_addr,
 input wire s_req_we,input wire [511:0] s_req_wdata,
 input wire [SRC_W-1:0] s_req_src,input wire [TAG_W-1:0] s_req_tag,
 output wire s_rsp_valid,output wire [511:0] s_rsp_rdata,
 output wire [SRC_W-1:0] s_rsp_src,output wire [TAG_W-1:0] s_rsp_tag,
 output wire m_req_valid,input wire m_req_ready,output wire [63:0] m_req_addr,
 output wire m_req_we,output wire [511:0] m_req_wdata,
 output wire [SRC_W-1:0] m_req_src,output wire [TAG_W-1:0] m_req_tag,
 input wire m_rsp_valid,input wire [511:0] m_rsp_rdata,
 input wire [SRC_W-1:0] m_rsp_src,input wire [TAG_W-1:0] m_rsp_tag,
 output wire idle
);
 localparam QW=64+1+512+SRC_W+TAG_W,RW=512+SRC_W+TAG_W;
 wire [STAGES:0] valid,ready,reply_valid;
 wire [(STAGES+1)*QW-1:0] payload;
 wire [(STAGES+1)*RW-1:0] reply;
 wire [STAGES:0] occupied;
 assign valid[0]=s_req_valid;assign s_req_ready=ready[0];
 assign payload[0+:QW]={s_req_addr,s_req_we,s_req_wdata,s_req_src,s_req_tag};
 assign m_req_valid=valid[STAGES];assign ready[STAGES]=m_req_ready;
 assign {m_req_addr,m_req_we,m_req_wdata,m_req_src,m_req_tag}=payload[STAGES*QW+:QW];
 assign reply_valid[STAGES]=m_rsp_valid;
 assign reply[STAGES*RW+:RW]={m_rsp_rdata,m_rsp_src,m_rsp_tag};
 assign s_rsp_valid=reply_valid[0];assign {s_rsp_rdata,s_rsp_src,s_rsp_tag}=reply[0+:RW];
 assign occupied[STAGES]=m_rsp_valid;assign idle=!(|occupied);
 generate for(genvar s=0;s<STAGES;s=s+1)begin:g_stage
  wire [1:0] count;
  tnca_fifo #(.W(QW),.DEPTH(2),.AW(1)) request(
   .clk(clk),.rst_n(rst_n),.in_valid(valid[s]),.in_ready(ready[s]),.in_data(payload[s*QW+:QW]),
   .out_valid(valid[s+1]),.out_ready(ready[s+1]),.out_data(payload[(s+1)*QW+:QW]),.count(count));
  reg rv;reg [RW-1:0] rd;
  always @(posedge clk)begin
   if(!rst_n)begin rv<=0;rd<=0;end
   else begin rv<=reply_valid[s+1];if(reply_valid[s+1])rd<=reply[(s+1)*RW+:RW];end
  end
  assign reply_valid[s]=rv;assign reply[s*RW+:RW]=rd;
  assign occupied[s]=(|count) || rv;
 end endgenerate
 initial if(STAGES<0)$fatal(1,"negative link stage count");
endmodule
`default_nettype wire
