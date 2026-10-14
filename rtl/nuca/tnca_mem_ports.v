// Independent memory outlets selected before arbitration. A home slice is
// bound to one controller (home_id % PORTS), so simultaneous controller
// replies cannot collide at a slice's non-backpressured response port.
// Address placement remains the existing unique-home policy. Controllers
// see byte line addresses and must preserve the complete home source/tag.
`default_nettype none
module tnca_mem_ports #(
 parameter N=3,PORTS=3,N_W=(N>4?$clog2(N):2),ADDR_W=64,LINE_BITS=512,TAG_W=3
)(
 input wire clk,rst_n,
 input wire [N-1:0] s_req_valid,output wire [N-1:0] s_req_ready,
 input wire [N*ADDR_W-1:0] s_req_addr,input wire [N-1:0] s_req_we,
 input wire [N*LINE_BITS-1:0] s_req_wdata,input wire [N*TAG_W-1:0] s_req_tag,
 output wire [N-1:0] s_rsp_valid,output wire [N*TAG_W-1:0] s_rsp_tag,
 output wire [N*LINE_BITS-1:0] s_rsp_rdata,
 output wire [PORTS-1:0] m_req_valid,input wire [PORTS-1:0] m_req_ready,
 output wire [PORTS*ADDR_W-1:0] m_req_addr,output wire [PORTS-1:0] m_req_we,
 output wire [PORTS*LINE_BITS-1:0] m_req_wdata,output wire [PORTS*TAG_W-1:0] m_req_tag,
 output wire [PORTS*N_W-1:0] m_req_src,
 input wire [PORTS-1:0] m_rsp_valid,input wire [PORTS*TAG_W-1:0] m_rsp_tag,
 input wire [PORTS*N_W-1:0] m_rsp_src,input wire [PORTS*LINE_BITS-1:0] m_rsp_rdata,
 output reg [PORTS-1:0] protocol_fault
);
 generate
  if(PORTS<1 || PORTS>N)begin:g_bad_ports
   tnca_invalid_memory_port_count invalid_configuration();
  end
  if((1<<N_W)<N)begin:g_bad_source_width
   tnca_invalid_memory_source_width invalid_configuration();
  end
 endgenerate
 genvar p,s;
 generate for(p=0;p<PORTS;p=p+1) begin: g_port
  wire [N-1:0] requests,ready,replies;
  wire [N*TAG_W-1:0] tags;
  wire [N*LINE_BITS-1:0] data;
  wire [N_W-1:0] source=m_rsp_src[p*N_W+:N_W];
  wire legal_source=source<N && source%PORTS==p;
  for(s=0;s<N;s=s+1) begin: g_home
   assign requests[s]=(s%PORTS==p) && s_req_valid[s];
   if(s%PORTS==p)begin
    assign s_req_ready[s]=ready[s];
    assign s_rsp_valid[s]=replies[s];
    assign s_rsp_tag[s*TAG_W+:TAG_W]=tags[s*TAG_W+:TAG_W];
    assign s_rsp_rdata[s*LINE_BITS+:LINE_BITS]=data[s*LINE_BITS+:LINE_BITS];
   end
  end
  tnca_mem_arbiter #(.N(N),.N_W(N_W),.ADDR_W(ADDR_W),.LINE_BITS(LINE_BITS),.TAG_W(TAG_W)) arb(
   .clk(clk),.rst_n(rst_n),.s_req_valid(requests),.s_req_ready(ready),
   .s_req_addr(s_req_addr),.s_req_we(s_req_we),.s_req_wdata(s_req_wdata),.s_req_tag(s_req_tag),
   .s_rsp_valid(replies),.s_rsp_tag(tags),.s_rsp_rdata(data),
   .m_req_valid(m_req_valid[p]),.m_req_ready(m_req_ready[p]),
   .m_req_addr(m_req_addr[p*ADDR_W+:ADDR_W]),.m_req_we(m_req_we[p]),
   .m_req_wdata(m_req_wdata[p*LINE_BITS+:LINE_BITS]),.m_req_tag(m_req_tag[p*TAG_W+:TAG_W]),
   .m_req_src(m_req_src[p*N_W+:N_W]),.m_rsp_valid(m_rsp_valid[p] && legal_source),
   .m_rsp_src(source),.m_rsp_tag(m_rsp_tag[p*TAG_W+:TAG_W]),.m_rsp_rdata(m_rsp_rdata[p*LINE_BITS+:LINE_BITS]));
  always @(posedge clk)begin
   if(!rst_n)protocol_fault[p]<=0;
   else if(m_rsp_valid[p] && !legal_source)protocol_fault[p]<=1;
  end
 end endgenerate
endmodule
`default_nettype wire
