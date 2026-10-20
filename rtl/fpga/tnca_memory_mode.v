// Optional, quiescent-only one/all-outlet routing. Physical home addresses and
// full home source/tag identities never change. Caller changes single_port
// only after all upstream work, links and AXI adapters have drained.
// single_port is used through a registered local copy, so the routing,
// response mux and legality check never depend combinationally on the
// control register (in another SLR). The controller holds new work off for
// longer than this copy takes (tnca_multicore_mmio SETTLE).
`default_nettype none
module tnca_memory_mode #(
 parameter PORTS=3,N_SRC=3,SRC_W=2,TAG_W=3
)(
 input wire clk,rst_n,
 input wire single_port,
 input wire [PORTS-1:0] s_req_valid,output wire [PORTS-1:0] s_req_ready,
 input wire [PORTS*64-1:0] s_req_addr,input wire [PORTS-1:0] s_req_we,
 input wire [PORTS*512-1:0] s_req_wdata,input wire [PORTS*TAG_W-1:0] s_req_tag,
 input wire [PORTS*SRC_W-1:0] s_req_src,
 output wire [PORTS-1:0] s_rsp_valid,output wire [PORTS*TAG_W-1:0] s_rsp_tag,
 output wire [PORTS*SRC_W-1:0] s_rsp_src,output wire [PORTS*512-1:0] s_rsp_rdata,
 output wire [PORTS-1:0] m_req_valid,input wire [PORTS-1:0] m_req_ready,
 output wire [PORTS*64-1:0] m_req_addr,output wire [PORTS-1:0] m_req_we,
 output wire [PORTS*512-1:0] m_req_wdata,output wire [PORTS*TAG_W-1:0] m_req_tag,
 output wire [PORTS*SRC_W-1:0] m_req_src,
 input wire [PORTS-1:0] m_rsp_valid,input wire [PORTS*TAG_W-1:0] m_rsp_tag,
 input wire [PORTS*SRC_W-1:0] m_rsp_src,input wire [PORTS*512-1:0] m_rsp_rdata,
 output reg fault
);
 reg single;
 always @(posedge clk) single<=single_port;
`ifndef SYNTHESIS
 always @(posedge clk) if(rst_n && single!==single_port && ((|s_req_valid) || (|m_rsp_valid)))
  $fatal(1,"tnca_memory_mode: traffic while the routing copy differs from its control");
`endif
 wire merged_valid,merged_we;
 wire [PORTS-1:0] merged_ready;
 wire [63:0] merged_addr;
 wire [511:0] merged_data;
 wire [TAG_W-1:0] merged_tag;
 wire [SRC_W-1:0] selected;
 tnca_mem_arbiter #(.N(PORTS),.N_W(SRC_W),.TAG_W(TAG_W)) arb(
  .clk(clk),.rst_n(rst_n),.s_req_valid(s_req_valid & {PORTS{single}}),
  .s_req_ready(merged_ready),.s_req_addr(s_req_addr),.s_req_we(s_req_we),
  .s_req_wdata(s_req_wdata),.s_req_tag(s_req_tag),
  .s_rsp_valid(),.s_rsp_tag(),.s_rsp_rdata(),
  .m_req_valid(merged_valid),.m_req_ready(m_req_ready[0] && single),
  .m_req_addr(merged_addr),.m_req_we(merged_we),.m_req_wdata(merged_data),
  .m_req_tag(merged_tag),.m_req_src(selected),
  .m_rsp_valid(1'b0),.m_rsp_tag({TAG_W{1'b0}}),
  .m_rsp_src({SRC_W{1'b0}}),.m_rsp_rdata(512'd0));
 genvar p;
 generate for(p=0;p<PORTS;p=p+1)begin:g_port
  wire [SRC_W-1:0] source=single ? m_rsp_src[0+:SRC_W] : m_rsp_src[p*SRC_W+:SRC_W];
  wire legal_source=source<N_SRC && source%PORTS==p;
  assign s_req_ready[p]=single ? merged_ready[p] : m_req_ready[p];
  assign m_req_valid[p]=single ? (p==0 && merged_valid) : s_req_valid[p];
  assign m_req_addr[p*64+:64]=single ? merged_addr : s_req_addr[p*64+:64];
  assign m_req_we[p]=single ? merged_we : s_req_we[p];
  assign m_req_wdata[p*512+:512]=single ? merged_data : s_req_wdata[p*512+:512];
  assign m_req_tag[p*TAG_W+:TAG_W]=single ? merged_tag : s_req_tag[p*TAG_W+:TAG_W];
  assign m_req_src[p*SRC_W+:SRC_W]=single ? s_req_src[selected*SRC_W+:SRC_W] : s_req_src[p*SRC_W+:SRC_W];
  assign s_rsp_valid[p]=(single ? m_rsp_valid[0] : m_rsp_valid[p]) && legal_source;
  assign s_rsp_tag[p*TAG_W+:TAG_W]=single ? m_rsp_tag[0+:TAG_W] : m_rsp_tag[p*TAG_W+:TAG_W];
  assign s_rsp_src[p*SRC_W+:SRC_W]=source;
  assign s_rsp_rdata[p*512+:512]=single ? m_rsp_rdata[0+:512] : m_rsp_rdata[p*512+:512];
 end endgenerate
 integer k;
 always @(posedge clk)begin
  if(!rst_n)fault<=0;
  else for(k=0;k<PORTS;k=k+1)begin
   if(m_rsp_valid[k] && (m_rsp_src[k*SRC_W+:SRC_W]>=N_SRC ||
      (single ? k!=0 : m_rsp_src[k*SRC_W+:SRC_W]%PORTS!=k)))fault<=1;
  end
 end
 generate if(PORTS<1 || PORTS>N_SRC || (1<<SRC_W)<N_SRC)begin:g_bad_geometry
  tnca_invalid_memory_mode_geometry invalid_configuration();
 end endgenerate
endmodule
`default_nettype wire
