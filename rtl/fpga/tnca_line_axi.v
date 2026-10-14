// NUCA full-line port to AXI4 INCR bursts. One instance per memory outlet.
// Unique home ownership implements atomics/masks above this adapter; writes
// here replace a complete 64-byte line. Independent AXI IDs retain source/tag.
// NUCA writebacks are fire-and-forget: B releases the internal slot but
// produces no home response. WRITE_RESPONSES=1 is an explicit diagnostic mode.
// Same-line accesses with a write wait for completion, preserving acceptance
// order across AXI's independent channels. Different lines overlap.
// Reset MUST reset the attached AXI fabric too (including pending AW/W).
// Fault stops admission and drains well-formed accepted responses. Malformed
// or missing replies require platform recovery; no completion is fabricated.
`default_nettype none
module tnca_line_axi #(
 parameter SRC_W=2,TAG_W=3,OUT_N=8,DATA_W=256,
 parameter ID_W=(OUT_N>1?$clog2(OUT_N):1),
 parameter MEM_BASE=64'h20000,WRITE_RESPONSES=0
)(
 input wire clk,rst_n,
 input wire req_valid,output wire req_ready,
 input wire [63:0] req_addr,input wire req_we,
 input wire [511:0] req_wdata,input wire [TAG_W-1:0] req_tag,
 input wire [SRC_W-1:0] req_src,
 output reg rsp_valid,output reg [511:0] rsp_rdata,
 output reg [TAG_W-1:0] rsp_tag,output reg [SRC_W-1:0] rsp_src,
 input wire [63:0] axi_base,axi_size,
 output reg fault,output wire idle,
 output reg [63:0] read_lines,write_lines,read_beats,write_beats,
 output wire m_awvalid,input wire m_awready,output wire [63:0] m_awaddr,
 output wire [ID_W-1:0] m_awid,output wire [7:0] m_awlen,
 output wire [2:0] m_awsize,output wire [1:0] m_awburst,
 output wire m_wvalid,input wire m_wready,output wire [DATA_W-1:0] m_wdata,
 output wire [DATA_W/8-1:0] m_wstrb,output wire m_wlast,
 input wire m_bvalid,output wire m_bready,input wire [ID_W-1:0] m_bid,
 input wire [1:0] m_bresp,
 output wire m_arvalid,input wire m_arready,output wire [63:0] m_araddr,
 output wire [ID_W-1:0] m_arid,output wire [7:0] m_arlen,
 output wire [2:0] m_arsize,output wire [1:0] m_arburst,
 input wire m_rvalid,output wire m_rready,input wire [DATA_W-1:0] m_rdata,
 input wire [ID_W-1:0] m_rid,input wire [1:0] m_rresp,input wire m_rlast
);
 localparam BEATS=512/DATA_W;
 localparam BEAT_W=(BEATS>1?$clog2(BEATS):1);
 reg [OUT_N-1:0] active,writing,sent;
 reg [63:0] address[0:OUT_N-1],bus_address[0:OUT_N-1];
 reg [511:0] line[0:OUT_N-1];
 reg [SRC_W-1:0] source[0:OUT_N-1];
 reg [TAG_W-1:0] tag[0:OUT_N-1];
 reg [BEAT_W-1:0] rbeat[0:OUT_N-1];
 reg issue_valid,aw_done,w_done;
 reg [ID_W-1:0] issue_slot;
 reg [BEAT_W-1:0] wbeat;
 integer i,free_slot;
 reg hazard;
 wire legal_address=req_addr>=MEM_BASE && req_addr[5:0]==0 &&
   axi_base[5:0]==0 && axi_size>=64 && req_addr-MEM_BASE<=axi_size-64 &&
   axi_base<=64'hffffffffffffffc0-(req_addr-MEM_BASE);
 always @* begin
  free_slot=-1;hazard=0;
  for(integer k=OUT_N-1;k>=0;k=k-1)begin
   if(!active[k])free_slot=k;
   if(active[k] && address[k]==req_addr && (writing[k] || req_we))hazard=1;
  end
 end
 // Invalid addresses are refused and latched as faults, never translated into
 // zero-valued successful memory responses.
 assign req_ready=rst_n && !fault && !issue_valid && free_slot>=0 && !hazard && legal_address;
 assign idle=!(|active) && !issue_valid;
 assign m_arvalid=issue_valid && !writing[issue_slot];
 assign m_araddr=bus_address[issue_slot];
 assign m_arid=issue_slot;assign m_arlen=BEATS-1;
 assign m_arsize=$clog2(DATA_W/8);assign m_arburst=2'b01;
 assign m_awvalid=issue_valid && writing[issue_slot] && !aw_done;
 assign m_awaddr=m_araddr;assign m_awid=issue_slot;
 assign m_awlen=BEATS-1;assign m_awsize=m_arsize;assign m_awburst=2'b01;
 assign m_wvalid=issue_valid && writing[issue_slot] && !w_done;
 assign m_wdata=line[issue_slot][wbeat*DATA_W+:DATA_W];
 assign m_wstrb={DATA_W/8{1'b1}};assign m_wlast=wbeat==BEATS-1;
 wire read_match=m_rid<OUT_N && active[m_rid] && !writing[m_rid] && sent[m_rid];
 wire read_final=m_rvalid && read_match && rbeat[m_rid]==BEATS-1 && m_rlast;
 wire write_match=m_bid<OUT_N && active[m_bid] && writing[m_bid] && sent[m_bid];
 assign m_rready=rst_n;
 // Only one line completion per cycle at the non-backpressured home port.
 assign m_bready=rst_n && !read_final;
 generate
  if(!(DATA_W==64 || DATA_W==128 || DATA_W==256 || DATA_W==512))begin:g_bad_width
   tnca_invalid_axi_data_width invalid_configuration();
  end
  if(OUT_N<1 || OUT_N>(1<<ID_W))begin:g_bad_slots
   tnca_invalid_axi_slot_geometry invalid_configuration();
  end
  if(MEM_BASE%64!=0)begin:g_bad_base
   tnca_invalid_axi_base_alignment invalid_configuration();
  end
 endgenerate
 always @(posedge clk)begin
  if(!rst_n)begin
   active<=0;writing<=0;sent<=0;issue_valid<=0;issue_slot<=0;
   aw_done<=0;w_done<=0;wbeat<=0;fault<=0;rsp_valid<=0;
   rsp_src<=0;rsp_tag<=0;rsp_rdata<=0;
   read_lines<=0;write_lines<=0;read_beats<=0;write_beats<=0;
   for(i=0;i<OUT_N;i=i+1)begin address[i]<=0;bus_address[i]<=0;line[i]<=0;source[i]<=0;tag[i]<=0;rbeat[i]<=0;end
  end else begin
   rsp_valid<=0;
   if(req_valid && !fault && !legal_address)fault<=1;
   if(req_valid && req_ready)begin
    active[free_slot]<=1;writing[free_slot]<=req_we;sent[free_slot]<=0;
    address[free_slot]<=req_addr;bus_address[free_slot]<=axi_base+(req_addr-MEM_BASE);line[free_slot]<=req_wdata;
    source[free_slot]<=req_src;tag[free_slot]<=req_tag;rbeat[free_slot]<=0;
    issue_slot<=free_slot;issue_valid<=1;aw_done<=0;w_done<=0;wbeat<=0;
   end
   if(m_arvalid && m_arready)begin
    issue_valid<=0;sent[issue_slot]<=1;read_lines<=read_lines+1;
   end
   if(m_awvalid && m_awready)begin aw_done<=1;write_lines<=write_lines+1;end
   if(m_wvalid && m_wready)begin
    write_beats<=write_beats+1;
    if(m_wlast)w_done<=1;else wbeat<=wbeat+1;
   end
   if(issue_valid && writing[issue_slot] &&
     (aw_done || (m_awvalid && m_awready)) &&
     (w_done || (m_wvalid && m_wready && m_wlast)))begin
    issue_valid<=0;sent[issue_slot]<=1;
   end
   if(m_rvalid && m_rready)begin
    if(!read_match)fault<=1;
    else begin
     read_beats<=read_beats+1;
     if(m_rresp!=0 || m_rlast!=(rbeat[m_rid]==BEATS-1))fault<=1;
     line[m_rid][rbeat[m_rid]*DATA_W+:DATA_W]<=m_rdata;
     if(read_final)begin
      active[m_rid]<=0;rsp_valid<=1;rsp_src<=source[m_rid];rsp_tag<=tag[m_rid];
      rsp_rdata<=line[m_rid];rsp_rdata[rbeat[m_rid]*DATA_W+:DATA_W]<=m_rdata;
     end else if(rbeat[m_rid]<BEATS-1)rbeat[m_rid]<=rbeat[m_rid]+1;
    end
   end
   if(m_bvalid && m_bready)begin
    if(!write_match)fault<=1;
    else begin
     if(m_bresp!=0)fault<=1;
     active[m_bid]<=0;
     if(WRITE_RESPONSES)begin rsp_valid<=1;rsp_src<=source[m_bid];rsp_tag<=tag[m_bid];rsp_rdata<=0;end
    end
   end
  end
 end
endmodule
`default_nettype wire
