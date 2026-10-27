// Ordered per-PC stream engine, independent of the serialized core/NUCA path.
// This component alone is not an image: the binding must own/cache-maintain
// both ranges, serialize commands against core work, and route this AXI port
// to its PC. All beats are complete 64-byte lines. Copy ranges cannot overlap.
// Modes: 0 byte-exact copy, 1 repeat a 32-bit bit pattern, 2 sum all uint32
// words modulo 2^32. Mode 2 does not implement floating-point reduction.
// AXI ID zero orders bursts; BRAM buffers copy data without a per-line CAM.
// Faults require coordinated fabric reset; no successful completion is forged.
`default_nettype none
module tnca_stream_axi #(
 parameter APERTURE_BITS=28,BURST_BEATS=16,DATA_DEPTH=128,BURST_OUT=8,
 parameter CREDIT_W=$clog2(DATA_DEPTH+1),META_AW=$clog2(BURST_OUT)
)(
 input wire clk,rst_n,
 input wire cmd_valid,output wire cmd_ready,
 input wire [1:0] cmd_mode,input wire [63:0] cmd_pc_base,
 input wire [APERTURE_BITS-1:0] cmd_src,cmd_dst,
 input wire [APERTURE_BITS-6:0] cmd_lines,input wire [31:0] cmd_pattern,
 output reg busy,fault,output reg done_valid,input wire done_ready,
 output reg [3:0] done_status,output reg [31:0] result_sum,
 output reg [63:0] cycles,read_lines,write_lines,
 output reg m_arvalid,input wire m_arready,output reg [63:0] m_araddr,
 output wire m_arid,output wire [7:0] m_arlen,output wire [2:0] m_arsize,
 output wire [1:0] m_arburst,
 input wire m_rvalid,output wire m_rready,input wire [511:0] m_rdata,
 input wire m_rid,input wire [1:0] m_rresp,input wire m_rlast,
 output reg m_awvalid,input wire m_awready,output reg [63:0] m_awaddr,
 output wire m_awid,output wire [7:0] m_awlen,output wire [2:0] m_awsize,
 output wire [1:0] m_awburst,
 output wire m_wvalid,input wire m_wready,output wire [511:0] m_wdata,
 output wire [63:0] m_wstrb,output wire m_wlast,
 input wire m_bvalid,output wire m_bready,input wire m_bid,input wire [1:0] m_bresp
);
 localparam COPY=0,FILL=1,SUM_U32=2;
 localparam [APERTURE_BITS:0] PC_BYTES=64'd1<<APERTURE_BITS;
 reg [1:0] mode;
 reg [31:0] pattern;
 reg [APERTURE_BITS-6:0] read_left,write_left;
 reg [63:0] next_read,next_write;
 reg [6:0] ar_length,aw_length,read_pos,write_pos;
 reg [CREDIT_W:0] reserved;
 reg [META_AW:0] write_bursts,write_sent;
 wire [APERTURE_BITS:0] byte_count={cmd_lines,6'd0};
 wire [APERTURE_BITS:0] src_end={1'b0,cmd_src}+byte_count;
 wire [APERTURE_BITS:0] dst_end={1'b0,cmd_dst}+byte_count;
 wire legal=cmd_mode<=SUM_U32 && cmd_pc_base[APERTURE_BITS-1:0]==0 &&
   cmd_pc_base<=64'hffffffffffffffff-(PC_BYTES-1) &&
   cmd_lines<=(1<<(APERTURE_BITS-6)) &&
   (cmd_mode==FILL || (cmd_src[5:0]==0 && src_end<=PC_BYTES)) &&
   (cmd_mode==SUM_U32 || (cmd_dst[5:0]==0 && dst_end<=PC_BYTES)) &&
   (cmd_mode!=COPY || cmd_lines==0 || src_end<=cmd_dst || dst_end<=cmd_src);
 assign cmd_ready=rst_n && !busy && !done_valid && !fault && !m_rvalid && !m_bvalid;
 function [6:0] burst_length;
  input [APERTURE_BITS-6:0] remaining;
  input [63:0] address;
  reg [6:0] n,boundary;
  begin
   boundary=7'd64-{1'b0,address[11:6]};
   n=BURST_BEATS;
   if(remaining<n)n=remaining;
   if(boundary<n)n=boundary;
   burst_length=n;
  end
 endfunction
 wire [6:0] read_length=burst_length(read_left,next_read);
 wire [6:0] write_length=burst_length(write_left,next_write);
 wire ar_fire=m_arvalid && m_arready,aw_fire=m_awvalid && m_awready;
 wire r_fire=m_rvalid && m_rready,w_fire=m_wvalid && m_wready;
 wire b_fire=m_bvalid && m_bready;
 wire rmeta_ready,rmeta_valid,wmeta_ready,wmeta_valid;
 wire [6:0] rmeta_length,wmeta_length;
 wire read_last=read_pos+7'd1==rmeta_length;
 wire write_last=write_pos+7'd1==wmeta_length;
 tnca_fifo #(.W(7),.DEPTH(BURST_OUT),.AW(META_AW)) read_metadata(
  .clk(clk),.rst_n(rst_n),.in_valid(ar_fire),.in_ready(rmeta_ready),.in_data(ar_length),
  .out_valid(rmeta_valid),.out_ready(r_fire && read_last),.out_data(rmeta_length),.count());
 tnca_fifo #(.W(7),.DEPTH(BURST_OUT),.AW(META_AW)) write_metadata(
  .clk(clk),.rst_n(rst_n),.in_valid(aw_fire),.in_ready(wmeta_ready),.in_data(aw_length),
  .out_valid(wmeta_valid),.out_ready(w_fire && write_last),.out_data(wmeta_length),.count());
 wire data_ready,data_valid;wire [511:0] data;
 tnca_stream_fifo #(.W(512),.DEPTH(DATA_DEPTH)) copy_data(
  .clk(clk),.rst_n(rst_n),.in_valid(r_fire && mode==COPY),.in_ready(data_ready),.in_data(m_rdata),
  .out_valid(data_valid),.out_ready(w_fire && mode==COPY),.out_data(data));
 assign m_arid=0;assign m_arlen=ar_length-7'd1;assign m_arsize=6;assign m_arburst=1;
 assign m_awid=0;assign m_awlen=aw_length-7'd1;assign m_awsize=6;assign m_awburst=1;
 assign m_rready=rst_n && busy && rmeta_valid && (mode!=COPY || data_ready);
 assign m_bready=rst_n;
 assign m_wvalid=rst_n && busy && wmeta_valid && (mode==FILL || data_valid);
 assign m_wdata=mode==FILL ? {16{pattern}} : data;
 assign m_wstrb=64'hffffffffffffffff;assign m_wlast=write_last;
 wire release_credit=mode==COPY ? w_fire : r_fire;
 reg [3:0] sum_valid;
 reg [31:0] sum8[0:7],sum4[0:3],sum2[0:1],sum1;
 integer k;
 generate
  if(APERTURE_BITS<12 || APERTURE_BITS>32 || BURST_BEATS<1 || BURST_BEATS>64 ||
     DATA_DEPTH<BURST_BEATS || BURST_OUT<2 || (1<<META_AW)!=BURST_OUT)begin:g_bad_geometry
   tnca_invalid_stream_geometry invalid_configuration();
  end
 endgenerate
 always @(posedge clk)begin
  if(!rst_n)begin
   busy<=0;fault<=0;done_valid<=0;done_status<=0;result_sum<=0;cycles<=0;read_lines<=0;write_lines<=0;
   mode<=0;pattern<=0;read_left<=0;write_left<=0;next_read<=0;next_write<=0;
   m_arvalid<=0;m_awvalid<=0;m_araddr<=0;m_awaddr<=0;ar_length<=0;aw_length<=0;
   read_pos<=0;write_pos<=0;reserved<=0;write_bursts<=0;write_sent<=0;sum_valid<=0;
  end else begin
   if(done_valid && done_ready)done_valid<=0;
   if(busy)cycles<=cycles+1'b1;
   sum_valid<={sum_valid[2:0],r_fire && mode==SUM_U32};
   if(r_fire && mode==SUM_U32)for(k=0;k<8;k=k+1)sum8[k]<=m_rdata[k*64+:32]+m_rdata[k*64+32+:32];
   if(sum_valid[0])for(k=0;k<4;k=k+1)sum4[k]<=sum8[k*2]+sum8[k*2+1];
   if(sum_valid[1])for(k=0;k<2;k=k+1)sum2[k]<=sum4[k*2]+sum4[k*2+1];
   if(sum_valid[2])sum1<=sum2[0]+sum2[1];
   if(sum_valid[3])result_sum<=result_sum+sum1;
   if(busy && !fault)begin
    if(!m_arvalid && read_left!=0 && rmeta_ready && reserved+read_length<=DATA_DEPTH)begin
     m_arvalid<=1;m_araddr<=next_read;ar_length<=read_length;
    end
    if(!m_awvalid && write_left!=0 && wmeta_ready && write_bursts<BURST_OUT)begin
     m_awvalid<=1;m_awaddr<=next_write;aw_length<=write_length;
    end
   end
   if(ar_fire)begin
    m_arvalid<=0;next_read<=next_read+{ar_length,6'd0};read_left<=read_left-ar_length;
   end
   if(aw_fire)begin
    m_awvalid<=0;next_write<=next_write+{aw_length,6'd0};write_left<=write_left-aw_length;
   end
   reserved<=reserved+(ar_fire ? ar_length : 0)-(release_credit ? 1 : 0);
   write_bursts<=write_bursts+(aw_fire ? 1 : 0)-(b_fire && write_sent!=0 ? 1 : 0);
   write_sent<=write_sent+(w_fire && write_last ? 1 : 0)-(b_fire && write_sent!=0 ? 1 : 0);
   if(r_fire)begin
    read_lines<=read_lines+1'b1;read_pos<=read_last ? 0 : read_pos+1'b1;
    if(m_rid!=0 || m_rresp!=0 || m_rlast!=read_last)fault<=1;
   end
   if(m_rvalid && (!busy || !rmeta_valid || m_rid!=0))fault<=1;
   if(w_fire)begin write_lines<=write_lines+1'b1;write_pos<=write_last ? 0 : write_pos+1'b1;end
   if(b_fire && (m_bid!=0 || m_bresp!=0 || write_sent==0))fault<=1;
   if(busy && !fault && read_left==0 && write_left==0 && !m_arvalid && !m_awvalid &&
      !rmeta_valid && !wmeta_valid && !m_rvalid && !m_bvalid && write_bursts==0 && reserved==0 && !data_valid && sum_valid==0)begin
    busy<=0;done_valid<=1;done_status<=0;
   end
   if(cmd_valid && cmd_ready)begin
    mode<=cmd_mode;pattern<=cmd_pattern;result_sum<=0;cycles<=0;read_lines<=0;write_lines<=0;
    next_read<=cmd_pc_base+cmd_src;next_write<=cmd_pc_base+cmd_dst;
    read_left<=cmd_mode==FILL ? 0 : cmd_lines;write_left<=cmd_mode==SUM_U32 ? 0 : cmd_lines;
    read_pos<=0;write_pos<=0;sum_valid<=0;
    if(!legal)begin done_valid<=1;done_status<=3;read_left<=0;write_left<=0;end
    else busy<=1;
   end
  end
 end
endmodule
`default_nettype wire
