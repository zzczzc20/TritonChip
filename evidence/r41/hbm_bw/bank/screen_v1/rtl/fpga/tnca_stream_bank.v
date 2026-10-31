// Flat-register binding for independent ordered stream lanes. The runtime
// must serialize this CU with the core CU and perform cache ownership handoff.
// Full command validation precedes every lane admission. Config writes while
// busy are ignored. Faults require coordinated engine/fabric reset.
`default_nettype none
module tnca_stream_bank #(
 parameter LANES=32,SEL_W=(LANES>1?$clog2(LANES):1)
)(
 input wire clk,rst_n,input wire [11:0] reg_addr,input wire [63:0] reg_wdata,
 input wire reg_we,output reg [63:0] reg_rdata,output wire busy,fault,
 output wire [LANES-1:0] m_arvalid,input wire [LANES-1:0] m_arready,
 output wire [LANES*64-1:0] m_araddr,output wire [LANES-1:0] m_arid,
 output wire [LANES*8-1:0] m_arlen,output wire [LANES*3-1:0] m_arsize,output wire [LANES*2-1:0] m_arburst,
 input wire [LANES-1:0] m_rvalid,output wire [LANES-1:0] m_rready,
 input wire [LANES*512-1:0] m_rdata,input wire [LANES-1:0] m_rid,
 input wire [LANES*2-1:0] m_rresp,input wire [LANES-1:0] m_rlast,
 output wire [LANES-1:0] m_awvalid,input wire [LANES-1:0] m_awready,
 output wire [LANES*64-1:0] m_awaddr,output wire [LANES-1:0] m_awid,
 output wire [LANES*8-1:0] m_awlen,output wire [LANES*3-1:0] m_awsize,output wire [LANES*2-1:0] m_awburst,
 output wire [LANES-1:0] m_wvalid,input wire [LANES-1:0] m_wready,
 output wire [LANES*512-1:0] m_wdata,output wire [LANES*64-1:0] m_wstrb,output wire [LANES-1:0] m_wlast,
 input wire [LANES-1:0] m_bvalid,output wire [LANES-1:0] m_bready,
 input wire [LANES-1:0] m_bid,input wire [LANES*2-1:0] m_bresp
);
 localparam IDLE=0,DISPATCH=1,WAIT=2,SUM=3,ACK=4;
 reg [2:0] state;
 reg [63:0] request_seq,served,done_seq,cycles,aperture_base,aperture_size;
 reg [3:0] status;
 reg [1:0] mode;
 reg [31:0] pattern,result_sum;
 reg [LANES-1:0] enabled,active,pending;
 reg [SEL_W-1:0] selected,cursor;
 reg [27:0] source[0:LANES-1],destination[0:LANES-1];
 reg [22:0] lines[0:LANES-1];
 reg [63:0] total_reads,total_writes;
 wire [LANES-1:0] ready,done,lane_fault,legal;
 wire [LANES*4-1:0] lane_status;
 wire [LANES*32-1:0] lane_sum;
 wire [LANES*64-1:0] lane_cycles,lane_reads,lane_writes;
 assign busy=state!=IDLE;
 assign fault=|lane_fault;
 generate for(genvar p=0;p<LANES;p=p+1)begin:g_lane
  wire [28:0] bytes={lines[p],6'd0};
  wire [28:0] src_end={1'b0,source[p]}+bytes,dst_end={1'b0,destination[p]}+bytes;
  assign legal[p]=lines[p]<=23'd4194304 &&
   (mode==1 || (source[p][5:0]==0 && src_end<=29'h10000000)) &&
   (mode==2 || (destination[p][5:0]==0 && dst_end<=29'h10000000)) &&
   (mode!=0 || lines[p]==0 || src_end<=destination[p] || dst_end<=source[p]);
  localparam [63:0] PC_BASE=64'h10000000*p;
  wire [63:0] araddr,awaddr;
  // The port is physically pinned to PC p. Local address wrap after the last
  // valid beat cannot create an access to another pseudo-channel.
  assign m_araddr[p*64+:64]={31'd0,5'(p),araddr[27:0]};
  assign m_awaddr[p*64+:64]={31'd0,5'(p),awaddr[27:0]};
  tnca_stream_axi lane(.clk(clk),.rst_n(rst_n),.cmd_valid(pending[p] && !fault),.cmd_ready(ready[p]),
   .cmd_mode(mode),.cmd_pc_base(PC_BASE),.cmd_src(source[p]),.cmd_dst(destination[p]),
   .cmd_lines(lines[p]),.cmd_pattern(pattern),.busy(),.fault(lane_fault[p]),
   .done_valid(done[p]),.done_ready(state==ACK && active[p]),.done_status(lane_status[p*4+:4]),
   .result_sum(lane_sum[p*32+:32]),.cycles(lane_cycles[p*64+:64]),
   .read_lines(lane_reads[p*64+:64]),.write_lines(lane_writes[p*64+:64]),
   .m_arvalid(m_arvalid[p]),.m_arready(m_arready[p]),.m_araddr(araddr),.m_arid(m_arid[p]),
   .m_arlen(m_arlen[p*8+:8]),.m_arsize(m_arsize[p*3+:3]),.m_arburst(m_arburst[p*2+:2]),
   .m_rvalid(m_rvalid[p]),.m_rready(m_rready[p]),.m_rdata(m_rdata[p*512+:512]),.m_rid(m_rid[p]),
   .m_rresp(m_rresp[p*2+:2]),.m_rlast(m_rlast[p]),
   .m_awvalid(m_awvalid[p]),.m_awready(m_awready[p]),.m_awaddr(awaddr),.m_awid(m_awid[p]),
   .m_awlen(m_awlen[p*8+:8]),.m_awsize(m_awsize[p*3+:3]),.m_awburst(m_awburst[p*2+:2]),
   .m_wvalid(m_wvalid[p]),.m_wready(m_wready[p]),.m_wdata(m_wdata[p*512+:512]),
   .m_wstrb(m_wstrb[p*64+:64]),.m_wlast(m_wlast[p]),
   .m_bvalid(m_bvalid[p]),.m_bready(m_bready[p]),.m_bid(m_bid[p]),.m_bresp(m_bresp[p*2+:2]));
 end endgenerate
 always @*begin
  reg_rdata=0;
  case(reg_addr)
   12'h000:reg_rdata=64'h52343153424e4b01;
   12'h008:reg_rdata=(64'(LANES)<<48)|(64'd512<<32)|(64'd16<<16)|64'd128;
   12'h010:reg_rdata=request_seq;
   12'h020:reg_rdata={61'd0,fault,(done_seq!=0),busy};
   12'h028:reg_rdata=done_seq;12'h030:reg_rdata=status;12'h038:reg_rdata=cycles;
   12'h040:reg_rdata={30'd0,mode,pattern};12'h048:reg_rdata=enabled;
   12'h050:reg_rdata=selected;12'h058:reg_rdata=source[selected];
   12'h060:reg_rdata=destination[selected];12'h068:reg_rdata=lines[selected];
   12'h070:reg_rdata=result_sum;12'h078:reg_rdata=total_reads;12'h080:reg_rdata=total_writes;
   12'h088:reg_rdata=aperture_size;12'h090:reg_rdata=aperture_base;
   12'h098:reg_rdata=lane_sum[selected*32+:32];12'h0a0:reg_rdata=lane_fault;
   12'h0a8:reg_rdata=lane_cycles[selected*64+:64];
   12'h0b0:reg_rdata=lane_reads[selected*64+:64];12'h0b8:reg_rdata=lane_writes[selected*64+:64];
   default:;
  endcase
 end
 generate if(LANES<1 || LANES>32 || (1<<SEL_W)<LANES)begin:g_bad_geometry
  tnca_invalid_stream_bank_geometry invalid_configuration();
 end endgenerate
 integer i;
 always @(posedge clk)begin
  if(!rst_n)begin
   state<=IDLE;request_seq<=0;served<=0;done_seq<=0;cycles<=0;status<=0;
   aperture_base<=0;aperture_size<=0;mode<=0;pattern<=0;result_sum<=0;
   enabled<=0;active<=0;pending<=0;selected<=0;cursor<=0;total_reads<=0;total_writes<=0;
   for(i=0;i<LANES;i=i+1)begin source[i]<=0;destination[i]<=0;lines[i]<=0;end
  end else begin
   if(busy)cycles<=cycles+1'b1;
   if(reg_we && !busy && !fault)case(reg_addr)
    12'h010:request_seq<=reg_wdata;
    12'h040:if(reg_wdata[63:34]==0)begin mode<=reg_wdata[33:32];pattern<=reg_wdata[31:0];end
    12'h048:if((reg_wdata>>LANES)==0)enabled<=reg_wdata[LANES-1:0];
    12'h050:if(reg_wdata<LANES)selected<=reg_wdata[SEL_W-1:0];
    12'h058:if(reg_wdata[63:28]==0)source[selected]<=reg_wdata[27:0];
    12'h060:if(reg_wdata[63:28]==0)destination[selected]<=reg_wdata[27:0];
    12'h068:if(reg_wdata[63:23]==0)lines[selected]<=reg_wdata[22:0];
    12'h088:aperture_size<=reg_wdata;12'h090:aperture_base<=reg_wdata;
    12'h018:if(reg_wdata[0] && request_seq>served)begin
     served<=request_seq;done_seq<=0;cycles<=0;status<=0;
     result_sum<=0;total_reads<=0;total_writes<=0;cursor<=0;
     if(mode>2 || aperture_base!=0 || aperture_size!=64'h200000000 || |(enabled & ~legal))begin
      status<=3;done_seq<=request_seq;
     end else begin pending<=enabled;active<=enabled;state<=DISPATCH;end
    end
    default:;
   endcase
   if(state==DISPATCH && !fault)begin
    pending<=pending & ~ready;
    if((pending & ~ready)==0)state<=WAIT;
   end
   if(state==WAIT && !fault && (done & active)==active)begin state<=SUM;cursor<=0;end
   // Sequential aggregation avoids wide reduction trees and their routing.
   // The measured window includes these bounded LANES+1 completion cycles.
   if(state==SUM && !fault)begin
    if(active[cursor])begin
     result_sum<=result_sum+lane_sum[cursor*32+:32];
     total_reads<=total_reads+lane_reads[cursor*64+:64];
     total_writes<=total_writes+lane_writes[cursor*64+:64];
     status<=status | lane_status[cursor*4+:4];
    end
    if(cursor==LANES-1)state<=ACK;else cursor<=cursor+1'b1;
   end
   if(state==ACK && !fault)begin state<=IDLE;done_seq<=served;active<=0;end
  end
 end
endmodule
`default_nettype wire
