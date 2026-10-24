// Test-only full-line AXI slave with independent stalls and interleaved IDs.
// The window starts at WIN_BASE (the stand-in BO address). Dense (default):
// LINES consecutive lines. SPARSE=1: only the ranges of the SPARSE_MAP file
// (tnca_sparse_map.vh) are backed, LINES is their total, and an access to any
// other window line is fatal; this lets an 8 GiB window be simulated.
`timescale 1ns/1ps
module tnca_axi_line_model #(
 parameter DATA_W=256,OUT_N=8,MODE=0,LINES=2048,LATENCY=25,WRITE_DELAY=15,
 parameter ID_W=(OUT_N>1?$clog2(OUT_N):1),
 parameter [63:0] WIN_BASE=64'h100000000,
 parameter SPARSE=0,parameter SPARSE_MAP="none"
)(
 input wire clk,rst_n,
 input wire m_awvalid,m_wvalid,m_arvalid,m_wlast,m_rready,m_bready,
 input wire [63:0] m_awaddr,m_araddr,input wire [ID_W-1:0] m_awid,m_arid,
 input wire [7:0] m_awlen,m_arlen,input wire [2:0] m_awsize,m_arsize,
 input wire [1:0] m_awburst,m_arburst,
 input wire [DATA_W-1:0] m_wdata,input wire [DATA_W/8-1:0] m_wstrb,
 output reg m_awready=0,m_wready=0,m_arready=0,m_rvalid=0,m_bvalid=0,m_rlast=0,
 output reg [ID_W-1:0] m_rid=0,m_bid=0,output reg [1:0] m_rresp=0,m_bresp=0,
 output reg [DATA_W-1:0] m_rdata=0
);
 localparam BEATS=512/DATA_W;
 reg [511:0] memory[0:LINES-1];
 initial for(integer init_line=0;init_line<LINES;init_line=init_line+1)memory[init_line]=0;
`include "tb/common/tnca_sparse_map.vh"
 initial if(SPARSE)begin
  sparse_load(SPARSE_MAP);
  if(sparse_total>LINES)$fatal(1,"sparse map needs %0d lines, model has %0d",sparse_total,LINES);
 end
 // Compact line index of a window byte address, or -1 outside the backing.
 function integer line_index;
  input [63:0] a;
  begin
   if(a<WIN_BASE || a[5:0]!=0)line_index=-1;
   else if(SPARSE)line_index=sparse_index((a-WIN_BASE)>>6);
   else if((a-WIN_BASE)/64>=LINES)line_index=-1;
   else line_index=(a-WIN_BASE)/64;
  end
 endfunction
 reg [OUT_N-1:0] reads=0,writes=0;
 reg [511:0] write_data[0:OUT_N-1];
 integer location[0:OUT_N-1],beat[0:OUT_N-1],due[0:OUT_N-1];
 reg aw_have=0;integer aw_slot=0,aw_line=0,wcount=0;
 reg [511:0] incoming=0;
 integer cycle=0,collisions=0;
 integer i,j,pick,seq,index,seed=123;
 reg [511:0] payload;
 reg ar_stalled=0,aw_stalled=0,w_stalled=0;
 reg [63:0] old_araddr,old_awaddr;reg [ID_W-1:0] old_arid,old_awid;
 reg [DATA_W-1:0] old_wdata;reg old_wlast;
 // AXI slave: independent AW/W acceptance, delayed B, interleaved read IDs.
 always @(posedge clk)begin
  if(!rst_n)begin
   reads=0;writes=0;aw_have=0;wcount=0;ar_stalled=0;aw_stalled=0;w_stalled=0;
   m_rvalid<=0;m_bvalid<=0;m_awready<=0;m_wready<=0;m_arready<=0;
  end else begin
   cycle=cycle+1;
   if(ar_stalled && (!m_arvalid || m_araddr!==old_araddr || m_arid!==old_arid))$fatal(1,"AR changed under stall");
   if(aw_stalled && (!m_awvalid || m_awaddr!==old_awaddr || m_awid!==old_awid))$fatal(1,"AW changed under stall");
   if(w_stalled && (!m_wvalid || m_wdata!==old_wdata || m_wlast!==old_wlast))$fatal(1,"W changed under stall");
   ar_stalled=m_arvalid&&!m_arready;old_araddr=m_araddr;old_arid=m_arid;
   aw_stalled=m_awvalid&&!m_awready;old_awaddr=m_awaddr;old_awid=m_awid;
   w_stalled=m_wvalid&&!m_wready;old_wdata=m_wdata;old_wlast=m_wlast;
   if(m_arvalid && m_arready)begin
    if(m_arlen!=BEATS-1 || m_arsize!=$clog2(DATA_W/8) || m_arburst!=1)$fatal(1,"read geometry");
    if(reads[m_arid] || writes[m_arid])$fatal(1,"ID reused before completion");
    if(line_index(m_araddr)<0)$fatal(1,"read outside model window: %h",m_araddr);
    reads[m_arid]=1;location[m_arid]=line_index(m_araddr);
    beat[m_arid]=0;due[m_arid]=cycle+LATENCY+(m_arid*11)%31;
   end
   if(m_awvalid && m_awready)begin
    if(aw_have || m_awlen!=BEATS-1 || m_awsize!=$clog2(DATA_W/8) || m_awburst!=1)$fatal(1,"write geometry/order");
    if(line_index(m_awaddr)<0)$fatal(1,"write outside model window: %h",m_awaddr);
    aw_have=1;aw_slot=m_awid;aw_line=line_index(m_awaddr);
   end
   if(m_wvalid && m_wready)begin
    if(m_wstrb!=={DATA_W/8{1'b1}} || m_wlast!=(wcount==BEATS-1))$fatal(1,"write beat geometry");
    incoming[wcount*DATA_W+:DATA_W]=m_wdata;wcount=wcount+1;
   end
   if(aw_have && wcount==BEATS)begin
    if(reads[aw_slot] || writes[aw_slot])$fatal(1,"write ID reused");
    writes[aw_slot]=1;location[aw_slot]=aw_line;write_data[aw_slot]=incoming;
    due[aw_slot]=cycle+LATENCY+WRITE_DELAY+(aw_slot*7)%23;aw_have=0;wcount=0;
   end
   if(m_rvalid && m_rready)begin
    if(m_rlast)reads[m_rid]=0;else beat[m_rid]=beat[m_rid]+1;
   end
   if(m_bvalid && m_bready)begin
    memory[location[m_bid]]=write_data[m_bid];writes[m_bid]=0;
   end
   if(m_rvalid && m_bvalid)collisions=collisions+1;
   if(!m_rvalid || m_rready)begin
    pick=-1;
    for(j=0;j<OUT_N;j=j+1)if(reads[j] && due[j]<=cycle && (pick<0 || ($random(seed)&1)))pick=j;
    m_rvalid<=pick>=0;
    if(pick>=0)begin
     m_rid<=pick;m_rdata<=memory[location[pick]][beat[pick]*DATA_W+:DATA_W];m_rlast<=beat[pick]==BEATS-1;
     m_rresp<=MODE==1?2:0;
    end
   end
   if(!m_bvalid || m_bready)begin
    pick=-1;for(j=OUT_N-1;j>=0;j=j-1)if(writes[j] && due[j]<=cycle)pick=j;
    m_bvalid<=pick>=0;if(pick>=0)begin m_bid<=pick;m_bresp<=MODE==2?2:0;end
   end
   // Initially allow W well before AW, then independent random channel stalls.
   m_awready<=cycle>12 && (($random(seed)&32'h7fffffff)%100)>35;
   m_wready<=(($random(seed)&32'h7fffffff)%100)>40;
   m_arready<=(($random(seed)&32'h7fffffff)%100)>30;
   
  end
 end
endmodule
