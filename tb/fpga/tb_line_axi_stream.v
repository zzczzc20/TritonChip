`timescale 1ns/1ps
// Sustained 512-bit line source and a deterministic 16-cycle AXI responder.
// The randomized shared testbench separately covers independent stalls,
// interleaved replies, same-line ordering, faults and all data widths.
module tb_line_axi_stream;
 parameter PIPELINED_ISSUE=1,WRITE=0,OUT_N=32;
 localparam ID_W=$clog2(OUT_N),COUNT=256,DELAY=16;
 reg clk=0,rst_n=0;always #5 clk=~clk;
 integer accepted=0,completed=0,issued=0,cycle=0,first_issue=-1,last_issue=-1;
 wire req_ready,rsp_valid,idle,fault;
 wire [7:0] rsp_tag;wire rsp_src;wire [511:0] rsp_data;
 wire [63:0] rd_lines,wr_lines,rd_beats,wr_beats;
 wire arvalid,awvalid,wvalid,wlast,rready,bready;
 wire [63:0] araddr,awaddr;wire [ID_W-1:0] arid,awid;
 wire [7:0] arlen,awlen;wire [2:0] arsize,awsize;
 wire [1:0] arburst,awburst;wire [511:0] wdata;wire [63:0] wstrb;
 reg [DELAY-1:0] replies=0;
 reg [ID_W-1:0] reply_id[0:DELAY-1];
 reg [31:0] reply_value[0:DELAY-1];
 wire [31:0] payload=32'h12340000+accepted;
 wire response=replies[DELAY-1];
 wire bus_fire=WRITE ? awvalid && wvalid : arvalid;
 wire [63:0] bus_addr=WRITE ? awaddr : araddr;
 wire [ID_W-1:0] bus_id=WRITE ? awid : arid;
 tnca_line_axi #(.SRC_W(1),.TAG_W(8),.DATA_W(512),.OUT_N(OUT_N),
   .WRITE_RESPONSES(1),.PIPELINED_ISSUE(PIPELINED_ISSUE)) dut(
  .clk(clk),.rst_n(rst_n),.admission_limit(8'd0),
  .req_valid(rst_n && accepted<COUNT),.req_ready(req_ready),
  .req_addr(64'h20000+accepted*64),.req_we(WRITE!=0),.req_wdata({16{payload}}),
  .req_tag(8'(accepted)),.req_src(1'b0),
  .rsp_valid(rsp_valid),.rsp_rdata(rsp_data),.rsp_tag(rsp_tag),.rsp_src(rsp_src),
  .axi_base(64'h100000000),.axi_size(64'd65536),.fault(fault),.idle(idle),
  .read_lines(rd_lines),.write_lines(wr_lines),.read_beats(rd_beats),.write_beats(wr_beats),
  .m_awvalid(awvalid),.m_awready(1'b1),.m_awaddr(awaddr),.m_awid(awid),
  .m_awlen(awlen),.m_awsize(awsize),.m_awburst(awburst),
  .m_wvalid(wvalid),.m_wready(1'b1),.m_wdata(wdata),.m_wstrb(wstrb),.m_wlast(wlast),
  .m_bvalid(WRITE && response),.m_bready(bready),.m_bid(reply_id[DELAY-1]),.m_bresp(2'b0),
  .m_arvalid(arvalid),.m_arready(1'b1),.m_araddr(araddr),.m_arid(arid),
  .m_arlen(arlen),.m_arsize(arsize),.m_arburst(arburst),
  .m_rvalid(!WRITE && response),.m_rready(rready),.m_rdata({16{reply_value[DELAY-1]}}),
  .m_rid(reply_id[DELAY-1]),.m_rresp(2'b0),.m_rlast(1'b1));
 integer i;
 always @(posedge clk)if(rst_n)begin
  cycle<=cycle+1;
  if(accepted<COUNT && req_ready)accepted<=accepted+1;
  if(WRITE && (awvalid!==wvalid || (wvalid && (!wlast || wstrb!==64'hffffffffffffffff))))
   $fatal(1,"write beat geometry");
  if(bus_fire)begin
   if(first_issue<0)first_issue<=cycle;
   last_issue<=cycle;issued<=issued+1;
   if(bus_addr!==64'h100000000+issued*64)$fatal(1,"issue address sequence");
   if(WRITE && (wdata!=={16{32'h12340000+32'(issued)}} || awlen!=0 || awsize!=6 || awburst!=1))
    $fatal(1,"write payload or geometry");
   if(!WRITE && (arlen!=0 || arsize!=6 || arburst!=1))$fatal(1,"read geometry");
  end
  replies<={replies[DELAY-2:0],bus_fire};
  reply_id[0]<=bus_id;reply_value[0]<=32'h12340000+issued;
  for(i=1;i<DELAY;i=i+1)begin reply_id[i]<=reply_id[i-1];reply_value[i]<=reply_value[i-1];end
  if(response && (WRITE ? !bready : !rready))$fatal(1,"unexpected response backpressure");
  if(rsp_valid)begin
   if(rsp_tag!==8'(completed) || rsp_src!==0)$fatal(1,"completion identity");
   if(rsp_data!==(WRITE ? 512'd0 : {16{32'h12340000+32'(completed)}}))$fatal(1,"completion payload");
   completed<=completed+1;
  end
  if(fault)$fatal(1,"unexpected adapter fault");
  if(cycle>2000)$fatal(1,"stream timeout");
 end
 initial begin
  repeat(4)@(negedge clk);rst_n=1;
  wait(completed==COUNT && idle);@(negedge clk);
  if(accepted!=COUNT || issued!=COUNT || rd_lines+wr_lines!=COUNT || rd_beats+wr_beats!=COUNT)
   $fatal(1,"traffic accounting");
  if(last_issue-first_issue!=(COUNT-1)*(PIPELINED_ISSUE ? 1 : 2))$fatal(1,"issue throughput");
  $display("LINE STREAM PASS pipeline=%0d write=%0d slots=%0d lines=%0d issue_span=%0d cycles=%0d",
   PIPELINED_ISSUE,WRITE,OUT_N,issued,last_issue-first_issue,cycle);
  $finish;
 end
endmodule
