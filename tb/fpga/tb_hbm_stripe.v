`timescale 1ns/1ps
// The host test generates independent page/PC oracle vectors. Real line
// requests traverse the adapter; the AXI address is returned as read data.
module tb_hbm_stripe;
 parameter BAD_WINDOW=0;
 reg clk=0,rst_n=0;always #5 clk=~clk;
 reg valid=0;wire ready,arvalid,rready,rsp_valid,fault,idle;
 reg [63:0] address=0;wire [63:0] araddr;
 wire [2:0] arid;reg rv=0;reg [2:0] rid=0;reg [511:0] data=0;
 wire [511:0] rsp_data;wire [63:0] rd_lines;
 reg [127:0] vectors[0:255];reg [1023:0] path;
 integer count,i;
 tnca_line_axi #(.DATA_W(512),.OUT_N(8),.HBM_STRIPED(1)) dut(
  .clk(clk),.rst_n(rst_n),.admission_limit(8'd0),.req_valid(valid),.req_ready(ready),.req_addr(address),
  .req_we(1'b0),.req_wdata(512'd0),.req_tag(3'd0),.req_src(2'd0),
  .rsp_valid(rsp_valid),.rsp_rdata(rsp_data),.rsp_tag(),.rsp_src(),
  .axi_base(BAD_WINDOW==1?64'h10000000:64'd0),.axi_size(BAD_WINDOW==2?64'h100000000:64'h200000000),
  .fault(fault),.idle(idle),.read_lines(rd_lines),.write_lines(),.read_beats(),.write_beats(),
  .m_awvalid(),.m_awready(1'b1),.m_awaddr(),.m_awid(),.m_awlen(),.m_awsize(),.m_awburst(),
  .m_wvalid(),.m_wready(1'b1),.m_wdata(),.m_wstrb(),.m_wlast(),
  .m_bvalid(1'b0),.m_bready(),.m_bid(3'd0),.m_bresp(2'd0),
  .m_arvalid(arvalid),.m_arready(1'b1),.m_araddr(araddr),.m_arid(arid),.m_arlen(),.m_arsize(),.m_arburst(),
  .m_rvalid(rv),.m_rready(rready),.m_rdata(data),.m_rid(rid),.m_rresp(2'd0),.m_rlast(1'b1));
 always @(posedge clk)if(rst_n)begin
  rv<=arvalid;
  if(arvalid)begin rid<=arid;data<={8{araddr}};end
 end
 initial begin
  if(!$value$plusargs("vectors=%s",path) || !$value$plusargs("count=%d",count))$fatal(1,"missing vectors");
  $readmemh(path,vectors,0,count-1);
  repeat(4)@(negedge clk);rst_n=1;repeat(3)@(negedge clk);
  for(i=0;i<count;i=i+1)begin
   address=64'h20000+vectors[i][127:64];valid=1;
   if(BAD_WINDOW)begin
    @(posedge clk);@(negedge clk);
    if(!fault || ready || arvalid || rd_lines || !idle)$fatal(1,"bad striped window admitted traffic");
    $display("HBM STRIPE REFUSAL PASS bad_window=%0d",BAD_WINDOW);$finish;
   end
   @(posedge clk);while(!ready)@(posedge clk);
   @(negedge clk);valid=0;wait(rsp_valid);@(negedge clk);
   if(rsp_data!=={8{vectors[i][63:0]}} || fault)$fatal(1,"striped address mismatch vector=%0d",i);
  end
  if(rd_lines!=count)$fatal(1,"striped read count");
  $display("HBM STRIPE AXI PASS vectors=%0d",count);$finish;
 end
 initial begin #1000000;$fatal(1,"striped adapter timeout");end
endmodule
