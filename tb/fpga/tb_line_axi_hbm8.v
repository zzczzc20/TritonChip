// R41 hbm8: tnca_line_axi aperture arithmetic for an 8 GiB window.
//
// One outlet adapter, the B40 image configuration (256-bit beats, 8 slots,
// dynamic admission limit), and a recording AXI slave. For each BO base the
// bench checks, at the window's 64-bit edges:
//   * legal lines -- first, the last below 2/4 GiB, the first at 4 GiB, the
//     last line of 8 GiB -- issue exactly base + (addr - MEM_BASE) on AR/AW
//     (so no bit 32/33/63 is dropped or added) and complete;
//   * the first line past the window, an address below MEM_BASE, an
//     unaligned line and an aperture that would wrap 2^64 latch a fault,
//     drop ready and put nothing on AR/AW (a refused request is never
//     translated into a successful zero response).
// Runs against either the image source (git 54249f7) or the current tree:
// both are compiled from -y rtl/fpga by the caller.
`timescale 1ns/1ps
`default_nettype none
module tb_line_axi_hbm8;
 localparam [63:0] MEM_BASE=64'h20000;
 localparam [63:0] GIB=64'h4000_0000;
 reg clk=0,rst_n=0;always #5 clk=~clk;
 reg req_valid=0,req_we=0;reg [63:0] req_addr=0;reg [511:0] req_wdata=0;
 reg [2:0] req_tag=0;reg [1:0] req_src=0;
 reg [63:0] axi_base=0,axi_size=0;
 wire req_ready,rsp_valid,fault,idle;wire [511:0] rsp_rdata;wire [2:0] rsp_tag;wire [1:0] rsp_src;
 wire [63:0] read_lines,write_lines,read_beats,write_beats;
 wire m_awvalid,m_wvalid,m_wlast,m_bready,m_arvalid,m_rready;
 wire [63:0] m_awaddr,m_araddr;wire [2:0] m_awid,m_arid;wire [7:0] m_awlen,m_arlen;
 wire [2:0] m_awsize,m_arsize;wire [1:0] m_awburst,m_arburst;wire [255:0] m_wdata;wire [31:0] m_wstrb;
 reg m_awready=1,m_wready=1,m_arready=1,m_rvalid=0,m_rlast=0,m_bvalid=0;
 reg [2:0] m_rid=0,m_bid=0;reg [255:0] m_rdata=0;
 tnca_line_axi #(.SRC_W(2),.TAG_W(3),.OUT_N(8),.DATA_W(256),.DYNAMIC_LIMIT(1),.ID_W(3),.MEM_BASE(MEM_BASE)) dut(
  .clk(clk),.rst_n(rst_n),.admission_limit(8'd8),.req_valid(req_valid),.req_ready(req_ready),
  .req_addr(req_addr),.req_we(req_we),.req_wdata(req_wdata),.req_tag(req_tag),.req_src(req_src),
  .rsp_valid(rsp_valid),.rsp_rdata(rsp_rdata),.rsp_tag(rsp_tag),.rsp_src(rsp_src),
  .axi_base(axi_base),.axi_size(axi_size),.fault(fault),.idle(idle),
  .read_lines(read_lines),.write_lines(write_lines),.read_beats(read_beats),.write_beats(write_beats),
  .m_awvalid(m_awvalid),.m_awready(m_awready),.m_awaddr(m_awaddr),.m_awid(m_awid),.m_awlen(m_awlen),
  .m_awsize(m_awsize),.m_awburst(m_awburst),.m_wvalid(m_wvalid),.m_wready(m_wready),.m_wdata(m_wdata),
  .m_wstrb(m_wstrb),.m_wlast(m_wlast),.m_bvalid(m_bvalid),.m_bready(m_bready),.m_bid(m_bid),.m_bresp(2'd0),
  .m_arvalid(m_arvalid),.m_arready(m_arready),.m_araddr(m_araddr),.m_arid(m_arid),.m_arlen(m_arlen),
  .m_arsize(m_arsize),.m_arburst(m_arburst),.m_rvalid(m_rvalid),.m_rready(m_rready),.m_rdata(m_rdata),
  .m_rid(m_rid),.m_rresp(2'd0),.m_rlast(m_rlast));
 // Recording slave: every AR/AW handshake is counted and its address kept.
 integer ar_count=0,aw_count=0,checks=0;reg [63:0] last_ar=0,last_aw=0;
 always @(posedge clk)begin
  if(m_arvalid && m_arready)begin ar_count<=ar_count+1;last_ar<=m_araddr;end
  if(m_awvalid && m_awready)begin aw_count<=aw_count+1;last_aw<=m_awaddr;end
 end
 task settle;begin repeat(4)@(posedge clk);#1;end endtask
 task configure;input [63:0] base,size;begin
  @(negedge clk);rst_n=0;req_valid=0;axi_base=base;axi_size=size;settle;
  @(negedge clk);rst_n=1;settle;
 end endtask
 // A legal read and a legal write of one line; both must issue base+offset.
 task legal;input [63:0] addr;reg [63:0] want;integer ar0,aw0,k;begin
  want=axi_base+(addr-MEM_BASE);
  @(negedge clk);ar0=ar_count;req_valid=1;req_we=0;req_addr=addr;req_tag=3'd5;req_src=2'd1;
  // req_ready read at the edge (before nonblocking updates) is the handshake.
  k=0;begin:handshake_r forever begin @(posedge clk);if(req_ready)disable handshake_r;k=k+1;if(k>20)$fatal(1,"legal read %h not admitted",addr);end end
  #1;req_valid=0;
  k=0;while(ar_count==ar0)begin @(posedge clk);#1;k=k+1;if(k>20)$fatal(1,"no AR for %h",addr);end
  if(last_ar!==want)$fatal(1,"read %h issued AR %h, expected %h",addr,last_ar,want);
  // Two beats back on the slot's ID.
  @(negedge clk);m_rvalid=1;m_rid=m_arid;m_rdata={4{addr}};m_rlast=0;
  @(negedge clk);m_rdata=~{4{addr}};m_rlast=1;
  @(negedge clk);m_rvalid=0;m_rlast=0;
  k=0;while(!idle)begin @(negedge clk);k=k+1;if(k>20)$fatal(1,"read %h did not complete",addr);end
  @(negedge clk);aw0=aw_count;req_valid=1;req_we=1;req_addr=addr;req_wdata={8{addr}};
  k=0;begin:handshake_w forever begin @(posedge clk);if(req_ready)disable handshake_w;k=k+1;if(k>20)$fatal(1,"legal write %h not admitted",addr);end end
  #1;req_valid=0;req_we=0;
  k=0;while(aw_count==aw0)begin @(posedge clk);#1;k=k+1;if(k>20)$fatal(1,"no AW for %h",addr);end
  if(last_aw!==want)$fatal(1,"write %h issued AW %h, expected %h",addr,last_aw,want);
  repeat(4)@(posedge clk);
  @(negedge clk);m_bvalid=1;m_bid=m_awid;@(negedge clk);m_bvalid=0;
  k=0;while(!idle)begin @(negedge clk);k=k+1;if(k>20)$fatal(1,"write %h did not complete",addr);end
  if(fault)$fatal(1,"fault after legal accesses at %h",addr);
  checks=checks+1;
 end endtask
 // A refused request: fault latches, ready stays low, nothing reaches AXI.
 task refused;input [63:0] addr;integer ar0,aw0;begin
  @(negedge clk);ar0=ar_count;aw0=aw_count;
  if(fault)$fatal(1,"fault already set before refusing %h",addr);
  req_valid=1;req_we=0;req_addr=addr;
  repeat(6)begin @(posedge clk);#1;if(req_ready)$fatal(1,"illegal %h was offered ready",addr);end
  @(negedge clk);req_valid=0;
  repeat(4)@(posedge clk);#1;
  if(!fault)$fatal(1,"illegal %h did not latch a fault",addr);
  if(ar_count!=ar0 || aw_count!=aw0)$fatal(1,"illegal %h reached AXI",addr);
  checks=checks+1;
 end endtask
 reg [63:0] base;integer b;
 initial begin
  for(b=0;b<3;b=b+1)begin
   // XRT's HBM[0:31] group BO (0), a BO at 4 GiB, and one ending at 2^64.
   base=b==0 ? 64'd0 : (b==1 ? 64'h1_0000_0000 : 64'hFFFF_FFFE_0000_0000);
   configure(base,8*GIB);
   legal(MEM_BASE);
   legal(MEM_BASE+2*GIB-64);
   legal(MEM_BASE+4*GIB-64);
   legal(MEM_BASE+4*GIB);
   legal(MEM_BASE+6*GIB+64);
   legal(MEM_BASE+8*GIB-64);
   refused(MEM_BASE+8*GIB);
   configure(base,8*GIB);refused(MEM_BASE+16*GIB);
   configure(base,8*GIB);refused(MEM_BASE+8*GIB+4*GIB);
   configure(base,8*GIB);refused(MEM_BASE-64);
   configure(base,8*GIB);refused(MEM_BASE+4*GIB+8);
   $display("tb_line_axi_hbm8: base %h, 8 GiB window: legal edges and refusals pass",base);
  end
  // An aperture whose last line would cross 2^64 is refused, not wrapped.
  configure(64'hFFFF_FFFE_0000_0000,8*GIB+64);
  legal(MEM_BASE+8*GIB-64);
  refused(MEM_BASE+8*GIB);
  $display("tb_line_axi_hbm8 PASS checks=%0d",checks);
  $finish;
 end
 initial begin #20000000;$fatal(1,"tb_line_axi_hbm8 timeout");end
endmodule
`default_nettype wire
