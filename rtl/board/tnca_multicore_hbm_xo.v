// U280 platform binding for three physical tiles. Placement is external.
// All named masters address the same global allocation. Kernel arguments are
// aliases of one base register: a runtime must supply the identical buffer.
// ap_rst_n must reset the AXI fabric/controllers too; reset alone cannot undo
// an AW accepted without its W or repair a lost response.
`default_nettype none
module tnca_multicore_hbm_xo #(
 parameter DCO_REGISTERED_RANGES=0,
 parameter S_AXI_ADDR_W=12,MEM_PORTS=3,CODE_WORDS=16384,CODE_LINK_STAGES=2,CACHE_MAINTENANCE=1,
 parameter NOC_LINK_STAGES=2,DISPATCH_STAGES=2,LINK_STAGES=2,MEM_HOME_STAGES=2,
 parameter AXI_DATA_W=256,AXI_OUT_N=8,
 parameter AXI_ID_W=(AXI_OUT_N>1?$clog2(AXI_OUT_N):1),
 parameter NUM_RCTX=4,RCTX_W=2,PHYS_VREGS=256,PHYS_W=8,VREGS_PER_RCTX=64,
 parameter HAS_SFU=1,HAS_ATOMICS=1,HAS_SYS_ATOMICS=0,USE_BANKED_VRF=1,VRF_BYPASS=1,FE_NOBUBBLE=1,
 parameter VRF_PREFETCH=1,RCTX_SWITCH=2,MTX_ABUF=1,LSU_INFLIGHT=2,
 parameter FPU_OVERLAP=1,RSLICE=1,RPAIR=1,HAS_TENSOR_XPOSE=1,
 parameter SETS=64,SET_W=6,WAYS=8,WAY_W=3
) (
 input wire ap_clk,ap_rst_n,
    input  wire                    s_axi_tnca_awvalid,
    output wire                    s_axi_tnca_awready,
    input  wire [S_AXI_ADDR_W-1:0] s_axi_tnca_awaddr,
    input  wire                    s_axi_tnca_wvalid,
    output wire                    s_axi_tnca_wready,
    input  wire [31:0]             s_axi_tnca_wdata,
    input  wire [3:0]              s_axi_tnca_wstrb,
    input  wire                    s_axi_tnca_arvalid,
    output wire                    s_axi_tnca_arready,
    input  wire [S_AXI_ADDR_W-1:0] s_axi_tnca_araddr,
    output wire                    s_axi_tnca_rvalid,
    input  wire                    s_axi_tnca_rready,
    output wire [31:0]             s_axi_tnca_rdata,
    output wire [1:0]              s_axi_tnca_rresp,
    output wire                    s_axi_tnca_bvalid,
    input  wire                    s_axi_tnca_bready,
    output wire [1:0]              s_axi_tnca_bresp,
 output wire m_axi_hbm0_awvalid,
 input wire m_axi_hbm0_awready,
 output wire [64-1:0] m_axi_hbm0_awaddr,
 output wire [AXI_ID_W-1:0] m_axi_hbm0_awid,
 output wire [8-1:0] m_axi_hbm0_awlen,
 output wire [3-1:0] m_axi_hbm0_awsize,
 output wire [2-1:0] m_axi_hbm0_awburst,
 output wire m_axi_hbm0_wvalid,
 input wire m_axi_hbm0_wready,
 output wire [AXI_DATA_W-1:0] m_axi_hbm0_wdata,
 output wire [AXI_DATA_W/8-1:0] m_axi_hbm0_wstrb,
 output wire m_axi_hbm0_wlast,
 input wire m_axi_hbm0_bvalid,
 output wire m_axi_hbm0_bready,
 input wire [AXI_ID_W-1:0] m_axi_hbm0_bid,
 input wire [2-1:0] m_axi_hbm0_bresp,
 output wire m_axi_hbm0_arvalid,
 input wire m_axi_hbm0_arready,
 output wire [64-1:0] m_axi_hbm0_araddr,
 output wire [AXI_ID_W-1:0] m_axi_hbm0_arid,
 output wire [8-1:0] m_axi_hbm0_arlen,
 output wire [3-1:0] m_axi_hbm0_arsize,
 output wire [2-1:0] m_axi_hbm0_arburst,
 input wire m_axi_hbm0_rvalid,
 output wire m_axi_hbm0_rready,
 input wire [AXI_DATA_W-1:0] m_axi_hbm0_rdata,
 input wire [AXI_ID_W-1:0] m_axi_hbm0_rid,
 input wire [2-1:0] m_axi_hbm0_rresp,
 input wire m_axi_hbm0_rlast,
 output wire m_axi_hbm1_awvalid,
 input wire m_axi_hbm1_awready,
 output wire [64-1:0] m_axi_hbm1_awaddr,
 output wire [AXI_ID_W-1:0] m_axi_hbm1_awid,
 output wire [8-1:0] m_axi_hbm1_awlen,
 output wire [3-1:0] m_axi_hbm1_awsize,
 output wire [2-1:0] m_axi_hbm1_awburst,
 output wire m_axi_hbm1_wvalid,
 input wire m_axi_hbm1_wready,
 output wire [AXI_DATA_W-1:0] m_axi_hbm1_wdata,
 output wire [AXI_DATA_W/8-1:0] m_axi_hbm1_wstrb,
 output wire m_axi_hbm1_wlast,
 input wire m_axi_hbm1_bvalid,
 output wire m_axi_hbm1_bready,
 input wire [AXI_ID_W-1:0] m_axi_hbm1_bid,
 input wire [2-1:0] m_axi_hbm1_bresp,
 output wire m_axi_hbm1_arvalid,
 input wire m_axi_hbm1_arready,
 output wire [64-1:0] m_axi_hbm1_araddr,
 output wire [AXI_ID_W-1:0] m_axi_hbm1_arid,
 output wire [8-1:0] m_axi_hbm1_arlen,
 output wire [3-1:0] m_axi_hbm1_arsize,
 output wire [2-1:0] m_axi_hbm1_arburst,
 input wire m_axi_hbm1_rvalid,
 output wire m_axi_hbm1_rready,
 input wire [AXI_DATA_W-1:0] m_axi_hbm1_rdata,
 input wire [AXI_ID_W-1:0] m_axi_hbm1_rid,
 input wire [2-1:0] m_axi_hbm1_rresp,
 input wire m_axi_hbm1_rlast,
 output wire m_axi_hbm2_awvalid,
 input wire m_axi_hbm2_awready,
 output wire [64-1:0] m_axi_hbm2_awaddr,
 output wire [AXI_ID_W-1:0] m_axi_hbm2_awid,
 output wire [8-1:0] m_axi_hbm2_awlen,
 output wire [3-1:0] m_axi_hbm2_awsize,
 output wire [2-1:0] m_axi_hbm2_awburst,
 output wire m_axi_hbm2_wvalid,
 input wire m_axi_hbm2_wready,
 output wire [AXI_DATA_W-1:0] m_axi_hbm2_wdata,
 output wire [AXI_DATA_W/8-1:0] m_axi_hbm2_wstrb,
 output wire m_axi_hbm2_wlast,
 input wire m_axi_hbm2_bvalid,
 output wire m_axi_hbm2_bready,
 input wire [AXI_ID_W-1:0] m_axi_hbm2_bid,
 input wire [2-1:0] m_axi_hbm2_bresp,
 output wire m_axi_hbm2_arvalid,
 input wire m_axi_hbm2_arready,
 output wire [64-1:0] m_axi_hbm2_araddr,
 output wire [AXI_ID_W-1:0] m_axi_hbm2_arid,
 output wire [8-1:0] m_axi_hbm2_arlen,
 output wire [3-1:0] m_axi_hbm2_arsize,
 output wire [2-1:0] m_axi_hbm2_arburst,
 input wire m_axi_hbm2_rvalid,
 output wire m_axi_hbm2_rready,
 input wire [AXI_DATA_W-1:0] m_axi_hbm2_rdata,
 input wire [AXI_ID_W-1:0] m_axi_hbm2_rid,
 input wire [2-1:0] m_axi_hbm2_rresp,
 input wire m_axi_hbm2_rlast
);
 (* ASYNC_REG="TRUE" *) reg [1:0] reset_sync=0;
 always @(posedge ap_clk or negedge ap_rst_n)begin
  if(!ap_rst_n)reset_sync<=0;else reset_sync<={reset_sync[0],1'b1};
 end
 wire rst_n=reset_sync[1];
 wire [8:0] reg_addr;wire [63:0] reg_wdata,reg_rdata;wire reg_we,reg_re;
 tnca_axi_lite #(.ADDR_BITS(9),.S_AXI_ADDR_W(S_AXI_ADDR_W)) registers(
  .s_axi_aclk(ap_clk),.s_axi_aresetn(rst_n),.reg_clk(ap_clk),.reg_rst_n(rst_n),
  .reg_addr(reg_addr),.reg_wdata(reg_wdata),.reg_we(reg_we),.reg_re(reg_re),.reg_rdata(reg_rdata),
  .s_axi_awvalid(s_axi_tnca_awvalid),
  .s_axi_awready(s_axi_tnca_awready),
  .s_axi_awaddr(s_axi_tnca_awaddr),
  .s_axi_wvalid(s_axi_tnca_wvalid),
  .s_axi_wready(s_axi_tnca_wready),
  .s_axi_wdata(s_axi_tnca_wdata),
  .s_axi_wstrb(s_axi_tnca_wstrb),
  .s_axi_arvalid(s_axi_tnca_arvalid),
  .s_axi_arready(s_axi_tnca_arready),
  .s_axi_araddr(s_axi_tnca_araddr),
  .s_axi_rvalid(s_axi_tnca_rvalid),
  .s_axi_rready(s_axi_tnca_rready),
  .s_axi_rdata(s_axi_tnca_rdata),
  .s_axi_rresp(s_axi_tnca_rresp),
  .s_axi_bvalid(s_axi_tnca_bvalid),
  .s_axi_bready(s_axi_tnca_bready),
  .s_axi_bresp(s_axi_tnca_bresp)
 );
 wire [MEM_PORTS*(1)-1:0] m_awvalid;
 wire [MEM_PORTS*(1)-1:0] m_awready;
 wire [MEM_PORTS*(64)-1:0] m_awaddr;
 wire [MEM_PORTS*(AXI_ID_W)-1:0] m_awid;
 wire [MEM_PORTS*(8)-1:0] m_awlen;
 wire [MEM_PORTS*(3)-1:0] m_awsize;
 wire [MEM_PORTS*(2)-1:0] m_awburst;
 wire [MEM_PORTS*(1)-1:0] m_wvalid;
 wire [MEM_PORTS*(1)-1:0] m_wready;
 wire [MEM_PORTS*(AXI_DATA_W)-1:0] m_wdata;
 wire [MEM_PORTS*(AXI_DATA_W/8)-1:0] m_wstrb;
 wire [MEM_PORTS*(1)-1:0] m_wlast;
 wire [MEM_PORTS*(1)-1:0] m_bvalid;
 wire [MEM_PORTS*(1)-1:0] m_bready;
 wire [MEM_PORTS*(AXI_ID_W)-1:0] m_bid;
 wire [MEM_PORTS*(2)-1:0] m_bresp;
 wire [MEM_PORTS*(1)-1:0] m_arvalid;
 wire [MEM_PORTS*(1)-1:0] m_arready;
 wire [MEM_PORTS*(64)-1:0] m_araddr;
 wire [MEM_PORTS*(AXI_ID_W)-1:0] m_arid;
 wire [MEM_PORTS*(8)-1:0] m_arlen;
 wire [MEM_PORTS*(3)-1:0] m_arsize;
 wire [MEM_PORTS*(2)-1:0] m_arburst;
 wire [MEM_PORTS*(1)-1:0] m_rvalid;
 wire [MEM_PORTS*(1)-1:0] m_rready;
 wire [MEM_PORTS*(AXI_DATA_W)-1:0] m_rdata;
 wire [MEM_PORTS*(AXI_ID_W)-1:0] m_rid;
 wire [MEM_PORTS*(2)-1:0] m_rresp;
 wire [MEM_PORTS*(1)-1:0] m_rlast;
 generate if(MEM_PORTS>0)begin:g_port0
  assign m_axi_hbm0_awvalid=m_awvalid[0*(1)+:(1)];
  assign m_awready[0*(1)+:(1)]=m_axi_hbm0_awready;
  assign m_axi_hbm0_awaddr=m_awaddr[0*(64)+:(64)];
  assign m_axi_hbm0_awid=m_awid[0*(AXI_ID_W)+:(AXI_ID_W)];
  assign m_axi_hbm0_awlen=m_awlen[0*(8)+:(8)];
  assign m_axi_hbm0_awsize=m_awsize[0*(3)+:(3)];
  assign m_axi_hbm0_awburst=m_awburst[0*(2)+:(2)];
  assign m_axi_hbm0_wvalid=m_wvalid[0*(1)+:(1)];
  assign m_wready[0*(1)+:(1)]=m_axi_hbm0_wready;
  assign m_axi_hbm0_wdata=m_wdata[0*(AXI_DATA_W)+:(AXI_DATA_W)];
  assign m_axi_hbm0_wstrb=m_wstrb[0*(AXI_DATA_W/8)+:(AXI_DATA_W/8)];
  assign m_axi_hbm0_wlast=m_wlast[0*(1)+:(1)];
  assign m_bvalid[0*(1)+:(1)]=m_axi_hbm0_bvalid;
  assign m_axi_hbm0_bready=m_bready[0*(1)+:(1)];
  assign m_bid[0*(AXI_ID_W)+:(AXI_ID_W)]=m_axi_hbm0_bid;
  assign m_bresp[0*(2)+:(2)]=m_axi_hbm0_bresp;
  assign m_axi_hbm0_arvalid=m_arvalid[0*(1)+:(1)];
  assign m_arready[0*(1)+:(1)]=m_axi_hbm0_arready;
  assign m_axi_hbm0_araddr=m_araddr[0*(64)+:(64)];
  assign m_axi_hbm0_arid=m_arid[0*(AXI_ID_W)+:(AXI_ID_W)];
  assign m_axi_hbm0_arlen=m_arlen[0*(8)+:(8)];
  assign m_axi_hbm0_arsize=m_arsize[0*(3)+:(3)];
  assign m_axi_hbm0_arburst=m_arburst[0*(2)+:(2)];
  assign m_rvalid[0*(1)+:(1)]=m_axi_hbm0_rvalid;
  assign m_axi_hbm0_rready=m_rready[0*(1)+:(1)];
  assign m_rdata[0*(AXI_DATA_W)+:(AXI_DATA_W)]=m_axi_hbm0_rdata;
  assign m_rid[0*(AXI_ID_W)+:(AXI_ID_W)]=m_axi_hbm0_rid;
  assign m_rresp[0*(2)+:(2)]=m_axi_hbm0_rresp;
  assign m_rlast[0*(1)+:(1)]=m_axi_hbm0_rlast;
 end else begin:g_unused0
  assign m_axi_hbm0_awvalid=0;
  assign m_axi_hbm0_awaddr=0;
  assign m_axi_hbm0_awid=0;
  assign m_axi_hbm0_awlen=0;
  assign m_axi_hbm0_awsize=0;
  assign m_axi_hbm0_awburst=0;
  assign m_axi_hbm0_wvalid=0;
  assign m_axi_hbm0_wdata=0;
  assign m_axi_hbm0_wstrb=0;
  assign m_axi_hbm0_wlast=0;
  assign m_axi_hbm0_bready=1;
  assign m_axi_hbm0_arvalid=0;
  assign m_axi_hbm0_araddr=0;
  assign m_axi_hbm0_arid=0;
  assign m_axi_hbm0_arlen=0;
  assign m_axi_hbm0_arsize=0;
  assign m_axi_hbm0_arburst=0;
  assign m_axi_hbm0_rready=1;
 end endgenerate
 generate if(MEM_PORTS>1)begin:g_port1
  assign m_axi_hbm1_awvalid=m_awvalid[1*(1)+:(1)];
  assign m_awready[1*(1)+:(1)]=m_axi_hbm1_awready;
  assign m_axi_hbm1_awaddr=m_awaddr[1*(64)+:(64)];
  assign m_axi_hbm1_awid=m_awid[1*(AXI_ID_W)+:(AXI_ID_W)];
  assign m_axi_hbm1_awlen=m_awlen[1*(8)+:(8)];
  assign m_axi_hbm1_awsize=m_awsize[1*(3)+:(3)];
  assign m_axi_hbm1_awburst=m_awburst[1*(2)+:(2)];
  assign m_axi_hbm1_wvalid=m_wvalid[1*(1)+:(1)];
  assign m_wready[1*(1)+:(1)]=m_axi_hbm1_wready;
  assign m_axi_hbm1_wdata=m_wdata[1*(AXI_DATA_W)+:(AXI_DATA_W)];
  assign m_axi_hbm1_wstrb=m_wstrb[1*(AXI_DATA_W/8)+:(AXI_DATA_W/8)];
  assign m_axi_hbm1_wlast=m_wlast[1*(1)+:(1)];
  assign m_bvalid[1*(1)+:(1)]=m_axi_hbm1_bvalid;
  assign m_axi_hbm1_bready=m_bready[1*(1)+:(1)];
  assign m_bid[1*(AXI_ID_W)+:(AXI_ID_W)]=m_axi_hbm1_bid;
  assign m_bresp[1*(2)+:(2)]=m_axi_hbm1_bresp;
  assign m_axi_hbm1_arvalid=m_arvalid[1*(1)+:(1)];
  assign m_arready[1*(1)+:(1)]=m_axi_hbm1_arready;
  assign m_axi_hbm1_araddr=m_araddr[1*(64)+:(64)];
  assign m_axi_hbm1_arid=m_arid[1*(AXI_ID_W)+:(AXI_ID_W)];
  assign m_axi_hbm1_arlen=m_arlen[1*(8)+:(8)];
  assign m_axi_hbm1_arsize=m_arsize[1*(3)+:(3)];
  assign m_axi_hbm1_arburst=m_arburst[1*(2)+:(2)];
  assign m_rvalid[1*(1)+:(1)]=m_axi_hbm1_rvalid;
  assign m_axi_hbm1_rready=m_rready[1*(1)+:(1)];
  assign m_rdata[1*(AXI_DATA_W)+:(AXI_DATA_W)]=m_axi_hbm1_rdata;
  assign m_rid[1*(AXI_ID_W)+:(AXI_ID_W)]=m_axi_hbm1_rid;
  assign m_rresp[1*(2)+:(2)]=m_axi_hbm1_rresp;
  assign m_rlast[1*(1)+:(1)]=m_axi_hbm1_rlast;
 end else begin:g_unused1
  assign m_axi_hbm1_awvalid=0;
  assign m_axi_hbm1_awaddr=0;
  assign m_axi_hbm1_awid=0;
  assign m_axi_hbm1_awlen=0;
  assign m_axi_hbm1_awsize=0;
  assign m_axi_hbm1_awburst=0;
  assign m_axi_hbm1_wvalid=0;
  assign m_axi_hbm1_wdata=0;
  assign m_axi_hbm1_wstrb=0;
  assign m_axi_hbm1_wlast=0;
  assign m_axi_hbm1_bready=1;
  assign m_axi_hbm1_arvalid=0;
  assign m_axi_hbm1_araddr=0;
  assign m_axi_hbm1_arid=0;
  assign m_axi_hbm1_arlen=0;
  assign m_axi_hbm1_arsize=0;
  assign m_axi_hbm1_arburst=0;
  assign m_axi_hbm1_rready=1;
 end endgenerate
 generate if(MEM_PORTS>2)begin:g_port2
  assign m_axi_hbm2_awvalid=m_awvalid[2*(1)+:(1)];
  assign m_awready[2*(1)+:(1)]=m_axi_hbm2_awready;
  assign m_axi_hbm2_awaddr=m_awaddr[2*(64)+:(64)];
  assign m_axi_hbm2_awid=m_awid[2*(AXI_ID_W)+:(AXI_ID_W)];
  assign m_axi_hbm2_awlen=m_awlen[2*(8)+:(8)];
  assign m_axi_hbm2_awsize=m_awsize[2*(3)+:(3)];
  assign m_axi_hbm2_awburst=m_awburst[2*(2)+:(2)];
  assign m_axi_hbm2_wvalid=m_wvalid[2*(1)+:(1)];
  assign m_wready[2*(1)+:(1)]=m_axi_hbm2_wready;
  assign m_axi_hbm2_wdata=m_wdata[2*(AXI_DATA_W)+:(AXI_DATA_W)];
  assign m_axi_hbm2_wstrb=m_wstrb[2*(AXI_DATA_W/8)+:(AXI_DATA_W/8)];
  assign m_axi_hbm2_wlast=m_wlast[2*(1)+:(1)];
  assign m_bvalid[2*(1)+:(1)]=m_axi_hbm2_bvalid;
  assign m_axi_hbm2_bready=m_bready[2*(1)+:(1)];
  assign m_bid[2*(AXI_ID_W)+:(AXI_ID_W)]=m_axi_hbm2_bid;
  assign m_bresp[2*(2)+:(2)]=m_axi_hbm2_bresp;
  assign m_axi_hbm2_arvalid=m_arvalid[2*(1)+:(1)];
  assign m_arready[2*(1)+:(1)]=m_axi_hbm2_arready;
  assign m_axi_hbm2_araddr=m_araddr[2*(64)+:(64)];
  assign m_axi_hbm2_arid=m_arid[2*(AXI_ID_W)+:(AXI_ID_W)];
  assign m_axi_hbm2_arlen=m_arlen[2*(8)+:(8)];
  assign m_axi_hbm2_arsize=m_arsize[2*(3)+:(3)];
  assign m_axi_hbm2_arburst=m_arburst[2*(2)+:(2)];
  assign m_rvalid[2*(1)+:(1)]=m_axi_hbm2_rvalid;
  assign m_axi_hbm2_rready=m_rready[2*(1)+:(1)];
  assign m_rdata[2*(AXI_DATA_W)+:(AXI_DATA_W)]=m_axi_hbm2_rdata;
  assign m_rid[2*(AXI_ID_W)+:(AXI_ID_W)]=m_axi_hbm2_rid;
  assign m_rresp[2*(2)+:(2)]=m_axi_hbm2_rresp;
  assign m_rlast[2*(1)+:(1)]=m_axi_hbm2_rlast;
 end else begin:g_unused2
  assign m_axi_hbm2_awvalid=0;
  assign m_axi_hbm2_awaddr=0;
  assign m_axi_hbm2_awid=0;
  assign m_axi_hbm2_awlen=0;
  assign m_axi_hbm2_awsize=0;
  assign m_axi_hbm2_awburst=0;
  assign m_axi_hbm2_wvalid=0;
  assign m_axi_hbm2_wdata=0;
  assign m_axi_hbm2_wstrb=0;
  assign m_axi_hbm2_wlast=0;
  assign m_axi_hbm2_bready=1;
  assign m_axi_hbm2_arvalid=0;
  assign m_axi_hbm2_araddr=0;
  assign m_axi_hbm2_arid=0;
  assign m_axi_hbm2_arlen=0;
  assign m_axi_hbm2_arsize=0;
  assign m_axi_hbm2_arburst=0;
  assign m_axi_hbm2_rready=1;
 end endgenerate
 tnca_multicore_fpga #(.NX(3),.NY(1),.REG_ADDR_W(9),
  .CACHE_MAINTENANCE(CACHE_MAINTENANCE), .DCO_REGISTERED_RANGES(DCO_REGISTERED_RANGES),
  .MEM_PORTS(MEM_PORTS),
  .CODE_WORDS(CODE_WORDS),
  .CODE_LINK_STAGES(CODE_LINK_STAGES),
  .NOC_LINK_STAGES(NOC_LINK_STAGES),
  .DISPATCH_STAGES(DISPATCH_STAGES),
  .MEM_HOME_STAGES(MEM_HOME_STAGES),
  .LINK_STAGES(LINK_STAGES),
  .AXI_DATA_W(AXI_DATA_W),
  .AXI_OUT_N(AXI_OUT_N),
  .AXI_ID_W(AXI_ID_W),
  .NUM_RCTX(NUM_RCTX),
  .RCTX_W(RCTX_W),
  .PHYS_VREGS(PHYS_VREGS),
  .PHYS_W(PHYS_W),
  .VREGS_PER_RCTX(VREGS_PER_RCTX),
  .HAS_SFU(HAS_SFU),
  .HAS_ATOMICS(HAS_ATOMICS),
  .HAS_SYS_ATOMICS(HAS_SYS_ATOMICS),
  .USE_BANKED_VRF(USE_BANKED_VRF),
  .VRF_BYPASS(VRF_BYPASS),
  .FE_NOBUBBLE(FE_NOBUBBLE),
  .VRF_PREFETCH(VRF_PREFETCH),
  .RCTX_SWITCH(RCTX_SWITCH),
  .MTX_ABUF(MTX_ABUF),
  .LSU_INFLIGHT(LSU_INFLIGHT),
  .FPU_OVERLAP(FPU_OVERLAP),
  .RSLICE(RSLICE),
  .RPAIR(RPAIR),
  .HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE),
  .SETS(SETS),
  .SET_W(SET_W),
  .WAYS(WAYS),
  .WAY_W(WAY_W)
 ) device(
  .clk(ap_clk),.rst_n(rst_n),.reg_addr(reg_addr),.reg_wdata(reg_wdata),.reg_we(reg_we),.reg_re(reg_re),.reg_rdata(reg_rdata),
  .m_awvalid(m_awvalid),
  .m_awready(m_awready),
  .m_awaddr(m_awaddr),
  .m_awid(m_awid),
  .m_awlen(m_awlen),
  .m_awsize(m_awsize),
  .m_awburst(m_awburst),
  .m_wvalid(m_wvalid),
  .m_wready(m_wready),
  .m_wdata(m_wdata),
  .m_wstrb(m_wstrb),
  .m_wlast(m_wlast),
  .m_bvalid(m_bvalid),
  .m_bready(m_bready),
  .m_bid(m_bid),
  .m_bresp(m_bresp),
  .m_arvalid(m_arvalid),
  .m_arready(m_arready),
  .m_araddr(m_araddr),
  .m_arid(m_arid),
  .m_arlen(m_arlen),
  .m_arsize(m_arsize),
  .m_arburst(m_arburst),
  .m_rvalid(m_rvalid),
  .m_rready(m_rready),
  .m_rdata(m_rdata),
  .m_rid(m_rid),
  .m_rresp(m_rresp),
  .m_rlast(m_rlast)
 );
 generate if(MEM_PORTS<1 || MEM_PORTS>3)begin:g_bad_ports
  tnca_invalid_u280_memory_ports invalid_configuration();
 end endgenerate
endmodule
`default_nettype wire
