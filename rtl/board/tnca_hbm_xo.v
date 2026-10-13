//==============================================================================
// tnca_hbm_xo -- the .xo top of the R40-04 image: the device with HBM
//
// rtl/board/tnca_u50_xo.v with one AXI4 master more, m_axi_hbm, through which
// every device address at or above LOCAL_LIMIT reaches the HBM the link
// connects it to (flow/fpga/link.sh, LINK_SP; rtl/fpga/tnca_axi_bridge.v
// says what the port promises). Everything else -- the register port, the
// power-on reset, the register map -- is the other top's, and its header is
// the argument for each of them.
//
// THE CLOCK
//
// The device runs on ap_clk itself, with no BUFGCE_DIV in between. The other
// top divides by DEV_DIV = 1 and so already runs the device at ap_clk's
// frequency; here the master's channels have to meet the platform's
// interconnect, which is clocked by ap_clk, and one clock on both sides of
// that interface is a synchronous interface rather than a crossing between
// two clock objects Vivado has to relate. tnca_u50_top's register-port
// crossing (toggle and synchronisers) works the same with one clock as with
// two; its header says why the ratio does not matter to it. A device slower
// than ap_clk would need an AXI clock converter here, and this image does not
// offer one.
//
// The master's address is 64 bits and its data 64: one 8-byte beat per
// access, which is the capacity step. kernel.xml (fpga/kernel_hbm.xml)
// declares the port and an argument on it, `hbm`, at register offset 0xA8 --
// R_AXI_BASE -- so the connectivity option has an argument to name and an
// XRT that writes the argument writes the window's base.
//==============================================================================
`default_nettype none

module tnca_hbm_xo #(
    parameter S_AXI_ADDR_W   = 12,
    parameter ADDR_BITS      = 8,
    parameter WORDS          = 65536,
    parameter ADDR_W         = 16,
    parameter NUM_RCTX       = 2,
    parameter RCTX_W         = 1,
    parameter VREGS_PER_RCTX = 64,
    // Architectural switches passed from the image profile, with legacy defaults.
    parameter HAS_SFU              = 1,
    parameter HAS_ATOMICS          = 1,
    parameter HAS_SYS_ATOMICS      = 0,
    parameter USE_BANKED_VRF       = 1,
    parameter VRF_BYPASS           = 1,
    parameter FE_NOBUBBLE          = 1,
    parameter VRF_PREFETCH         = 1,
    parameter RCTX_SWITCH          = 0,
    parameter LSU_INFLIGHT         = 1,
    parameter FPU_OVERLAP          = 0,
    parameter RSLICE               = 0,
    parameter RPAIR                = 0,
    parameter MTX_ABUF             = 0,
    parameter HAS_TENSOR_XPOSE     = 0,

    // The on-chip RAM keeps addresses below this (the code region and the
    // runtime's own area); the AXI window starts here.
    parameter LOCAL_LIMIT    = 64'h20000,
    parameter AXI_OUT_N      = 8,
    // The burst mechanism (rtl/fpga/tnca_fpga_top.v AXI_LINE): 0 is one beat
    // per access, the image R40-04 linked first.
    parameter AXI_LINE       = 0,
    parameter AXI_LB_N       = 8,
    parameter AXI_PF         = 1,
    parameter AXI_LOCAL_PIPE = 0,
    parameter AXI_RAM_CACHE  = 0,
    parameter AXI_WAIT_N     = 8
) (
    //---- the platform's clock ----------------------------------------------
    input  wire                    ap_clk,

    //---- AXI4-Lite slave, s_axi_tnca ---------------------------------------
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

    //---- AXI4 master, m_axi_hbm --------------------------------------------
    output wire                    m_axi_hbm_awvalid,
    input  wire                    m_axi_hbm_awready,
    output wire [63:0]             m_axi_hbm_awaddr,
    output wire [4:0]              m_axi_hbm_awid,
    output wire [7:0]              m_axi_hbm_awlen,
    output wire [2:0]              m_axi_hbm_awsize,
    output wire [1:0]              m_axi_hbm_awburst,
    output wire                    m_axi_hbm_wvalid,
    input  wire                    m_axi_hbm_wready,
    output wire [63:0]             m_axi_hbm_wdata,
    output wire [7:0]              m_axi_hbm_wstrb,
    output wire                    m_axi_hbm_wlast,
    input  wire                    m_axi_hbm_bvalid,
    output wire                    m_axi_hbm_bready,
    input  wire [4:0]              m_axi_hbm_bid,
    input  wire [1:0]              m_axi_hbm_bresp,
    output wire                    m_axi_hbm_arvalid,
    input  wire                    m_axi_hbm_arready,
    output wire [63:0]             m_axi_hbm_araddr,
    output wire [4:0]              m_axi_hbm_arid,
    output wire [7:0]              m_axi_hbm_arlen,
    output wire [2:0]              m_axi_hbm_arsize,
    output wire [1:0]              m_axi_hbm_arburst,
    input  wire                    m_axi_hbm_rvalid,
    output wire                    m_axi_hbm_rready,
    input  wire [63:0]             m_axi_hbm_rdata,
    input  wire [4:0]              m_axi_hbm_rid,
    input  wire [1:0]              m_axi_hbm_rresp,
    input  wire                    m_axi_hbm_rlast
);

    //==========================================================================
    // Power-on reset: tnca_u50_xo's, unchanged (its header says why 64)
    //==========================================================================
    reg [6:0] por_cnt;

    always @(posedge ap_clk)
        if (!por_cnt[6])
            por_cnt <= por_cnt + 7'd1;

    wire por_rst_n = por_cnt[6];

    tnca_u50_top #(
        .S_AXI_ADDR_W(S_AXI_ADDR_W),
        .ADDR_BITS(ADDR_BITS),
        .WORDS(WORDS),
        .ADDR_W(ADDR_W),
        .NUM_RCTX(NUM_RCTX),
        .RCTX_W(RCTX_W),
        .VREGS_PER_RCTX(VREGS_PER_RCTX),
        .HAS_SFU(HAS_SFU),
        .HAS_ATOMICS(HAS_ATOMICS),
        .HAS_SYS_ATOMICS(HAS_SYS_ATOMICS),
        .USE_BANKED_VRF(USE_BANKED_VRF),
        .VRF_BYPASS(VRF_BYPASS),
        .FE_NOBUBBLE(FE_NOBUBBLE),
        .VRF_PREFETCH(VRF_PREFETCH),
        .RCTX_SWITCH(RCTX_SWITCH),
        .LSU_INFLIGHT(LSU_INFLIGHT),
        .FPU_OVERLAP(FPU_OVERLAP),
        .RSLICE(RSLICE),
        .RPAIR(RPAIR),
        .MTX_ABUF(MTX_ABUF),
        .HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE),
        .DATA_AXI(1),
        .LOCAL_LIMIT(LOCAL_LIMIT),
        .AXI_OUT_N(AXI_OUT_N),
        .AXI_LINE(AXI_LINE),
        .AXI_LB_N(AXI_LB_N),
        .AXI_PF(AXI_PF),
        .AXI_LOCAL_PIPE(AXI_LOCAL_PIPE),
        .AXI_RAM_CACHE(AXI_RAM_CACHE),
        .AXI_WAIT_N(AXI_WAIT_N)
    ) u_top (
        .ap_clk(ap_clk),
        .clk_dev(ap_clk),
        .rst_n_async(por_rst_n),

        .s_axi_tnca_awaddr(s_axi_tnca_awaddr),
        .s_axi_tnca_awvalid(s_axi_tnca_awvalid),
        .s_axi_tnca_awready(s_axi_tnca_awready),
        .s_axi_tnca_wdata(s_axi_tnca_wdata),
        .s_axi_tnca_wstrb(s_axi_tnca_wstrb),
        .s_axi_tnca_wvalid(s_axi_tnca_wvalid),
        .s_axi_tnca_wready(s_axi_tnca_wready),
        .s_axi_tnca_bresp(s_axi_tnca_bresp),
        .s_axi_tnca_bvalid(s_axi_tnca_bvalid),
        .s_axi_tnca_bready(s_axi_tnca_bready),

        .s_axi_tnca_araddr(s_axi_tnca_araddr),
        .s_axi_tnca_arvalid(s_axi_tnca_arvalid),
        .s_axi_tnca_arready(s_axi_tnca_arready),
        .s_axi_tnca_rdata(s_axi_tnca_rdata),
        .s_axi_tnca_rresp(s_axi_tnca_rresp),
        .s_axi_tnca_rvalid(s_axi_tnca_rvalid),
        .s_axi_tnca_rready(s_axi_tnca_rready),

        .m_axi_hbm_awvalid(m_axi_hbm_awvalid), .m_axi_hbm_awready(m_axi_hbm_awready),
        .m_axi_hbm_awaddr(m_axi_hbm_awaddr), .m_axi_hbm_awid(m_axi_hbm_awid),
        .m_axi_hbm_awlen(m_axi_hbm_awlen), .m_axi_hbm_awsize(m_axi_hbm_awsize),
        .m_axi_hbm_awburst(m_axi_hbm_awburst),
        .m_axi_hbm_wvalid(m_axi_hbm_wvalid), .m_axi_hbm_wready(m_axi_hbm_wready),
        .m_axi_hbm_wdata(m_axi_hbm_wdata), .m_axi_hbm_wstrb(m_axi_hbm_wstrb),
        .m_axi_hbm_wlast(m_axi_hbm_wlast),
        .m_axi_hbm_bvalid(m_axi_hbm_bvalid), .m_axi_hbm_bready(m_axi_hbm_bready),
        .m_axi_hbm_bid(m_axi_hbm_bid), .m_axi_hbm_bresp(m_axi_hbm_bresp),
        .m_axi_hbm_arvalid(m_axi_hbm_arvalid), .m_axi_hbm_arready(m_axi_hbm_arready),
        .m_axi_hbm_araddr(m_axi_hbm_araddr), .m_axi_hbm_arid(m_axi_hbm_arid),
        .m_axi_hbm_arlen(m_axi_hbm_arlen), .m_axi_hbm_arsize(m_axi_hbm_arsize),
        .m_axi_hbm_arburst(m_axi_hbm_arburst),
        .m_axi_hbm_rvalid(m_axi_hbm_rvalid), .m_axi_hbm_rready(m_axi_hbm_rready),
        .m_axi_hbm_rdata(m_axi_hbm_rdata), .m_axi_hbm_rid(m_axi_hbm_rid),
        .m_axi_hbm_rresp(m_axi_hbm_rresp), .m_axi_hbm_rlast(m_axi_hbm_rlast)
    );

endmodule

`default_nettype wire
