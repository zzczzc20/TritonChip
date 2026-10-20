//==============================================================================
// tnca_u50_top -- the device and its bus adapter, joined on one clock pair
//
// doc/phase3_6_tasks_and_acceptance.md R36-HAND-02, at the level above the one
// that file stops at.
//
// rtl/fpga/tnca_fpga_top.v is the device with a flat register port and one
// clock. rtl/fpga/tnca_axi_lite.v turns a 32-bit AXI4-Lite slave into that port
// across two clock domains. Neither names the other: tnca_fpga_top's header
// says an AXI-Lite adapter attaches above it, and tnca_axi_lite's header says a
// wrapper has to hand it two resets that are already synchronised and released
// together. This file is that wrapper. It is the last thing before the board
// that can still be simulated, and the split is arranged so that it is.
//
// WHAT IS HERE
//
//   THE JOIN          reg_* on tnca_axi_lite's device side, straight onto
//                     tnca_fpga_top's register port. There is no glue because
//                     the port was written to be joined, and the place that
//                     would have gone is where a bug would hide.
//
//   THE RESET         One asynchronous source, released in a fixed order. The
//                     order is not decoration and is argued below.
//
//   THE SECOND CLOCK  A PORT, not a primitive. On the board clk_dev comes from
//                     ap_clk through a BUFGCE_DIV; generating it here would put
//                     a Xilinx primitive in a file iverilog has to read, and
//                     `make lint` would stop being able to see any of this. The
//                     file that owns the primitive is rtl/board/tnca_u50_xo.v
//                     and it adds nothing else -- no logic, no third name for
//                     anything, so there is no layer for a signal to go missing
//                     in.
//
// WHAT IS NOT HERE
//
//   THE BUS           The AXI4-Lite ports are this module's ports. They are not
//                     re-registered, re-named, or crossed a second time, and
//                     the names are the platform's (`s_axi_tnca_*`) rather than
//                     the adapter's, so that the .xo above this is a pure
//                     pass-through. A rename is a place for a typo and an
//                     unconnected port in a .xo is a typo that does not always
//                     announce itself.
//
//   EVERYTHING ELSE   No pin, no MMCM, no PCIe, no DMA. All 512 KiB of storage
//                     is the block RAM behind tnca_fpga_mem.v. A kernel that
//                     wants HBM is a different file and a different memory, not
//                     a parameter of this one.
//
// WHY THE RESET IS RELEASED IN AN ORDER, AND WHICH ORDER
//
// The bus domain must not leave reset while the device domain is still in it.
//
// tnca_axi_lite crosses on a toggle: the bus side flips a bit, and waits for the
// device side to flip one back. Both sides reset those bits to zero. If the bus
// side came out first it could flip its bit and start waiting, and the device
// side's reset would then release with that bit already high -- so the device
// would see, in its first cycles, a toggle that arrived before it was listening.
// It would answer a write it had no record of, at whatever address its reset
// value happens to be. The address register is the one that would be reset, and
// the address the device would act on is zero.
//
// So the device domain runs first and the bus domain follows it, and never the
// other way round. `dev_rst_sync` below is the device side's own synchroniser,
// which makes it the reset synchroniser and the head start in one: the bus side
// cannot take a single edge until the device side has taken four.
//
// Assertion is not sequenced and does not need to be. Both are reset by the same
// asynchronous signal, so asserting rst_n_async drops dev_ready_axi on the same
// edge and axi_rst_n with it. There is no window in which one is held and the
// other is running -- only a window during release, and that is the one the
// ordering closes.
//
// WHAT THIS DOES NOT ESTABLISH
//
// Not that it fits, not that it meets timing at any frequency, and not that it
// runs on a board. Nothing in this repository has been on a board. The clock
// pair is taken as given and nothing here assumes a frequency, which is the
// same position tnca_fpga_top takes and for the same reason.
//==============================================================================
`default_nettype none

module tnca_u50_top #(
    // ---- AXI4-Lite, host side ----------------------------------------------
    // Twelve bits of byte address covers 0x000..0xFFF. The register map ends at
    // 0x0BC (R40-04 added 0xA8..0xBC), and tnca_axi_lite decodes the low eight
    // bits, so the four bits above them are ignored by design rather than by
    // accident.
    parameter S_AXI_ADDR_W   = 12,

    // ---- the device --------------------------------------------------------
    parameter ADDR_BITS      = 8,

    // WORDS is not free. It is the size of the device the profile describes --
    // 65536 words of 64 bits, 512 KiB -- and the profile is what every compiled
    // artifact was built against. rtl/fpga/tnca_fpga_mem.v carries the argument;
    // the short version is that a board which is smaller is a different machine
    // and one which is larger is a machine nothing has compiled for.
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


    // R40-04: the data memory above LOCAL_LIMIT is an AXI4 memory reached
    // through m_axi_hbm (rtl/fpga/tnca_axi_bridge.v). The master is in the
    // clk_dev domain: an image that sets DATA_AXI gives the device ap_clk
    // itself (rtl/board/tnca_hbm_xo.v), so the platform's interconnect and
    // the bridge share a clock. 0: the ports are driven idle.
    parameter DATA_AXI       = 0,
    parameter LOCAL_LIMIT    = 64'h20000,
    parameter AXI_OUT_N      = 8,
    parameter AXI_LINE       = 0,           // tnca_fpga_top: the burst mechanism
    parameter AXI_LB_N       = 8,
    parameter AXI_PF         = 1,
    parameter AXI_LOCAL_PIPE = 0,
    parameter AXI_RAM_CACHE  = 0,
    parameter AXI_WAIT_N     = 8
) (
    //---- bus side, ap_clk domain -------------------------------------------
    input  wire                    ap_clk,

    //---- device side, clk_dev domain ---------------------------------------
    input  wire                    clk_dev,

    //---- reset, asynchronous to both ---------------------------------------
    // Asserted asynchronously and held until the board is ready. A pulse
    // narrower than a clk_dev period can be missed entirely, and no synchroniser
    // can recover a reset that was never observed; what is synchronised here is
    // the release.
    input  wire                    rst_n_async,

    //---- AXI4-Lite slave, s_axi_tnca, ap_clk domain ------------------------
    input  wire [S_AXI_ADDR_W-1:0] s_axi_tnca_awaddr,
    input  wire                    s_axi_tnca_awvalid,
    output wire                    s_axi_tnca_awready,
    input  wire [31:0]             s_axi_tnca_wdata,
    input  wire [3:0]              s_axi_tnca_wstrb,
    input  wire                    s_axi_tnca_wvalid,
    output wire                    s_axi_tnca_wready,
    output wire [1:0]              s_axi_tnca_bresp,
    output wire                    s_axi_tnca_bvalid,
    input  wire                    s_axi_tnca_bready,

    input  wire [S_AXI_ADDR_W-1:0] s_axi_tnca_araddr,
    input  wire                    s_axi_tnca_arvalid,
    output wire                    s_axi_tnca_arready,
    output wire [31:0]             s_axi_tnca_rdata,
    output wire [1:0]              s_axi_tnca_rresp,
    output wire                    s_axi_tnca_rvalid,
    input  wire                    s_axi_tnca_rready,

    //---- R40-04 AXI4 master, clk_dev domain ---------------------------------
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
    // Reset
    //
    // Two synchronisers and no handshake, because the ordering above is enough
    // and a handshake here would be a state machine that has to survive the
    // reset it is generating. See the header for why the device side is the one
    // that leads.
    //==========================================================================

    // The device side. Four flops, so that the bus side's release -- which waits
    // on dev_ready_axi, taken from the last of them -- is at least four clk_dev
    // edges behind the device actually running.
    (* ASYNC_REG = "TRUE" *) reg [3:0] dev_rst_sync;

    always @(posedge clk_dev or negedge rst_n_async) begin
        if (!rst_n_async)
            dev_rst_sync <= 4'b0000;
        else
            dev_rst_sync <= {dev_rst_sync[2:0], 1'b1};
    end

    wire dev_rst_n = dev_rst_sync[3];

    // The bus side. Its release is dev_rst_n, so the ordering is a property of
    // what is fed in and not of a delay that has to be kept long enough by hand.
    (* ASYNC_REG = "TRUE" *) reg [1:0] dev_ready_axi;

    always @(posedge ap_clk or negedge rst_n_async) begin
        if (!rst_n_async)
            dev_ready_axi <= 2'b00;
        else
            dev_ready_axi <= {dev_ready_axi[0], dev_rst_n};
    end

    wire axi_rst_n = dev_ready_axi[1];

    //==========================================================================
    // The device's register port, and the only wires between the two blocks
    //==========================================================================
    wire [ADDR_BITS-1:0] reg_addr;
    wire [63:0]          reg_wdata;
    wire                 reg_we;
    wire                 reg_re;
    wire [63:0]          reg_rdata;

    //==========================================================================
    // The bus adapter
    //==========================================================================
    tnca_axi_lite #(
        .ADDR_BITS(ADDR_BITS),
        .S_AXI_ADDR_W(S_AXI_ADDR_W)
    ) u_axi (
        .s_axi_aclk(ap_clk),
        .s_axi_aresetn(axi_rst_n),

        .s_axi_awaddr(s_axi_tnca_awaddr),
        .s_axi_awvalid(s_axi_tnca_awvalid),
        .s_axi_awready(s_axi_tnca_awready),
        .s_axi_wdata(s_axi_tnca_wdata),
        .s_axi_wstrb(s_axi_tnca_wstrb),
        .s_axi_wvalid(s_axi_tnca_wvalid),
        .s_axi_wready(s_axi_tnca_wready),
        .s_axi_bresp(s_axi_tnca_bresp),
        .s_axi_bvalid(s_axi_tnca_bvalid),
        .s_axi_bready(s_axi_tnca_bready),

        .s_axi_araddr(s_axi_tnca_araddr),
        .s_axi_arvalid(s_axi_tnca_arvalid),
        .s_axi_arready(s_axi_tnca_arready),
        .s_axi_rdata(s_axi_tnca_rdata),
        .s_axi_rresp(s_axi_tnca_rresp),
        .s_axi_rvalid(s_axi_tnca_rvalid),
        .s_axi_rready(s_axi_tnca_rready),

        .reg_clk(clk_dev),
        .reg_rst_n(dev_rst_n),

        .reg_addr(reg_addr),
        .reg_wdata(reg_wdata),
        .reg_we(reg_we),
        .reg_re(reg_re),
        .reg_rdata(reg_rdata)
    );

    //==========================================================================
    // The device
    //
    // rst_n_async rather than dev_rst_n. tnca_fpga_top synchronises its own
    // release, so handing it the already-synchronised dev_rst_n would cost two
    // more cycles and buy nothing -- and handing it the raw source keeps the one
    // signal from which every other reset here is derived visible as one signal.
    //==========================================================================
    tnca_fpga_top #(
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
        .WORDS(WORDS),
        .ADDR_W(ADDR_W),
        .ADDR_BITS(ADDR_BITS),
        .LSU_TAG_W(5),
        .DATA_AXI(DATA_AXI),
        .LOCAL_LIMIT(LOCAL_LIMIT),
        .AXI_OUT_N(AXI_OUT_N),
        .AXI_LINE(AXI_LINE),
        .AXI_LB_N(AXI_LB_N),
        .AXI_PF(AXI_PF),
        .AXI_LOCAL_PIPE(AXI_LOCAL_PIPE),
        .AXI_RAM_CACHE(AXI_RAM_CACHE),
        .AXI_WAIT_N(AXI_WAIT_N)
    ) u_device (
        .clk(clk_dev),
        .rst_n_async(rst_n_async),

        .reg_addr(reg_addr),
        .reg_wdata(reg_wdata),
        .reg_we(reg_we),
        .reg_re(reg_re),
        .reg_rdata(reg_rdata),

        .m_axi_awvalid(m_axi_hbm_awvalid), .m_axi_awready(m_axi_hbm_awready),
        .m_axi_awaddr(m_axi_hbm_awaddr), .m_axi_awid(m_axi_hbm_awid),
        .m_axi_awlen(m_axi_hbm_awlen), .m_axi_awsize(m_axi_hbm_awsize),
        .m_axi_awburst(m_axi_hbm_awburst),
        .m_axi_wvalid(m_axi_hbm_wvalid), .m_axi_wready(m_axi_hbm_wready),
        .m_axi_wdata(m_axi_hbm_wdata), .m_axi_wstrb(m_axi_hbm_wstrb),
        .m_axi_wlast(m_axi_hbm_wlast),
        .m_axi_bvalid(m_axi_hbm_bvalid), .m_axi_bready(m_axi_hbm_bready),
        .m_axi_bid(m_axi_hbm_bid), .m_axi_bresp(m_axi_hbm_bresp),
        .m_axi_arvalid(m_axi_hbm_arvalid), .m_axi_arready(m_axi_hbm_arready),
        .m_axi_araddr(m_axi_hbm_araddr), .m_axi_arid(m_axi_hbm_arid),
        .m_axi_arlen(m_axi_hbm_arlen), .m_axi_arsize(m_axi_hbm_arsize),
        .m_axi_arburst(m_axi_hbm_arburst),
        .m_axi_rvalid(m_axi_hbm_rvalid), .m_axi_rready(m_axi_hbm_rready),
        .m_axi_rdata(m_axi_hbm_rdata), .m_axi_rid(m_axi_hbm_rid),
        .m_axi_rresp(m_axi_hbm_rresp), .m_axi_rlast(m_axi_hbm_rlast)
    );

endmodule

`default_nettype wire
