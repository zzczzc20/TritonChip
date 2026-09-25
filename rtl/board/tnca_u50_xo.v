//==============================================================================
// tnca_u50_xo -- the .xo top: the platform's ports, and the device's clock
//
// This is the file package_xo is pointed at and the only file in the repository
// that contains a Xilinx primitive. It is deliberately the thinnest file here.
//
// WHY IT IS SEPARATE FROM rtl/fpga/tnca_u50_top.v
//
// Because of the primitive. `make lint` elaborates every file under rtl/fpga/
// with iverilog, and iverilog has no BUFGCE_DIV; putting one in a file the
// linter reads would trade a whole directory's coverage for one line of
// convenience. So the divider lives here, in a directory the linter does not
// glob, and rtl/fpga/tnca_u50_top.v takes `clk_dev` as a port and stays
// lintable. What is left here is connectivity and a counter.
//
// Nothing in this file has a name of its own except `ap_clk`, which the
// platform supplies, and the ports of tnca_u50_top, which are passed straight
// through with no re-naming. That is on purpose: a rename is a place to type a
// name wrong, and an unconnected port in a .xo does not always announce itself
// -- it can reach the board as a register that reads zero for ever.
//
// THE PLATFORM'S CONTRACT, AS THE TOOL'S OWN OUTPUT STATES IT
//
// Taken from the RTL Kernel Wizard's shipped example rather than from memory
// (Vitis 2022.2, cli/examples/xo_kernel_srcs/rtl_kernel_wizard_0.xo, unpacked):
//
//   * The top has ONE system port, `ap_clk`. There is no `ap_rst_n`: the
//     wizard's own top has no such port and instantiates its AXI control slave
//     with `.ap_rst_n(1'b1)`.
//   * Its out-of-context constraints are one line:
//         create_clock -period 4.000 [get_ports ap_clk]
//   * An AXI4-Lite slave declared `range="0x1000"` has a twelve-bit address,
//     which is the `S_AXI_ADDR_W` tnca_axi_lite was written with.
//
// THE RESET, SINCE THERE ISN'T ONE
//
// No external reset means the reset has to be generated, and a counter off
// `ap_clk` is the source that is always there. On a Xilinx part every
// flip-flop powers up in a defined state -- the configuration bitstream sets
// it, which is why `por_cnt` below has no `initial` and would be wrong to be
// given one -- so the counter starts at zero and the reset is held for 64
// `ap_clk` edges, which at 300 MHz is 213 ns.
//
// It is released as a level, synchronously to `ap_clk`, and tnca_u50_top treats
// it as an asynchronous reset. A synchronous release into an asynchronous reset
// input is the safe direction of that mismatch: the synchronisers in
// tnca_u50_top see a clean single edge, and the assertion they are supposed to
// catch at power-up has already happened before `ap_clk` started.
//
// THE DEVICE CLOCK, AND WHY IT IS EIGHT
//
// 300 MHz for the bus, divided by eight for the device, is 37.5 MHz. The
// measured worst path through this design, placed and routed out of context,
// is 18.829 ns -- 53 MHz -- so eight leaves about 40% of the period unspent.
//
// Eight is also the largest divide BUFGCE_DIV has. Anything slower needs an
// MMCM and a re-link, and saying so here is the point: a larger number in
// BUFGCE_DIVIDE is not an edit, and the divider is not the knob to reach for
// if this does not close.
//
// Nothing here writes a constraint. ap_clk's period belongs to the platform
// and to flow/fpga/ooc.xdc, and the device clock is not separately constrained
// because Vivado propagates the divide through BUFGCE_DIV on its own. A second
// create_clock on clk_dev would not be a constraint, it would be a contradiction.
//==============================================================================
`default_nettype none

module tnca_u50_xo #(
    parameter S_AXI_ADDR_W   = 12,
    parameter ADDR_BITS      = 8,
    parameter WORDS          = 65536,
    parameter ADDR_W         = 16,
    parameter NUM_RCTX       = 2,
    parameter RCTX_W         = 1,
    parameter VREGS_PER_RCTX = 64
) (
    //---- the platform's clock ----------------------------------------------
    input  wire                    ap_clk,

    //---- AXI4-Lite slave, s_axi_tnca ---------------------------------------
    // The order is the wizard's, so that a diff against its output reads as a
    // shorter diff than it is.
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
    output wire [1:0]              s_axi_tnca_bresp
);

    //==========================================================================
    // Power-on reset
    //
    // 64 edges is 213 ns at 300 MHz: long enough that every clock in the design
    // has started, short enough that it is over before the host can finish
    // opening the device. Nothing depends on it being exactly 64.
    //==========================================================================
    reg [6:0] por_cnt;

    always @(posedge ap_clk)
        if (!por_cnt[6])
            por_cnt <= por_cnt + 7'd1;

    wire por_rst_n = por_cnt[6];

    //==========================================================================
    // The device's clock
    //
    // CE tied high and CLR tied low: the divider is never stopped and never
    // cleared, so clk_dev's relationship to ap_clk is fixed from the moment the
    // bitstream is loaded. Clearing it would shift that relationship by a cycle
    // and buy nothing -- the reset synchronisers downstream do not care about
    // phase, only about having edges.
    //==========================================================================
    wire clk_dev;

    BUFGCE_DIV #(
        .BUFGCE_DIVIDE  (8),
        .CE_TYPE        ("SYNC"),
        .IS_CE_INVERTED (1'b0),
        .IS_CLR_INVERTED(1'b0),
        .IS_I_INVERTED  (1'b0)
    ) u_clk_div (
        .O  (clk_dev),
        .CE (1'b1),
        .CLR(1'b0),
        .I  (ap_clk)
    );

    //==========================================================================
    // The wrapper, passed straight through
    //==========================================================================
    tnca_u50_top #(
        .S_AXI_ADDR_W(S_AXI_ADDR_W),
        .ADDR_BITS(ADDR_BITS),
        .WORDS(WORDS),
        .ADDR_W(ADDR_W),
        .NUM_RCTX(NUM_RCTX),
        .RCTX_W(RCTX_W),
        .VREGS_PER_RCTX(VREGS_PER_RCTX)
    ) u_top (
        .ap_clk(ap_clk),
        .clk_dev(clk_dev),
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
        .s_axi_tnca_rready(s_axi_tnca_rready)
    );

endmodule

`default_nettype wire
