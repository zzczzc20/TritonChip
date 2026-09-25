//==============================================================================
// tnca_axi_lite -- the flat register port, on a host bus, across a clock
//
// doc/phase3_6_tasks_and_acceptance.md R36-RTL-02, R36-IO-01 and R36-IO-02.
//
// rtl/fpga/tnca_fpga_top.v takes `reg_addr`/`reg_wdata`/`reg_we`/`reg_re` and
// returns `reg_rdata`, and its header names the gap this file fills: "`reg_*`
// is a flat register port, deliberately not AXI. An AXI-Lite adapter attaches
// above this module." This is that adapter. It changes the medium only -- the
// protocol and the register map are the ones in rtl/fpga/tnca_mmio.v and are
// not renegotiated here.
//
// WHY THERE ARE TWO CLOCKS IN IT
//
// The platform's kernel clock is 300 MHz and cannot be lowered; of the two
// scalable clocks this U50 platform declares, 300 and 500, neither is reachable
// by this design. So the device is given a divided clock by the wrapper above,
// and the register port is on that divided clock while the host bus is not.
// Every clock crossing in the design is in this file.
//
// The crossing is a toggle handshake, not a FIFO. The host polls with one
// register access outstanding at a time, so there is never a second
// transaction to queue and a FIFO would be a queue for one item.
//
// WHAT 64 BITS ON A 32-BIT BUS MEANS
//
// The registers are 64 bits and the bus is not; XRT has no 64-bit register
// access at all (`xclRegRead`'s data parameter is `uint32_t *`), so a register
// is two bus accesses, at `offset` and `offset + 4`. The high half is what
// completes the write: only then is `reg_we` raised, once, with the assembled
// 64-bit value.
//
// That is not an optimisation. `R_HOST_CTRL` bit 1 (START) and `R_DOORBELL` are
// level-sensitive in tnca_mmio.v and need a one-cycle pulse, because holding
// one high re-issues the access every time the host window returns to idle. A
// design that pulsed `reg_we` once per bus write would fire START twice per
// register, and the second one would be a second launch.
//
// A write to the low half alone is therefore latched and does nothing, which is
// what makes the two accesses of a register indivisible from the device's side:
// nothing observes the register until both halves have arrived.
//
// WHAT THIS FILE IS NOT
//
// It is not the board and it does not make a clock. `s_axi_aclk` and `reg_clk`
// are both inputs; where they come from is the wrapper's business, and keeping
// the primitive that makes them out of here is what lets this file run under
// iverilog and be elaborated by `make lint` like every other rtl/fpga file.
//
// `reg_rst_n` is taken the way tnca_fpga_top.v takes its own reset: asserted
// asynchronously and released synchronously to `reg_clk`, by the wrapper, with
// the same two-flop pattern. This module does not make a second one.
//
// The two resets must also be released together. A `wr_ack_tog` or `rd_ack_tog`
// going back to zero on its own is indistinguishable from an acknowledgement to
// the far side, so a slow-domain reset on its own would be read as one. The
// wrapper drives both from the same source, which is what makes this hold.
//
// The slave does not decode the high address bits: a byte offset is eight bits
// and the aperture declared in kernel.xml is 0x100, so the aliasing above that
// is outside what any master should generate. An access that aliases anyway is
// answered OKAY and lands on the mirrored register rather than being refused.
//==============================================================================

`default_nettype none

module tnca_axi_lite #(
    // The device's register address is a byte offset with the low three bits
    // zero -- tnca_mmio.v decodes on exact equality and its `default` arm does
    // nothing. Eight bits covers the 0x00..0xA0 the map uses.
    parameter ADDR_BITS    = 8,

    // AXI4-Lite address width, in bytes. Twelve bits covers 0x000..0xFFF, and
    // the highest offset anything here touches is 0xA0 + 4 (the high half of
    // SERVED).
    parameter S_AXI_ADDR_W = 12
) (
    //---- host side, s_axi_aclk domain ---------------------------------------
    input  wire                    s_axi_aclk,
    input  wire                    s_axi_aresetn,

    input  wire [S_AXI_ADDR_W-1:0] s_axi_awaddr,
    input  wire                    s_axi_awvalid,
    output wire                    s_axi_awready,
    input  wire [31:0]             s_axi_wdata,
    input  wire [3:0]              s_axi_wstrb,
    input  wire                    s_axi_wvalid,
    output wire                    s_axi_wready,
    output wire [1:0]              s_axi_bresp,
    output wire                    s_axi_bvalid,
    input  wire                    s_axi_bready,

    input  wire [S_AXI_ADDR_W-1:0] s_axi_araddr,
    input  wire                    s_axi_arvalid,
    output wire                    s_axi_arready,
    output wire [31:0]             s_axi_rdata,
    output wire [1:0]              s_axi_rresp,
    output wire                    s_axi_rvalid,
    input  wire                    s_axi_rready,

    //---- device side, reg_clk domain ----------------------------------------
    input  wire                    reg_clk,
    input  wire                    reg_rst_n,

    output wire [ADDR_BITS-1:0]    reg_addr,
    output wire [63:0]             reg_wdata,
    output wire                    reg_we,
    output wire                    reg_re,
    input  wire [63:0]             reg_rdata
);

    // The device's bus has no byte enables: `R_HOST_BE` is a register the host
    // writes, not a property of the access, and the payload window applies it
    // inside tnca_mmio.v. A narrow `wstrb` is therefore not something this
    // adapter can honour, and pretending otherwise would write a full word for
    // a request that asked for a byte. It is referenced here so that the port
    // is visibly read rather than dropped without a word.
    wire unused_wstrb = |s_axi_wstrb;

    //==========================================================================
    // The crossing, declared first
    //
    // Four buses and three toggles. Each bus goes still, then its toggle says
    // it has gone still; the toggle coming back says the far side acted. The
    // buses are deliberately not synchronised -- they are sampled only while
    // they are known to be still, which is what the toggle buys and what no
    // synchroniser on a 64-bit bus could buy.
    //==========================================================================
    // host -> device
    reg  [ADDR_BITS-1:0] wr_addr_f;     // byte offset, low three bits zero
    reg  [63:0]          wr_data_f;     // assembled 64-bit value
    reg                  wr_tog;
    reg  [ADDR_BITS-1:0] rd_addr_f;     // byte offset, low three bits zero
    reg                  rd_tog;

    // device -> host
    reg  [63:0]          rd_data_s;
    reg                  rd_ack_tog;
    reg                  wr_ack_tog;

    //==========================================================================
    // Host side: the write channel
    //
    // AW and W are captured separately because AXI does not require them to
    // arrive together, and a slave that assumes they do deadlocks on the first
    // master that does not. Each ready is held low once its half is taken, so
    // neither can be taken twice.
    //==========================================================================
    localparam W_IDLE = 2'd0,   // take AW and W
               W_SEND = 2'd1,   // the value is assembled and crossing
               W_RESP = 2'd2;   // hold B until bready

    reg [1:0]              wstate;
    reg                    aw_have;
    reg                    w_have;
    reg [S_AXI_ADDR_W-1:0] awaddr_q;
    reg [31:0]             wdata_q;
    reg [31:0]             shadow_lo;   // low half of a register not yet complete

    (* ASYNC_REG = "TRUE" *) reg [2:0] wr_ack_sync;

    wire wr_ack_pulse = wr_ack_sync[2] ^ wr_ack_sync[1];

    assign s_axi_awready = (wstate == W_IDLE) && !aw_have;
    assign s_axi_wready  = (wstate == W_IDLE) && !w_have;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_bvalid  = (wstate == W_RESP);

    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            wstate      <= W_IDLE;
            aw_have     <= 1'b0;
            w_have      <= 1'b0;
            awaddr_q    <= {S_AXI_ADDR_W{1'b0}};
            wdata_q     <= 32'd0;
            shadow_lo   <= 32'd0;
            wr_addr_f   <= {ADDR_BITS{1'b0}};
            wr_data_f   <= 64'd0;
            wr_tog      <= 1'b0;
            wr_ack_sync <= 3'b000;
        end else begin
            wr_ack_sync <= {wr_ack_sync[1:0], wr_ack_tog};

            case (wstate)
                W_IDLE: begin
                    if (s_axi_awvalid && s_axi_awready) begin
                        awaddr_q <= s_axi_awaddr;
                        aw_have  <= 1'b1;
                    end
                    if (s_axi_wvalid && s_axi_wready) begin
                        wdata_q <= s_axi_wdata;
                        w_have  <= 1'b1;
                    end

                    // Both halves in hand. This tests the registered pair, so it
                    // cannot fire in the cycle the second one lands -- and that
                    // extra cycle is the price of not assigning to the same
                    // signal from two places, which is what yosys warned about
                    // 648 times over when tnca_mmio.v was first written.
                    if (aw_have && w_have) begin
                        aw_have <= 1'b0;
                        w_have  <= 1'b0;
                        if (awaddr_q[2]) begin
                            // High half: assemble and cross. This is the write.
                            wr_addr_f <= {awaddr_q[ADDR_BITS-1:3], 3'b000};
                            wr_data_f <= {wdata_q, shadow_lo};
                            wr_tog    <= ~wr_tog;
                            wstate    <= W_SEND;
                        end else begin
                            // Low half alone: latch it and answer. The device
                            // sees nothing, which is the point -- a register is
                            // not observable until both halves have arrived.
                            shadow_lo <= wdata_q;
                            wstate    <= W_RESP;
                        end
                    end
                end

                W_SEND: begin
                    // The toggle has been up for a cycle; wait for the device
                    // side to say it acted. There is no timeout here; a device
                    // that never answers leaves the response channel held, and
                    // the host's own deadline is what ends the wait.
                    if (wr_ack_pulse)
                        wstate <= W_RESP;
                end

                W_RESP: begin
                    if (s_axi_bready)
                        wstate <= W_IDLE;
                end

                default: wstate <= W_IDLE;
            endcase
        end
    end

    //==========================================================================
    // Host side: the read channel
    //
    // The device's read is combinational from `reg_addr`, so the device side
    // only has to put the address down, wait one of its own cycles, and take
    // what `reg_rdata` says. The address it puts down is the 64-bit register's,
    // so the half is selected here rather than there.
    //==========================================================================
    localparam R_IDLE = 2'd0,
               R_WAIT = 2'd1,
               R_RESP = 2'd2;

    reg [1:0]              rstate;
    reg [S_AXI_ADDR_W-1:0] araddr_q;

    (* ASYNC_REG = "TRUE" *) reg [2:0] rd_ack_sync;

    wire rd_ack_pulse = rd_ack_sync[2] ^ rd_ack_sync[1];

    assign s_axi_arready = (rstate == R_IDLE);
    assign s_axi_rresp   = 2'b00;
    assign s_axi_rvalid  = (rstate == R_RESP);
    assign s_axi_rdata   = araddr_q[2] ? rd_data_s[63:32] : rd_data_s[31:0];

    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            rstate      <= R_IDLE;
            araddr_q    <= {S_AXI_ADDR_W{1'b0}};
            rd_addr_f   <= {ADDR_BITS{1'b0}};
            rd_tog      <= 1'b0;
            rd_ack_sync <= 3'b000;
        end else begin
            rd_ack_sync <= {rd_ack_sync[1:0], rd_ack_tog};

            case (rstate)
                R_IDLE: begin
                    if (s_axi_arvalid && s_axi_arready) begin
                        araddr_q  <= s_axi_araddr;
                        rd_addr_f <= {s_axi_araddr[ADDR_BITS-1:3], 3'b000};
                        rd_tog    <= ~rd_tog;
                        rstate    <= R_WAIT;
                    end
                end

                R_WAIT: begin
                    if (rd_ack_pulse)
                        rstate <= R_RESP;
                end

                R_RESP: begin
                    if (s_axi_rready)
                        rstate <= R_IDLE;
                end

                default: rstate <= R_IDLE;
            endcase
        end
    end

    //==========================================================================
    // Device side
    //
    // Everything that drives the device is registered here, in this domain.
    // Nothing on the device port is combinationally a host-domain signal: the
    // address presented to a write is `wr_addr_q`, not `wr_addr_f`, because the
    // one-cycle strobe is longer than the host needs to turn around and start
    // the next one, and a write whose address moved under it would land on the
    // wrong register.
    //
    // The three-flop chains are synchroniser and edge detector in one: `sync[2]`
    // against `sync[1]` compares the synchronised toggle with a copy of itself
    // one cycle older, which is the only pair that is in one domain. Comparing
    // against the raw far-side toggle would compare across the boundary, which
    // is the thing this block exists to not do.
    //==========================================================================
    (* ASYNC_REG = "TRUE" *) reg [2:0] wr_sync;
    (* ASYNC_REG = "TRUE" *) reg [2:0] rd_sync;

    wire wr_pulse = wr_sync[2] ^ wr_sync[1];
    wire rd_pulse = rd_sync[2] ^ rd_sync[1];

    reg [ADDR_BITS-1:0] wr_addr_q;
    reg [ADDR_BITS-1:0] rd_addr_q;
    reg [63:0]          reg_wdata_q;
    reg                 reg_we_q;
    reg                 rd_pending;

    // A read and a write can arrive on the same device cycle: the host's read
    // and write channels are independent and nothing above serialises them.
    // The write takes the address for its one cycle -- it has to, since `reg_we`
    // is a pulse and the address must be up while it is -- so the read waits
    // for that cycle to end. The read is the one that waits because a read only
    // observes and a write changes state: delaying an observation costs
    // nothing, and delaying a write would lose it.
    wire rd_take = rd_pending && !reg_we_q;

    always @(posedge reg_clk or negedge reg_rst_n) begin
        if (!reg_rst_n) begin
            wr_sync     <= 3'b000;
            rd_sync     <= 3'b000;
            wr_addr_q   <= {ADDR_BITS{1'b0}};
            rd_addr_q   <= {ADDR_BITS{1'b0}};
            reg_wdata_q <= 64'd0;
            reg_we_q    <= 1'b0;
            rd_pending  <= 1'b0;
            wr_ack_tog  <= 1'b0;
            rd_data_s   <= 64'd0;
            rd_ack_tog  <= 1'b0;
        end else begin
            wr_sync <= {wr_sync[1:0], wr_tog};
            rd_sync <= {rd_sync[1:0], rd_tog};

            // The write strobe is one cycle, always, and this is the only place
            // `reg_we` is ever raised. Address and data are loaded on the same
            // edge, so the device sees a complete access for exactly one cycle
            // -- which is what R_DOORBELL and R_HOST_CTRL bit 1 need and what a
            // level held across cycles would break.
            reg_we_q <= wr_pulse;
            if (wr_pulse) begin
                wr_addr_q   <= wr_addr_f;
                reg_wdata_q <= wr_data_f;
                wr_ack_tog  <= ~wr_ack_tog;
            end

            // The read address goes down on the pulse; the data comes back
            // combinationally from it, so it is taken one cycle later, and the
            // acknowledgement is raised from the same edge the data is taken.
            if (rd_pulse)
                rd_pending <= 1'b1;
            else if (rd_take)
                rd_pending <= 1'b0;

            if (rd_pulse)
                rd_addr_q <= rd_addr_f;

            if (rd_take) begin
                rd_data_s  <= reg_rdata;
                rd_ack_tog <= ~rd_ack_tog;
            end
        end
    end

    assign reg_addr  = reg_we_q ? wr_addr_q : rd_addr_q;
    assign reg_wdata = reg_wdata_q;
    assign reg_we    = reg_we_q;

    // `reg_re` is an input the device does not read -- tnca_mmio.v says so in
    // as many words, because the read mux there is combinational and
    // `reg_rdata` is valid without a strobe. It is driven low rather than left
    // unconnected so that the port list matches the device's, and so that a
    // device that later does read it sees an honest zero rather than a float.
    assign reg_re    = 1'b0;

endmodule

`default_nettype wire
