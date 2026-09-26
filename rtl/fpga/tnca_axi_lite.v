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
// Because they belong to different things. The host bus is on the platform's
// kernel clock, which a design does not get to choose -- it is asked for at link
// time (`--kernel_frequency` in flow/fpga/link.sh, 150 MHz) and handed over by
// the shell. The register port is on the device clock, because the register file
// it addresses is there and a port on any other clock would be a crossing per
// access. So the wrapper above divides the bus clock and passes the result down
// as `clk_dev`; this file is what sits between the two. Every clock crossing in
// the design is in this file.
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
    //
    // What each bus is worth, and why no bus is wider than it has to be: the
    // design is a multicycle one in both directions, and nothing in the linked
    // design says so -- the platform declares no clock relationship between the
    // bus clock and the device clock, so every bus is analysed as if a single
    // fast edge had to carry it. That analysis is not wrong about the hardware,
    // it is wrong about the intent: the toggles are what make it a multicycle
    // design, and a bus that is still for a whole device cycle is still for
    // eight bus cycles. The buses are sized so that the analysis is survivable
    // anyway, because a bus that needs the multicycle assumption to pass is a
    // bus that cannot be checked.
    //==========================================================================
    // host -> device
    reg  [ADDR_BITS-1:0] wr_addr_f;     // byte offset, low three bits zero
    reg  [63:0]          wr_data_f;     // assembled 64-bit value
    reg                  wr_tog;
    reg  [ADDR_BITS-1:0] rd_addr_f;     // byte offset, low three bits zero
    reg                  rd_tog;

    // device -> host
    //
    // `rd_data_s` carries the half the host asked for, already selected. The
    // 64-to-32 mux is on this side because of how little room the crossing had:
    // `reg_clk` is the bus clock divided by four, so its tree reaches 172884
    // loads and spreads across the whole part, while the bus clock's reaches
    // 449. The device's edge therefore lands about 2.6ns into the bus period.
    //
    // At 300 MHz that period was 3.333ns and left the far side under 0.7ns; the
    // linked design measured its own acknowledgement toggles at 0.588ns to cross
    // one bit, which is the whole of it. At 150 MHz the period is 6.667ns and
    // there is room. The mux stays on this side anyway: room bought by lowering
    // a clock is not room to move work back into, and this side is where the
    // measurement was made.
    reg  [31:0]          rd_data_s;
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

    // One-hot by the time it is placed, so each state is one net. The linked
    // design charges 3.731ns to the W_IDLE net -- twelve loads, with the driver
    // and one of the loads in the same slice -- while a thirty-two load net
    // spanning further in the same block costs 0.886ns. That is a router
    // serving a net's far loads along one tree, not a logic depth problem: the
    // whole path is 0.237ns of logic inside 4.854ns. An RTL edit cannot buy
    // that back, and the only force it can apply is to ask for the state to be
    // near its consumers more than once. This is that ask, and it is a hint
    // rather than an instruction -- if the tool declines it, the answer is a
    // floorplan and not a rewrite.
    (* max_fanout = 4 *) reg [1:0] wstate;
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
    // and the half wanted travels with it as `rd_sel_f`; the mux itself is on
    // the device's side, for the reason given at `rd_data_s`.
    //==========================================================================
    localparam R_IDLE = 2'd0,
               R_WAIT = 2'd1,
               R_RESP = 2'd2;

    reg [1:0]              rstate;
    reg                    rd_sel_f;    // bit 2 of the address, crossing with it
    reg [31:0]             rd_data_q;   // what `rd_data_s` last said, in this domain

    (* ASYNC_REG = "TRUE" *) reg [2:0] rd_ack_sync;

    wire rd_ack_pulse = rd_ack_sync[2] ^ rd_ack_sync[1];

    assign s_axi_arready = (rstate == R_IDLE);
    assign s_axi_rresp   = 2'b00;
    assign s_axi_rvalid  = (rstate == R_RESP);

    // A register, not a mux over a far-side bus. `rvalid` is held from the same
    // edge this is loaded, so data and valid are up together exactly as they
    // were when the mux was here.
    assign s_axi_rdata   = rd_data_q;

    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            rstate      <= R_IDLE;
            rd_sel_f    <= 1'b0;
            rd_data_q   <= 32'd0;
            rd_addr_f   <= {ADDR_BITS{1'b0}};
            rd_tog      <= 1'b0;
            rd_ack_sync <= 3'b000;
        end else begin
            rd_ack_sync <= {rd_ack_sync[1:0], rd_ack_tog};

            // Taken on the acknowledgement: `rd_data_s` went still on the same
            // device edge that moved `rd_ack_tog`, so by the time that edge has
            // been synchronised and detected here the value has been quiet for
            // longer than the two cycles `rstate` spends in R_WAIT. Same edge
            // as `rstate` reaching R_RESP, so nothing about the response timing
            // changes -- only which side of the crossing the mux is on.
            if (rd_ack_pulse)
                rd_data_q <= rd_data_s;

            case (rstate)
                R_IDLE: begin
                    if (s_axi_arvalid && s_axi_arready) begin
                        rd_sel_f  <= s_axi_araddr[2];
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
    reg                 rd_sel_q;       // loaded with rd_addr_q, held with it
    reg [63:0]          reg_wdata_q;
    reg                 reg_we_q;
    reg                 rd_pending;

    // The half the host asked for, chosen here rather than on the far side of
    // the crossing. `rd_sel_q` is loaded on the same edge as `rd_addr_q` and
    // held until the next read arrives, so both are valid on the cycle that
    // `rd_take` fires and the read costs exactly what it cost before.
    wire [31:0] reg_rdata_sel = rd_sel_q ? reg_rdata[63:32] : reg_rdata[31:0];

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
            rd_sel_q    <= 1'b0;
            reg_wdata_q <= 64'd0;
            reg_we_q    <= 1'b0;
            rd_pending  <= 1'b0;
            wr_ack_tog  <= 1'b0;
            rd_data_s   <= 32'd0;
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

            if (rd_pulse) begin
                rd_addr_q <= rd_addr_f;
                rd_sel_q  <= rd_sel_f;
            end

            if (rd_take) begin
                rd_data_s  <= reg_rdata_sel;
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
