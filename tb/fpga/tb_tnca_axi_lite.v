//==============================================================================
// tb_tnca_axi_lite -- the register port, on a bus, across a clock
//
// doc/phase3_6_tasks_and_acceptance.md R36-IO-01, R36-IO-02 and R36-RTL-02.
//
// tb/fpga/tb_tnca_fpga.v drives the flat register port the way a host drives
// it. tb/fpga/tnca_fpga_regs.v puts a file protocol on the same port so that
// python/tnca/backends/u50_sim.py can drive it. Neither of them crosses a
// clock and neither of them sees a bus. rtl/fpga/tnca_axi_lite.v does both,
// and what goes wrong in it is not visible from either side: the device sees
// a strobe that may be one cycle too long, and the host sees a transaction
// that may never complete. So this bench sits ON the adapter, with a stub
// register file on the device port where strobes can be counted, and an
// AXI4-Lite master on the other side where transactions can be timed.
//
// WHAT IS ACTUALLY UNDER TEST
//
// One property above all, because everything downstream rests on it: **a
// 64-bit register write raises `reg_we` exactly once.** Not zero times, not
// twice. R_HOST_CTRL bit 1 and R_DOORBELL are level-sensitive in tnca_mmio.v;
// a second strobe is a second launch, and a lost one is a command that never
// runs while the host waits for a completion that is not coming. The 64-bit
// register is two bus writes, so the arithmetic that has to come out right is
// 2 bus accesses -> 1 strobe, and it is checked by counting strobes rather
// than by reading a register back -- a duplicated launch overwrites the
// evidence that would have shown it.
//
// The rest:
//
//   * a write to the low half alone reaches the device not at all
//   * a read returns the half that was asked for, and both halves of a value
//     that went in through the bus
//   * a read and a write in flight together both complete -- the bus channels
//     are independent, and a design that let them collide would answer a read
//     with the data of a write
//   * `reg_we` is one `reg_clk` cycle wide, never two
//   * AW and W offered in different cycles still complete
//   * a read between the two halves of a register does not lose the write
//
// CLOCK RATIO
//
// The board runs 150 MHz into the bus and 37.5 MHz into the device, a ratio of
// 4. The suite is run at more than one ratio, because a crossing that happens
// to work at one is not a crossing that works. The two parameters are clock
// half-periods in ns and the Makefile passes them.
//
// Plusargs:
//   +quiet          only the verdict
//==============================================================================
`timescale 1ns/1ps

module tb_tnca_axi_lite;

    // 2 ns and 26 ns: 500 MHz and 38.5 MHz, a ratio of 13.
    parameter FAST_HALF  = 1;
    parameter SLOW_HALF  = 13;

    parameter ADDR_BITS   = 8;
    parameter AXI_ADDR_W  = 12;
    parameter WATCHDOG_NS = 4000000;

    reg s_axi_aclk = 1'b0;
    reg reg_clk    = 1'b0;

    always #FAST_HALF s_axi_aclk = ~s_axi_aclk;
    always #SLOW_HALF reg_clk    = ~reg_clk;

    // Two counters and not one. The monitor process below runs on `reg_clk`
    // and the driving process runs in the initial block; a single variable
    // written by both is a race, and an error counter that loses a race is an
    // error counter that reports PASS.
    integer errors     = 0;
    integer mon_errors = 0;
    reg     quiet      = 1'b0;

    //==========================================================================
    // Reset
    //
    // Released together, and the slow one through its own two flops, which is
    // the contract tnca_axi_lite.v states the wrapper keeps: `reg_rst_n` is
    // asserted asynchronously and released synchronously to `reg_clk`. A
    // domain that reset on its own would present the far side with a toggle
    // that looks exactly like an acknowledgement.
    //==========================================================================
    reg       s_axi_aresetn = 1'b0;
    reg       reg_rst_n     = 1'b0;
    reg       rst_release   = 1'b0;
    reg [1:0] rst_sync_s    = 2'b00;

    always @(posedge reg_clk) begin
        if (!rst_release) rst_sync_s <= 2'b00;
        else              rst_sync_s <= {rst_sync_s[0], 1'b1};
    end

    //==========================================================================
    // The device port, watched
    //
    // A stub register file, and a strobe counter beside it. The counter is the
    // instrument: it turns "the register holds the right value" into "the
    // register was written the right number of times", and the first of those
    // is still true after a duplicated write while the second is not.
    //==========================================================================
    wire [ADDR_BITS-1:0] reg_addr;
    wire [63:0]          reg_wdata;
    wire                 reg_we;
    wire                 reg_re;
    wire [63:0]          reg_rdata;

    reg [63:0] dev_regs [0:31];

    assign reg_rdata = dev_regs[reg_addr[ADDR_BITS-1:3]];

    integer             we_count     = 0;
    reg [ADDR_BITS-1:0] last_we_addr = {ADDR_BITS{1'b0}};
    reg [63:0]          last_we_data = 64'd0;

    reg we_d = 1'b0;

    always @(posedge reg_clk) begin
        we_d <= reg_we;

        if (reg_we) begin
            we_count     <= we_count + 1;
            last_we_addr <= reg_addr;
            last_we_data <= reg_wdata;
            dev_regs[reg_addr[ADDR_BITS-1:3]] <= reg_wdata;
        end

        // Two cycles high in a row is the failure this whole bench is about.
        // A level held across cycles re-issues against tnca_mmio.v every time
        // the host window returns to idle. Caught here rather than at the
        // mailbox, because by the time it reaches the mailbox it is a second
        // command being executed and the first one's evidence is gone.
        if (reg_we && we_d) begin
            $display("FAIL reg_we held high for more than one reg_clk cycle");
            mon_errors <= mon_errors + 1;
        end
    end

    // `reg_re` is an input the device does not read, and a byte enable is not
    // something a 64-bit port with a byte-mask register behind it can honour.
    // Both are checked rather than assumed.
    always @(posedge reg_clk) begin
        if (reg_re !== 1'b0) begin
            $display("FAIL reg_re is not held low");
            mon_errors <= mon_errors + 1;
        end
    end

    //==========================================================================
    // AXI4-Lite master
    //
    // The sampling convention every wait below depends on: a handshake is
    // offered with a nonblocking assignment on one rising edge and looked for
    // just after the NEXT one. The ready signals here are combinational from a
    // state register, so reading one immediately after an edge gives the state
    // the slave was in during the cycle that just ended -- which is the cycle
    // the handshake completed in. Reading it after the register has updated
    // would be reading it a cycle late, and every transaction below would then
    // be one cycle too long and every timeout one cycle too short.
    //
    // `bready` and `rready` are held low until the response is wanted, so that
    // a slave which dropped its response early would be seen to have dropped
    // it rather than being covered for.
    //==========================================================================
    reg  [AXI_ADDR_W-1:0] awaddr  = {AXI_ADDR_W{1'b0}};
    reg                   awvalid = 1'b0;
    wire                  awready;
    reg  [31:0]           wdata    = 32'd0;
    reg  [3:0]            wstrb    = 4'hF;
    reg                   wvalid   = 1'b0;
    wire                  wready;
    wire [1:0]            bresp;
    wire                  bvalid;
    reg                   bready   = 1'b0;

    reg  [AXI_ADDR_W-1:0] araddr  = {AXI_ADDR_W{1'b0}};
    reg                   arvalid = 1'b0;
    wire                  arready;
    wire [31:0]           rdata;
    wire [1:0]            rresp;
    wire                  rvalid;
    reg                   rready   = 1'b0;

    tnca_axi_lite #(
        .ADDR_BITS(ADDR_BITS), .S_AXI_ADDR_W(AXI_ADDR_W)
    ) u_dut (
        .s_axi_aclk(s_axi_aclk), .s_axi_aresetn(s_axi_aresetn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid),
        .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid),
        .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid),
        .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .reg_clk(reg_clk), .reg_rst_n(reg_rst_n),
        .reg_addr(reg_addr), .reg_wdata(reg_wdata), .reg_we(reg_we),
        .reg_re(reg_re), .reg_rdata(reg_rdata)
    );

    //==========================================================================
    // Sticky responses
    //
    // Only test 5 needs these. It issues a read and a write together, and the
    // order their responses come back in is not fixed -- it depends on which
    // device-side pulse won the race, which is the thing being varied. Waiting
    // for B and then for R is an assumption about that order, and when it is
    // wrong the read's response is missed, the read state machine is left
    // holding it, and the next read in the suite waits on a slave that is
    // still busy. That is a hang, and a hang is a slow way to find out that a
    // bench assumed an order the bus never promised.
    //
    // So a response is latched where it is taken and waited for afterwards.
    // The flags are cleared only when nothing is in flight, which is why the
    // clearing sits inside the loop rather than at the top of it.
    //==========================================================================
    reg        b_seen      = 1'b0;
    reg        r_seen      = 1'b0;
    reg [31:0] r_seen_data = 32'd0;

    always @(posedge s_axi_aclk) begin
        if (bvalid && bready) b_seen <= 1'b1;
        if (rvalid && rready) begin
            r_seen      <= 1'b1;
            r_seen_data <= rdata;
        end
    end

    //==========================================================================
    // Bus transactions
    //==========================================================================
    task bus_write;
        input [AXI_ADDR_W-1:0] off;
        input [31:0]           val;
        begin
            @(posedge s_axi_aclk);
            awaddr <= off;  awvalid <= 1'b1;
            wdata  <= val;  wvalid  <= 1'b1;  wstrb <= 4'hF;

            // Both ready signals are looked for on the same edge, because they
            // are independent. A slave that took AW and then waited for W to
            // be offered again would pass a serial master and hang against a
            // real one. Valid is held until seen, which is what valid means.
            while (!(awready && wready)) @(posedge s_axi_aclk);
            @(posedge s_axi_aclk);
            awvalid <= 1'b0;  wvalid <= 1'b0;

            @(posedge s_axi_aclk);
            bready <= 1'b1;
            while (!bvalid) @(posedge s_axi_aclk);
            @(posedge s_axi_aclk);
            bready <= 1'b0;
        end
    endtask

    // The same write with AW and W offered cycles apart, which AXI permits and
    // a slave that assumed otherwise would deadlock on.
    task bus_write_split;
        input [AXI_ADDR_W-1:0] off;
        input [31:0]           val;
        begin
            @(posedge s_axi_aclk);
            awaddr <= off;  awvalid <= 1'b1;
            while (!awready) @(posedge s_axi_aclk);
            @(posedge s_axi_aclk);
            awvalid <= 1'b0;

            repeat (5) @(posedge s_axi_aclk);

            @(posedge s_axi_aclk);
            wdata <= val;  wvalid <= 1'b1;  wstrb <= 4'hF;
            while (!wready) @(posedge s_axi_aclk);
            @(posedge s_axi_aclk);
            wvalid <= 1'b0;

            @(posedge s_axi_aclk);
            bready <= 1'b1;
            while (!bvalid) @(posedge s_axi_aclk);
            @(posedge s_axi_aclk);
            bready <= 1'b0;
        end
    endtask

    task bus_read;
        input  [AXI_ADDR_W-1:0] off;
        output [31:0]           val;
        begin
            @(posedge s_axi_aclk);
            araddr <= off;  arvalid <= 1'b1;
            while (!arready) @(posedge s_axi_aclk);
            @(posedge s_axi_aclk);
            arvalid <= 1'b0;

            @(posedge s_axi_aclk);
            rready <= 1'b1;
            while (!rvalid) @(posedge s_axi_aclk);
            val = rdata;
            @(posedge s_axi_aclk);
            rready <= 1'b0;
        end
    endtask

    // One 64-bit register write: the low half, then the high half.
    task reg_write64;
        input [AXI_ADDR_W-1:0] off;
        input [63:0]           val;
        begin
            bus_write(off,     val[31:0]);
            bus_write(off + 4, val[63:32]);
        end
    endtask

    reg [31:0] rd_lo, rd_hi;

    task reg_read64;
        input  [AXI_ADDR_W-1:0] off;
        output [63:0]           val;
        begin
            bus_read(off,     rd_lo);
            bus_read(off + 4, rd_hi);
            val = {rd_hi, rd_lo};
        end
    endtask

    //==========================================================================
    // Checks
    //==========================================================================
    task check64;
        input [511:0] what;
        input [63:0]  got;
        input [63:0]  want;
        begin
            if (got !== want) begin
                $display("FAIL %0s: got %016x want %016x", what, got, want);
                errors = errors + 1;
            end else if (!quiet)
                $display("ok   %0s = %016x", what, got);
        end
    endtask

    task check_int;
        input [511:0] what;
        input integer got;
        input integer want;
        begin
            if (got !== want) begin
                $display("FAIL %0s: got %0d want %0d", what, got, want);
                errors = errors + 1;
            end else if (!quiet)
                $display("ok   %0s = %0d", what, got);
        end
    endtask

    //==========================================================================
    // The suite
    //==========================================================================
    integer    base;
    integer    phase;
    reg [63:0] got64;

    // How many device cycles carried a read pulse and a write pulse at once.
    // Counted, not assumed: the phase sweep below exists to produce these, and
    // a sweep that produced none would test nothing while still reporting
    // PASS. The verdict at the end fails if this is still zero.
    integer    collisions = 0;

    always @(posedge reg_clk)
        if (u_dut.rd_pulse && u_dut.wr_pulse) collisions <= collisions + 1;

    initial begin
        if ($test$plusargs("quiet")) quiet = 1'b1;

        #(8 * SLOW_HALF);
        rst_release = 1'b1;
        @(posedge reg_clk);
        while (!rst_sync_s[1]) @(posedge reg_clk);
        reg_rst_n     = 1'b1;
        s_axi_aresetn = 1'b1;
        #(4 * SLOW_HALF);

        if (!quiet)
            $display("tb_tnca_axi_lite: fast half %0d ns, slow half %0d ns",
                     FAST_HALF, SLOW_HALF);

        //----------------------------------------------------------------------
        // 1. A write to the low half alone must not reach the device.
        //----------------------------------------------------------------------
        base = we_count;
        bus_write(12'h040, 32'hDEAD_BEEF);          // low half of REQ_DEADLINE
        #(4 * SLOW_HALF);
        check_int("low half alone: no strobe", we_count - base, 0);

        //----------------------------------------------------------------------
        // 2. The high half completes it: one strobe, right address, assembled
        //    value. This is the property the whole file rests on.
        //----------------------------------------------------------------------
        bus_write(12'h044, 32'h0000_0007);          // high half
        #(4 * SLOW_HALF);
        check_int("two bus writes: exactly one strobe", we_count - base, 1);
        check64("the value that reached the device",
                last_we_data, 64'h0000_0007_DEAD_BEEF);
        check_int("the address that reached the device", last_we_addr, 8'h40);

        //----------------------------------------------------------------------
        // 3. A read returns the half that was asked for.
        //----------------------------------------------------------------------
        dev_regs[8'h20 >> 3] = 64'h1122_3344_5566_7788;
        reg_read64(12'h020, got64);
        check64("read back through the bus", got64, 64'h1122_3344_5566_7788);

        bus_read(12'h020, rd_lo);
        check64("low half read alone", {32'd0, rd_lo}, 64'h0000_0000_5566_7788);
        bus_read(12'h024, rd_hi);
        check64("high half read alone", {32'd0, rd_hi}, 64'h0000_0000_1122_3344);

        //----------------------------------------------------------------------
        // 4. Back to back: two completed writes are two strobes. Not one, which
        //    would mean the second was swallowed; not three.
        //----------------------------------------------------------------------
        base = we_count;
        reg_write64(12'h048, 64'h1);                // R_DOORBELL, twice
        reg_write64(12'h048, 64'h1);
        #(4 * SLOW_HALF);
        check_int("two writes: exactly two strobes", we_count - base, 2);
        check64("the second doorbell arrived", last_we_data, 64'h1);

        //----------------------------------------------------------------------
        // 5. A read and a write in flight together. The bus channels are
        //    independent and this is where a design that let them collide
        //    answers the read with the data of a write.
        //
        //    The gap between the two device-side pulses is what has to be
        //    swept, so the delay is put on the READ. The write's own
        //    address-to-toggle latency is fixed and cannot be removed, so
        //    delaying the write instead would start the sweep above zero and
        //    never reach the near-miss cases -- and a version of this test did
        //    exactly that, swept 0..27, produced no collision at all, and
        //    passed a design with the arbitration deleted. With the delay on
        //    the read the gap is `phase` fast cycles and starts at zero.
        //    `collisions` is what says the sweep arrived.
        //----------------------------------------------------------------------
        for (phase = 0; phase < 48; phase = phase + 1) begin
            dev_regs[8'h30 >> 3] = {32'hAAAA_BBBB, {16'd0, phase[15:0]}};
            base = we_count;

            // The low half first and alone, so that the write whose pulse has
            // to collide is a single bus write.
            bus_write(12'h060, 32'h0000_0001);

            b_seen = 1'b0;  r_seen = 1'b0;   // nothing is in flight here

            // The write is offered and taken in the next cycle, and its valid
            // is dropped on that edge. `phase` is then spent before the read
            // is offered, so the gap between the two device-side pulses is
            // `phase` fast cycles and sweeps from zero.
            //
            // Every valid here is dropped on the edge of its own handshake and
            // none is held a cycle longer. A valid left standing is a second
            // transaction: the slave takes it as soon as it returns to idle,
            // which at a fast device clock is sooner than the bench expects.
            // That is exactly what a version of this test did -- it held
            // `awvalid` until both responses had come back, the write finished
            // first, and the slave ran the same register a second time. At the
            // board's ratio the write outlasts the hold and nothing shows; at
            // 1:1 every phase produced two or three, and the check that says
            // "one strobe" is the only reason any of it was visible.
            @(posedge s_axi_aclk);
            awaddr <= 12'h064; awvalid <= 1'b1;
            wdata  <= 32'h0000_0001; wvalid <= 1'b1; wstrb <= 4'hF;
            bready <= 1'b1;  rready <= 1'b1;

            @(posedge s_axi_aclk);                  // AW and W are taken here
            awvalid <= 1'b0; wvalid <= 1'b0;

            if (phase > 0) repeat (phase) @(posedge s_axi_aclk);
            araddr <= 12'h030; arvalid <= 1'b1;

            @(posedge s_axi_aclk);                  // and the read here
            arvalid <= 1'b0;

            // Both ready signals stay up, so neither response can be held
            // behind the other and the collision cannot be hidden by a stall.
            while (!(b_seen && r_seen)) @(posedge s_axi_aclk);
            got64 = {32'd0, r_seen_data};
            @(posedge s_axi_aclk);
            bready <= 1'b0;  rready <= 1'b0;

            #(2 * SLOW_HALF);
            check64("concurrent read/write: the read is the read",
                    got64, {32'd0, dev_regs[8'h30 >> 3][31:0]});
            check_int("concurrent read/write: one strobe", we_count - base, 1);
        end
        check_int("the collision was actually exercised", collisions > 0, 1);

        //----------------------------------------------------------------------
        // 6. AW and W offered in different cycles, both halves of a register.
        //----------------------------------------------------------------------
        base = we_count;
        bus_write_split(12'h070, 32'h0000_0002);    // low half
        bus_write_split(12'h074, 32'h0000_0000);    // high half
        #(4 * SLOW_HALF);
        check_int("AW and W apart: one strobe", we_count - base, 1);
        check64("AW and W apart: the value", last_we_data, 64'h0000_0002);

        //----------------------------------------------------------------------
        // 7. The low half, then a read of something else, then the high half.
        //    The read must not disturb what is waiting to be completed.
        //----------------------------------------------------------------------
        base = we_count;
        bus_write(12'h088, 32'h0000_0001);          // low half of R_HOST_CTRL
        bus_read(12'h030, rd_lo);                   // a read in between
        bus_write(12'h08C, 32'h0000_0000);          // high half
        #(4 * SLOW_HALF);
        check_int("a read between the halves does not lose the write",
                  we_count - base, 1);
        check64("and the value is whole", last_we_data, 64'h0000_0001);

        //----------------------------------------------------------------------
        // Verdict
        //----------------------------------------------------------------------
        if (errors + mon_errors == 0)
            $display("PASS tb_tnca_axi_lite: %0d ns/%0d ns", FAST_HALF, SLOW_HALF);
        else
            $display("FAIL tb_tnca_axi_lite: %0d error(s)",
                     errors + mon_errors);
        $finish;
    end

    initial begin
        #WATCHDOG_NS;
        $display("FAIL tb_tnca_axi_lite: watchdog at %0d ns", WATCHDOG_NS);
        $finish;
    end

endmodule
