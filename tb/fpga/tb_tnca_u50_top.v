//==============================================================================
// tb_tnca_u50_top -- the top the board uses, with the real device behind it.
//
// tb/fpga/tb_tnca_axi_lite.v tests the adapter against a stub register file:
// thirty-two words of behavioural memory with a strobe counter beside them. It
// sweeps eleven clock ratios over that stub and breaks the adapter six ways.
// Everything it establishes is about the adapter, and it says so.
//
// tb/fpga/tnca_fpga_regs.v tests the device through a flat register port, one
// clock, no adapter. Everything it establishes is about the device.
//
// Between those two files is rtl/fpga/tnca_u50_top.v -- the join, the second
// clock, and the reset ordering -- and until this file existed, nothing drove
// it. The only thing that instantiated it was rtl/board/tnca_u50_xo.v, which is
// the board. So the layer whose header argues at length about the order the two
// resets are released in was the layer with no bench under it, and the argument
// was the only evidence there was.
//
// This file is that bench. The DUT is tnca_u50_top itself: the AXI master below
// talks to s_axi_tnca_*, the adapter inside crosses to clk_dev, and what answers
// on the far side is tnca_fpga_top with its real memory, its real mailbox, and
// its real core.
//
// WHAT IT DOES NOT DO
//
// It does not sweep clock ratios. That is R36-FPGA-05's job and it is done
// there, against a stub, twelve times. Here the clocks are locked at the ratio
// the board runs -- see CLOCKS below -- and the subject is the join.
//
// It does not establish that any of this fits, meets timing, or runs on a
// board. Nothing in this repository has been on a board.
//
// It does not run a program. A ping is a mailbox round trip and it reads
// tnca_mmio.v's registers; it never reaches the core's fetch port, and a design
// that had the core wired to nothing would pass every check below. The upload
// and launch path is gated by tb/fpga/tb_tnca_fpga.v, which drives the same
// device through a flat port -- so the fetch and data ports are covered, and
// what is not is the pair of those in series. Saying which is the point of
// writing it down.
//
//------------------------------------------------------------------------------
// CLOCKS
//
// FAST_HALF and SLOW_HALF are half-periods in nanoseconds, as in
// tb_tnca_axi_lite.v: ap_clk is the fast one and clk_dev the slow one. The
// default pair is 1 and 4, which is the board's -- ap_clk 150 MHz divided by
// BUFGCE_DIV's 4 in rtl/board/tnca_u50_xo.v is 37.5 MHz, and 4 is the ratio.
// The absolute rate here differs from the board's by the same factor on both
// sides, so the ratio is real and the frequencies are not.
//
// The two clocks are locked rather than free-running, because that is what a
// divided clock is: BUFGCE_DIV's output edges coincide with every fourth
// ap_clk edge rather than drifting against it. A bench that let them drift
// would spend most of its time in a phase relationship the board never has.
//==============================================================================

`timescale 1ns / 1ps
`default_nettype none

module tb_tnca_u50_top;

    parameter FAST_HALF   = 1;             // ap_clk half-period, ns
    parameter SLOW_HALF   = 4;             // clk_dev half-period, ns

    parameter ADDR_BITS   = 8;
    parameter AXI_ADDR_W  = 12;

    // 200 us. tb_tnca_fpga.v uses 400 ms, which at these periods is more
    // simulated time than this suite spends by six orders of magnitude -- and a
    // watchdog that takes longer to fire than the gate's own subprocess timeout
    // is not a watchdog, it is a second timeout that loses the race. This one
    // fires inside the budget and says so in the log.
    parameter WATCHDOG_NS = 200000;

    // Every wait below is bounded, and these are the bounds. A poll here is a
    // 64-bit register read: two AXI reads, two clock crossings, and the device
    // port before it -- call it twenty ap_clk cycles, forty nanoseconds. A
    // correct access completes in a handful of clk_dev cycles, so five polls;
    // two thousand is four hundred times that, and the worst case is 80 us of
    // simulated time rather than the eight milliseconds 200000 would be.
    parameter POLL_MAX    = 2000;

    //==========================================================================
    // Register map, names, and the two constants this bench checks against.
    //
    // Copied from rtl/fpga/tnca_mmio.v's localparams rather than from
    // python/tnca/transport.py, because the numbers being checked are the
    // device's and the file that holds them is the authority. A drift between
    // the two would be a real finding and is not something to paper over by
    // reading one of them from Python.
    //==========================================================================
    localparam [7:0] R_REQ_SEQ    = 8'h00;
    localparam [7:0] R_REQ_OP     = 8'h08;
    localparam [7:0] R_REQ_A0     = 8'h10;
    localparam [7:0] R_REQ_A1     = 8'h18;
    localparam [7:0] R_REQ_A2     = 8'h20;
    localparam [7:0] R_REQ_A3     = 8'h28;
    localparam [7:0] R_REQ_A4     = 8'h30;
    localparam [7:0] R_REQ_A5     = 8'h38;
    localparam [7:0] R_REQ_DEAD   = 8'h40;
    localparam [7:0] R_DOORBELL   = 8'h48;
    localparam [7:0] R_RSP_SEQ    = 8'h50;
    localparam [7:0] R_RSP_STATUS = 8'h58;
    localparam [7:0] R_RSP_R0     = 8'h60;
    localparam [7:0] R_RSP_R1     = 8'h68;
    localparam [7:0] R_HOST_ADDR  = 8'h70;
    localparam [7:0] R_HOST_DATA  = 8'h78;
    localparam [7:0] R_HOST_BE    = 8'h80;
    localparam [7:0] R_HOST_CTRL  = 8'h88;
    localparam [7:0] R_HOST_STAT  = 8'h90;
    localparam [7:0] R_MAGIC      = 8'h98;
    localparam [7:0] R_SERVED     = 8'hA0;

    localparam [3:0] OP_PING = 4'd6;

    localparam [3:0] ST_OK      = 4'd0;
    localparam [3:0] ST_TIMEOUT = 4'd2;

    // tnca_mmio.v:241. Version 1.0, "TNCA".
    localparam [63:0] MAGIC = {16'h0001, 16'h0000, 32'h544E_4341};

    localparam [63:0] CTRL_WE    = 64'h1;
    localparam [63:0] CTRL_START = 64'h2;

    localparam [63:0] BE_ALL   = 64'hFF;
    localparam [63:0] DEADLINE = 64'd2000000;

    // A pattern with no half equal to another and no half all-zero, so that a
    // join which dropped a half, swapped the halves, or returned the low half
    // twice would be visible as a wrong value rather than as a plausible one.
    localparam [63:0] WANT_WINDOW = 64'h5A5A_0F0F_1234_ABCD;

    // A second word with no half equal to any half of the first, for the
    // two-address test in the suite: a value that shared a half with
    // WANT_WINDOW could survive a join that swapped or duplicated one.
    localparam [63:0] WANT_OTHER  = 64'hA5A5_F0F0_DEAD_1234;

    // A byte address, not a word index: tnca_mmio.v's guard divides by eight.
    // 0x40 is word 8 and 0x48 is word 9, so the two are different words of the
    // same memory and a window that ignored its address would land one on the
    // other.
    localparam [63:0] WIN_ADDR    = 64'h0000_0000_0000_0040;

    //==========================================================================
    // Clocks
    //==========================================================================
    reg ap_clk  = 1'b0;
    reg clk_dev = 1'b0;

    always #FAST_HALF ap_clk  = ~ap_clk;
    always #SLOW_HALF clk_dev = ~clk_dev;

    // Two counters and not one. The driving process runs in this file's initial
    // block and the monitor processes below run on clk_dev; one variable written
    // by both is a race, and a counter that loses a race is a counter that
    // reports PASS. Same split as tb_tnca_axi_lite.v, for the same reason.
    integer errors     = 0;
    integer mon_errors = 0;
    reg     quiet      = 1'b0;

    //==========================================================================
    // Reset
    //
    // One source. tnca_u50_top derives both domains' resets from it, and that
    // is the thing under test -- the bench does not hand it two, because
    // handing it two is what the adapter bench does when it stands in for this
    // wrapper.
    //==========================================================================
    reg rst_n_async = 1'b0;

    //==========================================================================
    // The reset ordering, watched at the two wires the DUT derives
    //
    // tnca_u50_top's header makes a claim about a sequence and not about a
    // value: the device side leads, and the bus side cannot take an edge until
    // it has. Both wires are inside the DUT and these are the only two places
    // the claim is visible as itself rather than as something downstream of it
    // which could be true for another reason.
    //
    // Sampled on clk_dev, which is the slower of the two and therefore cannot
    // miss a level that is held for many of its periods.
    //==========================================================================
    integer dev_edges       = 0;
    integer dev_rel_edge    = -1;
    integer axi_rel_edge    = -1;
    integer core_rel_edge   = -1;
    reg     axi_before_dev  = 1'b0;
    reg     axi_before_core = 1'b0;
    reg     we_before_axi   = 1'b0;
    integer writes_at_rel   = -1;

    //==========================================================================
    // The device's register port, watched
    //
    // This is the join. tnca_u50_top connects these wires straight from
    // tnca_axi_lite to tnca_fpga_top with no glue between them, and the
    // argument for that is that there is nowhere for a bug to hide. So the
    // instrument goes on the wires themselves: how many strobes crossed, what
    // the last one carried, and whether any strobe was a level rather than a
    // pulse.
    //
    // WHAT `dev_last_data` DOES NOT SEE, measured and not assumed. These are
    // the wires, and a port connection is a transform below them: when the
    // device instance's own connection was mutated to swap the halves of
    // reg_wdata, this monitor still read the unswapped value and the check on
    // it passed, while the device received halves in the wrong order. The
    // mutation was caught further down -- by the value the payload window read
    // back -- and not here. So the value checks below are worth more than this
    // wire is, and a check that read only this wire would be watching the
    // adapter rather than the join.
    //==========================================================================
    integer    dev_writes    = 0;
    reg [63:0] dev_last_data = 64'd0;
    reg [7:0]  dev_last_addr = 8'd0;
    reg        dev_we_d      = 1'b0;
    integer    dev_re_pulses = 0;

    always @(posedge clk_dev) begin
        dev_edges <= dev_edges + 1;

        //------------------------------------------------------------------------
        // The ordering claim, at the instant it can be false. If the bus side
        // came out of reset while the device side was still held, this is where
        // that is: axi_rst_n high with dev_rst_n low. Since dev_rst_n rises and
        // stays, "ever" and "at the rising edge" are the same question.
        //------------------------------------------------------------------------
        if (u_dut.axi_rst_n === 1'b1 && u_dut.dev_rst_n !== 1'b1)
            axi_before_dev <= 1'b1;

        // The third wire, and the only one of the three that is not a reset
        // this wrapper generates: tnca_fpga_top's own synchroniser output,
        // which is the earliest moment the device is actually running rather
        // than merely un-held. The header's claim is measured against this one
        // -- "the bus side cannot take an edge until the device side has taken
        // four" is a claim about the device, and dev_rst_n is just the signal
        // the claim is implemented with.
        if (u_dut.u_device.rst_n === 1'b1 && core_rel_edge < 0)
            core_rel_edge <= dev_edges;

        if (u_dut.axi_rst_n === 1'b1 && u_dut.u_device.rst_n !== 1'b1)
            axi_before_core <= 1'b1;

        if (u_dut.dev_rst_n === 1'b1 && dev_rel_edge < 0)
            dev_rel_edge <= dev_edges;

        if (u_dut.axi_rst_n === 1'b1 && axi_rel_edge < 0) begin
            axi_rel_edge  <= dev_edges;
            writes_at_rel <= dev_writes;
            if (u_dut.dev_rst_n !== 1'b1) axi_before_dev  <= 1'b1;
            if (u_dut.u_device.rst_n !== 1'b1) axi_before_core <= 1'b1;
        end

        // A strobe offered before the bus side was released is a device being
        // driven by a domain that had not been let go yet -- which is the
        // failure the ordering exists to prevent, seen from the device's side
        // rather than from the reset wires.
        if (u_dut.reg_we === 1'b1 && u_dut.axi_rst_n !== 1'b1)
            we_before_axi <= 1'b1;

        if (u_dut.reg_we === 1'b1) begin
            dev_writes    <= dev_writes + 1;
            dev_last_data <= u_dut.reg_wdata;
            dev_last_addr <= u_dut.reg_addr;
        end

        // Two cycles high in a row. The adapter pulses this; a level held
        // across cycles would re-issue the write against tnca_mmio.v every
        // time the window returned to idle. Checked here as well as at the
        // adapter's own port, because what is between the two is the join.
        if (u_dut.reg_we === 1'b1 && dev_we_d === 1'b1) begin
            $display("FAIL reg_we held high for more than one clk_dev cycle at the device port");
            mon_errors <= mon_errors + 1;
        end
        dev_we_d <= u_dut.reg_we;

        if (u_dut.reg_re === 1'b1) dev_re_pulses <= dev_re_pulses + 1;
    end

    //==========================================================================
    // AXI4-Lite master
    //
    // The sampling convention is tb_tnca_axi_lite.v's, and it is load-bearing
    // for every wait below: a handshake is offered with a nonblocking assignment
    // on one rising edge and looked for just after the NEXT one. The ready
    // signals here are combinational from a state register, so reading one
    // immediately after an edge gives the state the slave was in during the
    // cycle that just ended -- which is the cycle the handshake completed in.
    // Reading it after the register has updated would read it a cycle late, and
    // every transaction below would then be one cycle too long.
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

    //==========================================================================
    // The DUT
    //==========================================================================
    tnca_u50_top #(
        .S_AXI_ADDR_W(AXI_ADDR_W),
        .ADDR_BITS(ADDR_BITS)
    ) u_dut (
        .ap_clk(ap_clk),
        .clk_dev(clk_dev),
        .rst_n_async(rst_n_async),

        .s_axi_tnca_awaddr(awaddr),   .s_axi_tnca_awvalid(awvalid),
        .s_axi_tnca_awready(awready),
        .s_axi_tnca_wdata(wdata),     .s_axi_tnca_wstrb(wstrb),
        .s_axi_tnca_wvalid(wvalid),   .s_axi_tnca_wready(wready),
        .s_axi_tnca_bresp(bresp),     .s_axi_tnca_bvalid(bvalid),
        .s_axi_tnca_bready(bready),

        .s_axi_tnca_araddr(araddr),   .s_axi_tnca_arvalid(arvalid),
        .s_axi_tnca_arready(arready),
        .s_axi_tnca_rdata(rdata),     .s_axi_tnca_rresp(rresp),
        .s_axi_tnca_rvalid(rvalid),   .s_axi_tnca_rready(rready)
    );

    //==========================================================================
    // Bus transactions
    //==========================================================================
    task bus_write;
        input [AXI_ADDR_W-1:0] off;
        input [31:0]           val;
        begin
            @(posedge ap_clk);
            awaddr <= off;  awvalid <= 1'b1;
            wdata  <= val;  wvalid  <= 1'b1;  wstrb <= 4'hF;

            // Both readys looked for on the same edge, because they are
            // independent. A slave that took AW and then waited for W to be
            // offered again would pass a serial master and hang against a real
            // one. Valid is held until seen, which is what valid means.
            while (!(awready && wready)) @(posedge ap_clk);
            @(posedge ap_clk);
            awvalid <= 1'b0;  wvalid <= 1'b0;

            @(posedge ap_clk);
            bready <= 1'b1;
            while (!bvalid) @(posedge ap_clk);
            @(posedge ap_clk);
            bready <= 1'b0;
        end
    endtask

    // The same write with AW and W offered cycles apart, which AXI permits and
    // a slave that assumed otherwise would deadlock on.
    task bus_write_split;
        input [AXI_ADDR_W-1:0] off;
        input [31:0]           val;
        begin
            @(posedge ap_clk);
            awaddr <= off;  awvalid <= 1'b1;
            while (!awready) @(posedge ap_clk);
            @(posedge ap_clk);
            awvalid <= 1'b0;

            repeat (5) @(posedge ap_clk);

            @(posedge ap_clk);
            wdata <= val;  wvalid <= 1'b1;  wstrb <= 4'hF;
            while (!wready) @(posedge ap_clk);
            @(posedge ap_clk);
            wvalid <= 1'b0;

            @(posedge ap_clk);
            bready <= 1'b1;
            while (!bvalid) @(posedge ap_clk);
            @(posedge ap_clk);
            bready <= 1'b0;
        end
    endtask

    task bus_read;
        input  [AXI_ADDR_W-1:0] off;
        output [31:0]           val;
        begin
            @(posedge ap_clk);
            araddr <= off;  arvalid <= 1'b1;
            while (!arready) @(posedge ap_clk);
            @(posedge ap_clk);
            arvalid <= 1'b0;

            @(posedge ap_clk);
            rready <= 1'b1;
            while (!rvalid) @(posedge ap_clk);
            val = rdata;
            @(posedge ap_clk);
            rready <= 1'b0;
        end
    endtask

    // One 64-bit register write: the low half, then the high half. tnca_axi_lite
    // shadow-latches the low half and the high write completes it, so this
    // ordering is the adapter's contract and not a convention of the bench.
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

    task check_status;
        input [511:0] what;
        input [3:0]   got;
        input [3:0]   want;
        begin
            if (got !== want) begin
                $display("FAIL %0s: status %0d want %0d", what, got, want);
                errors = errors + 1;
            end else if (!quiet)
                $display("ok   %0s status %0d", what, got);
        end
    endtask

    //==========================================================================
    // The mailbox
    //
    // The descriptor is written one register at a time and then rung for, and
    // the response is waited for by sequence number rather than by "something
    // is there". A driver that took the first response it saw would read the
    // previous request's answer after any timeout and read it as this one's.
    //==========================================================================
    reg [63:0] seq = 64'd0;

    task mailbox;
        input  [3:0]  op;
        input  [63:0] a0, a1, a2, a3, a4, a5;
        output [3:0]  st_out;
        output [63:0] r0_out, r1_out;
        reg [63:0] seen;
        integer    spins;
        begin
            seq = seq + 64'd1;
            reg_write64(R_REQ_SEQ,  seq);
            reg_write64(R_REQ_OP,   {60'd0, op});
            reg_write64(R_REQ_A0,   a0);
            reg_write64(R_REQ_A1,   a1);
            reg_write64(R_REQ_A2,   a2);
            reg_write64(R_REQ_A3,   a3);
            reg_write64(R_REQ_A4,   a4);
            reg_write64(R_REQ_A5,   a5);
            reg_write64(R_REQ_DEAD, DEADLINE);
            reg_write64(R_DOORBELL, 64'd1);

            seen  = 64'd0;
            spins = 0;
            while ((seen !== seq) && (spins < POLL_MAX)) begin
                reg_read64(R_RSP_SEQ, seen);
                spins = spins + 1;
            end
            if (spins >= POLL_MAX) begin
                $display("FAIL no response to sequence %0d after %0d polls of RSP_SEQ (it read %0d)",
                         seq, POLL_MAX, seen);
                errors = errors + 1;
                // Handed back as the status that means "nothing came", so that
                // the caller's check_status reports the same fault at the point
                // the request was made rather than only here.
                st_out = ST_TIMEOUT;
                r0_out = 64'd0;
                r1_out = 64'd0;
                disable mailbox;
            end

            reg_read64(R_RSP_STATUS, seen);
            st_out = seen[3:0];
            reg_read64(R_RSP_R0, r0_out);
            reg_read64(R_RSP_R1, r1_out);
        end
    endtask

    //==========================================================================
    // The payload window
    //==========================================================================
    task wait_not_busy;
        reg [63:0] st;
        integer    spins;
        begin
            st    = 64'h1;
            spins = 0;
            while ((st[0] === 1'b1) && (spins < POLL_MAX)) begin
                reg_read64(R_HOST_STAT, st);
                spins = spins + 1;
            end
            if (spins >= POLL_MAX) begin
                $display("FAIL the payload window never went idle: HOST_STATUS stayed busy for %0d polls",
                         POLL_MAX);
                errors = errors + 1;
            end else if (st[2] === 1'b1) begin
                $display("FAIL the payload window refused an address it was given");
                errors = errors + 1;
            end
        end
    endtask

    task win_put;
        input [63:0] devaddr;
        input [63:0] w;
        begin
            reg_write64(R_HOST_ADDR, devaddr);
            reg_write64(R_HOST_DATA, w);
            reg_write64(R_HOST_CTRL, CTRL_START | CTRL_WE);
            wait_not_busy();
        end
    endtask

    task win_get;
        input  [63:0] devaddr;
        output [63:0] w;
        begin
            reg_write64(R_HOST_ADDR, devaddr);
            reg_write64(R_HOST_CTRL, CTRL_START);
            wait_not_busy();
            reg_read64(R_HOST_DATA, w);
        end
    endtask

    //==========================================================================
    // The suite
    //==========================================================================
    integer    base;
    reg [63:0] got64;
    reg [3:0]  st;
    reg [63:0] r0, r1;
    integer    head_start;

    initial begin
        if ($test$plusargs("quiet")) quiet = 1'b1;

        // Held long enough for both domains to see it. tnca_u50_top's header
        // says a pulse narrower than a clk_dev period can be missed entirely and
        // that no synchroniser can recover a reset that was never observed, so
        // the hold is eight clk_dev periods rather than a token delay.
        #(8 * SLOW_HALF);
        rst_n_async = 1'b1;

        // Release, then wait for it to have happened. Both wires are watched
        // rather than timed out on, and the wait is on the slower of the two
        // releases.
        @(posedge clk_dev);
        while (core_rel_edge < 0) @(posedge clk_dev);
        while (dev_rel_edge  < 0) @(posedge clk_dev);
        while (axi_rel_edge  < 0) @(posedge clk_dev);
        #(8 * SLOW_HALF);

        if (!quiet) begin
            $display("tb_tnca_u50_top: fast half %0d ns, slow half %0d ns (ratio %0d)",
                     FAST_HALF, SLOW_HALF, SLOW_HALF / FAST_HALF);
            $display("    clk_dev edge when the device core reset released: %0d",
                     core_rel_edge);
            $display("    clk_dev edge when dev_rst_n released:              %0d",
                     dev_rel_edge);
            $display("    clk_dev edge when axi_rst_n released:              %0d",
                     axi_rel_edge);
        end

        //----------------------------------------------------------------------
        // 1. The ordering, which is what this file exists for.
        //----------------------------------------------------------------------
        check_int("the bus side never came out of reset first",
                  axi_before_dev, 0);
        check_int("the bus side never came out before the device core was running",
                  axi_before_core, 0);
        check_int("no strobe was offered to the device before the bus side was released",
                  we_before_axi, 0);
        check_int("the device port was quiet until the bus side was released",
                  writes_at_rel, 0);

        // Reported rather than asserted, and over all three wires. How much wall
        // clock a head start is in clk_dev edges depends on the ratio, and a
        // bench that pinned it to a number would be testing its own arithmetic
        // rather than the design. What is asserted is the direction, above.
        head_start = axi_rel_edge - core_rel_edge;
        if (!quiet)
            $display("    head start over the device core: %0d clk_dev edge(s), %0d ns",
                     head_start, head_start * 2 * SLOW_HALF);

        //----------------------------------------------------------------------
        // 2. The device is there, and it is the one that was expected.
        //
        //    MAGIC is a localparam in tnca_mmio.v, so this reads through the
        //    adapter, across both clock domains, and back without touching
        //    memory or the core. It is the cheapest thing that could possibly
        //    answer and the first thing that would not.
        //----------------------------------------------------------------------
        reg_read64(R_MAGIC, got64);
        check64("MAGIC through the whole top", got64, MAGIC);

        //----------------------------------------------------------------------
        // 3. A write split across the AW and W channels still completes exactly
        //    one register write.
        //
        //    Done on R_HOST_BE because the window needs a byte enable and
        //    because the device resets that register to 8'hFF -- so the value
        //    written is checked against a register that had a different value
        //    before, rather than confirming a register that was already right.
        //----------------------------------------------------------------------
        base = dev_writes;
        bus_write_split(R_HOST_BE, 32'h0000_00FF);   // low half, AW and W apart
        bus_write(R_HOST_BE + 4, 32'h0000_0000);     // high half, together
        #(4 * SLOW_HALF);
        check_int("AW and W apart: exactly one device strobe",
                  dev_writes - base, 1);
        check64("AW and W apart: the value that reached the device",
                dev_last_data, 64'hFF);
        check64("AW and W apart: the address that reached the device",
                {56'd0, dev_last_addr}, 64'h80);

        //----------------------------------------------------------------------
        // 4. Two 64-bit writes are two device strobes. Not one, which would
        //    mean the second was swallowed; not three.
        //----------------------------------------------------------------------
        base = dev_writes;
        reg_write64(R_HOST_ADDR, WIN_ADDR);
        reg_write64(R_HOST_DATA, WANT_WINDOW);
        #(4 * SLOW_HALF);
        check_int("two 64-bit writes: exactly two device strobes",
                  dev_writes - base, 2);
        check64("the second one carried the data register",
                dev_last_data, WANT_WINDOW);
        check64("and it was addressed to R_HOST_DATA",
                {56'd0, dev_last_addr}, 64'h78);

        //----------------------------------------------------------------------
        // 5. A payload round trip through the window.
        //
        //    The address is not zero, so a join that dropped the address and
        //    answered from the base of memory would be caught. The word is
        //    written and read back through the same path the backend uses.
        //----------------------------------------------------------------------
        win_put(WIN_ADDR, WANT_WINDOW);
        win_get(WIN_ADDR, got64);
        check64("a word written through the window reads back", got64,
                WANT_WINDOW);

        // Two addresses, so that the address is tested and not only the path.
        // A single write-then-read at one address passes on a window that
        // ignores the address entirely: both halves of the round trip go to the
        // same wrong word and the value that comes back is the value that went
        // in. The second write is what makes the first read mean something --
        // if the two collapse, the first word is gone.
        //
        // This has no entry in the mutation table, and that is not an oversight
        // to be papered over with a mutation that would be caught by something
        // else anyway. It guards a design error rather than verifying a check:
        // a window that dropped its address would fail MAGIC long before it got
        // here, and one that dropped it only for the payload window is not a
        // one-line mutation. Kept because it is the difference between testing
        // the storage and testing the path to it.
        win_put(WIN_ADDR + 64'd8, WANT_OTHER);
        win_get(WIN_ADDR, got64);
        check64("writing the next word left the first one alone", got64,
                WANT_WINDOW);
        win_get(WIN_ADDR + 64'd8, got64);
        check64("and the next word holds its own value", got64, WANT_OTHER);

        //----------------------------------------------------------------------
        // 6. A mailbox round trip.
        //
        //    PING is the one operation tnca_mmio.v answers itself: it needs no
        //    program, no memory and no core, and it returns the sequence number
        //    it was asked about. That makes it the whole request/response
        //    protocol -- descriptor write, doorbell, response by sequence --
        //    with nothing else in the way.
        //----------------------------------------------------------------------
        mailbox(OP_PING, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, st, r0, r1);
        check_status("PING", st, ST_OK);
        check64("PING answers the sequence it was given", r0, seq);

        //----------------------------------------------------------------------
        // 7. And a second one, one sequence later.
        //
        //    A mailbox that latched the first answer and replayed it would pass
        //    every check above. This is the one that would not.
        //----------------------------------------------------------------------
        mailbox(OP_PING, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, st, r0, r1);
        check_status("PING, second request", st, ST_OK);
        check64("PING answers the second sequence, not the first", r0, seq);

        //----------------------------------------------------------------------
        // 8. The device saw exactly the writes the host issued.
        //
        //    Counted over the whole run and not per group: every reg_write64
        //    above issues precisely one completed 64-bit write, and one bus
        //    write to a low half alone issues none. A join that dropped one,
        //    duplicated one, or let the shadow latch leak a strobe would be
        //    visible here and nowhere else.
        //----------------------------------------------------------------------
        check_int("the device never saw a read pulse", dev_re_pulses, 0);

        //----------------------------------------------------------------------
        // Verdict
        //----------------------------------------------------------------------
        if (errors + mon_errors == 0)
            $display("PASS tb_tnca_u50_top: %0d ns/%0d ns", FAST_HALF, SLOW_HALF);
        else
            $display("FAIL tb_tnca_u50_top: %0d error(s)", errors + mon_errors);
        $finish;
    end

    initial begin
        #WATCHDOG_NS;
        $display("FAIL tb_tnca_u50_top: watchdog at %0d ns", WATCHDOG_NS);
        $finish;
    end

endmodule
