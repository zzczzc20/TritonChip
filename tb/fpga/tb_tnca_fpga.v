//==============================================================================
// tb_tnca_fpga -- the register protocol, driven the way a host drives it
//
// doc/phase3_6_tasks_and_acceptance.md R36-IO-01, R36-IO-02, R36-IO-03 and
// R36-MEM-02.
//
// This bench does not reach inside the device. It writes registers and reads
// registers, and everything a host can do to the device it does here: the
// program goes in through the payload window, the command goes in through the
// mailbox, the answers come back out through the window. Every register
// address below is the one a driver would use, and this file is therefore the
// protocol reference that python/tnca/backends/u50.py is written against --
// when the two disagree, one of them is wrong and it is worth knowing which.
//
// WHAT IT IS FOR
//
// tb/device/tnca_shell_tb.v drives tnca_shell's ports directly. This drives
// them through rtl/fpga/tnca_mmio.v, which is the layer that turns the ports
// into a medium. So the thing under test is the medium: that a descriptor
// written one register at a time arrives whole, that a doorbell rings once,
// that a completion is not answered twice and not answered late, and that the
// payload window puts words where it says it will.
//
// The program is tb/core/progs/t30_memboundary.s -- the same one, assembled by
// the same assembler, and the expected values are the same hand-computed ones
// tnca_shell_tb.v checks. That is deliberate: a result that agreed with a
// number this bench made up would prove the bench agrees with itself.
//
// Plusargs:
//   +hex=<file>     the assembled program (required)
//   +quiet          only the verdict
//==============================================================================
`timescale 1ns/1ps

module tb_tnca_fpga;
    // PERF_NUM_COUNTERS: the index past the last counter is read from the
    // generated header, not written down -- R39-00 added a counter and a
    // hard-coded 20 became a real one.
`include "tnca_perf_count.vh"

    parameter WORDS          = 65536;
    parameter ADDR_W         = 16;
    parameter IMAGE_WORDS    = 4096;
    parameter WATCHDOG_NS    = 400000000;
    parameter LSU_TAG_W      = 5;
    parameter NUM_RCTX       = 2;
    parameter RCTX_W         = 1;

    // How much of memory the host initialises. The program lives at 0 and
    // writes its results at 0x1000 and its scratch at 0x1200; 1024 words is
    // 8 KiB and covers all of it with room to spare. See the note beside
    // `initialise` for why this is the host's job and not the device's.
    parameter INIT_WORDS     = 1024;

    // ---- the register map, from rtl/fpga/tnca_mmio.v -------------------------
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

    localparam [3:0] OP_UPLOAD   = 4'd1;
    localparam [3:0] OP_DOWNLOAD = 4'd2;
    localparam [3:0] OP_LAUNCH   = 4'd3;
    localparam [3:0] OP_QUIESCE  = 4'd4;
    localparam [3:0] OP_SHUTDOWN = 4'd5;
    localparam [3:0] OP_PING     = 4'd6;
    localparam [3:0] OP_PERF_READ = 4'd7;

    localparam [3:0] ST_OK      = 4'd0;
    localparam [3:0] ST_FAULT   = 4'd1;
    localparam [3:0] ST_TIMEOUT = 4'd2;
    localparam [3:0] ST_REFUSED = 4'd3;

    localparam [63:0] MAGIC       = 64'h0001_0000_544E_4341;
    localparam [63:0] RESULT_BASE = 64'h1000;
    localparam [63:0] DEADLINE    = 64'd2000000;

    // HOST_CTRL bits
    localparam [63:0] CTRL_WE    = 64'h1;
    localparam [63:0] CTRL_START = 64'h2;

    // ---- what the program must produce -------------------------------------
    // Hand-computed from tb/core/progs/t30_memboundary.s, and the same numbers
    // tnca_shell_tb.v checks. Not read back from a run: "the two agreed" is not
    // evidence, "both agreed with a number written down in advance" is.
    localparam [63:0] WANT_PARTIAL = 64'h33333333_2222EE11;
    localparam [63:0] WANT_VSUM    = 64'd240;
    localparam [63:0] WANT_GATHER  = 64'd120;
    localparam [63:0] WANT_AMO_OLD = 64'd7;
    localparam [63:0] WANT_AMO_NEW = 64'd12;
    localparam [63:0] SENTINEL     = 64'hDEAD_BEEF_FACE_F00D;

    // ---- the device ---------------------------------------------------------
    reg          clk = 1'b0;
    reg          rst_n_async = 1'b0;
    always #5 clk = ~clk;

    reg  [7:0]  reg_addr  = 8'd0;
    reg  [63:0] reg_wdata = 64'd0;
    reg         reg_we    = 1'b0;
    reg         reg_re    = 1'b0;
    wire [63:0] reg_rdata;

    tnca_fpga_top #(
        .WORDS(WORDS), .ADDR_W(ADDR_W), .ADDR_BITS(8),
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .LSU_TAG_W(LSU_TAG_W),
        .USE_BANKED_VRF(1)
    ) dut (
        .clk(clk), .rst_n_async(rst_n_async),
        .reg_addr(reg_addr), .reg_wdata(reg_wdata),
        .reg_we(reg_we), .reg_re(reg_re), .reg_rdata(reg_rdata)
    );

    // ---- bench state --------------------------------------------------------
    reg [63:0] image [0:IMAGE_WORDS-1];
    reg [1023:0] hexfile;
    reg [63:0] seq;
    reg [63:0] got, r0, r1;
    reg [63:0] perf_a;          // a PERF_READ result kept for the second read
    reg [3:0]  status;
    reg        quiet;
    integer    i, fails, step;
    reg [63:0] ctrl_wdata;

    initial begin
        #WATCHDOG_NS;
        $display("FAIL tb_tnca_fpga: watchdog at step %0d", step);
        $finish;
    end

    //==========================================================================
    // The register interface
    //
    // Driven on the falling edge and sampled on the falling edge, so nothing
    // in this bench races the device's rising edge. A read is combinational in
    // the device -- tnca_mmio's read mux has no clock in it -- so the value is
    // taken after the address settles.
    //==========================================================================
    task reg_wr;
        input [7:0]  a;
        input [63:0] d;
        begin
            @(negedge clk);
            reg_addr  = a;
            reg_wdata = d;
            reg_we    = 1'b1;
            @(negedge clk);
            reg_we    = 1'b0;
        end
    endtask

    task reg_rd;
        input  [7:0]  a;
        output [63:0] d;
        begin
            @(negedge clk);
            reg_addr = a;
            reg_re   = 1'b1;
            #1;
            d = reg_rdata;
            reg_re = 1'b0;
        end
    endtask

    //==========================================================================
    // The payload window
    //
    // One 64-bit word of device memory per access. The address register
    // advances by eight when an access completes, so a bulk transfer is one
    // address write followed by a run of data -- but this bench writes the
    // address every time anyway, because a driver that depended on the
    // auto-increment would be depending on something the mailbox protocol does
    // not promise, and the first transfer after a refusal is where that would
    // show up as a word landing in the wrong place.
    //==========================================================================
    // Every spin in this file is bounded, and the bound is not decoration.
    //
    // The first version of this bench polled until the device answered, and
    // when the device did not answer the simulation ran until the harness
    // killed it: no failure, no line of output, and a transcript ending at
    // whatever the last successful step printed -- which reads as the bench
    // having stopped rather than as a device that stopped responding. The
    // window sequencer had deadlocked on its first access, and the bench sat
    // in PS_WAIT's equivalent for five minutes without saying so.
    //
    // A bound turns that into one line naming what did not happen. 200000
    // polls is far past any correct access here -- the whole of memory is
    // 16384 words and every one of them completes in well under ten register
    // reads -- and far short of a wall-clock timeout.
    localparam integer POLL_MAX = 200000;

    task wait_not_busy;
        reg [63:0] st;
        integer    spins;
        begin
            st    = 64'h1;
            spins = 0;
            while ((st[0] === 1'b1) && (spins < POLL_MAX)) begin
                reg_rd(R_HOST_STAT, st);
                spins = spins + 1;
            end
            if (spins >= POLL_MAX) begin
                $display("FAIL the payload window never went idle: HOST_STATUS stayed busy for %0d polls",
                         POLL_MAX);
                fails = fails + 1;
            end else if (st[2] === 1'b1) begin
                $display("FAIL the payload window refused an address it was given");
                fails = fails + 1;
            end
        end
    endtask

    task win_put;
        input [63:0] devaddr;
        input [63:0] w;
        begin
            reg_wr(R_HOST_ADDR, devaddr);
            reg_wr(R_HOST_DATA, w);
            reg_wr(R_HOST_CTRL, CTRL_START | CTRL_WE);
            wait_not_busy();
        end
    endtask

    task win_get;
        input  [63:0] devaddr;
        output [63:0] w;
        begin
            reg_wr(R_HOST_ADDR, devaddr);
            reg_wr(R_HOST_CTRL, CTRL_START);
            wait_not_busy();
            reg_rd(R_HOST_DATA, w);
        end
    endtask

    //==========================================================================
    // The mailbox
    //
    // The descriptor is written one register at a time and then rung for, and
    // the response is waited for by sequence number rather than by "something
    // is there". That distinction is the whole point of the field: a driver
    // that took the first response it saw would read the previous request's
    // answer after any timeout, and would read it as this request's.
    //==========================================================================
    task mailbox;
        input  [3:0]  op;
        input  [63:0] a0, a1, a2, a3, a4, a5;
        output [3:0]  st_out;
        output [63:0] r0_out, r1_out;
        reg [63:0] seen;
        integer    spins;
        begin
            seq = seq + 64'd1;
            reg_wr(R_REQ_SEQ,  seq);
            reg_wr(R_REQ_OP,   {60'd0, op});
            reg_wr(R_REQ_A0,   a0);
            reg_wr(R_REQ_A1,   a1);
            reg_wr(R_REQ_A2,   a2);
            reg_wr(R_REQ_A3,   a3);
            reg_wr(R_REQ_A4,   a4);
            reg_wr(R_REQ_A5,   a5);
            reg_wr(R_REQ_DEAD, DEADLINE);
            reg_wr(R_DOORBELL, 64'd1);

            seen  = 64'd0;
            spins = 0;
            while ((seen !== seq) && (spins < POLL_MAX)) begin
                reg_rd(R_RSP_SEQ, seen);
                spins = spins + 1;
            end
            if (spins >= POLL_MAX) begin
                $display("FAIL no response to sequence %0d after %0d polls of RSP_SEQ (it read %0d)",
                         seq, POLL_MAX, seen);
                fails  = fails + 1;
                // Handed back as the status that means "nothing came", so a
                // caller's `check_status` reports the same fault at the point
                // the request was made rather than only here.
                st_out = ST_TIMEOUT;
                r0_out = 64'd0;
                r1_out = 64'd0;
                disable mailbox;
            end

            reg_rd(R_RSP_STATUS, got);
            st_out = got[3:0];
            reg_rd(R_RSP_R0, r0_out);
            reg_rd(R_RSP_R1, r1_out);
        end
    endtask

    task check;
        input [511:0] what;
        input [63:0]  saw;
        input [63:0]  want;
        begin
            if (saw !== want) begin
                $display("FAIL %0s: got %016h want %016h", what, saw, want);
                fails = fails + 1;
            end else if (!quiet) begin
                $display("    %0s ok (%016h)", what, saw);
            end
        end
    endtask

    task check_status;
        input [511:0] what;
        input [3:0]   saw;
        input [3:0]   want;
        begin
            if (saw !== want) begin
                $display("FAIL %0s: status %0d want %0d", what, saw, want);
                fails = fails + 1;
            end else if (!quiet) begin
                $display("    %0s ok (status %0d)", what, saw);
            end
        end
    endtask

    //==========================================================================
    // Initialising memory, which is the host's job
    //
    // rtl/fpga/tnca_fpga_mem.v has no initial block and neither does any block
    // RAM: what a location holds before it is written is not zero, it is
    // whatever the array came up holding. Both simulation memories zero
    // themselves, so a kernel that read an unwritten location would work in
    // simulation and not here.
    //
    // This program does not do that -- every location it reads it wrote first,
    // which is a property of t30_memboundary.s and not of the device. So what
    // this task buys is the region's definedness and not the program's
    // correctness: the image goes in, the rest of the region goes in as zeros,
    // and every address the program can reach is then a value somebody chose.
    //==========================================================================
    task initialise;
        begin
            for (i = 0; i < INIT_WORDS; i = i + 1)
                win_put(i * 8, image[i]);
        end
    endtask

    //==========================================================================
    initial begin
        fails = 0;
        step  = 0;
        seq   = 64'd0;
        quiet = $test$plusargs("quiet");

        if (!$value$plusargs("hex=%s", hexfile)) begin
            $display("FAIL tb_tnca_fpga: pass +hex=<file>");
            $finish;
        end
        for (i = 0; i < IMAGE_WORDS; i = i + 1) image[i] = 64'd0;
        $readmemh(hexfile, image);

        //---- 0: reset, and the device says who it is ------------------------
        step = 0;
        repeat (4) @(posedge clk);
        #1 rst_n_async = 1'b1;
        repeat (4) @(posedge clk);

        reg_rd(R_MAGIC, got);
        check("the identity register", got, MAGIC);

        //---- 1: ping, which is the cheapest proof the mailbox turns ---------
        step = 1;
        mailbox(OP_PING, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("ping", status, ST_OK);
        check("ping reports the sequence it answered", r0, seq);

        //---- 2: the program, and the results area ---------------------------
        // The sentinel goes down first so a result the program never wrote is
        // distinguishable from one it wrote correctly by accident.
        step = 2;
        initialise();
        mailbox(OP_UPLOAD, 64'd0, INIT_WORDS, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("upload", status, ST_OK);

        for (i = 0; i < 8; i = i + 1)
            win_put(RESULT_BASE + i * 8, SENTINEL);

        //---- 3: launch ------------------------------------------------------
        step = 3;
        // a3 = pid_x | pid_y<<16, a4 = pid_z | npid_x<<16,
        // a5 = npid_y | npid_z<<16 -- one program, so pid 0 and grid 1.
        mailbox(OP_LAUNCH, 64'd0, 64'd0, 64'd1,
                64'd0, 64'd1 << 16, (64'd1) | (64'd1 << 16),
                status, r0, r1);
        if (status !== ST_OK) begin
            $display("FAIL the program returned status %0d (r0=%0h r1=%0h)",
                     status, r0, r1);
            fails = fails + 1;
        end else if (!quiet) begin
            $display("    launch ok (%0d cycles, %0d retired)",
                     r0, r1);
        end

        //---- 4: the results -------------------------------------------------
        step = 4;
        win_get(RESULT_BASE + 0,  got);
        check("partial stores", got, WANT_PARTIAL);
        win_get(RESULT_BASE + 8,  got);
        check("vector round trip", got, WANT_VSUM);
        win_get(RESULT_BASE + 16, got);
        check("gather", got, WANT_GATHER);
        win_get(RESULT_BASE + 24, got);
        check("atomic, old value", got, WANT_AMO_OLD);
        win_get(RESULT_BASE + 32, got);
        check("atomic, new value", got, WANT_AMO_NEW);

        // Ownership: five results and nothing past them. A store that ran off
        // the end would have eaten a sentinel.
        step = 5;
        for (i = 5; i < 8; i = i + 1) begin
            win_get(RESULT_BASE + i * 8, got);
            check("a slot the program must not have touched", got, SENTINEL);
        end

        //---- 6: the device refuses, and keeps answering ---------------------
        //
        // R36-IO-01 wants illegal input to have no device-side effect, and
        // R36-IO-03 wants a refusal to be distinguishable from a device that
        // stopped. So each of these is checked for the status it should get
        // AND the session is checked to still be alive afterwards -- a refusal
        // that ended the session would pass the first half of this and fail
        // the second.
        step = 6;
        mailbox(OP_UPLOAD, WORDS * 8, 64'd1, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("upload past the end of memory", status, ST_REFUSED);

        mailbox(OP_UPLOAD, 64'd5, 64'd1, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("upload at an unaligned address", status, ST_REFUSED);

        mailbox(OP_UPLOAD, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("upload of zero words", status, ST_REFUSED);

        mailbox(4'd9, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("an opcode the device does not have", status, ST_REFUSED);

        // PERF_READ (op 7, transport 1.1): a counter in two 32-bit halves,
        // as the simulation bench answers it. After the launch above the
        // cycle and issue counters are past zero; nothing runs between two
        // reads, so a second read is not smaller; an index past the last
        // counter is refused, never answered with a zero.
        mailbox(OP_PERF_READ, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("PERF_READ perf_cycles", status, ST_OK);
        check("PERF_READ answers in 32-bit halves", {32'd0, r0[63:32] | r1[63:32]}, 64'd0);
        perf_a = {r1[31:0], r0[31:0]};
        check("perf_cycles is past zero after a launch", {63'd0, perf_a != 64'd0}, 64'd1);
        mailbox(OP_PERF_READ, 64'd1, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("PERF_READ perf_issued", status, ST_OK);
        check("perf_issued is past zero after a launch",
              {63'd0, {r1[31:0], r0[31:0]} != 64'd0}, 64'd1);
        mailbox(OP_PERF_READ, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check("perf_cycles does not go backwards between reads",
              {63'd0, {r1[31:0], r0[31:0]} >= perf_a}, 64'd1);
        mailbox(OP_PERF_READ, PERF_NUM_COUNTERS, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("PERF_READ past the last counter", status, ST_REFUSED);

        // The results must be exactly where they were: a refused upload that
        // wrote something anyway is the failure this checks for.
        win_get(RESULT_BASE + 0, got);
        check("a refused upload left memory alone", got, WANT_PARTIAL);

        // And the session still works.
        mailbox(OP_PING, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0, 64'd0,
                status, r0, r1);
        check_status("ping after three refusals", status, ST_OK);
        check("ping after three refusals reports its own sequence", r0, seq);

        //---- 7: drain -------------------------------------------------------
        // A second launch of the same program gives the same answers. Anything
        // left in flight from the first lands in the middle of this.
        step = 7;
        for (i = 0; i < 8; i = i + 1)
            win_put(RESULT_BASE + i * 8, SENTINEL);
        mailbox(OP_LAUNCH, 64'd0, 64'd0, 64'd1,
                64'd0, 64'd1 << 16, (64'd1) | (64'd1 << 16),
                status, r0, r1);
        check_status("launch, second time", status, ST_OK);
        win_get(RESULT_BASE + 0,  got);
        check("partial stores, second launch", got, WANT_PARTIAL);
        win_get(RESULT_BASE + 16, got);
        check("gather, second launch", got, WANT_GATHER);

        //---- 8: a duplicate ring is not a second completion -----------------
        //
        // The sequence has already been served, so re-ringing it must do
        // nothing: not serve it again, and not answer it. The response
        // register must still hold the previous answer rather than a new one.
        step = 8;
        reg_wr(R_DOORBELL, 64'd1);
        repeat (20) @(posedge clk);
        reg_rd(R_RSP_SEQ, got);
        check("a duplicate ring produced no new completion", got, seq);

        //---- the verdict ----------------------------------------------------
        if (fails == 0)
            $display("PASS tb_tnca_fpga");
        else
            $display("FAIL tb_tnca_fpga: %0d failure(s)", fails);
        $finish;
    end

endmodule
