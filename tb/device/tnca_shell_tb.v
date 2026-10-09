//==============================================================================
// tnca_shell_tb -- the public memory and fetch boundary, under two memories
//
// doc/phase3_6_tasks_and_acceptance.md R36-MEM-02 and R36-RST-01.
//
// The bench drives the shell through its own ports only: the program goes in
// through `host_*` a word at a time, the command goes in through `cmd_*`, the
// answers come back out through `host_*`. Nothing reaches inside.
//
// What it is for: the core has only ever run against one memory, so every
// habit of that memory is indistinguishable from a requirement of the
// contract. MEM_IMPL selects a second one -- a byte array with a shift-
// register pipeline and an adversarial swap stage, written from the port list
// -- and the same program must give the same answers.
//
// Every expected value is computed here, by hand, from what the program says
// it does. "The two memories agreed" would not be evidence; "both agreed with
// a number written down in advance" is.
//
// Plusargs:
//   +hex=<file>     the assembled program (required)
//   +reset          run the reset-in-flight case as well
//   +quiet          only the verdict
//==============================================================================
`timescale 1ns/1ps

module tnca_shell_tb;

    parameter MEM_IMPL       = 0;      // 0 = tnca_mem_model, 1 = tnca_mem_ref
    parameter MEM_LATENCY    = 1;
    parameter RANDOM_STALL   = 0;
    parameter MEM_REORDER    = 0;
    parameter USE_BANKED_VRF = 0;
    parameter VRF_BYPASS     = 0;
    parameter FE_NOBUBBLE    = 0;
    parameter VRF_PREFETCH   = 0;
    parameter RCTX_SWITCH    = 0;
    parameter LSU_INFLIGHT   = 1;
    parameter FPU_OVERLAP    = 0;
    parameter RSLICE         = 0;
    parameter WORDS          = 4096;
    parameter NUM_RCTX       = 2;
    parameter RCTX_W         = 1;
    parameter LSU_TAG_W      = 5;
    parameter PROGRAM_WORDS  = 4096;
    parameter WATCHDOG_NS    = 40000000;

    localparam [3:0] CMD_NOP     = 4'd0;
    localparam [3:0] CMD_LAUNCH  = 4'd1;
    localparam [3:0] CMD_QUIESCE = 4'd2;

    localparam [63:0] RESULT_BASE = 64'h1000;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    wire [63:0] imem_addr, imem_data;
    wire        mreq_valid, mreq_ready, mreq_we, mreq_amo, mrsp_valid;
    wire [63:0] mreq_addr, mreq_wdata, mreq_cmp, mrsp_rdata;
    wire [7:0]  mreq_be;
    wire [3:0]  mreq_aop;
    wire [1:0]  mreq_hint;
    wire [LSU_TAG_W-1:0] mreq_tag, mrsp_tag;

    reg         cmd_valid = 1'b0;
    reg  [3:0]  cmd_op = CMD_NOP;
    reg  [63:0] cmd_pc = 64'd0, cmd_arg_base = 64'd0;
    reg  [NUM_RCTX-1:0] cmd_role_mask = 1;
    reg                 cmd_slice = 0;
    reg  [31:0] px = 0, py = 0, pz = 0, nx = 1, ny = 1, nz = 1;
    reg  [63:0] cmd_deadline = 64'd2000000;
    wire        cmd_ready, done_valid, busy, core_idle;
    reg         done_ready = 1'b0;
    wire [3:0]  done_status;
    wire [63:0] done_cycles, done_retired, done_fault_pc;
    wire [7:0]  done_fault_code;
    wire [RCTX_W-1:0] done_fault_rctx;
    wire [23:0] done_fault_detail;
    wire [63:0] perf_cycles, perf_issued, perf_stall_hazard,
                perf_stall_lsu, perf_stall_wb, perf_squash;

    reg         hq_valid = 1'b0, hq_we = 1'b0;
    reg  [63:0] hq_addr = 0, hq_wdata = 0;
    reg  [7:0]  hq_be = 8'hFF;
    wire        hq_ready, hs_valid;
    wire [63:0] hs_rdata;

    tnca_shell #(
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .LSU_TAG_W(LSU_TAG_W),
        .USE_BANKED_VRF(USE_BANKED_VRF),
        .VRF_BYPASS(VRF_BYPASS), .FE_NOBUBBLE(FE_NOBUBBLE),
        .VRF_PREFETCH(VRF_PREFETCH), .RCTX_SWITCH(RCTX_SWITCH),
        .LSU_INFLIGHT(LSU_INFLIGHT), .FPU_OVERLAP(FPU_OVERLAP), .RSLICE(RSLICE)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_op(cmd_op),
        .cmd_pc(cmd_pc), .cmd_arg_base(cmd_arg_base),
        .cmd_role_mask(cmd_role_mask), .cmd_slice(cmd_slice),
        .cmd_pid_x(px), .cmd_pid_y(py), .cmd_pid_z(pz),
        .cmd_npid_x(nx), .cmd_npid_y(ny), .cmd_npid_z(nz),
        .cmd_deadline(cmd_deadline),
        .done_valid(done_valid), .done_ready(done_ready),
        .done_status(done_status), .done_cycles(done_cycles),
        .done_retired(done_retired),
        .done_fault_code(done_fault_code), .done_fault_pc(done_fault_pc),
        .done_fault_rctx(done_fault_rctx),
        .done_fault_detail(done_fault_detail),
        .host_req_valid(hq_valid), .host_req_ready(hq_ready),
        .host_req_we(hq_we), .host_req_addr(hq_addr),
        .host_req_be(hq_be), .host_req_wdata(hq_wdata),
        .host_rsp_valid(hs_valid), .host_rsp_rdata(hs_rdata),
        .busy(busy), .core_idle(core_idle),
        .perf_cycles(perf_cycles), .perf_issued(perf_issued),
        .perf_stall_hazard(perf_stall_hazard),
        .perf_stall_lsu(perf_stall_lsu),
        .perf_stall_wb(perf_stall_wb), .perf_squash(perf_squash),
        .perf_sel(5'd0), .perf_rdata(),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .mem_req_valid(mreq_valid), .mem_req_ready(mreq_ready),
        .mem_req_addr(mreq_addr), .mem_req_we(mreq_we),
        .mem_req_be(mreq_be), .mem_req_wdata(mreq_wdata),
        .mem_req_tag(mreq_tag), .mem_req_hint(mreq_hint),
        .mem_req_amo(mreq_amo), .mem_req_aop(mreq_aop),
        .mem_req_cmp(mreq_cmp),
        .mem_rsp_valid(mrsp_valid), .mem_rsp_tag(mrsp_tag),
        .mem_rsp_rdata(mrsp_rdata)
    );

    generate
    if (MEM_IMPL == 0) begin : g_mem
        tnca_mem_model #(
            .WORDS(WORDS), .MEM_DW(64), .MEM_BYTES(8), .TAG_W(LSU_TAG_W),
            .LATENCY(MEM_LATENCY), .RANDOM_STALL(RANDOM_STALL),
            .REORDER(MEM_REORDER)
        ) u (
            .clk(clk), .rst_n(rst_n),
            .imem_addr(imem_addr), .imem_data(imem_data),
            .req_valid(mreq_valid), .req_ready(mreq_ready),
            .req_addr(mreq_addr), .req_we(mreq_we),
            .req_be(mreq_be), .req_wdata(mreq_wdata), .req_tag(mreq_tag),
            .req_amo(mreq_amo), .req_aop(mreq_aop), .req_cmp(mreq_cmp),
            .rsp_valid(mrsp_valid), .rsp_tag(mrsp_tag),
            .rsp_rdata(mrsp_rdata)
        );
    end else begin : g_mem
        tnca_mem_ref #(
            .WORDS(WORDS), .MEM_DW(64), .MEM_BYTES(8), .TAG_W(LSU_TAG_W),
            .LATENCY(MEM_LATENCY), .RANDOM_STALL(RANDOM_STALL),
            .REORDER(MEM_REORDER)
        ) u (
            .clk(clk), .rst_n(rst_n),
            .imem_addr(imem_addr), .imem_data(imem_data),
            .req_valid(mreq_valid), .req_ready(mreq_ready),
            .req_addr(mreq_addr), .req_we(mreq_we),
            .req_be(mreq_be), .req_wdata(mreq_wdata), .req_tag(mreq_tag),
            .req_amo(mreq_amo), .req_aop(mreq_aop), .req_cmp(mreq_cmp),
            .rsp_valid(mrsp_valid), .rsp_tag(mrsp_tag),
            .rsp_rdata(mrsp_rdata)
        );
    end
    endgenerate

    //--------------------------------------------------------------------------
    // What the program must produce. Written from tb/core/progs/t30_memboundary.s
    // by hand, not read back from a run.
    //--------------------------------------------------------------------------
    // byte 0 = 0x11, byte 1 untouched 0xEE, bytes 2..3 = 0x2222,
    // bytes 4..7 = 0x33333333, little endian.
    localparam [63:0] WANT_PARTIAL = 64'h33333333_2222EE11;
    localparam [63:0] WANT_VSUM    = 64'd240;   // sum of 2*i, i = 0..15
    localparam [63:0] WANT_GATHER  = 64'd120;   // sum of 15-i, i = 0..15
    localparam [63:0] WANT_AMO_OLD = 64'd7;
    localparam [63:0] WANT_AMO_NEW = 64'd12;

    localparam [63:0] SENTINEL = 64'hDEAD_BEEF_FACE_F00D;

    reg [63:0] image [0:PROGRAM_WORDS-1];
    reg [1023:0] hexfile;
    // The host-access monitor. `io_watch` is raised around the launch
    // in step 9 so this counts only that window.
    reg io_watch, io_saw_busy, io_open_while_busy;
    integer io_busy_cycles;
    always @(posedge clk) begin
        if (!rst_n) begin
            io_saw_busy <= 1'b0;
            io_open_while_busy <= 1'b0;
            io_busy_cycles <= 0;
        end else if (io_watch && busy) begin
            io_saw_busy <= 1'b1;
            io_busy_cycles <= io_busy_cycles + 1;
            // valid was offered and the shell said ready: it took it.
            if (hq_valid && hq_ready) io_open_while_busy <= 1'b1;
        end
    end

    reg [63:0] got;
    reg settled;
    integer i, fails, nwords, step;
    reg do_reset, quiet;


    initial begin
        #WATCHDOG_NS;
        $display("FAIL tnca_shell_tb: watchdog at step %0d", step);
        $finish;
    end

    task host_xfer;
        input        we;
        input [63:0] addr;
        input [63:0] wdata;
        output [63:0] rdata;
        begin
            settled = 1'b0;
            while (hq_ready !== 1'b1) begin @(posedge clk); #1; end
            hq_valid = 1'b1; hq_we = we; hq_addr = addr;
            hq_be = 8'hFF; hq_wdata = wdata;
            @(posedge clk);
            #1 hq_valid = 1'b0;
            rdata = 64'd0;
            while (settled !== 1'b1) begin
                if (hs_valid === 1'b1) begin
                    rdata = hs_rdata; settled = 1'b1;
                end else begin @(posedge clk); #1; end
            end
        end
    endtask

    task run_cmd;
        input [3:0]  cop;
        input [63:0] pc;
        begin
            while (cmd_ready !== 1'b1) begin @(posedge clk); #1; end
            cmd_op = cop; cmd_pc = pc; cmd_valid = 1'b1;
            @(posedge clk);
            #1 cmd_valid = 1'b0;
            settled = 1'b0;
            while (settled !== 1'b1) begin
                if (done_valid === 1'b1) settled = 1'b1;
                else begin @(posedge clk); #1; end
            end
            done_ready = 1'b1;
            @(posedge clk);
            #1 done_ready = 1'b0;
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

    // The program, and the results area scribbled over first so a result
    // that was never written is distinguishable from one that was written
    // correctly by accident.
    task load_and_prime;
        begin
            step = 1;
            for (i = 0; i < nwords; i = i + 1)
                host_xfer(1'b1, i*8, image[i], got);
            step = 2;
            for (i = 0; i < 8; i = i + 1)
                host_xfer(1'b1, RESULT_BASE + i*8, SENTINEL, got);
        end
    endtask

    initial begin
        fails = 0;
        step  = 0;
        io_watch = 1'b0;
        quiet = $test$plusargs("quiet");
        do_reset = $test$plusargs("reset");
        if (!$value$plusargs("hex=%s", hexfile)) begin
            $display("FAIL tnca_shell_tb: pass +hex=<file>");
            $finish;
        end
        for (i = 0; i < PROGRAM_WORDS; i = i + 1) image[i] = 64'd0;
        $readmemh(hexfile, image);
        // Only the words the assembler actually emitted are worth pushing
        // through a one-word-at-a-time port.
        nwords = 0;
        for (i = 0; i < PROGRAM_WORDS; i = i + 1)
            if (image[i] !== 64'd0) nwords = i + 1;
        if (nwords < 64) nwords = 64;

        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;
        repeat (2) @(posedge clk);

        load_and_prime();

        step = 3;
        run_cmd(CMD_LAUNCH, 64'd0);
        // Printed, so a caller can compare the same program at several
        // latencies. A memory that ignores the latency it was given
        // returns the right answers at every setting and the same
        // number of cycles at every setting, and only the second of
        // those is observable.
        $display("CYCLES %0d", done_cycles);
        if (done_status !== 4'd0) begin
            $display("FAIL the program returned status %0d (fault %02h at %0h, detail %06h)",
                     done_status, done_fault_code, done_fault_pc,
                     done_fault_detail);
            fails = fails + 1;
        end

        step = 4;
        host_xfer(1'b0, RESULT_BASE + 0,  64'd0, got);
        check("partial stores", got, WANT_PARTIAL);
        host_xfer(1'b0, RESULT_BASE + 8,  64'd0, got);
        check("vector round trip", got, WANT_VSUM);
        host_xfer(1'b0, RESULT_BASE + 16, 64'd0, got);
        check("gather", got, WANT_GATHER);
        host_xfer(1'b0, RESULT_BASE + 24, 64'd0, got);
        check("atomic, old value", got, WANT_AMO_OLD);
        host_xfer(1'b0, RESULT_BASE + 32, 64'd0, got);
        check("atomic, new value", got, WANT_AMO_NEW);

        // Ownership: the program wrote five results and nothing past them.
        // A store that ran off the end would have eaten a sentinel.
        step = 5;
        for (i = 5; i < 8; i = i + 1) begin
            host_xfer(1'b0, RESULT_BASE + i*8, 64'd0, got);
            check("a slot the program must not have touched", got, SENTINEL);
        end

        // Drain: a second launch of the same program gives the same answers.
        // If anything were left in flight from the first, this is where it
        // would land in the middle of the second.
        step = 6;
        for (i = 0; i < 8; i = i + 1)
            host_xfer(1'b1, RESULT_BASE + i*8, SENTINEL, got);
        run_cmd(CMD_LAUNCH, 64'd0);
        host_xfer(1'b0, RESULT_BASE + 0, 64'd0, got);
        check("partial stores, second launch", got, WANT_PARTIAL);
        host_xfer(1'b0, RESULT_BASE + 16, 64'd0, got);
        check("gather, second launch", got, WANT_GATHER);

        //----------------------------------------------------------------------
        // R36-IO-02: the device refuses host access while a kernel runs.
        //
        // Not "the protocol happens not to offer one" -- an access is
        // actually attempted, mid-launch, and the port must be closed
        // for every cycle the shell is busy. A monitor watches the whole
        // window rather than sampling once, because a port that opened
        // for a single cycle in the middle would pass a single sample.
        //----------------------------------------------------------------------
        step = 9;
        io_saw_busy = 1'b0;
        io_open_while_busy = 1'b0;
        io_busy_cycles = 0;
        io_watch = 1'b1;
        while (cmd_ready !== 1'b1) begin @(posedge clk); #1; end
        cmd_op = CMD_LAUNCH; cmd_pc = 64'd0; cmd_valid = 1'b1;
        @(posedge clk);
        #1 cmd_valid = 1'b0;
        // Offer a host write for the whole run. The shell must not take
        // it; it is accepted only once the kernel is done.
        hq_valid = 1'b1; hq_we = 1'b1; hq_addr = RESULT_BASE + 64'd56;
        hq_be = 8'hFF; hq_wdata = 64'hFEEDFACECAFEBEEF;
        settled = 1'b0;
        while (settled !== 1'b1) begin
            if (done_valid === 1'b1) settled = 1'b1;
            else begin @(posedge clk); #1; end
        end
        io_watch = 1'b0;
        #1 hq_valid = 1'b0;
        done_ready = 1'b1;
        @(posedge clk);
        #1 done_ready = 1'b0;
        if (!io_saw_busy) begin
            $display("FAIL the host-access case never saw a busy cycle");
            fails = fails + 1;
        end else if (io_open_while_busy) begin
            $display("FAIL the host port accepted a write while a kernel ran");
            fails = fails + 1;
        end else if (!quiet) begin
            $display("    the host port refused a real offer for all %0d busy cycle(s)",
                     io_busy_cycles);
        end

        //----------------------------------------------------------------------
        // R36-RST-01: reset while a kernel is in flight.
        //
        // The kernel is launched, given long enough to have requests
        // outstanding, and then reset underneath. What must be true
        // afterwards is that the device comes back, that a new launch is
        // correct, and that nothing from the abandoned epoch appears in it.
        //----------------------------------------------------------------------
        if (do_reset) begin
            step = 7;
            // Scribble, so anything the abandoned run wrote is visible.
            for (i = 0; i < 8; i = i + 1)
                host_xfer(1'b1, RESULT_BASE + i*8, SENTINEL, got);

            while (cmd_ready !== 1'b1) begin @(posedge clk); #1; end
            cmd_op = CMD_LAUNCH; cmd_pc = 64'd0; cmd_valid = 1'b1;
            @(posedge clk);
            #1 cmd_valid = 1'b0;
            // Far enough in to have memory requests in flight, and not far
            // enough to have finished.
            repeat (30) @(posedge clk);
            if (busy !== 1'b1) begin
                $display("FAIL the reset case reset an idle device, which tests nothing");
                fails = fails + 1;
            end
            #1 rst_n = 1'b0;
            repeat (4) @(posedge clk);
            #1 rst_n = 1'b1;
            repeat (4) @(posedge clk);
            #1;

            // The shell must be idle and accepting again, with no stale
            // completion waiting to be read.
            if (done_valid !== 1'b0) begin
                $display("FAIL a completion survived the reset");
                fails = fails + 1;
            end
            if (busy !== 1'b0) begin
                $display("FAIL the shell is still busy after reset");
                fails = fails + 1;
            end

            step = 8;
            // The program image survived in memory; the results area did not
            // necessarily. Prime and run again.
            for (i = 0; i < 8; i = i + 1)
                host_xfer(1'b1, RESULT_BASE + i*8, SENTINEL, got);
            run_cmd(CMD_LAUNCH, 64'd0);
            if (done_status !== 4'd0) begin
                $display("FAIL the launch after reset returned status %0d",
                         done_status);
                fails = fails + 1;
            end
            host_xfer(1'b0, RESULT_BASE + 0, 64'd0, got);
            check("partial stores, after a reset in flight", got,
                  WANT_PARTIAL);
            host_xfer(1'b0, RESULT_BASE + 8, 64'd0, got);
            check("vector round trip, after a reset in flight", got,
                  WANT_VSUM);
            host_xfer(1'b0, RESULT_BASE + 24, 64'd0, got);
            check("atomic, after a reset in flight", got, WANT_AMO_OLD);
        end

        if (fails == 0)
            $display("PASS tnca_shell_tb impl=%0d lat=%0d stall=%0d reorder=%0d%0s",
                     MEM_IMPL, MEM_LATENCY, RANDOM_STALL, MEM_REORDER,
                     do_reset ? " +reset" : "");
        else
            $display("FAIL tnca_shell_tb impl=%0d lat=%0d stall=%0d reorder=%0d: %0d failure(s)",
                     MEM_IMPL, MEM_LATENCY, RANDOM_STALL, MEM_REORDER, fails);
        $finish;
    end

endmodule
