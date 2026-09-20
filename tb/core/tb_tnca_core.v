//==============================================================================
// tb_tnca_core -- runs a self-checking TNCA program on a single core
//
// Plusargs:
//   +hex=<file>       memory image from tools/asm/tnca_asm.py
//   +argbase=<n>      byte address of the kernel argument block (ISA sec 5.1
//                     ARG_BASE). Default 0. A kernel compiled for dynamic
//                     parameters reads its addresses and scalars from here,
//                     so this is what lets one program image run with
//                     different arguments.
//   +maxcycles=<n>    timeout (default 200000)
//   +dump=<file>      write a memory region out after the run
//   +dumpfirst=<n>    first 64-bit word to dump (default 512)
//   +dumpcount=<n>    words to dump (default 256)
//   +expectfault=<hex> the run is expected to fault with this trap code; a
//                     clean exit is then the failure
//   +trace            print one line per retired instruction
//   +vcd=<file>       waveform dump
//
// The program signals success with `exit` from every role and failure with
// `trap #id`; the id identifies which check failed. That keeps the pass/fail
// decision inside the program under test rather than in the harness.
//==============================================================================
`timescale 1ns/1ps

module tb_tnca_core;

    parameter NUM_RCTX       = 2;
    parameter RCTX_W         = 1;
    parameter VLEN           = 16;
    parameter ELEN           = 32;
    parameter VREGS_PER_RCTX = 64;
    parameter PHYS_VREGS     = NUM_RCTX * VREGS_PER_RCTX;
    parameter PHYS_W         = 7;
    parameter WORDS          = 16384;
    parameter MEM_LATENCY    = 1;
    parameter RANDOM_STALL   = 0;
    parameter MEM_REORDER    = 0;
    parameter LSU_TAG_W      = 5;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    wire [63:0] imem_addr, imem_data;
    wire        mreq_valid, mreq_ready, mreq_we;
    wire [63:0] mreq_addr;
    wire [7:0]  mreq_be;
    wire        mreq_amo;
    wire [3:0]  mreq_aop;
    wire [63:0] mreq_cmp;
    wire [63:0] mreq_wdata;
    wire        mrsp_valid;
    wire [63:0] mrsp_rdata;
    wire [LSU_TAG_W-1:0] mreq_tag, mrsp_tag;
    wire [1:0]  mreq_hint;

    reg         launch_valid = 1'b0;
    reg [63:0]  launch_pc = 64'd0;
    reg [NUM_RCTX-1:0] launch_role_mask;
    reg [31:0]  pid_x = 0, pid_y = 0, pid_z = 0;
    reg [31:0]  npid_x = 1, npid_y = 1, npid_z = 1;
    reg [63:0]  arg_base = 64'd0;
    reg [NUM_RCTX*PHYS_W-1:0] launch_vbase;

    wire        core_idle;
    wire        fault_valid;
    wire [7:0]  fault_code;
    wire [63:0] fault_pc;
    wire [RCTX_W-1:0] fault_rctx;
    wire [23:0] fault_detail;

    wire [63:0] perf_cycles, perf_issued, perf_stall_hazard,
                perf_stall_lsu, perf_stall_wb, perf_squash;

    tnca_core #(
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN), .ELEN(ELEN), .LW(4),
        .VREGS_PER_RCTX(VREGS_PER_RCTX), .PHYS_VREGS(PHYS_VREGS), .PHYS_W(PHYS_W),
        .CORE_ID(0), .NUM_CORES(1), .HAS_SFU(1)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .launch_valid(launch_valid), .launch_pc(launch_pc),
        .launch_role_mask(launch_role_mask),
        .launch_pid_x(pid_x), .launch_pid_y(pid_y), .launch_pid_z(pid_z),
        .launch_npid_x(npid_x), .launch_npid_y(npid_y), .launch_npid_z(npid_z),
        .launch_arg_base(arg_base),
        .launch_vbase(launch_vbase),
        .core_idle(core_idle),
        .fault_valid(fault_valid), .fault_code(fault_code),
        .fault_pc(fault_pc), .fault_rctx(fault_rctx), .fault_detail(fault_detail),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .mem_req_valid(mreq_valid), .mem_req_ready(mreq_ready),
        .mem_req_amo(mreq_amo), .mem_req_aop(mreq_aop), .mem_req_cmp(mreq_cmp),
        .mem_req_addr(mreq_addr), .mem_req_we(mreq_we),
        .mem_req_be(mreq_be), .mem_req_wdata(mreq_wdata),
        .mem_req_tag(mreq_tag), .mem_req_hint(mreq_hint),
        .mem_rsp_valid(mrsp_valid), .mem_rsp_tag(mrsp_tag),
        .mem_rsp_rdata(mrsp_rdata),
        .perf_cycles(perf_cycles), .perf_issued(perf_issued),
        .perf_stall_hazard(perf_stall_hazard), .perf_stall_lsu(perf_stall_lsu),
        .perf_stall_wb(perf_stall_wb), .perf_squash(perf_squash)
    );

    tnca_mem_model #(
        .WORDS(WORDS), .MEM_DW(64), .MEM_BYTES(8), .TAG_W(LSU_TAG_W),
        .LATENCY(MEM_LATENCY), .RANDOM_STALL(RANDOM_STALL), .REORDER(MEM_REORDER)
    ) u_mem (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .req_valid(mreq_valid), .req_ready(mreq_ready),
        .req_addr(mreq_addr), .req_we(mreq_we),
        .req_be(mreq_be), .req_wdata(mreq_wdata), .req_tag(mreq_tag),
        .req_amo(mreq_amo), .req_aop(mreq_aop), .req_cmp(mreq_cmp),
        .rsp_valid(mrsp_valid), .rsp_tag(mrsp_tag), .rsp_rdata(mrsp_rdata)
    );

    reg [1023:0] hexfile, dumpfile, vcdfile;
    integer maxcycles, dumpfirst, dumpcount, k;
    integer argbase;
    integer cyc;
    integer expectfault;
    reg do_trace, do_dump, want_fault;

    initial begin
        if (!$value$plusargs("hex=%s", hexfile)) begin
            $display("FATAL: pass +hex=<image>");
            $finish;
        end
        if (!$value$plusargs("maxcycles=%d", maxcycles)) maxcycles = 200000;
        if (!$value$plusargs("argbase=%d", argbase)) argbase = 0;
        arg_base = argbase;
        if (!$value$plusargs("dumpfirst=%d", dumpfirst))  dumpfirst = 512;
        if (!$value$plusargs("dumpcount=%d", dumpcount))  dumpcount = 256;
        want_fault = $value$plusargs("expectfault=%h", expectfault);
        do_trace = $test$plusargs("trace");
        do_dump  = $value$plusargs("dump=%s", dumpfile);

        if ($value$plusargs("vcd=%s", vcdfile)) begin
            $dumpfile(vcdfile);
            $dumpvars(0, tb_tnca_core);
        end

        u_mem.load_hex(hexfile);

        launch_role_mask = {{(NUM_RCTX-1){1'b0}}, 1'b1};
        if ($test$plusargs("allroles"))
            launch_role_mask = {NUM_RCTX{1'b1}};

        // Per-RCTX physical vector base. Disjoint by default; +aliasvrf gives
        // every context the same base, which is what a producer/consumer role
        // pair needs to share a window with no copy.
        for (k = 0; k < NUM_RCTX; k = k + 1)
            launch_vbase[k*PHYS_W +: PHYS_W] =
                $test$plusargs("aliasvrf") ? 0 : (k * VREGS_PER_RCTX);

        // Stimulus is driven one delta after the clock edge, never on it: a
        // blocking assignment in the active region of the same edge the DUT
        // samples is a race, and it silently swallowed the launch pulse.
        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;
        repeat (2) @(posedge clk);

        #1;
        launch_pc    = 64'd0;
        launch_valid = 1'b1;
        @(posedge clk);
        #1 launch_valid = 1'b0;
        // let the launch take effect before polling core_idle, which is only
        // meaningful once `running` has been written
        repeat (3) @(posedge clk);

        cyc = 0;
        while (!core_idle && cyc < maxcycles) begin
            @(posedge clk);
            cyc = cyc + 1;
        end

        if (do_dump) u_mem.dump_hex(dumpfile, dumpfirst, dumpcount);

        $display("---- tnca core run ----");
        $display("cycles=%0d issued=%0d hazard=%0d lsu=%0d wb=%0d squash=%0d",
                 cyc, perf_issued, perf_stall_hazard, perf_stall_lsu,
                 perf_stall_wb, perf_squash);
        // What the memory model's seed actually changed. A matrix that runs
        // three seeds and reports three identical traces has run one seed
        // three times, and R1-08 asks for the difference to be visible.
        $display("memtrace=%016h memstall=%0d memreorder=%0d memmulti=%0d",
                 u_mem.mem_trace, u_mem.n_stall_cycles, u_mem.n_reorder,
                 u_mem.n_multi_eligible);
        // Two engine completions landing on one barrier in one cycle. The
        // barrier sums three independent ports so it cannot drop one; this
        // says whether the case actually occurred.
        $display("barcollide=%0d", dut.u_bar.n_multi_complete);

        if (cyc >= maxcycles) begin
            $display("FAIL: timeout after %0d cycles", cyc);
        end else if (want_fault) begin
            // A fault is the expected outcome; reaching idle without one, or
            // with the wrong code, is the failure.
            if (!fault_valid)
                $display("FAIL: expected trap 0x%02x, ran to completion", expectfault);
            else if (fault_code !== expectfault[7:0])
                $display("FAIL: expected trap 0x%02x, got 0x%02x at pc=0x%0h",
                         expectfault, fault_code, fault_pc);
            else
                $display("PASS  (expected trap 0x%02x at pc=0x%0h rctx=%0d)",
                         fault_code, fault_pc, fault_rctx);
        end else if (fault_valid) begin
            $display("FAIL: trap code=0x%02x id=%0d pc=0x%0h rctx=%0d",
                     fault_code, fault_detail, fault_pc, fault_rctx);
        end else begin
            $display("PASS");
        end
        $finish;
    end

    // Completion cycles, for the collision experiment. +donetrace prints the
    // cycle each engine signals a barrier, which is what makes "arrange for
    // them to land together" an experiment rather than a guess.
    always @(posedge clk) begin
        if (rst_n && $test$plusargs("donetrace")) begin
            if (dut.txf_done && dut.txf_done_bar_en)
                $display("[done] cyc=%0d txf bar=%0d", cyc, dut.txf_done_bar);
            if (dut.mtx_done && dut.mtx_done_bar_en)
                $display("[done] cyc=%0d mtx bar=%0d", cyc, dut.mtx_done_bar);
        end
    end

    // instruction trace
    always @(posedge clk) begin
        if (rst_n && do_trace && dut.x_valid)
            $display("[%0t] rctx=%0d pc=%08h maj=%02h fn=%02h dst=%0d/%0d",
                     $time, dut.x_rctx, dut.x_pc, dut.x_major, dut.x_func,
                     dut.x_dstc, dut.x_dsti);
    end

endmodule
