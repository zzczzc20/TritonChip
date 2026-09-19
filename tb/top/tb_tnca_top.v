//==============================================================================
// tb_tnca_top -- runs a program on all four tiles of the 2x2 system
//
// Plusargs:
//   +hex=<file>       program image, loaded into every tile's instruction
//                     memory and used to preload DRAM
//   +maxcycles=<n>
//   +allroles         launch every role context on every tile
//
// The DRAM is preloaded from the same image so the program's .data lands where
// the cache will fetch it from. The test asserts three things beyond "the
// program did not trap":
//
//   - every tile reached idle (nothing deadlocked in the mesh or a slice)
//   - the mesh actually carried cross-tile traffic (a test where every access
//     happened to home locally would prove nothing about the mesh)
//   - the slices actually missed, evicted and refilled
//==============================================================================
`timescale 1ns/1ps

module tb_tnca_top;

    parameter NX = 2, NY = 2;
    parameter N  = NX*NY;
    parameter NUM_RCTX = 2;
    parameter LINE_BITS = 512;
    parameter MSHR_W   = 3;
    parameter DRAM_LINES = 8192;      // 512 KiB
    parameter DRAM_LAT   = 20;
    parameter DRAM_STALL = 1;
    parameter DRAM_OOO   = 1;
    parameter IWORDS     = 4096;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg                    launch_valid = 1'b0;
    reg  [63:0]            launch_pc = 64'd0;
    reg  [NUM_RCTX-1:0]    launch_role_mask;
    reg  [63:0]            launch_arg_base = 64'd0;
    wire                   all_idle;

    wire [N-1:0]      fault_valid;
    wire [N*8-1:0]    fault_code;
    wire [N*64-1:0]   fault_pc;
    wire [N*24-1:0]   fault_detail;

    wire [N*64-1:0]   imem_addr;
    reg  [N*64-1:0]   imem_data;

    wire              m_req_valid, m_req_we;
    reg               m_req_ready_unused;
    wire              m_req_ready;
    wire [63:0]       m_req_addr;
    wire [LINE_BITS-1:0] m_req_wdata;
    wire [MSHR_W-1:0] m_req_tag;
    wire [1:0]        m_req_src;
    wire              m_rsp_valid;
    wire [MSHR_W-1:0] m_rsp_tag;
    wire [1:0]        m_rsp_src;
    wire [LINE_BITS-1:0] m_rsp_rdata;

    wire [N*64-1:0] perf_cycles, perf_issued;
    wire [N*32-1:0] perf_hit, perf_miss, perf_merge, perf_wb,
                    perf_mshr_full, perf_hit_past_block;

    tnca_top #(
        .NX(NX), .NY(NY), .NUM_RCTX(NUM_RCTX), .LINE_BITS(LINE_BITS),
        .MSHR_W(MSHR_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .launch_valid(launch_valid), .launch_pc(launch_pc),
        .launch_role_mask(launch_role_mask), .launch_arg_base(launch_arg_base),
        .all_idle(all_idle),
        .fault_valid(fault_valid), .fault_code(fault_code),
        .fault_pc(fault_pc), .fault_detail(fault_detail),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .mem_req_valid(m_req_valid), .mem_req_ready(m_req_ready),
        .mem_req_addr(m_req_addr), .mem_req_we(m_req_we),
        .mem_req_wdata(m_req_wdata), .mem_req_tag(m_req_tag),
        .mem_req_src(m_req_src),
        .mem_rsp_valid(m_rsp_valid), .mem_rsp_tag(m_rsp_tag),
        .mem_rsp_src(m_rsp_src), .mem_rsp_rdata(m_rsp_rdata),
        .perf_cycles(perf_cycles), .perf_issued(perf_issued),
        .perf_hit(perf_hit), .perf_miss(perf_miss), .perf_merge(perf_merge),
        .perf_wb(perf_wb), .perf_mshr_full(perf_mshr_full),
        .perf_hit_past_block(perf_hit_past_block)
    );

    tnca_dram_model #(
        .ADDR_W(64), .LINE_BYTES(64), .LINE_BITS(LINE_BITS),
        .LINES(DRAM_LINES), .TAG_W(MSHR_W), .SRC_W(2),
        .LATENCY(DRAM_LAT), .RANDOM_STALL(DRAM_STALL), .REORDER(DRAM_OOO)
    ) u_dram (
        .clk(clk), .rst_n(rst_n),
        .req_valid(m_req_valid), .req_ready(m_req_ready),
        .req_addr(m_req_addr), .req_we(m_req_we), .req_wdata(m_req_wdata),
        .req_tag(m_req_tag), .req_src(m_req_src),
        .rsp_valid(m_rsp_valid), .rsp_tag(m_rsp_tag), .rsp_src(m_rsp_src),
        .rsp_rdata(m_rsp_rdata)
    );

    // ---- per-tile instruction memory (sec 1.1 allows a local buffer) --------
    reg [63:0] imem [0:IWORDS-1];
    integer t;
    always @(*) begin
        for (t = 0; t < N; t = t + 1)
            imem_data[t*64 +: 64] =
                imem[(imem_addr[t*64 +: 64] >> 3) % IWORDS];
    end

    // ---- mesh traffic observation -------------------------------------------
    integer remote_flits, local_flits;
    integer i;
    always @(posedge clk) begin
        if (rst_n) begin
            for (i = 0; i < N; i = i + 1) begin
                if (dut.li_valid[i*2 + 0] && dut.li_ready[i*2 + 0]) begin
                    // request flit injected by tile i; its destination is in
                    // the flit header
                    if (({dut.li_flit[(i*2)*148 + 146 +: 2],
                          dut.li_flit[(i*2)*148 + 144 +: 2]} ==
                         {i[1], i[0]}))
                        local_flits = local_flits + 1;
                    else
                        remote_flits = remote_flits + 1;
                end
            end
        end
    end

    reg [1023:0] hexfile;
    integer maxcycles, cyc, k;
    integer tot_hit, tot_miss, tot_wb, tot_mshr_full, tot_block;

    initial begin
        remote_flits = 0; local_flits = 0;
        if (!$value$plusargs("hex=%s", hexfile)) begin
            $display("FATAL: pass +hex=<image>"); $finish;
        end
        if (!$value$plusargs("maxcycles=%d", maxcycles)) maxcycles = 2000000;

        for (k = 0; k < IWORDS; k = k + 1) imem[k] = 64'd0;
        $readmemh(hexfile, imem);
        // The same image seeds DRAM, so .data reaches the caches.
        for (k = 0; k < IWORDS; k = k + 1)
            u_dram.preload_word(k*8, imem[k]);

        // shared table at 0x8000: 0..63
        for (k = 0; k < 64; k = k + 1)
            u_dram.preload_word(64'h8000 + k*8, k);

        launch_role_mask = $test$plusargs("allroles") ? {NUM_RCTX{1'b1}}
                                                     : {{(NUM_RCTX-1){1'b0}}, 1'b1};

        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;
        repeat (2) @(posedge clk);
        #1;
        launch_pc    = 64'd0;
        launch_valid = 1'b1;
        @(posedge clk);
        #1 launch_valid = 1'b0;
        repeat (4) @(posedge clk);

        cyc = 0;
        while (!all_idle && cyc < maxcycles) begin
            @(posedge clk);
            cyc = cyc + 1;
        end

        tot_hit = 0; tot_miss = 0; tot_wb = 0; tot_mshr_full = 0; tot_block = 0;
        for (k = 0; k < N; k = k + 1) begin
            tot_hit       = tot_hit  + perf_hit[k*32 +: 32];
            tot_miss      = tot_miss + perf_miss[k*32 +: 32];
            tot_wb        = tot_wb   + perf_wb[k*32 +: 32];
            tot_mshr_full = tot_mshr_full + perf_mshr_full[k*32 +: 32];
            tot_block     = tot_block + perf_hit_past_block[k*32 +: 32];
        end

        $display("---- tnca 2x2 ----");
        $display("cycles=%0d", cyc);
        for (k = 0; k < N; k = k + 1)
            $display("  tile%0d issued=%0d  slice: hit=%0d miss=%0d wb=%0d",
                     k, perf_issued[k*64 +: 64], perf_hit[k*32 +: 32],
                     perf_miss[k*32 +: 32], perf_wb[k*32 +: 32]);
        $display("totals: hit=%0d miss=%0d wb=%0d mshr_full=%0d hit_past_block=%0d",
                 tot_hit, tot_miss, tot_wb, tot_mshr_full, tot_block);
        $display("mesh: local_home=%0d remote_home=%0d", local_flits, remote_flits);

        if (cyc >= maxcycles) begin
            $display("FAIL: timeout after %0d cycles (all_idle never asserted)", cyc);
        end else if (|fault_valid) begin
            for (k = 0; k < N; k = k + 1)
                if (fault_valid[k])
                    $display("FAIL: tile%0d trap code=0x%02x id=%0d pc=0x%0h",
                             k, fault_code[k*8 +: 8], fault_detail[k*24 +: 24],
                             fault_pc[k*64 +: 64]);
        end else if (remote_flits == 0) begin
            $display("FAIL: no request ever left its tile -- the mesh was not tested");
        end else if (tot_miss == 0 || tot_wb == 0) begin
            $display("FAIL: slices never missed or never evicted");
        end else begin
            $display("PASS");
        end
        $finish;
    end

endmodule
