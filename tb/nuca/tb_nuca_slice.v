//==============================================================================
// tb_nuca_slice -- randomised differential test of one NUCA slice
//
// A traffic generator drives reads and writes into the slice while a shadow
// memory records what every read should return. Every response is checked
// against the shadow by tag.
//
// The address range is chosen to be several times the slice's capacity, so the
// test spends most of its time evicting dirty lines rather than hitting -- the
// paths that a small directed test would never reach. Requests are issued as
// fast as the slice accepts them, so the MSHR file and the target pool fill up
// and the sec 9.3 "hits continue past a blocked miss" path is exercised.
//
// The shadow is updated at ISSUE time, not at response time, because the slice
// orders conflicting accesses to one line by arrival (sec 14.1): a read issued
// after a write to the same address must see that write, whatever order the
// responses come back in.
//==============================================================================
`timescale 1ns/1ps

module tb_nuca_slice;

    parameter DRAM_LAT   = 40;
    parameter DRAM_STALL = 1;
    parameter DRAM_OOO   = 1;

    localparam ADDR_W     = 64;
    localparam LINE_BYTES = 64;
    localparam LINE_BITS  = 512;
    localparam WORD_BITS  = 64;
    localparam WORD_BYTES = 8;
    localparam SETS       = 16;
    localparam SET_W      = 4;
    localparam WAYS       = 4;
    localparam WAY_W      = 2;
    localparam MSHR_N     = 4;
    localparam MSHR_W     = 2;
    localparam TARGET_N   = 8;
    localparam TARGET_W   = 3;
    localparam RQ_N       = 8;
    localparam RQ_W       = 3;
    localparam SRC_W      = 4;
    localparam TAG_W      = 5;

    // Address space: 8x the slice capacity, so eviction is the common case.
    localparam N_LINES    = SETS * WAYS * 8;
    localparam N_WORDS    = N_LINES * (LINE_BYTES / WORD_BYTES);

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---- DUT wiring ---------------------------------------------------------
    reg                    req_valid;
    wire                   req_ready;
    reg  [ADDR_W-1:0]      req_addr;
    reg                    req_we;
    reg  [WORD_BYTES-1:0]  req_be;
    reg  [WORD_BITS-1:0]   req_wdata;
    reg  [SRC_W-1:0]       req_src;
    reg  [TAG_W-1:0]       req_tag;

    wire                   rsp_valid;
    reg                    rsp_ready;
    wire [SRC_W-1:0]       rsp_src;
    wire [TAG_W-1:0]       rsp_tag;
    wire [WORD_BITS-1:0]   rsp_rdata;

    wire                   m_req_valid;
    wire                   m_req_ready;
    wire [ADDR_W-1:0]      m_req_addr;
    wire                   m_req_we;
    wire [LINE_BITS-1:0]   m_req_wdata;
    wire [MSHR_W-1:0]      m_req_tag;
    wire                   m_rsp_valid;
    wire [MSHR_W-1:0]      m_rsp_tag;
    wire [LINE_BITS-1:0]   m_rsp_rdata;

    wire [31:0] perf_hit, perf_miss, perf_merge, perf_mshr_full,
                perf_target_full, perf_wb, perf_hit_past_block;

    tnca_nuca_slice #(
        .ADDR_W(ADDR_W), .LINE_BYTES(LINE_BYTES), .LINE_BITS(LINE_BITS),
        .WORD_BITS(WORD_BITS), .WORD_BYTES(WORD_BYTES),
        .SETS(SETS), .SET_W(SET_W), .WAYS(WAYS), .WAY_W(WAY_W),
        .MSHR_N(MSHR_N), .MSHR_W(MSHR_W),
        .TARGET_N(TARGET_N), .TARGET_W(TARGET_W),
        .RQ_N(RQ_N), .RQ_W(RQ_W), .SRC_W(SRC_W), .TAG_W(TAG_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .req_ready(req_ready), .req_valid(req_valid), .req_addr(req_addr),
        .req_we(req_we), .req_be(req_be), .req_wdata(req_wdata),
        .req_src(req_src), .req_tag(req_tag),
        .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_src(rsp_src),
        .rsp_tag(rsp_tag), .rsp_rdata(rsp_rdata),
        .mem_req_valid(m_req_valid), .mem_req_ready(m_req_ready),
        .mem_req_addr(m_req_addr), .mem_req_we(m_req_we),
        .mem_req_wdata(m_req_wdata), .mem_req_tag(m_req_tag),
        .mem_rsp_valid(m_rsp_valid), .mem_rsp_tag(m_rsp_tag),
        .mem_rsp_rdata(m_rsp_rdata),
        .perf_hit(perf_hit), .perf_miss(perf_miss), .perf_merge(perf_merge),
        .perf_mshr_full(perf_mshr_full), .perf_target_full(perf_target_full),
        .perf_wb(perf_wb), .perf_hit_past_block(perf_hit_past_block)
    );

    tnca_dram_model #(
        .ADDR_W(ADDR_W), .LINE_BYTES(LINE_BYTES), .LINE_BITS(LINE_BITS),
        .LINES(N_LINES), .TAG_W(MSHR_W), .SRC_W(1),
        .LATENCY(DRAM_LAT), .RANDOM_STALL(DRAM_STALL), .REORDER(DRAM_OOO)
    ) u_dram (
        .clk(clk), .rst_n(rst_n),
        .req_valid(m_req_valid), .req_ready(m_req_ready),
        .req_addr(m_req_addr), .req_we(m_req_we), .req_wdata(m_req_wdata),
        .req_tag(m_req_tag), .req_src(1'b0),
        .rsp_valid(m_rsp_valid), .rsp_tag(m_rsp_tag), .rsp_src(),
        .rsp_rdata(m_rsp_rdata)
    );

    // ---- shadow memory and in-flight expectations ---------------------------
    reg [WORD_BITS-1:0] shadow [0:N_WORDS-1];
    reg [TAG_W:0]       inflight;               // count, not a mask
    reg                 e_valid [0:(1<<TAG_W)-1];
    reg                 e_isread[0:(1<<TAG_W)-1];
    reg [WORD_BITS-1:0] e_exp   [0:(1<<TAG_W)-1];
    reg [ADDR_W-1:0]    e_addr  [0:(1<<TAG_W)-1];

    integer i, n_issued, n_checked, fails, nreq;
    integer widx, stream_ptr, min_wb, min_block;
    reg [WORD_BITS-1:0] cur;
    reg [TAG_W-1:0] t;

    // pick a free tag
    reg [TAG_W-1:0] free_tag; reg have_tag;
    always @(*) begin
        free_tag = {TAG_W{1'b0}}; have_tag = 1'b0;
        for (i = (1<<TAG_W)-1; i >= 0; i = i - 1)
            if (!e_valid[i]) begin free_tag = i[TAG_W-1:0]; have_tag = 1'b1; end
    end

    // ---- response checking ---------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && rsp_valid && rsp_ready) begin
            t = rsp_tag;
            if (!e_valid[t]) begin
                fails = fails + 1;
                $display("FAIL response for an unissued tag %0d", t);
            end else begin
                if (e_isread[t]) begin
                    n_checked = n_checked + 1;
                    if (rsp_rdata !== e_exp[t]) begin
                        fails = fails + 1;
                        if (fails < 12)
                            $display("FAIL read addr=%h tag=%0d got=%h want=%h",
                                     e_addr[t], t, rsp_rdata, e_exp[t]);
                    end
                end
                e_valid[t] = 1'b0;
                inflight = inflight - 1;
            end
        end
    end

    // ---- watchdog: a deadlocked slice must fail, not run forever ------------
    integer wd;
    initial begin
        wd = 0;
        forever begin
            @(posedge clk);
            wd = wd + 1;
            if (wd > 2000000) begin
                $display("FAIL: watchdog -- issued=%0d checked=%0d inflight=%0d est=%0d req_ready=%b",
                         n_issued, n_checked, inflight, dut.est, req_ready);
                $finish;
            end
        end
    end

    // ---- traffic -------------------------------------------------------------
    integer seed;
    reg [ADDR_W-1:0] a;
    reg [63:0] rnd;
    // The stimulus trace, hashed. R1-08 requires the seed to reach the PRNG
    // that actually controls the traffic rather than being recorded in a log
    // line, and the only way to show that is a digest of what was issued: two
    // seeds must produce different traces, and the same seed the same one.
    reg [63:0] trace_hash;
    integer    n_write;
    // $random mutates its seed argument, so the value at the end is not the
    // one the run started from. Keep the request separately or the log
    // reports a number nobody passed.
    integer    seed0;

    initial begin
        // +seed reaches $random itself. Without the argument $random draws
        // from one global stream, so every "seed" ran identical traffic and
        // three seeds were three copies of one test.
        if (!$value$plusargs("seed=%d", seed)) seed = 20260918;
        seed0 = seed;
        trace_hash = 64'hcbf29ce484222325;   // FNV-1a offset basis
        n_write = 0;
        fails = 0; n_issued = 0; n_checked = 0; inflight = 0; stream_ptr = 0;
        req_valid = 1'b0; rsp_ready = 1'b1;
        for (i = 0; i < (1<<TAG_W); i = i + 1) e_valid[i] = 1'b0;
        for (i = 0; i < N_WORDS; i = i + 1) shadow[i] = 64'd0;

        if (!$value$plusargs("nreq=%d", nreq)) nreq = 20000;
        // Coverage floors. A configuration with a fast, never-stalling memory
        // legitimately never exhausts an MSHR file, so requiring it everywhere
        // would be a false failure; the stressing configuration asks for it.
        if (!$value$plusargs("min_wb=%d", min_wb))       min_wb = 1;
        if (!$value$plusargs("min_block=%d", min_block)) min_block = 0;

        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;

        while (n_issued < nreq) begin
            @(posedge clk);
            #1;
            req_valid = 1'b0;
            if (have_tag && req_ready) begin
                rnd = {$random(seed), $random(seed)};
                // Two traffic modes, alternating every 64 requests.
                //
                //   hot       a working set that fits, so lines are reused and
                //             dirty-evicted repeatedly
                //   streaming a walk through addresses far larger than the
                //             slice, which misses on essentially every access
                //             and is the only way to drive the MSHR file and
                //             the target pool to exhaustion -- the sec 9.3
                //             "hits continue past a blocked miss" path
                if (n_issued[6]) begin
                    // Three streaming misses to one hot access, so there is
                    // always a hit available to be selected past a blocked
                    // miss. Without that mix, sec 9.3's path is never taken and
                    // the counter that proves it stays at zero.
                    if (rnd[1:0] == 2'b00) begin
                        widx = rnd[5:0] % 64;
                    end else begin
                        widx = stream_ptr % N_WORDS;
                        stream_ptr = stream_ptr + 8;  // one word per line
                    end
                end else if (rnd[3:0] < 12) begin
                    widx = rnd[9:0] % 64;
                end else begin
                    widx = rnd[31:0] % N_WORDS;
                end

                a = widx * WORD_BYTES;
                req_addr  = a;
                req_src   = 4'd1;
                req_tag   = free_tag;
                req_be    = 8'hff;
                if (rnd[5:4] == 2'b00) begin
                    // write
                    req_we    = 1'b1;
                    req_wdata = {$random(seed), $random(seed)};
                    n_write = n_write + 1;
                    shadow[widx] = req_wdata;
                    e_isread[free_tag] = 1'b0;
                end else begin
                    req_we    = 1'b0;
                    req_wdata = 64'd0;
                    e_isread[free_tag] = 1'b1;
                    e_exp[free_tag]    = shadow[widx];
                end
                e_valid[free_tag] = 1'b1;
                e_addr[free_tag]  = a;
                // One step of FNV-1a over {address, direction}: what was
                // asked for, in the order it was asked.
                trace_hash = (trace_hash ^ {31'd0, req_we, a}) * 64'd1099511628211;
                inflight  = inflight + 1;
                req_valid = 1'b1;
                n_issued  = n_issued + 1;
            end
        end
        // The last request has only been PRESENTED; it is sampled at the next
        // edge. Dropping req_valid here would silently lose it.
        @(posedge clk);
        #1 req_valid = 1'b0;

        // drain
        i = 0;
        while (inflight > 0 && i < 20000) begin
            @(posedge clk); #1;
            i = i + 1;
        end

        $display("---- nuca slice ----");
        $display("seed=%0d trace_hash=%016h writes=%0d", seed0, trace_hash, n_write);
        $display("issued=%0d reads_checked=%0d inflight_left=%0d", n_issued, n_checked, inflight);
        $display("hit=%0d miss=%0d merge=%0d wb=%0d mshr_full=%0d target_full=%0d hit_past_block=%0d",
                 perf_hit, perf_miss, perf_merge, perf_wb,
                 perf_mshr_full, perf_target_full, perf_hit_past_block);

        if (inflight > 0) begin
            fails = fails + 1;
            $display("FAIL: %0d requests never completed", inflight);
        end
        if (perf_wb < min_wb) begin
            fails = fails + 1;
            $display("FAIL: only %0d dirty writebacks, wanted %0d", perf_wb, min_wb);
        end
        if (min_block > 0) begin
            if ((perf_mshr_full + perf_target_full) < min_block) begin
                fails = fails + 1;
                $display("FAIL: resources ran out only %0d times, wanted %0d",
                         perf_mshr_full + perf_target_full, min_block);
            end
            if (perf_hit_past_block == 0) begin
                fails = fails + 1;
                $display("FAIL: no hit was ever selected past a blocked miss");
            end
        end
        if (fails == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end

endmodule
