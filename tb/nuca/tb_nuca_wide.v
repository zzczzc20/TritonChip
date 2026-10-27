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

module tb_nuca_wide;
    parameter CACHE_DATA_PIPELINE=1;
    parameter PRESSURE=0;
    parameter HIT_CHAIN=0;
    parameter RSP_STALL=0;
    parameter SERVICE_MODE=0; // 1: hot reads, 2: streaming reads
    parameter DCO_REGISTERED_RANGES = 0;
    parameter SYNC_TAG_BANK = 1;
    parameter BALANCED_VICTIM = 1;
    parameter REGISTER_VICTIM_OWNER = 0;
    parameter EPOCH_CHURN = 0;

    parameter DRAM_LAT   = 40;
    parameter DRAM_STALL = 1;
    parameter DRAM_OOO   = 1;
    // DCO policy under test (rtl/nuca/tnca_tmu.v POLICY): 0 is the round-robin
    // baseline. Whatever the policy, every read must still return the shadow
    // -- replacement, bypass and dead-block prediction move only where a line
    // lives, never what it holds.
    parameter POLICY     = 0;
    parameter MAINTENANCE = 0;

    localparam ADDR_W     = 64;
    localparam LINE_BYTES = 64;
    localparam LINE_BITS  = 512;
    parameter WORD_BITS  = 512;
    localparam WORD_BYTES = WORD_BITS/8;
    localparam WORDS_PER_LINE=LINE_BYTES/WORD_BYTES;
    parameter SETS       = 16;
    parameter SET_W      = 4;
    parameter WAYS       = 4;
    parameter WAY_W      = 2;
    localparam MSHR_N     = 4;
    localparam MSHR_W     = 2;
    localparam TARGET_N   = 8;
    localparam TARGET_W   = 3;
    localparam RQ_N       = 8;
    localparam RQ_W       = 3;
    localparam SRC_W      = 4;
    localparam TAG_W      = 5;
    localparam SLICE_ID   = 0;

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
    wire [31:0] perf_bypass, perf_evict, perf_dead_evict;
    wire [3:0]  perf_gear;

    // ---- DCO stimulus ---------------------------------------------------------
    // One tensor registered as nAcc = 1 (bypass) over the second quarter of the
    // address space, one ordinary tensor over the rest, a dead-tile FIFO that a
    // random process rewrites, and a "slower core" flag that flips. A small
    // window and low thresholds keep the gear moving.
    localparam [7:0]  B_BITS_TB = 8'd3;
    localparam [63:0] T_BYP     = N_LINES * 64;
    wire [75:0] dco_cfg = {16'd2, 16'd6, 16'd64, 4'd2, 6'd0, 6'd8,
                           B_BITS_TB[3:0], POLICY[7:0]};
    reg  [15:0]             dco_slow;
    reg  [15:0]             dead_v;
    reg  [16*16-1:0]        dead_id;
    reg                     tl_ready;
    wire                    tl_valid, tl_drop, tmw_valid;
    wire [63:0]             tl_addr;

    tnca_nuca_slice #(
        .HIT_CHAIN(HIT_CHAIN),.CACHE_DATA_PIPELINE(CACHE_DATA_PIPELINE),.MAINTENANCE(MAINTENANCE), .DCO_REGISTERED_RANGES(DCO_REGISTERED_RANGES), .SYNC_TAG_BANK(SYNC_TAG_BANK), .BALANCED_VICTIM(BALANCED_VICTIM), .REGISTER_VICTIM_OWNER(REGISTER_VICTIM_OWNER),
        .ADDR_W(ADDR_W), .LINE_BYTES(LINE_BYTES), .LINE_BITS(LINE_BITS),
        .WORD_BITS(WORD_BITS), .WORD_BYTES(WORD_BYTES),
        .SETS(SETS), .SET_W(SET_W), .WAYS(WAYS), .WAY_W(WAY_W),
        .MSHR_N(MSHR_N), .MSHR_W(MSHR_W),
        .TARGET_N(TARGET_N), .TARGET_W(TARGET_W),
        .RQ_N(RQ_N), .RQ_W(RQ_W), .SRC_W(SRC_W), .TAG_W(TAG_W),
        .SLICE_ID(SLICE_ID), .DCO_T(2)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .maint_valid(1'b0), .maint_base(64'd0), .maint_size(64'd0),
        .req_ready(req_ready), .req_valid(req_valid), .req_addr(req_addr),
        .req_we(req_we), .req_be(req_be), .req_wdata(req_wdata),
        .req_src(req_src), .req_tag(req_tag),
        .req_amo(1'b0), .req_aop(4'd0), .req_cmp({WORD_BITS{1'b0}}),
        .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_src(rsp_src),
        .rsp_tag(rsp_tag), .rsp_rdata(rsp_rdata),
        .mem_req_valid(m_req_valid), .mem_req_ready(m_req_ready),
        .mem_req_addr(m_req_addr), .mem_req_we(m_req_we),
        .mem_req_wdata(m_req_wdata), .mem_req_tag(m_req_tag),
        .mem_rsp_valid(m_rsp_valid), .mem_rsp_tag(m_rsp_tag),
        .mem_rsp_rdata(m_rsp_rdata),
        .perf_hit(perf_hit), .perf_miss(perf_miss), .perf_merge(perf_merge),
        .perf_mshr_full(perf_mshr_full), .perf_target_full(perf_target_full),
        .perf_wb(perf_wb), .perf_hit_past_block(perf_hit_past_block),
        .dco_owner_epoch(owner_epoch), .dco_cfg(dco_cfg),
        .dco_t_valid(2'b11),
        .dco_t_base({64'd0, T_BYP}),
        .dco_t_size({64'd0, T_BYP}),
        .dco_t_end({1'b1, 64'd0, ({1'b0,T_BYP}+{1'b0,T_BYP})}),
        .dco_t_bypass({owner_bypass_flip,1'b1}), .dco_t_lsb({6'd0, 6'd9}),
        .dco_dead_valid(dead_v), .dco_dead_id(dead_id), .dco_slow(dco_slow),
        .tmu_wr_valid(tmw_valid), .tmu_wr_ready(1'b1), .tmu_wr_off(),
        .tmu_wr_data(), .tmu_wr_src(), .tmu_rd_off(), .tmu_rd_data(64'd0),
        .tmu_tl_valid(tl_valid), .tmu_tl_ready(tl_ready), .tmu_tl_addr(tl_addr),
        .tmu_tl_drop(tl_drop),
        .perf_bypass(perf_bypass), .perf_evict(perf_evict),
        .perf_dead_evict(perf_dead_evict), .perf_gear(perf_gear)
    );

    reg owner_epoch=0, owner_bypass_flip=0;
    integer owner_cycles=0, owner_retries=0;
    reg expect_owner_retry=0;
    always @(posedge clk) begin
        if (!rst_n) begin
            owner_epoch<=0; owner_bypass_flip<=0; owner_cycles<=0;
            owner_retries<=0; expect_owner_retry<=0;
        end else begin
            owner_cycles<=owner_cycles+1;
            if (EPOCH_CHURN && dut.est==1 && owner_cycles%4==0) begin
                owner_epoch<=~owner_epoch;
                owner_bypass_flip<=~owner_bypass_flip;
            end
            expect_owner_retry <= REGISTER_VICTIM_OWNER && dut.est==3 &&
                                  dut.owner_epoch_q != owner_epoch;
            if (REGISTER_VICTIM_OWNER && dut.est==3 && dut.owner_epoch_q != owner_epoch)
                owner_retries<=owner_retries+1;
        end
    end
    always @(negedge clk)
        if (rst_n && expect_owner_retry && dut.est!==1)
            $fatal(1,"stale owner classification consumed instead of retried");

    integer dk;
    integer dseed;
    always @(posedge clk) begin
        if (!rst_n) begin
            dco_slow <= 16'h0000; dead_v <= 16'd0; dead_id <= 0; tl_ready <= 1'b0;
        end else begin
            tl_ready <= $random(dseed);
            if (($random(dseed) & 255) == 0) dco_slow <= $random(dseed);
            if (($random(dseed) & 31) == 0) begin
                dk = ($random(dseed) & 15);
                dead_v[dk] <= 1'b1;
                // {tensor index, chunk}: tensor 0 (bypassed, 512-byte tiles)
                // or 1 (the default granule, D_LSB = 8)
                dead_id[dk*16 +: 16] <= {3'd0, 13'd0} | (($random(dseed) & 1) << 13)
                                        | ($random(dseed) & 255);
            end
        end
    end

    tnca_dram_model #(
        .ADDR_W(ADDR_W), .LINE_BYTES(LINE_BYTES), .LINE_BITS(LINE_BITS),
        .LINES(N_LINES * 4), .TAG_W(MSHR_W), .SRC_W(1),   // home-consistent lines span 4x
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
    integer service_cycle=0,accepted=0,responses=0,first_accept=-1,last_response=-1;
    always @(posedge clk)begin
        if(rst_n)begin
            service_cycle<=service_cycle+1;
            if(req_valid && req_ready)begin
                accepted<=accepted+1;
                if(first_accept<0)first_accept<=service_cycle;
            end
            if(rsp_valid && rsp_ready)begin responses<=responses+1;last_response<=service_cycle;end
        end
    end
    generate if(RSP_STALL)begin:g_rsp_stall
        always @(posedge clk)if(rst_n)rsp_ready <= service_cycle%13<5;
    end endgenerate
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
        dseed = seed ^ 32'h00dc0dc0;
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
                        stream_ptr = stream_ptr + WORDS_PER_LINE;  // one word per line
                    end
                end else if (rnd[3:0] < 12) begin
                    widx = rnd[9:0] % 64;
                end else begin
                    widx = rnd[31:0] % N_WORDS;
                end

                if(PRESSURE) widx=0;
                if(SERVICE_MODE==1)widx=(n_issued%16)*WORDS_PER_LINE;
                if(SERVICE_MODE==2)widx=n_issued%N_WORDS;

                // A home-consistent address: the mesh only ever delivers a
                // slice the lines whose home bits (line[1:0] ^ line[3:2]) are
                // its id, and the slice indexes on that. Line L of the shadow
                // becomes line {L, L[1:0] ^ SLICE_ID}, a bijection.
                a = ((((widx / WORDS_PER_LINE) << 2) | (((widx / WORDS_PER_LINE) & 3) ^ SLICE_ID)) << 6)
                    | ((widx % WORDS_PER_LINE) * WORD_BYTES);
                req_addr  = a;
                req_src   = 4'd1;
                req_tag   = free_tag;
                req_be    = {WORD_BYTES{1'b1}};
                if (SERVICE_MODE==0 && rnd[5:4] == 2'b00) begin
                    // write
                    req_we    = 1'b1;
                    for(integer bytepos=0;bytepos<WORD_BYTES;bytepos=bytepos+1)begin
                        req_wdata[bytepos*8+:8]=$random(seed);
                        req_be[bytepos]=$random(seed);
                    end
                    n_write = n_write + 1;
                    for(integer bytepos=0;bytepos<WORD_BYTES;bytepos=bytepos+1)
                        if(req_be[bytepos])shadow[widx][bytepos*8+:8]=req_wdata[bytepos*8+:8];
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

        if(SERVICE_MODE!=0)begin
            if(accepted!=nreq || responses!=nreq)$fatal(1,"service lost a handshake");
            $display("HOME SERVICE mode=%0d word_bytes=%0d accepted=%0d responses=%0d span_cycles=%0d hit=%0d miss=%0d",SERVICE_MODE,WORD_BYTES,accepted,responses,last_response-first_accept+1,perf_hit,perf_miss);
        end
        $display("---- nuca slice ----");
        $display("seed=%0d trace_hash=%016h writes=%0d", seed0, trace_hash, n_write);
        $display("issued=%0d reads_checked=%0d inflight_left=%0d", n_issued, n_checked, inflight);
        $display("hit=%0d miss=%0d merge=%0d wb=%0d mshr_full=%0d target_full=%0d hit_past_block=%0d",
                 perf_hit, perf_miss, perf_merge, perf_wb,
                 perf_mshr_full, perf_target_full, perf_hit_past_block);
        if (EPOCH_CHURN && REGISTER_VICTIM_OWNER && owner_retries==0)
            $fatal(1,"owner epoch mutation was not exercised");
        $display("OWNER_EPOCH retries=%0d",owner_retries);
        $display("policy=%0d bypass=%0d evict=%0d dead_evict=%0d gear=%0d",
                 POLICY, perf_bypass, perf_evict, perf_dead_evict, perf_gear);

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
