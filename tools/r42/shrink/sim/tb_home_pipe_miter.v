//==============================================================================
// tb_home_pipe_miter -- cycle-exact equivalence of tnca_home_pipe DATA_RAM = 0 (distributed RAM, asynchronous read: the shipped RTL)
// against DATA_RAM = DR (1 block RAM, 2 UltraRAM; synchronous read issued by S2), R42-shrink.
//
// Both instances get the same inputs every cycle: random requests held until accepted (reads, byte-enabled stores, full-line stores,
// atomics, TMU register accesses), random response back-pressure, a random-latency out-of-order memory model driven by instance A's
// memory requests (data = line hash, updated by A's writebacks), random TMU ready, random maintenance ranges. Every output of both
// instances is compared in every cycle (`!==`, so an X against a value fails). Parameters overridable with iverilog -P.
//==============================================================================
`timescale 1ns/1ps
module tb;
    parameter DR       = 1;            // DATA_RAM of instance B
    parameter AR       = 0;            // AUX_RAM of instance B
    parameter SEED     = 1;
    parameter CYCLES   = 200000;
    parameter NLINES   = 384;          // distinct lines touched (32 sets x 8 ways = 256 lines resident)
    parameter P_REQ    = 80;           // percent: a new request is offered in a cycle when none is pending
    parameter P_RSP    = 70;           // percent: rsp_ready
    parameter P_MEM    = 70;           // percent: mem_req_ready
    parameter P_MRSP   = 60;           // percent: an eligible memory response is delivered
    parameter MAXLAT   = 24;           // memory latency 1..MAXLAT cycles
    parameter P_WR     = 35;           // percent of requests that are stores
    parameter P_FULL   = 40;           // percent of stores that cover the whole line
    parameter P_AMO    = 10;           // percent of requests that are atomics
    parameter P_MMIO   = 3;            // percent of requests to the TMU register page
    parameter P_MAINT  = 1;            // per mille: a maintenance request in a cycle
    parameter HOT      = 0;            // 1: 80 percent of requests go to the 16 hottest lines (same-line ordering stress)
    parameter ADDR_W   = 35;
    parameter SETS     = 32;
    parameter SET_W    = 5;
    parameter WAYS     = 8;
    parameter WAY_W    = 3;
    parameter MSHR_N   = 64;
    parameter MSHR_W   = 6;
    parameter TARGET_N = 128;
    parameter TARGET_W = 7;
    parameter RQ_N     = 16;
    parameter RQ_W     = 4;
    parameter SRC_W    = 4;
    parameter TAG_W    = 8;
    parameter NOFILL   = 1;
    parameter CHAIN    = 1;
    parameter MAINT_EN = 1;
    parameter WBQ_N    = 4;

    localparam [63:0] TMU_BASE = 64'h0000_0000_7FFF_0000;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ---- stimulus (registered, shared) ----
    reg                 req_valid; reg [ADDR_W-1:0] req_addr; reg req_we; reg [63:0] req_be; reg [511:0] req_wdata;
    reg [SRC_W-1:0]     req_src; reg [TAG_W-1:0] req_tag; reg req_amo; reg [3:0] req_aop; reg [511:0] req_cmp;
    reg                 rsp_ready, mem_req_ready, mem_rsp_valid, tmu_wr_ready, maint_valid;
    reg [MSHR_W-1:0]    mem_rsp_tag; reg [511:0] mem_rsp_rdata; reg [63:0] tmu_rd_data;
    reg [ADDR_W-1:0]    maint_base, maint_size;
    wire [75:0] dco_cfg = 76'd0; wire [511:0] z512 = 512'd0;

    // ---- outputs of A and B ----
    wire a_req_ready, a_rsp_valid, a_mem_req_valid, a_mem_req_we, a_maint_ready, a_maint_busy, a_maint_done, a_tmu_wr_valid, a_tmu_tl_valid, a_tmu_tl_drop;
    wire b_req_ready, b_rsp_valid, b_mem_req_valid, b_mem_req_we, b_maint_ready, b_maint_busy, b_maint_done, b_tmu_wr_valid, b_tmu_tl_valid, b_tmu_tl_drop;
    wire [SRC_W-1:0] a_rsp_src, b_rsp_src, a_tmu_wr_src, b_tmu_wr_src; wire [TAG_W-1:0] a_rsp_tag, b_rsp_tag;
    wire [511:0] a_rsp_rdata, b_rsp_rdata, a_mem_req_wdata, b_mem_req_wdata;
    wire [ADDR_W-1:0] a_mem_req_addr, b_mem_req_addr; wire [MSHR_W-1:0] a_mem_req_tag, b_mem_req_tag;
    wire [11:0] a_tmu_wr_off, b_tmu_wr_off, a_tmu_rd_off, b_tmu_rd_off; wire [63:0] a_tmu_wr_data, b_tmu_wr_data, a_tmu_tl_addr, b_tmu_tl_addr;
    wire [31:0] a_p0,a_p1,a_p2,a_p3,a_p4,a_p5,a_p6,a_p7,a_p8,a_p9, b_p0,b_p1,b_p2,b_p3,b_p4,b_p5,b_p6,b_p7,b_p8,b_p9; wire [3:0] a_gear, b_gear;

`define HP_PARAMS \
        .ADDR_W(ADDR_W), .LINE_BYTES(64), .LINE_BITS(512), .WORD_BITS(512), .WORD_BYTES(64), .LINE_WRITE_NOFILL(NOFILL), \
        .SETS(SETS), .SET_W(SET_W), .WAYS(WAYS), .WAY_W(WAY_W), .MSHR_N(MSHR_N), .MSHR_W(MSHR_W), .TARGET_N(TARGET_N), .TARGET_W(TARGET_W), \
        .RQ_N(RQ_N), .RQ_W(RQ_W), .SRC_W(SRC_W), .TAG_W(TAG_W), .SLICE_ID(1), .FULL_LINE_TAG(1), .MAINTENANCE(MAINT_EN), .WBQ_N(WBQ_N), \
        .FILL_CHAIN(CHAIN), .TMU_BASE(TMU_BASE)
`define HP_PORTS(P) \
        .clk(clk), .rst_n(rst_n), .maint_valid(maint_valid), .maint_ready(P``_maint_ready), .maint_base(maint_base), .maint_size(maint_size), \
        .maint_done(P``_maint_done), .maint_busy(P``_maint_busy), .req_ready(P``_req_ready), .req_valid(req_valid), .req_addr(req_addr), .req_we(req_we), \
        .req_be(req_be), .req_wdata(req_wdata), .req_src(req_src), .req_tag(req_tag), .req_amo(req_amo), .req_aop(req_aop), .req_cmp(req_cmp), \
        .rsp_valid(P``_rsp_valid), .rsp_ready(rsp_ready), .rsp_src(P``_rsp_src), .rsp_tag(P``_rsp_tag), .rsp_rdata(P``_rsp_rdata), \
        .mem_req_valid(P``_mem_req_valid), .mem_req_ready(mem_req_ready), .mem_req_addr(P``_mem_req_addr), .mem_req_we(P``_mem_req_we), \
        .mem_req_wdata(P``_mem_req_wdata), .mem_req_tag(P``_mem_req_tag), .mem_rsp_valid(mem_rsp_valid), .mem_rsp_tag(mem_rsp_tag), .mem_rsp_rdata(mem_rsp_rdata), \
        .dco_cfg(dco_cfg), .dco_owner_epoch(1'b0), .dco_t_valid(8'd0), .dco_t_base(512'd0), .dco_t_size(512'd0), .dco_t_end(520'd0), .dco_t_bypass(8'd0), \
        .dco_t_lsb(48'd0), .dco_dead_valid(16'd0), .dco_dead_id(256'd0), .dco_slow(16'd0), \
        .tmu_wr_valid(P``_tmu_wr_valid), .tmu_wr_ready(tmu_wr_ready), .tmu_wr_off(P``_tmu_wr_off), .tmu_wr_data(P``_tmu_wr_data), .tmu_wr_src(P``_tmu_wr_src), \
        .tmu_rd_off(P``_tmu_rd_off), .tmu_rd_data(tmu_rd_data), .tmu_tl_valid(P``_tmu_tl_valid), .tmu_tl_ready(1'b1), .tmu_tl_addr(P``_tmu_tl_addr), .tmu_tl_drop(P``_tmu_tl_drop), \
        .perf_bypass(P``_p0), .perf_evict(P``_p1), .perf_dead_evict(P``_p2), .perf_gear(P``_gear), .perf_hit(P``_p3), .perf_miss(P``_p4), .perf_merge(P``_p5), \
        .perf_mshr_full(P``_p6), .perf_target_full(P``_p7), .perf_wb(P``_p8), .perf_hit_past_block(P``_p9)

    tnca_home_pipe #(`HP_PARAMS, .DATA_RAM(0))  ua (`HP_PORTS(a));
    tnca_home_pipe #(`HP_PARAMS, .DATA_RAM(DR), .AUX_RAM(AR)) ub (`HP_PORTS(b));

    // ---- random numbers ----
    reg [63:0] rng;
    function [63:0] xs; input [63:0] x; reg [63:0] y; begin y = x; y = y ^ (y << 13); y = y ^ (y >> 7); y = y ^ (y << 17); xs = y; end endfunction
    task rnd; begin rng = xs(rng); end endtask
    function integer pct; input integer p; input [63:0] r; begin pct = ((r >> 20) % 100) < p; end endfunction

    // ---- memory model (driven by A) ----
    reg [511:0] mem [0:4095];
    function [511:0] line_init; input [11:0] l; reg [63:0] h; integer k; begin
        h = {52'd0, l} * 64'h9E3779B97F4A7C15 + 64'h1234567; for (k = 0; k < 8; k = k + 1) begin h = xs(h); line_init[k*64 +: 64] = h; end end endfunction
    integer i;
    reg          pend_v [0:63]; reg [MSHR_W-1:0] pend_tag [0:63]; reg [511:0] pend_data [0:63]; integer pend_t [0:63];
    integer now;

    // ---- request generation ----
    reg pending, maint_pend, maint_want; reg [ADDR_W-1:0] line;
    integer cyc, mism, n_rsp, n_mrsp, n_maint, n_mdone, n_mmio, n_full, n_amo, n_reqs;
    reg [255:0] why;

    // compare vector
    wire a_eq_ctl = ({a_req_ready, a_rsp_valid, a_mem_req_valid, a_mem_req_we, a_maint_ready, a_maint_busy, a_maint_done, a_tmu_wr_valid, a_tmu_tl_valid, a_tmu_tl_drop,
                      a_rsp_src, a_rsp_tag, a_tmu_wr_src, a_mem_req_addr, a_mem_req_tag, a_tmu_wr_off, a_tmu_rd_off, a_tmu_wr_data, a_tmu_tl_addr,
                      a_p0, a_p1, a_p2, a_gear, a_p3, a_p4, a_p5, a_p6, a_p7, a_p8, a_p9} ===
                     {b_req_ready, b_rsp_valid, b_mem_req_valid, b_mem_req_we, b_maint_ready, b_maint_busy, b_maint_done, b_tmu_wr_valid, b_tmu_tl_valid, b_tmu_tl_drop,
                      b_rsp_src, b_rsp_tag, b_tmu_wr_src, b_mem_req_addr, b_mem_req_tag, b_tmu_wr_off, b_tmu_rd_off, b_tmu_wr_data, b_tmu_tl_addr,
                      b_p0, b_p1, b_p2, b_gear, b_p3, b_p4, b_p5, b_p6, b_p7, b_p8, b_p9});
    // data outputs only matter when valid, but compare them always: both instances must agree bit for bit
    wire a_eq_dat = ({a_rsp_rdata, a_mem_req_wdata} === {b_rsp_rdata, b_mem_req_wdata});

    integer k, r;
    reg [63:0] lr;
    initial begin
        rng = 64'hC0FFEE0000000000 ^ (SEED * 64'h9E3779B97F4A7C15) ^ 64'h1;
        for (i = 0; i < 4096; i = i + 1) mem[i] = line_init(i[11:0]);
        for (i = 0; i < 64; i = i + 1) pend_v[i] = 0;
        req_valid = 0; req_addr = 0; req_we = 0; req_be = 0; req_wdata = 0; req_src = 0; req_tag = 0; req_amo = 0; req_aop = 0; req_cmp = 0;
        rsp_ready = 0; mem_req_ready = 0; mem_rsp_valid = 0; mem_rsp_tag = 0; mem_rsp_rdata = 0; tmu_wr_ready = 0; tmu_rd_data = 0;
        maint_valid = 0; maint_base = 0; maint_size = 0; pending = 0; maint_pend = 0; maint_want = 0;
        n_mdone = 0; cyc = 0; mism = 0; n_rsp = 0; n_mrsp = 0; n_maint = 0; n_mmio = 0; n_full = 0; n_amo = 0; n_reqs = 0; now = 0;
        repeat (4) @(posedge clk);
        #1 rst_n = 1;
    end

    // inputs change just after each rising edge; outputs of both are compared just before the next edge
    always @(posedge clk) if (rst_n) begin
        #1;
        now = now + 1;
        // 1. events of the edge that just happened (A's outputs sampled before the edge are what fired; use the registered view)
        // handled below in the negedge block; here the new stimulus is drawn
        rnd; rsp_ready = pct(P_RSP, rng);
        rnd; mem_req_ready = pct(P_MEM, rng);
        rnd; tmu_wr_ready = pct(70, rng);
        rnd; tmu_rd_data = rng;
        // request: offered until accepted (the accepted-edge bookkeeping is in the negedge block)
        if (!pending && !maint_want) begin
            rnd;
            if (pct(P_REQ, rng)) begin
                pending = 1; n_reqs = n_reqs + 1;
                rnd; lr = rng;
                if (HOT && pct(80, lr)) line = (lr >> 40) % 16; else line = (lr >> 24) % NLINES;
                rnd;
                if (pct(P_MMIO, rng)) begin req_addr = {TMU_BASE[ADDR_W-1:12], 12'd0} | (((rng >> 32) % 512) * 8); n_mmio = n_mmio + 1; end
                else req_addr = (line << 6);
                rnd; lr = rng;
                req_amo = pct(P_AMO, lr);
                req_we = pct(P_WR, lr >> 8);
                if (req_amo) begin n_amo = n_amo + 1; req_aop = (lr >> 16) & 15; req_we = (lr >> 12) & 1; end else req_aop = 0;
                rnd; lr = rng;
                req_be = {64{1'b0}};
                if (req_we || req_amo) begin
                    if (!req_amo && pct(P_FULL, lr)) begin req_be = {64{1'b1}}; n_full = n_full + 1; end
                    else begin
                        // random contiguous or scattered byte masks; atomics: 4 or 8 byte groups
                        rnd;
                        if (req_amo) begin
                            req_be = 64'd0;
                            case ((rng >> 10) & 3)
                              0: req_be[((rng >> 20) % 8)*8 +: 8] = 8'hFF;
                              1: req_be[((rng >> 20) % 16)*4 +: 4] = 4'hF;
                              2: req_be = 64'hFF << (((rng >> 20) % 8)*8);
                              3: req_be = 64'h0F << (((rng >> 20) % 16)*4);
                            endcase
                        end else begin
                            case ((rng >> 10) & 3)
                              0: req_be = rng ^ (rng << 17);
                              1: req_be = ({64{1'b1}} << ((rng >> 20) % 64));
                              2: req_be = 64'hFFFF << (((rng >> 20) % 48));
                              3: req_be = 64'd1 << ((rng >> 20) % 64);
                            endcase
                        end
                    end
                end else if (pct(50, lr)) req_be = {64{1'b1}};   // read: be is ignored by the home; vary it anyway
                for (k = 0; k < 8; k = k + 1) begin rnd; req_wdata[k*64 +: 64] = rng; end
                rnd; req_cmp = {448'd0, rng};
                rnd; req_src = rng[SRC_W-1:0]; req_tag = rng[24 +: TAG_W];
            end
        end
        req_valid = pending;
        // maintenance request (held until the home takes it)
        // maintenance: stop offering requests, wait until the home reports idle (maint_ready), then offer the range once
        maint_valid = maint_pend;
        if (MAINT_EN && !maint_want && !maint_pend) begin
            rnd;
            if (((rng >> 24) % 1000) < P_MAINT) maint_want = 1;
        end
        if (maint_want && !pending && !maint_pend && a_maint_ready) begin
            maint_pend = 1; maint_valid = 1; maint_want = 0; rnd;
            maint_base = ((rng >> 8) % NLINES) << 6; maint_size = ((rng >> 32) % (NLINES/2 + 1)) << 6;
            if (((rng >> 50) & 7) == 0) maint_size = 0;
            if (((rng >> 53) & 7) == 0) maint_size = NLINES << 6;
            n_maint = n_maint + 1;
        end
        // memory response: pick a random pending one that is old enough
        mem_rsp_valid = 0;
        rnd;
        if (pct(P_MRSP, rng)) begin
            r = (rng >> 30) % 64;
            for (k = 0; k < 64 && !mem_rsp_valid; k = k + 1)
                if (pend_v[(r + k) % 64] && pend_t[(r + k) % 64] <= now) begin
                    mem_rsp_valid = 1; mem_rsp_tag = pend_tag[(r + k) % 64]; mem_rsp_rdata = pend_data[(r + k) % 64];
                    pend_v[(r + k) % 64] = 0; n_mrsp = n_mrsp + 1;
                end
        end
    end

    // the events of one edge, seen from A's outputs and the inputs held through the edge, are applied at the falling edge before it
    // (inputs are stable from just after the previous rising edge)
    always @(negedge clk) if (rst_n) begin
        cyc = cyc + 1;
        // compare
        if (!(a_eq_ctl && a_eq_dat)) begin
            mism = mism + 1;
            if (mism <= 3) begin
                $display("MISMATCH at cycle %0d (SEED %0d DR %0d AR %0d)", cyc, SEED, DR, AR);
                $display("  A: req_ready %b rsp_valid %b rsp_tag %h src %h mem_req_valid %b we %b addr %h tag %h maint r%b d%b b%b", a_req_ready, a_rsp_valid, a_rsp_tag, a_rsp_src, a_mem_req_valid, a_mem_req_we, a_mem_req_addr, a_mem_req_tag, a_maint_ready, a_maint_done, a_maint_busy);
                $display("  B: req_ready %b rsp_valid %b rsp_tag %h src %h mem_req_valid %b we %b addr %h tag %h maint r%b d%b b%b", b_req_ready, b_rsp_valid, b_rsp_tag, b_rsp_src, b_mem_req_valid, b_mem_req_we, b_mem_req_addr, b_mem_req_tag, b_maint_ready, b_maint_done, b_maint_busy);
                $display("  rsp_rdata A %h", a_rsp_rdata); $display("  rsp_rdata B %h", b_rsp_rdata);
                $display("  mem_wdata A %h", a_mem_req_wdata); $display("  mem_wdata B %h", b_mem_req_wdata); $display("  ctl_eq %b dat_eq %b", a_eq_ctl, a_eq_dat);
                $display("  A state: s3_v %b s3_kind %0d s3_index %0d s3_wb %b | B s3_v %b s3_kind %0d s3_index %0d s3_wb %b", ua.s3_v, ua.s3_kind, ua.s3_index, ua.s3_wb, ub.s3_v, ub.s3_kind, ub.s3_index, ub.s3_wb);
                $display("  perf A hit %0d miss %0d merge %0d wb %0d  B hit %0d miss %0d merge %0d wb %0d", a_p3, a_p4, a_p5, a_p8, b_p3, b_p4, b_p5, b_p8);
            end
            if (mism >= 3) begin $display("MITER FAIL"); $finish; end
        end
        // handshakes that fire at the coming rising edge (inputs stable, outputs of A settled)
        if (req_valid && a_req_ready) pending = 0;
        if (a_rsp_valid && rsp_ready) n_rsp = n_rsp + 1;
        if (maint_valid && a_maint_ready) maint_pend = 0;
        if (a_maint_done) n_mdone = n_mdone + 1;
        if (a_mem_req_valid && mem_req_ready) begin
            if (a_mem_req_we) mem[a_mem_req_addr[17:6]] = a_mem_req_wdata;
            else begin
                for (i = 0; i < 64; i = i + 1) if (!pend_v[i] && !(i < 0)) begin
                    // a free slot: record the response (tag, data of the memory now, ready time)
                    pend_v[i] = 1; pend_tag[i] = a_mem_req_tag; pend_data[i] = mem[a_mem_req_addr[17:6]];
                    rnd; pend_t[i] = now + 1 + ((rng >> 12) % MAXLAT);
                    i = 64;
                end
            end
        end
        if (cyc >= CYCLES) begin
            $display("  state: maint_valid %b maint_pend %b ready %b | rq_v %h s1_v %b s3_v %b rsq_cnt %0d tg_v!=0 %b ms_all_free %b fu_v %b wbq_cnt %0d m_state %0d mem_req_valid %b", maint_valid, maint_pend, a_maint_ready, ua.rq_v, ua.s1_v, ua.s3_v, ua.rsq_cnt, ua.tg_v != 0, ua.ms_all_free, ua.fu_v, ua.wbq_cnt, ua.m_state, a_mem_req_valid);
            $display("MITER %s: DR=%0d AR=%0d SEED=%0d cycles=%0d mismatches=%0d | requests=%0d responses=%0d memrsp=%0d maint=%0d(done %0d) mmio=%0d full=%0d amo=%0d | A: hit=%0d miss=%0d merge=%0d evict=%0d wb=%0d mshr_full=%0d target_full=%0d",
                     mism == 0 ? "PASS" : "FAIL", DR, AR, SEED, cyc, mism, n_reqs, n_rsp, n_mrsp, n_maint, n_mdone, n_mmio, n_full, n_amo, a_p3, a_p4, a_p5, a_p1, a_p8, a_p6, a_p7);
            $finish;
        end
    end
endmodule
