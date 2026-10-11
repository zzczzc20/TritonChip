//==============================================================================
// tb_axi_bridge -- rtl/fpga/tnca_axi_bridge.v against a reference, under an
// AXI memory that is as unhelpful as AXI allows
//
// R40-04 acceptance 3: out-of-order responses, random backpressure, sub-word
// stores to one word, atomics, out-of-range accesses. Random requests over a
// small pool of addresses (so they collide) in both regions -- the on-chip RAM
// below LOCAL_LIMIT and the AXI window above it -- each checked, at its
// response, against a reference memory that applies every request in the
// order the bridge ACCEPTED it, which is what the port promises; at the end,
// every word of both memories against the reference.
//
// The AXI slave below makes ordering mistakes visible: it samples a read's
// data when it sends the R beat and applies a write when it sends the B,
// each after a random latency, picks among pending reads at random when
// +reorder is set, and drops each READY with probability +bp percent. A
// bridge that let a read pass a write to the same word, or a write pass a
// read, would return a value the reference does not.
//
// +err: the slave answers SLVERR for the last ERR_WORDS words of the pool in
// the window (a read with zero data, a write dropped). Those words are then
// checked like words past the window, and axi_err must be set at the end
// exactly when one of them was accessed, with nothing hung.
//
// +rst: the bridge is reset RST_TIMES times in the middle of the traffic,
// with requests in flight on AXI, and the slave -- which is not reset --
// still answers them. No answer may reach the requester after a reset until
// it asks again; the reference is re-read from both memories once the
// bridge's quiet window is over, and checking resumes from there. A reset
// is placed only where no write is being offered: the reset is synchronous,
// so a write offered on its edge can have AW taken and W not, which is the
// one thing the bridge cannot recover (its header says why).
//
// +dma: now and then, with nothing outstanding, words of the window are
// written behind the bridge's back -- the host's DMA -- and `inv` is raised,
// as tnca_mmio raises it for the command that follows a DMA. Every later read
// must see them, whatever the bridge had buffered or was still fetching.
//
// Built with -DBRIDGE_LB=<beats> it tests rtl/fpga/tnca_axi_bridge_lb.v
// (LB_SLOTS lines, LB_PF ahead, LB_WAIT waiters; -DLB_SLOTS= -DLB_PF=
// -DLB_WAIT= to change them; -DBR_OUT= sets either bridge's OUT_N), whose
// reads are bursts; the slave below answers ARLEN+1 beats, interleaving them
// across IDs under +reorder. Under +err a refused word's read value is not
// checked there (a buffered line may hold what a refused write tried to put).
//
//   vvp tb_axi_bridge.vvp +seed=N +n=20000 +lat_min=1 +lat_max=40 +reorder +bp=30 [+err] [+rst] [+dma]
//==============================================================================
`timescale 1ns/1ps
`default_nettype none
`include "tnca_defs.vh"

`ifndef LB_SLOTS
`define LB_SLOTS 4
`endif
`ifndef LB_PF
`define LB_PF 1
`endif
`ifndef LB_WAIT
`define LB_WAIT 8
`endif
`ifndef BR_OUT
`define BR_OUT 8
`endif

module tb_axi_bridge;
`ifdef BRIDGE_LB
    localparam LB_MODE = 1;
`else
    localparam LB_MODE = 0;
`endif
    localparam TAG_W  = 5;
    localparam LOCAL_WORDS = 128;                 // 1 KiB of on-chip RAM
    localparam LOCAL_LIMIT = 64'h400;
    localparam AXI_WORDS   = 256;                 // 2 KiB window
    localparam [63:0] AXI_BASE = 64'h1_0000_0000;
    localparam POOL = 24;                          // words requests choose from, per region
    localparam ERR_WORDS = 3;                      // +err: pool words answered with SLVERR
    localparam QUIET = 1024;                       // the bridge's quiet window after reset
    localparam RST_TIMES = 4;                      // +rst: resets in one run

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    integer seed, seed0, n_req, lat_min, lat_max, bp, reorder, err_mode, rst_mode, dma_mode;
    initial begin
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        seed0 = seed;
        if (!$value$plusargs("n=%d", n_req)) n_req = 20000;
        if (!$value$plusargs("lat_min=%d", lat_min)) lat_min = 1;
        if (!$value$plusargs("lat_max=%d", lat_max)) lat_max = 20;
        if (!$value$plusargs("bp=%d", bp)) bp = 0;
        reorder = $test$plusargs("reorder");
        err_mode = $test$plusargs("err");
        rst_mode = $test$plusargs("rst");
        dma_mode = $test$plusargs("dma");
    end

    // +err: the window words the slave refuses
    function err_word;
        input [63:0] axi_a;
        reg [63:0] w;
        begin
            w = (axi_a - AXI_BASE) >> 3;
            err_word = err_mode && w >= POOL - ERR_WORDS && w < POOL;
        end
    endfunction

    // ---- the bridge ----------------------------------------------------------
    reg              req_valid;
    wire             req_ready;
    reg [63:0]       req_addr;
    reg              req_we, req_amo;
    reg [7:0]        req_be;
    reg [63:0]       req_wdata, req_cmp;
    reg [TAG_W-1:0]  req_tag;
    reg [3:0]        req_aop;
    wire             rsp_valid;
    wire [TAG_W-1:0] rsp_tag;
    wire [63:0]      rsp_rdata;

    wire l_req_valid, l_req_ready, l_req_we, l_req_amo, l_rsp_valid;
    wire [63:0] l_req_addr, l_req_wdata, l_req_cmp, l_rsp_rdata;
    wire [7:0] l_req_be;
    wire [TAG_W-1:0] l_req_tag, l_rsp_tag;
    wire [3:0] l_req_aop;

    wire m_awvalid, m_wvalid, m_arvalid, m_wlast, axi_err, axi_stray;
    wire [63:0] cnt_rd, cnt_wr, cnt_rd_lat;
    integer s_ar, s_aw;                     // the slave's own count since the last reset
    reg  [1:0] m_bresp = 2'b00, m_rresp = 2'b00;
    reg  m_awready, m_wready, m_arready;
    wire [63:0] m_awaddr, m_araddr, m_wdata;
    wire [7:0] m_wstrb, m_awlen, m_arlen;
    wire [2:0] m_awsize, m_arsize;
    wire [1:0] m_awburst, m_arburst;
    wire [TAG_W-1:0] m_awid, m_arid;
    reg  m_bvalid, m_rvalid, m_rlast = 1'b1;
    reg  inv = 1'b0;
    wire [63:0] cnt_rbeat;
    wire m_bready, m_rready;
    reg  [TAG_W-1:0] m_bid, m_rid;
    reg  [63:0] m_rdata;

`ifdef BRIDGE_LB
    tnca_axi_bridge_lb #(.TAG_W(TAG_W), .OUT_N(`BR_OUT), .LOCAL_LIMIT(LOCAL_LIMIT), .QUIET(QUIET),
                         .LINE_BEATS(`BRIDGE_LB), .LB_N(`LB_SLOTS), .WAIT_N(`LB_WAIT), .PF(`LB_PF)) dut (
`else
    tnca_axi_bridge #(.TAG_W(TAG_W), .OUT_N(`BR_OUT), .LOCAL_LIMIT(LOCAL_LIMIT), .QUIET(QUIET)) dut (
`endif
        .clk(clk), .rst_n(rst_n),
        .req_valid(req_valid), .req_ready(req_ready), .req_addr(req_addr), .req_we(req_we),
        .req_be(req_be), .req_wdata(req_wdata), .req_tag(req_tag), .req_amo(req_amo),
        .req_aop(req_aop), .req_cmp(req_cmp),
        .rsp_valid(rsp_valid), .rsp_tag(rsp_tag), .rsp_rdata(rsp_rdata),
        .l_req_valid(l_req_valid), .l_req_ready(l_req_ready), .l_req_addr(l_req_addr),
        .l_req_we(l_req_we), .l_req_be(l_req_be), .l_req_wdata(l_req_wdata), .l_req_tag(l_req_tag),
        .l_req_amo(l_req_amo), .l_req_aop(l_req_aop), .l_req_cmp(l_req_cmp),
        .l_rsp_valid(l_rsp_valid), .l_rsp_tag(l_rsp_tag), .l_rsp_rdata(l_rsp_rdata),
        .axi_base(AXI_BASE), .axi_size(64'd8 * AXI_WORDS), .inv(inv), .axi_err(axi_err),
        .axi_stray(axi_stray),
        .cnt_rd(cnt_rd), .cnt_wr(cnt_wr), .cnt_rd_lat(cnt_rd_lat), .cnt_rbeat(cnt_rbeat),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awid(m_awid),
        .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb),
        .m_wlast(m_wlast), .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bid(m_bid),
        .m_bresp(m_bresp),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_araddr(m_araddr), .m_arid(m_arid),
        .m_arlen(m_arlen), .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rdata(m_rdata), .m_rid(m_rid),
        .m_rresp(m_rresp), .m_rlast(m_rlast)
    );

    tnca_fpga_mem #(.WORDS(LOCAL_WORDS), .ADDR_W(7), .TAG_W(TAG_W)) u_local (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(64'd0), .imem_data(),
        .req_valid(l_req_valid), .req_ready(l_req_ready), .req_addr(l_req_addr),
        .req_we(l_req_we), .req_be(l_req_be), .req_wdata(l_req_wdata), .req_tag(l_req_tag),
        .req_amo(l_req_amo), .req_aop(l_req_aop), .req_cmp(l_req_cmp),
        .rsp_valid(l_rsp_valid), .rsp_tag(l_rsp_tag), .rsp_rdata(l_rsp_rdata)
    );

    // ---- the AXI memory ------------------------------------------------------
    reg [63:0] axi_mem [0:AXI_WORDS-1];
    // pending reads: address, id, cycles left
    reg [63:0] pr_addr [0:31]; reg [TAG_W-1:0] pr_id [0:31]; integer pr_t [0:31]; integer pr_n;
    integer pr_len [0:31]; integer pr_beat [0:31];
    reg [63:0] ba;
    // pending writes: address and strobe/data (AW and W joined), cycles left
    reg [63:0] aw_q [0:31]; integer aw_n;
    reg [63:0] w_q_data [0:31]; reg [7:0] w_q_strb [0:31]; integer w_n;
    reg [63:0] pw_addr [0:31]; reg [63:0] pw_data [0:31]; reg [7:0] pw_strb [0:31]; integer pw_t [0:31]; integer pw_n;
    integer i, j, pick;

    function [63:0] merge;
        input [63:0] old, data; input [7:0] be;
        integer b;
        begin
            merge = old;
            for (b = 0; b < 8; b = b + 1) if (be[b]) merge[b*8 +: 8] = data[b*8 +: 8];
        end
    endfunction

    function integer lat;
        input integer dummy;
        begin lat = lat_min + (($random(seed) & 32'h7fffffff) % (lat_max - lat_min + 1)); end
    endfunction

    function rnd_ready;
        input integer dummy;
        begin rnd_ready = (($random(seed) & 32'h7fffffff) % 100) >= bp; end
    endfunction

    reg slave_rst_n = 0;                    // the slave is reset once, at the start
    always @(posedge clk) begin
        if (!slave_rst_n) begin
            pr_n = 0; aw_n = 0; w_n = 0; pw_n = 0;
            m_rvalid <= 0; m_bvalid <= 0;
            m_arready <= 0; m_awready <= 0; m_wready <= 0;
        end else begin
            // channel acceptance
            if (m_arvalid && m_arready && rst_n) s_ar = s_ar + 1;
            if (m_awvalid && m_awready && rst_n) s_aw = s_aw + 1;
            if (m_arvalid && m_arready) begin
                pr_addr[pr_n] = m_araddr; pr_id[pr_n] = m_arid; pr_t[pr_n] = lat(0);
                pr_len[pr_n] = m_arlen; pr_beat[pr_n] = 0; pr_n = pr_n + 1;
            end
            if (m_awvalid && m_awready) begin aw_q[aw_n] = m_awaddr; aw_n = aw_n + 1; end
            if (m_wvalid && m_wready) begin
                w_q_data[w_n] = m_wdata; w_q_strb[w_n] = m_wstrb; w_n = w_n + 1;
            end
            while (aw_n > 0 && w_n > 0) begin
                pw_addr[pw_n] = aw_q[0]; pw_data[pw_n] = w_q_data[0]; pw_strb[pw_n] = w_q_strb[0];
                pw_t[pw_n] = lat(0); pw_n = pw_n + 1;
                for (j = 1; j < aw_n; j = j + 1) aw_q[j-1] = aw_q[j];
                for (j = 1; j < w_n; j = j + 1) begin w_q_data[j-1] = w_q_data[j]; w_q_strb[j-1] = w_q_strb[j]; end
                aw_n = aw_n - 1; w_n = w_n - 1;
            end
            // R: a beat taken, or a new one offered
            if (m_rvalid && m_rready) m_rvalid <= 0;
            if (!m_rvalid || m_rready) begin
                for (i = 0; i < pr_n; i = i + 1) if (pr_t[i] > 0) pr_t[i] = pr_t[i] - 1;
                pick = -1;
                for (i = 0; i < pr_n; i = i + 1)
                    if (pr_t[i] == 0 && (pick < 0 || (reorder && ($random(seed) & 1)))) pick = i;
                if (pick >= 0) begin
                    ba = pr_addr[pick] + pr_beat[pick] * 8;
                    m_rvalid <= 1; m_rid <= pr_id[pick];
                    // the data as it stands NOW, not as it stood at AR
                    m_rdata <= err_word(ba) ? 64'd0 : axi_mem[(ba - AXI_BASE) >> 3];
                    m_rresp <= err_word(ba) ? 2'b10 : 2'b00;
                    m_rlast <= pr_beat[pick] == pr_len[pick];
                    if (pr_beat[pick] < pr_len[pick]) begin
                        pr_beat[pick] = pr_beat[pick] + 1;
                    end else begin
                        for (j = pick + 1; j < pr_n; j = j + 1) begin
                            pr_addr[j-1] = pr_addr[j]; pr_id[j-1] = pr_id[j]; pr_t[j-1] = pr_t[j];
                            pr_len[j-1] = pr_len[j]; pr_beat[j-1] = pr_beat[j];
                        end
                        pr_n = pr_n - 1;
                    end
                end
            end
            // B: in order (one ID), the write applied only now
            if (m_bvalid && m_bready) m_bvalid <= 0;
            if (!m_bvalid || m_bready) begin
                for (i = 0; i < pw_n; i = i + 1) if (pw_t[i] > 0) pw_t[i] = pw_t[i] - 1;
                if (pw_n > 0 && pw_t[0] == 0) begin
                    if (!err_word(pw_addr[0]))
                        axi_mem[(pw_addr[0] - AXI_BASE) >> 3] =
                            merge(axi_mem[(pw_addr[0] - AXI_BASE) >> 3], pw_data[0], pw_strb[0]);
                    m_bvalid <= 1; m_bid <= 0;
                    m_bresp <= err_word(pw_addr[0]) ? 2'b10 : 2'b00;
                    for (j = 1; j < pw_n; j = j + 1) begin
                        pw_addr[j-1] = pw_addr[j]; pw_data[j-1] = pw_data[j];
                        pw_strb[j-1] = pw_strb[j]; pw_t[j-1] = pw_t[j];
                    end
                    pw_n = pw_n - 1;
                end
            end
            m_arready <= rnd_ready(0) && pr_n < 30;
            m_awready <= rnd_ready(0) && aw_n < 30;
            m_wready  <= rnd_ready(0) && w_n < 30;
        end
    end

    // ---- the reference, in acceptance order ----------------------------------
    reg [63:0] ref_local [0:LOCAL_WORDS-1];
    reg [63:0] ref_axi   [0:AXI_WORDS-1];
    reg [63:0] expect_data [0:(1<<TAG_W)-1];
    reg        expect_check [0:(1<<TAG_W)-1];
    reg        tag_busy [0:(1<<TAG_W)-1];
    integer issued, answered, errors, n_local, n_axi, n_amo, n_oob, n_err, n_rst, err_since, n_dma;

    `include "tnca_amo.vh"

    task accept_ref;
        input [63:0] addr; input we, amo; input [7:0] be; input [63:0] wdata, cmp;
        input [3:0] aop; input [TAG_W-1:0] tag;
        reg [63:0] old;
        reg loc, oob, bad;
        integer w;
        begin
            loc = addr < LOCAL_LIMIT;
            oob = !loc && (addr - LOCAL_LIMIT) >= AXI_WORDS * 8;
            w = loc ? (addr >> 3) : ((addr - LOCAL_LIMIT) >> 3);
            // a word the slave refuses reads as zero and keeps nothing, like one
            // past the window; it is counted apart, since it sets axi_err
            bad = !loc && !oob && err_word(AXI_BASE + w * 8);
            if (bad) begin n_err = n_err + 1; err_since = err_since + 1; end
            oob = oob || bad;
            old = oob ? 64'd0 : (loc ? ref_local[w] : ref_axi[w]);
            expect_data[tag] = old;
            expect_check[tag] = (!we || amo) && !(bad && LB_MODE);  // a store's answer carries nothing
            if (!oob) begin
                if (amo) begin
                    if (aop != `TNCA_AOP_LD) begin
                        if (loc) ref_local[w] = merge(old, amo_new(aop, be, old, wdata, cmp), be);
                        else     ref_axi[w]   = merge(old, amo_new(aop, be, old, wdata, cmp), be);
                    end
                end else if (we) begin
                    if (loc) ref_local[w] = merge(old, wdata, be);
                    else     ref_axi[w]   = merge(old, wdata, be);
                end
            end
            if (loc) n_local = n_local + 1; else if (oob && !bad) n_oob = n_oob + 1; else n_axi = n_axi + 1;
            if (amo) n_amo = n_amo + 1;
        end
    endtask

    // ---- +rst: a reset with AXI requests in flight ----------------------------
    // Called at a negedge; returns at one.
    task reset_in_flight;
        input integer dummy;
        integer q;
        begin
            rst_n = 0; req_valid = 0;
            for (q = 0; q < (1<<TAG_W); q = q + 1) tag_busy[q] = 0;
            answered = issued;                       // forgotten, as the core forgets them
            repeat (3) @(posedge clk);
            @(negedge clk);
            rst_n = 1;
            for (q = 0; q < QUIET + 8; q = q + 1) begin
                @(posedge clk);
                if (rsp_valid) begin
                    $display("ERROR: tag %0d answered %0d cycles after a reset", rsp_tag, q);
                    errors = errors + 1;
                end
            end
            if (pr_n || pw_n || aw_n || w_n) begin
                $display("ERROR: the slave still holds work %0d cycles after a reset", QUIET);
                errors = errors + 1;
            end
            // what the memories hold now is the new reference
            for (q = 0; q < AXI_WORDS; q = q + 1) ref_axi[q] = axi_mem[q];
            for (q = 0; q < LOCAL_WORDS; q = q + 1) ref_local[q] = u_local.mem[q];
            err_since = 0;
            s_ar = 0; s_aw = 0;
            n_rst = n_rst + 1;
            @(negedge clk);
        end
    endtask

    // ---- +dma: memory written behind the bridge, then inv --------------------
    // Called at a negedge with nothing outstanding; returns at one.
    task dma_behind;
        input integer dummy;
        integer q, w;
        reg [63:0] v;
        begin
            // the writes ahead of us have reached the memory (B sent = applied)
            while (pw_n != 0 || aw_n != 0 || w_n != 0 || dut.t_n != 0 || dut.m_awvalid || dut.m_wvalid)
                @(negedge clk);
            for (q = 0; q < 6; q = q + 1) begin
                w = ($random(seed) & 32'h7fffffff) % (POOL + 8);   // mostly where the traffic is
                v = {$random(seed), $random(seed)};
                if (!err_word(AXI_BASE + w * 8)) begin
                    axi_mem[w] = v; ref_axi[w] = v;
                end
            end
            inv = 1'b1;
            @(negedge clk);
            inv = 1'b0;
            n_dma = n_dma + 1;
        end
    endtask

    // ---- the requester -------------------------------------------------------
    function [63:0] pick_addr;
        input integer dummy;
        integer r, wsel;
        begin
            r = ($random(seed) & 32'h7fffffff) % 100;
            wsel = ($random(seed) & 32'h7fffffff) % POOL;
            if (r < 30)      pick_addr = wsel * 8 + 64'h100;                      // on-chip
            else if (r < 97) pick_addr = LOCAL_LIMIT + wsel * 8;                  // AXI
            else             pick_addr = LOCAL_LIMIT + AXI_WORDS * 8 + wsel * 8;  // past the window
        end
    endfunction

    reg [TAG_W-1:0] t;
    integer r, k2, free_found, cyc;
    initial begin
        for (i = 0; i < LOCAL_WORDS; i = i + 1) begin ref_local[i] = 0; u_local.mem[i] = 0; end
        for (i = 0; i < AXI_WORDS; i = i + 1) begin ref_axi[i] = 0; axi_mem[i] = 0; end
        for (i = 0; i < (1<<TAG_W); i = i + 1) tag_busy[i] = 0;
        issued = 0; answered = 0; errors = 0; n_local = 0; n_axi = 0; n_amo = 0; n_oob = 0;
        n_err = 0; n_rst = 0; err_since = 0; s_ar = 0; s_aw = 0; n_dma = 0;
        req_valid = 0;
        repeat (4) @(posedge clk);
        rst_n = 1; slave_rst_n = 1;
        @(posedge clk);
        cyc = 0;
        while ((issued < n_req || answered < issued) && cyc < n_req * 200) begin
            @(negedge clk);
            if (rst_mode && n_rst < RST_TIMES && issued >= (n_rst + 1) * (n_req / (RST_TIMES + 1))
                    && (dut.t_n != 0 || dut.a_st != 0) && !dut.m_awvalid && !dut.m_wvalid)
                reset_in_flight(0);
            if (dma_mode && answered == issued && !req_valid && issued / (n_req / 32 + 1) > n_dma)
                dma_behind(0);
            if (!req_valid && issued < n_req) begin
                free_found = 0;
                for (k2 = 0; k2 < (1<<TAG_W) && !free_found; k2 = k2 + 1)
                    if (!tag_busy[k2]) begin t = k2; free_found = 1; end
                if (free_found && (($random(seed) & 3) != 0)) begin
                    r = ($random(seed) & 32'h7fffffff) % 100;
                    req_addr  = pick_addr(0);
                    req_we    = r < 45;
                    req_amo   = r >= 85;
                    if (req_amo) req_we = 1;
                    case (($random(seed) & 32'h7fffffff) % 4)
                    0: req_be = 8'hff;
                    1: req_be = 8'h0f;
                    2: req_be = 8'hf0;
                    default: req_be = req_amo ? 8'hff : ($random(seed) & 8'hff);
                    endcase
                    req_wdata = {$random(seed), $random(seed)};
                    req_cmp   = (($random(seed) & 1) ? 64'd0 : {$random(seed), $random(seed)});
                    case (($random(seed) & 32'h7fffffff) % 6)
                    0: req_aop = `TNCA_AOP_ADD;
                    1: req_aop = `TNCA_AOP_XCHG;
                    2: req_aop = `TNCA_AOP_CAS;
                    3: req_aop = `TNCA_AOP_MAX_S;
                    4: req_aop = `TNCA_AOP_LD;
                    default: req_aop = `TNCA_AOP_OR;
                    endcase
                    req_tag = t;
                    req_valid = 1;
                end
            end
            @(posedge clk);
            if (rsp_valid) begin
                if (!tag_busy[rsp_tag]) begin
                    $display("ERROR: a response for tag %0d nobody is waiting for", rsp_tag);
                    errors = errors + 1;
                end else if (expect_check[rsp_tag] && rsp_rdata !== expect_data[rsp_tag]) begin
                    $display("ERROR: tag %0d got %h expected %h", rsp_tag, rsp_rdata, expect_data[rsp_tag]);
                    errors = errors + 1;
                end
                tag_busy[rsp_tag] = 0;
                answered = answered + 1;
            end
            if (req_valid && req_ready) begin
                accept_ref(req_addr, req_we, req_amo, req_be, req_wdata, req_cmp, req_aop, req_tag);
                tag_busy[req_tag] = 1;
                issued = issued + 1;
                req_valid = 0;
            end
            cyc = cyc + 1;
        end
        repeat (lat_max * 4 + 50) @(posedge clk);
        for (i = 0; i < AXI_WORDS; i = i + 1)
            if (axi_mem[i] !== ref_axi[i]) begin
                if (errors < 20) $display("ERROR: AXI word %0d is %h, reference %h", i, axi_mem[i], ref_axi[i]);
                errors = errors + 1;
            end
        for (i = 0; i < LOCAL_WORDS; i = i + 1)
            if (u_local.mem[i] !== ref_local[i]) begin
                if (errors < 20) $display("ERROR: RAM word %0d is %h, reference %h", i, u_local.mem[i], ref_local[i]);
                errors = errors + 1;
            end
        if (issued != n_req || answered != issued) begin
            $display("ERROR: %0d issued, %0d answered of %0d (a request or a response was lost)",
                     issued, answered, n_req);
            errors = errors + 1;
        end
        if (axi_err !== (err_since > 0)) begin
            $display("ERROR: axi_err is %0d after %0d refused accesses since the last reset", axi_err, err_since);
            errors = errors + 1;
        end
        if (axi_stray) begin $display("ERROR: axi_stray set"); errors = errors + 1; end
        if (cnt_rd != s_ar || cnt_wr != s_aw) begin
            $display("ERROR: counters say %0d reads and %0d writes, the slave took %0d and %0d",
                     cnt_rd, cnt_wr, s_ar, s_aw);
            errors = errors + 1;
        end
        if (s_ar > 0 && cnt_rd_lat < cnt_rd) begin
            $display("ERROR: %0d read-cycles in flight for %0d reads", cnt_rd_lat, cnt_rd);
            errors = errors + 1;
        end
        if (rst_mode && n_rst != RST_TIMES) begin
            $display("ERROR: %0d of %0d resets placed", n_rst, RST_TIMES);
            errors = errors + 1;
        end
        $display("tb_axi_bridge%s seed=%0d n=%0d lat=%0d..%0d reorder=%0d bp=%0d%% err=%0d rst=%0d dma=%0d: %0d on-chip, %0d AXI, %0d out of range, %0d refused, %0d atomics, %0d resets, %0d dmas, %0d ARs, %0d R beats, %0d cycles: %s",
                 LB_MODE ? "[lb]" : "", seed0, n_req, lat_min, lat_max, reorder, bp, err_mode, rst_mode, dma_mode,
                 n_local, n_axi, n_oob, n_err, n_amo, n_rst, n_dma, cnt_rd, cnt_rbeat, cyc, errors ? "FAIL" : "PASS");
        $finish;
    end
endmodule
`default_nettype wire
