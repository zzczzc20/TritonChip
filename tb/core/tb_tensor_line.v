// R41 G4: the line-wide, multi-command transfer engine (tnca_tensor
// LINE_ENGINE=1) against the original word engine (LINE_ENGINE=0), as a miter.
//
// Both engines run the same descriptors, windows and transfers -- rank 1 and
// 2, fp32, line-aligned and unaligned bases, in-bounds and partly outside the
// tensor -- against their own memory (64-bit words for the original, 64-byte
// lines for the line engine) with random latency, out-of-order responses and
// a register file that refuses writes at random. The original engine runs one
// command at a time; the line engine is given the next command as soon as it
// is free, so line commands overlap. Commands that could interfere are
// ordered by waiting for completions, as a program does with its barriers.
// At the end both memories, both register files and both completion counts
// must be identical, and the line engine must have moved lines.
`timescale 1ns/1ps
`default_nettype none
module tb_tensor_line;
parameter SEED = 1;
parameter NCMD = 400;
parameter MAXLAT = 40;
parameter VW_STALL = 20;      // % of cycles the register file refuses a write
parameter REQ_STALL = 20;     // % of cycles the memory refuses a request
// XPERF 1: instead of the random program, the compiled Hub FP8 matmul's weight
// transfer (DECODE_PROFILE.md part 1): 1-byte elements, source rows K = 2048 B
// apart, a 16-row x 32-byte box loaded transposed, each transfer waiting for
// the previous one (mb.arm / mb.wait), XN transfers along K. FIXLAT > 0 gives
// every memory response that latency (no random latency). Reports each
// engine's cycles.
parameter XPERF = 0;
parameter XN = 32;
parameter FIXLAT = 0;
// XRACE (with XPERF's matrix): another requester writes the weight lines.
// 1: between two transposed loads of the same box, at the point where the
// first has completed and the second has not started (a store ordered before
// the second command) -- both engines must return the new bytes, so the line
// buffer must not keep lines across commands. 2: while the second load is in
// flight, XRACE_AT cycles after it starts -- an unordered race, which the ISA
// leaves open; the run reports how many window bytes differ between the two
// engines instead of comparing them.
parameter XRACE = 0;
parameter XRACE_AT = 40;
localparam MEMB = 65536;      // bytes of memory
localparam DESC_AT = 32'h0000; // TensorMaps at 0x0000.. (16 x 512 B)
// every third map is not line-aligned, so the word sequence still runs
function integer desc_addr(input integer slot);
    desc_addr = DESC_AT + slot * 512 + ((slot % 3 == 1) ? 24 : 0);
endfunction

reg clk = 0; always #5 clk = ~clk;
reg rst_n = 0;
integer cycle = 0; always @(posedge clk) cycle <= cycle + 1;

// ---- the program, built before reset and run by each engine on its own ----
// kind 0: td.load of slot a; 1: w.def window a base b len c; 2: load, 3: store
// (descriptor a, window b, coordinates c0/c1). An entry starts once the engine
// has signalled `wait_done` completions (a program's barrier waits).
localparam PMAX = 2048;
reg [1:0]  p_kind [0:PMAX-1];
integer    p_a [0:PMAX-1], p_b [0:PMAX-1], p_c [0:PMAX-1], p_wait [0:PMAX-1];
reg [63:0] p_c0 [0:PMAX-1], p_c1 [0:PMAX-1];
reg [2:0]  p_esz [0:PMAX-1];
reg        p_x [0:PMAX-1];      // a transposed load (X8 path: 16-bit and 1-byte elements)
integer    plen, ncomp;   // entries; completions the whole program signals

// ---- per instance ----
wire [1:0] busy, done, done_ben, fault, done_err;
wire [4:0] done_bar [0:1];
reg  [7:0] mem [0:1][0:MEMB-1];
reg  [511:0] vrf [0:1][0:255];
integer ndone [0:1];
integer nfault [0:1];
integer nx;   // transposed loads issued
integer lines_moved;
integer fin [0:1];   // XPERF: the cycle each engine finished its program

genvar g;
generate for (g = 0; g < 2; g = g + 1) begin : inst
    localparam LEN = g;                      // 0: word engine, 1: line engine
    localparam PDW = LEN ? 512 : 64;
    localparam PBY = LEN ? 64 : 8;
    localparam PTW = LEN ? 6 : 4;
    wire vw_en; wire [0:0] vw_rctx; wire [7:0] vw_idx; wire [15:0] vw_lane; wire [511:0] vw_data;
    reg  vw_ready;
    wire vr_req; wire [0:0] vr_rctx; wire [7:0] vr_idx; reg vr_valid; reg [511:0] vr_data;
    wire mreq_v; reg mreq_r; wire [63:0] mreq_a; wire mreq_we; wire [PBY-1:0] mreq_be;
    wire [PDW-1:0] mreq_wd; wire [PTW-1:0] mreq_tag;
    reg  mrsp_v; reg [PTW-1:0] mrsp_tag; reg [PDW-1:0] mrsp_rd;
    wire bany; wire [1:0] infl;
    wire [7:0] fcode; wire [0:0] frctx; wire [63:0] fpc;
    // this engine's program counter and the entry it presents
    integer pc = 0;
    wire        pe   = rst_n && (pc < plen) && (ndone[g] >= p_wait[pc]) && !busy[g];
    wire        cfg_v = pe && (p_kind[pc] <= 2'd1);
    wire [1:0]  cfg_op = (p_kind[pc] == 2'd0) ? 2'd0 : 2'd2;
    wire [3:0]  cfg_desc = p_a[pc];
    wire [4:0]  cfg_win = p_a[pc];
    wire [63:0] cfg_addr = desc_addr(p_a[pc]);
    wire [7:0]  cfg_wbase = p_b[pc], cfg_wlen = p_c[pc];
    wire        cmd_v = pe && (p_kind[pc] >= 2'd2);
    wire        cmd_store = (p_kind[pc] == 2'd3);
    wire [3:0]  cmd_desc = p_a[pc];
    wire [4:0]  cmd_win = p_b[pc];
    wire [63:0] cmd_c0 = p_c0[pc], cmd_c1 = p_c1[pc];
    wire [2:0]  cmd_esz = p_esz[pc];
    wire        cmd_xpose = p_x[pc];
    wire [4:0]  cmd_bar = pc % 32;
    always @(posedge clk) if (pe) pc <= pc + 1;
    tnca_tensor #(.VLEN(16), .ELEN(32), .RCTX_W(1), .BAR_W(5), .MEM_DW(64), .MEM_BYTES(8),
                  .TAG_W(4), .NTAG(16), .HAS_XPOSE(2), .LINE_ENGINE(LEN), .LSLOT_N(32), .LSLOT_W(5),
                  .CMD_N(8), .CMD_W(3), .PTAG_W(PTW), .PORT_DW(PDW), .PORT_BYTES(PBY)) dut (
        .clk(clk), .rst_n(rst_n), .flush(1'b0),
        .cfg_valid(cfg_v), .cfg_op(cfg_op), .cfg_desc(cfg_desc), .cfg_win(cfg_win),
        .cfg_addr(cfg_addr), .cfg_wbase(cfg_wbase), .cfg_wlen(cfg_wlen),
        .cmd_valid(cmd_v), .cmd_store(cmd_store), .cmd_desc(cmd_desc), .cmd_win(cmd_win),
        .cmd_c0(cmd_c0), .cmd_c1(cmd_c1), .cmd_c2(64'd0), .cmd_c3(64'd0), .cmd_c4(64'd0),
        .cmd_sb(6'd1), .cmd_esz(cmd_esz), .cmd_xpose(cmd_xpose), .cmd_layout(2'd0),
        .cmd_bar_en(1'b1), .cmd_bar(cmd_bar), .cmd_rctx(1'b0), .cmd_pc(64'd0), .cmd_active(1'b1),
        .busy(busy[g]), .done(done[g]), .done_bar_en(done_ben[g]), .done_bar(done_bar[g]),
        .fault(fault[g]), .fault_code(fcode), .fault_rctx(frctx), .fault_pc(fpc), .done_err(done_err[g]),
        .vw_en(vw_en), .vw_rctx(vw_rctx), .vw_idx(vw_idx), .vw_lane_en(vw_lane), .vw_data(vw_data),
        .vw_ready(vw_ready),
        .vr_req(vr_req), .vr_rctx(vr_rctx), .vr_idx(vr_idx), .vr_valid(vr_valid), .vr_data(vr_data),
        .mem_req_valid(mreq_v), .mem_req_ready(mreq_r), .mem_req_addr(mreq_a), .mem_req_we(mreq_we),
        .mem_req_be(mreq_be), .mem_req_wdata(mreq_wd), .mem_req_tag(mreq_tag),
        .mem_rsp_valid(mrsp_v), .mem_rsp_tag(mrsp_tag), .mem_rsp_rdata(mrsp_rd),
        .busy_any(bany), .inflight_rctx(infl));

    // register file: lane-enabled writes, refused at random; reads a cycle later
    integer seed_v = SEED * 31 + g * 7 + 1;
    always @(negedge clk) vw_ready = ($unsigned($random(seed_v)) % 100) >= VW_STALL;
    always @(posedge clk) if (rst_n) begin
        if (vw_en && vw_ready) begin : wr
            integer l;
            for (l = 0; l < 16; l = l + 1)
                if (vw_lane[l]) vrf[g][vw_idx][l*32 +: 32] <= vw_data[l*32 +: 32];
        end
        vr_valid <= vr_req && !vr_valid;
        vr_data  <= vrf[g][vr_idx];
    end

    // memory: functional at acceptance, responses after a random delay, any order
    integer seed_m = SEED * 17 + g * 5 + 3;
    reg [PTW-1:0] q_tag [0:63]; reg [PDW-1:0] q_d [0:63]; integer q_due [0:63]; reg q_v [0:63];
    integer k, b, n, pick;
    always @(negedge clk) mreq_r = ($unsigned($random(seed_m)) % 100) >= REQ_STALL;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (k = 0; k < 64; k = k + 1) q_v[k] = 0;
            mrsp_v <= 0;
        end else begin
            if (mreq_v && mreq_r) begin : acc
                reg [63:0] base; reg [PDW-1:0] rd;
                base = LEN ? {mreq_a[63:6], 6'd0} : {mreq_a[63:3], 3'd0};
                if (base + PBY > MEMB) $fatal(1, "inst %0d: access past memory %h", g, mreq_a);
                rd = 0;
                for (b = 0; b < PBY; b = b + 1) begin
                    rd[b*8 +: 8] = mem[g][base + b];
                    if (mreq_we && mreq_be[b]) mem[g][base + b] = mreq_wd[b*8 +: 8];
                end
                if (LEN && !mreq_we) lines_moved = lines_moved + (mreq_tag[PTW-1] ? 1 : 0);
                if (LEN && mreq_we) lines_moved = lines_moved + (mreq_tag[PTW-1] ? 1 : 0);
                pick = -1;
                for (k = 63; k >= 0; k = k - 1) if (!q_v[k]) pick = k;
                if (pick < 0) $fatal(1, "inst %0d: response queue full", g);
                q_v[pick] = 1; q_tag[pick] = mreq_tag; q_d[pick] = rd;
                q_due[pick] = cycle + 1 + ((FIXLAT > 0) ? FIXLAT - 1 : ($unsigned($random(seed_m)) % MAXLAT));
            end
            mrsp_v <= 0;
            pick = -1; n = 0;
            for (k = 0; k < 64; k = k + 1)
                if (q_v[k] && q_due[k] <= cycle) begin
                    n = n + 1;
                    if (pick < 0 || ($unsigned($random(seed_m)) % n) == 0) pick = k;
                end
            if (pick >= 0) begin
                mrsp_v <= 1; mrsp_tag <= q_tag[pick]; mrsp_rd <= q_d[pick]; q_v[pick] = 0;
            end
        end
    end

    // XRACE: the other requester's store to the 16 weight lines (every byte inverted)
    integer xr_l, xr_b, xr_t2 = -1, xr_one = 0;
    always @(posedge clk) if (rst_n && XRACE) begin
        // once the first load has completed (whichever order this block and the
        // completion count run in at that edge); the second starts a cycle later
        if (XRACE == 1 && !xr_one && ndone[g] >= ncomp - 1) begin
            xr_one = 1;
            for (xr_l = 0; xr_l < 16; xr_l = xr_l + 1)
                for (xr_b = 0; xr_b < 64; xr_b = xr_b + 1)
                    mem[g][32'h8000 + xr_l * 2048 + xr_b] = ~mem[g][32'h8000 + xr_l * 2048 + xr_b];
        end
        if (XRACE == 2 && cmd_v && pc == plen - 1 && xr_t2 < 0) xr_t2 = cycle;
        if (XRACE == 2 && xr_t2 >= 0 && cycle == xr_t2 + XRACE_AT)
            for (xr_l = 0; xr_l < 16; xr_l = xr_l + 1)
                for (xr_b = 0; xr_b < 64; xr_b = xr_b + 1)
                    mem[g][32'h8000 + xr_l * 2048 + xr_b] = ~mem[g][32'h8000 + xr_l * 2048 + xr_b];
    end
    always @(posedge clk) if (rst_n) begin
        if (done[g] && ndone[g] + 1 == ncomp) fin[g] = cycle;
        if (done[g]) ndone[g] = ndone[g] + 1;
        if (fault[g]) nfault[g] = nfault[g] + 1;
    end
end endgenerate

// ---- the program ----
integer seed = SEED, i, j, k2, w, d, t, grp, gsize, nreg, ncyc;
integer rb [0:15], rs [0:15], re0 [0:15], re1 [0:15], rrank [0:15];   // slot d and d+8: region d
integer bl0 [0:15], bl1 [0:15];   // each slot's box (rows, row elements)
integer sesz [0:15], sdt [0:15];  // element size (log2 bytes) and dtype code: fp32, fp16, fp8
integer bb0, bb1;

task put64(input integer a, input [63:0] v);
    integer bb;
    begin
        for (bb = 0; bb < 8; bb = bb + 1) begin
            mem[0][a + bb] = v[bb*8 +: 8]; mem[1][a + bb] = v[bb*8 +: 8];
        end
    end
endtask

task put32(input integer a, input [31:0] v);
    integer bb;
    begin
        for (bb = 0; bb < 4; bb = bb + 1) begin
            mem[0][a + bb] = v[bb*8 +: 8]; mem[1][a + bb] = v[bb*8 +: 8];
        end
    end
endtask

// A TensorMap (ISA sec 9D.3): base, extents[5], strides[5] (bytes), block[5]
// (32-bit each), rank (byte 108), dtype (109: 2 = fp32), padding (110: zero).
task make_desc(input integer slot);
    integer a;
    begin
        a = desc_addr(slot);
        for (k2 = 0; k2 < 256; k2 = k2 + 8) put64(a + k2, 64'd0);
        put64(a + 0, rb[slot]);
        if (rrank[slot] == 1) begin
            put64(a + 8, re1[slot]);  put64(a + 48, 1 << sesz[slot]);
        end else begin
            put64(a + 8, re0[slot]);  put64(a + 16, re1[slot]);
            put64(a + 48, rs[slot]);  put64(a + 56, 1 << sesz[slot]);
        end
        if (rrank[slot] == 1) put64(a + 88, {32'd0, bl1[slot][31:0]});
        else                  put64(a + 88, {bl1[slot][31:0], bl0[slot][31:0]});
        mem[0][a + 108] = rrank[slot]; mem[1][a + 108] = rrank[slot];
        mem[0][a + 109] = sdt[slot];   mem[1][a + 109] = sdt[slot];
    end
endtask

task add(input [1:0] kind, input integer a, input integer b, input integer c,
         input [63:0] c0, input [63:0] c1, input integer wt);
    begin
        p_esz[plen] = (kind >= 2) ? sesz[a] : 3'd2; p_x[plen] = 1'b0;
        p_kind[plen] = kind; p_a[plen] = a; p_b[plen] = b; p_c[plen] = c;
        p_c0[plen] = c0; p_c1[plen] = c1; p_wait[plen] = wt;
        if (kind != 2'd1) ncomp = ncomp + 1;   // a w.def signals nothing
        plen = plen + 1;
    end
endtask

// A box for slot d's rank: rank 1 a row of bb1 elements, rank 2 bb0 rows of bb1.
task pick_box(input integer slot);
    integer rb_;
    begin
        case ($unsigned($random(seed)) % 6)
        0: rb_ = 64; 1: rb_ = 128; 2: rb_ = 256; 3: rb_ = 192; 4: rb_ = 32; default: rb_ = 8;
        endcase
        if (rrank[slot] == 1) begin
            bb0 = 1;
            if ($unsigned($random(seed)) % 2) rb_ = rb_ * 4;           // up to 1 KiB
        end else
            bb0 = 1 + $unsigned($random(seed)) % ((1024 / rb_) > 8 ? 8 : (1024 / rb_));
        bb1 = rb_ >> sesz[slot];
        // transposed-capable slots (8..15, 16-bit and 1-byte rank 2): 4, 8 or 16 rows
        if (slot >= 8 && rrank[slot] == 2 && sesz[slot] <= 1 && rs[slot] % 8 == 0 && rb[slot] % 8 == 0) begin
            case ($unsigned($random(seed)) % 3) 0: bb0 = 4; 1: bb0 = 8; default: bb0 = 16; endcase
            if (bb0 * rb_ > 1024) rb_ = 1024 / bb0;
            bb1 = rb_ >> sesz[slot];
        end
        if (bb1 > 256) bb1 = 256;          // a block dimension is at most 256 elements
    end
endtask

function [63:0] coord(input integer ext, input integer box);
    integer c;
    begin
        // mostly inside, sometimes straddling either edge
        case ($unsigned($random(seed)) % 12)
        0: c = -($unsigned($random(seed)) % 4) - 1;
        1: c = ext - box + 1 + $unsigned($random(seed)) % 4;
        default: c = (ext > box) ? $unsigned($random(seed)) % (ext - box + 1) : 0;
        endcase
        // whole-line boxes stay line-aligned when the coordinate is a multiple of 16
        if (($unsigned($random(seed)) % 3) != 0 && c >= 0) c = c & ~15;
        coord = c;
    end
endfunction

initial begin
    plen = 0; ncomp = 0; lines_moved = 0; nx = 0;
    ndone[0] = 0; ndone[1] = 0; nfault[0] = 0; nfault[1] = 0;
    for (i = 0; i < MEMB; i = i + 1) begin
        mem[0][i] = $random(seed); mem[1][i] = mem[0][i];
    end
    for (i = 0; i < 256; i = i + 1) begin
        vrf[0][i] = {16{$random(seed)}}; vrf[1][i] = vrf[0][i];
    end
    // Eight regions of 4 KiB from 0x8000; slot d describes region d. Some
    // bases and row strides are line multiples, some are not.
    for (d = 0; d < 16; d = d + 1) begin
        // element sizes 4, 2, 1 bytes; the aligned rank-2 slots 9 and 15 are
        // 16-bit and 1-byte so transposed loads through them are legal
        sesz[d] = (d == 9) ? 1 : (d == 15) ? 0 : (d % 3 == 0) ? 2 : (d % 3 == 1) ? 1 : 0;
        sdt[d]  = (sesz[d] == 2) ? 2 : (sesz[d] == 1) ? 0 : 7;
        if (d < 8) begin
            rrank[d] = (d % 2) ? 2 : 1;
            rb[d] = 32'h8000 + d * 4096 + ((d == 3) ? 4 * (1 + $unsigned($random(seed)) % 15) : 0);
            if (rrank[d] == 1) begin
                re1[d] = (2048 + $unsigned($random(seed)) % 1600) >> sesz[d]; re0[d] = 1; rs[d] = 0;
            end else begin
                rs[d] = (d == 5) ? 196 : 256;              // slot 5: a stride that is not a line multiple
                re1[d] = (rs[d] >> sesz[d]) - ($unsigned($random(seed)) % 2) * 3;
                re0[d] = 3000 / rs[d];
            end
        end else begin
            rrank[d] = rrank[d-8]; rb[d] = rb[d-8]; rs[d] = rs[d-8]; re0[d] = re0[d-8];
            // the same region seen with this slot's element size
            re1[d] = (rrank[d] == 1) ? ((re1[d-8] << sesz[d-8]) >> sesz[d]) : ((rs[d] >> sesz[d]) - 1);
        end
        pick_box(d);
        bl0[d] = bb0; bl1[d] = bb1;
        make_desc(d);
        add(0, d, 0, 0, 0, 0, ncomp);
    end
    if (XPERF) begin
        // slot 15: the weight matrix, fp8, 16 rows of 2048 bytes from 0x8000
        plen = 0; ncomp = 0;
        sesz[15] = 0; sdt[15] = 7; rrank[15] = 2; rb[15] = 32'h8000; rs[15] = 2048;
        re0[15] = 16; re1[15] = 2048; bl0[15] = 16; bl1[15] = 32;
        make_desc(15);
        add(0, 15, 0, 0, 0, 0, ncomp);
        add(1, 0, 0, 16, 0, 0, ncomp);
        if (XRACE) add(1, 1, 16, 16, 0, 0, ncomp);
        for (t = 0; t < (XRACE ? 2 : XN); t = t + 1) begin
            j = ncomp;
            add(2'd2, 15, (XRACE ? t : 0), 0, 64'd0, (XRACE ? 0 : t * 32), j);
            p_x[plen-1] = 1'b1;
            p_wait[plen] = ncomp;
        end
    end else begin
    // Eight windows of 16 registers (64 lanes x 16 = 256 fp32).
    for (w = 0; w < 8; w = w + 1) add(1, w, w * 16, 16, 0, 0, ncomp);
    // Groups: loads into distinct windows from anywhere, then stores from
    // distinct windows into distinct regions; every group waits for the last.
    t = 0;
    for (grp = 0; plen < PMAX - 20 && t < NCMD; grp = grp + 1) begin
        j = ncomp;      // completions before this group
        gsize = 1 + $unsigned($random(seed)) % 8;
        for (w = 0; w < gsize && t < NCMD; w = w + 1) begin
            // loads: any slot; stores: region w, through slot w or w + 8
            d = (grp % 2 == 0) ? $unsigned($random(seed)) % 16 : w + 8 * ($unsigned($random(seed)) % 2);
            add((grp % 2 == 0) ? 2'd2 : 2'd3, d, w, 0,
                (rrank[d] == 1) ? coord(re1[d], bl1[d]) : coord(re0[d], bl0[d]),
                (rrank[d] == 1) ? 64'd0 : coord(re1[d], bl1[d]), j);
            // some loads through a 16-bit or 1-byte rank-2 slot are transposed (X8)
            // (slots 11 and 13 are not transposable: their faults must match too)
            if (grp % 2 == 0 && d >= 8 && rrank[d] == 2 && sesz[d] <= 1 &&
                $unsigned($random(seed)) % ((d == 9 || d == 15) ? 2 : 6) == 0)
            begin
                p_x[plen-1] = 1'b1; nx = nx + 1;
                // whole 8-byte source units: column offset 8-byte aligned
                p_c1[plen-1] = p_c1[plen-1] & ~((64'd8 >> sesz[d]) - 64'd1);
                // half of them a whole-line box inside the tensor, which the
                // line path would take if it did not exclude transposed loads
                if ($unsigned($random(seed)) % 2 && re0[d] >= bl0[d] && re1[d] >= bl1[d] + (64 >> sesz[d])) begin
                    p_c0[plen-1] = $unsigned($random(seed)) % (re0[d] - bl0[d] + 1);
                    p_c1[plen-1] = ($unsigned($random(seed)) % ((re1[d] - bl1[d]) / (64 >> sesz[d]) + 1)) * (64 >> sesz[d]);
                end
            end
            t = t + 1;
        end
        // the next group starts after all of this one
        p_wait[plen] = ncomp;
    end
    end
    repeat (4) @(posedge clk); rst_n = 1;
    ncyc = 0;
    while (!(inst[0].pc == plen && inst[1].pc == plen && ndone[0] == ncomp && ndone[1] == ncomp
             && !busy[0] && !busy[1] && !inst[1].bany)) begin
        @(posedge clk); ncyc = ncyc + 1;
        if (ncyc > 3000000) $fatal(1, "timeout: pc %0d/%0d of %0d, done %0d/%0d of %0d",
                                inst[0].pc, inst[1].pc, plen, ndone[0], ndone[1], ncomp);
    end
    repeat (4) @(posedge clk);
    if (nfault[0] != nfault[1]) $fatal(1, "faults differ: %0d vs %0d", nfault[0], nfault[1]);
    for (i = 0; i < MEMB; i = i + 1)
        if (mem[0][i] !== mem[1][i]) $fatal(1, "memory byte %h differs: %h vs %h", i, mem[0][i], mem[1][i]);
    if (XRACE == 2) begin : race_report
        integer nb;
        nb = 0;
        for (i = 16; i < 32; i = i + 1)
            for (j = 0; j < 64; j = j + 1)
                if (vrf[0][i][j*8 +: 8] !== vrf[1][i][j*8 +: 8]) nb = nb + 1;
        $display("XRACE 2: store %0d cycles into the second load; window bytes that differ between the engines: %0d of 512 (both legal: the ISA does not order an unordered store against a command in flight)", XRACE_AT, nb);
    end else
    for (i = 0; i < 256; i = i + 1)
        if (vrf[0][i] !== vrf[1][i]) $fatal(1, "register %0d differs:\n%h\n%h", i, vrf[0][i], vrf[1][i]);
    if (lines_moved == 0 && !XPERF) $fatal(1, "the line engine moved no lines");
    if (XPERF) $display("XPERF transfers=%0d fixlat=%0d word_engine_cycles=%0d line_engine_cycles=%0d weight_bytes=%0d",
                        XN, FIXLAT, fin[0], fin[1], XN * 512);
    $display("TENSOR LINE MITER PASS seed=%0d commands=%0d completions=%0d faults=%0d xpose=%0d lines=%0d cycles=%0d",
             SEED, t, ncomp, nfault[0], nx, lines_moved, ncyc);
    $finish;
end
endmodule
`default_nettype wire
