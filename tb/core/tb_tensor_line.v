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
//
// R42 lineelig: LHALF 1 builds the line engine with LINE_HALF 1 (half-line rows:
// 32-byte pieces); HBIAS 1 adds boxes that only the half-line path moves (rows
// of 32, 96, 160, 224 bytes, 32-byte-aligned starts, strides of 96 and 224
// bytes, a base on a 32-byte boundary) and counts the pieces moved. LOCK 1 runs
// instead the engine as it was (tnca_tensor_ref, a renamed copy of the base
// commit's file) beside the current one with LINE_HALF 0, both line engines
// and the same environment, and compares every output of the two on every
// cycle: the default parameter keeps the engine cycle-identical.
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
parameter LHALF = 0;          // R42: LINE_HALF of the line engine
parameter HBIAS = 0;          // R42: boxes of 32-byte pieces
parameter DIRECTED = 0;       // R42: a fixed list of boxes, each with the number of 32-byte pieces the half-line path must move for it
parameter LSW = 5;            // line slots 2^LSW (the run-2 image: 6); the line engine's port tag is LSW + 1 bits
parameter LOCK = 0;           // R42: cycle-exact comparison of the reference and the current engine (both LINE_ENGINE 1, LINE_HALF 0)
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
reg        p_act [0:PMAX-1];    // R42: the command's predicate (an inactive command moves nothing)
integer    exp_cum [0:PMAX-1];  // R42 DIRECTED: pieces the half-line path must have moved when completion k has been signalled
integer    dcum;
integer    gate = 0;           // R42 DIRECTED: completions both engines have signalled and the checker has compared
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
integer halves_moved, halves_st;   // R42: piece requests of the half-line path, and of its stores
integer fin [0:1];   // XPERF: the cycle each engine finished its program

genvar g;
generate for (g = 0; g < 2; g = g + 1) begin : inst
    localparam LEN = LOCK ? 1 : g;           // 0: word engine, 1: line engine
    localparam REF = LOCK && (g == 0);       // R42 LOCK: the engine as it was
    localparam PDW = LEN ? 512 : 64;
    localparam PBY = LEN ? 64 : 8;
    localparam PTW = LEN ? LSW + 1 : 4;
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
    wire        pe   = rst_n && (pc < plen) && ((DIRECTED ? gate : ndone[g]) >= p_wait[pc]) && !busy[g];
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
    `define TL_PORTS \
        .clk(clk), .rst_n(rst_n), .flush(1'b0), \
        .cfg_valid(cfg_v), .cfg_op(cfg_op), .cfg_desc(cfg_desc), .cfg_win(cfg_win), \
        .cfg_addr(cfg_addr), .cfg_wbase(cfg_wbase), .cfg_wlen(cfg_wlen), \
        .cmd_valid(cmd_v), .cmd_store(cmd_store), .cmd_desc(cmd_desc), .cmd_win(cmd_win), \
        .cmd_c0(cmd_c0), .cmd_c1(cmd_c1), .cmd_c2(64'd0), .cmd_c3(64'd0), .cmd_c4(64'd0), \
        .cmd_sb(6'd1), .cmd_esz(cmd_esz), .cmd_xpose(cmd_xpose), .cmd_layout(2'd0), \
        .cmd_bar_en(1'b1), .cmd_bar(cmd_bar), .cmd_rctx(1'b0), .cmd_pc(64'd0), .cmd_active(p_act[pc]), \
        .busy(busy[g]), .done(done[g]), .done_bar_en(done_ben[g]), .done_bar(done_bar[g]), \
        .fault(fault[g]), .fault_code(fcode), .fault_rctx(frctx), .fault_pc(fpc), .done_err(done_err[g]), \
        .vw_en(vw_en), .vw_rctx(vw_rctx), .vw_idx(vw_idx), .vw_lane_en(vw_lane), .vw_data(vw_data), \
        .vw_ready(vw_ready), \
        .vr_req(vr_req), .vr_rctx(vr_rctx), .vr_idx(vr_idx), .vr_valid(vr_valid), .vr_data(vr_data), \
        .mem_req_valid(mreq_v), .mem_req_ready(mreq_r), .mem_req_addr(mreq_a), .mem_req_we(mreq_we), \
        .mem_req_be(mreq_be), .mem_req_wdata(mreq_wd), .mem_req_tag(mreq_tag), \
        .mem_rsp_valid(mrsp_v), .mem_rsp_tag(mrsp_tag), .mem_rsp_rdata(mrsp_rd), \
        .busy_any(bany), .inflight_rctx(infl)
    if (REF) begin : u_ref
        tnca_tensor_ref #(.VLEN(16), .ELEN(32), .RCTX_W(1), .BAR_W(5), .MEM_DW(64), .MEM_BYTES(8),
                      .TAG_W(4), .NTAG(16), .HAS_XPOSE(2), .LINE_ENGINE(LEN), .LSLOT_N(1 << LSW), .LSLOT_W(LSW),
                      .CMD_N(8), .CMD_W(3), .PTAG_W(PTW), .PORT_DW(PDW), .PORT_BYTES(PBY)) dut (`TL_PORTS);
    end else begin : u_cur
        tnca_tensor #(.VLEN(16), .ELEN(32), .RCTX_W(1), .BAR_W(5), .MEM_DW(64), .MEM_BYTES(8),
                      .TAG_W(4), .NTAG(16), .HAS_XPOSE(2), .LINE_ENGINE(LEN), .LINE_HALF(LEN ? LHALF : 0), .LSLOT_N(1 << LSW), .LSLOT_W(LSW),
                      .CMD_N(8), .CMD_W(3), .PTAG_W(PTW), .PORT_DW(PDW), .PORT_BYTES(PBY)) dut (`TL_PORTS);
    end
    // R42: piece requests of the half-line path (the line engine with LINE_HALF)
    if (!REF && LEN) begin : halfcnt
        always @(posedge clk) if (rst_n && mreq_v && mreq_r && mreq_tag[PTW-1] && u_cur.dut.hm) begin
            halves_moved = halves_moved + 1; if (mreq_we) halves_st = halves_st + 1;
        end
    end

    // register file: lane-enabled writes, refused at random; reads a cycle later
    integer seed_v = SEED * 31 + (LOCK ? 0 : g) * 7 + 1;
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
    integer seed_m = SEED * 17 + (LOCK ? 0 : g) * 5 + 3;
    reg [PTW-1:0] q_tag [0:255]; reg [PDW-1:0] q_d [0:255]; integer q_due [0:255]; reg q_v [0:255];
    integer k, b, n, pick;
    always @(negedge clk) mreq_r = ($unsigned($random(seed_m)) % 100) >= REQ_STALL;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (k = 0; k < 256; k = k + 1) q_v[k] = 0;
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
                for (k = 255; k >= 0; k = k - 1) if (!q_v[k]) pick = k;
                if (pick < 0) $fatal(1, "inst %0d: response queue full", g);
                q_v[pick] = 1; q_tag[pick] = mreq_tag; q_d[pick] = rd;
                q_due[pick] = cycle + 1 + ((FIXLAT > 0) ? FIXLAT - 1 : ($unsigned($random(seed_m)) % MAXLAT));
            end
            mrsp_v <= 0;
            pick = -1; n = 0;
            for (k = 0; k < 256; k = k + 1)
                if (q_v[k] && q_due[k] <= cycle) begin
                    n = n + 1;
                    if (pick < 0 || ($unsigned($random(seed_m)) % n) == 0) pick = k;
                end
            if (pick >= 0) begin
                mrsp_v <= 1; mrsp_tag <= q_tag[pick]; mrsp_rd <= q_d[pick]; q_v[pick] = 0;
            end
        end
    end

    // R42: the window read port's protocol -- a read request is held, with its register, until vr_valid
    reg vq_p = 1'b0; reg [7:0] vq_i;
    always @(posedge clk) if (rst_n) begin
        if (vq_p && (!vr_req || vr_idx !== vq_i)) $fatal(1, "inst %0d: a window read request was dropped or changed before vr_valid", g);
        vq_p <= vr_req && !vr_valid; vq_i <= vr_idx;
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
        if (DIRECTED && LHALF && LEN && !REF && done[g] && halves_moved != exp_cum[ndone[g] - 1])
            $fatal(1, "DIRECTED: after completion %0d the half-line path has moved %0d pieces, expected %0d", ndone[g] - 1, halves_moved, exp_cum[ndone[g] - 1]);
        if (fault[g]) nfault[g] = nfault[g] + 1;
    end
end endgenerate

// R42 LOCK: the two engines see the same environment and must agree on every output of every cycle
reg [1023:0] lk_what;
always @(posedge clk) if (LOCK && rst_n) begin
    lk_what = "";
    if (inst[0].vw_en !== inst[1].vw_en || (inst[0].vw_en && (inst[0].vw_rctx !== inst[1].vw_rctx || inst[0].vw_idx !== inst[1].vw_idx ||
        inst[0].vw_lane !== inst[1].vw_lane || inst[0].vw_data !== inst[1].vw_data))) lk_what = "vw";
    if (inst[0].vr_req !== inst[1].vr_req || (inst[0].vr_req && (inst[0].vr_idx !== inst[1].vr_idx || inst[0].vr_rctx !== inst[1].vr_rctx))) lk_what = "vr";
    if (inst[0].mreq_v !== inst[1].mreq_v || (inst[0].mreq_v && (inst[0].mreq_a !== inst[1].mreq_a || inst[0].mreq_we !== inst[1].mreq_we ||
        inst[0].mreq_be !== inst[1].mreq_be || inst[0].mreq_wd !== inst[1].mreq_wd || inst[0].mreq_tag !== inst[1].mreq_tag))) lk_what = "mem";
    if (busy[0] !== busy[1] || done[0] !== done[1] || done_ben[0] !== done_ben[1] || fault[0] !== fault[1] || done_err[0] !== done_err[1] ||
        inst[0].bany !== inst[1].bany || inst[0].infl !== inst[1].infl) lk_what = "status";
    if ((done[0] && (done_bar[0] !== done_bar[1])) || (fault[0] && (inst[0].fcode !== inst[1].fcode || inst[0].frctx !== inst[1].frctx ||
        inst[0].fpc !== inst[1].fpc))) lk_what = "done/fault";
    if (lk_what != "") $fatal(1, "LOCK: the engines differ at cycle %0d (%0s): the default parameter changed the engine", cycle, lk_what);
end

// R42 DIRECTED: after every command, once both engines have completed it and any held write is through, the
// memories and the register files of the two engines must be identical; only then does the next command start.
integer ck, cb, cr;
initial if (DIRECTED) begin
    wait (rst_n === 1'b1);
    for (ck = 1; ck <= ncomp; ck = ck + 1) begin
        while (!(ndone[0] >= ck && ndone[1] >= ck)) @(posedge clk);
        repeat (60) @(posedge clk);
        for (cb = 0; cb < MEMB; cb = cb + 1)
            if (mem[0][cb] !== mem[1][cb]) $fatal(1, "DIRECTED: after completion %0d memory byte %h differs: %h vs %h", ck, cb, mem[0][cb], mem[1][cb]);
        for (cr = 0; cr < 256; cr = cr + 1)
            if (vrf[0][cr] !== vrf[1][cr]) $fatal(1, "DIRECTED: after completion %0d register %0d differs:\n%h\n%h", ck, cr, vrf[0][cr], vrf[1][cr]);
        gate = ck;
    end
end

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

// R42 DIRECTED: a slot's geometry (element size log2, rank, base, row stride, extents, box)
task dslot(input integer d, input integer esz, input integer rank, input integer base, input integer stride,
           input integer e0, input integer e1, input integer b0, input integer b1);
    begin
        sesz[d] = esz; sdt[d] = (esz == 2) ? 2 : (esz == 1) ? 0 : 7; rrank[d] = rank; rb[d] = base;
        rs[d] = stride; re0[d] = e0; re1[d] = e1; bl0[d] = b0; bl1[d] = b1;
    end
endtask
// one more completion with `exp` pieces moved by the half-line path in it
task dnote(input integer pieces);
    begin
        dcum = dcum + pieces;
        exp_cum[ncomp - 1] = dcum;
    end
endtask
task dcmd(input [1:0] kind, input integer slot, input integer win, input integer c0, input integer c1, input integer pieces);
    begin
        if (rrank[slot] == 1) add(kind, slot, win, 0, c0, 64'd0, ncomp);
        else                  add(kind, slot, win, 0, c0, c1, ncomp);
        dnote(pieces);
        p_wait[plen] = ncomp;
    end
endtask

task add(input [1:0] kind, input integer a, input integer b, input integer c,
         input [63:0] c0, input [63:0] c1, input integer wt);
    begin
        p_esz[plen] = (kind >= 2) ? sesz[a] : 3'd2; p_x[plen] = 1'b0; p_act[plen] = 1'b1;
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
        // R42 HBIAS: rows that are whole 32-byte pieces but not whole lines
        if (HBIAS && ($unsigned($random(seed)) % 2)) begin
            case ($unsigned($random(seed)) % 5)
            0: rb_ = 32; 1: rb_ = 96; 2: rb_ = 160; 3: rb_ = 224; default: rb_ = 32;
            endcase
        end
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

function [63:0] coord(input integer ext, input integer box, input integer esz);
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
        // R42 HBIAS: or to a 32-byte boundary
        else if (HBIAS && ($unsigned($random(seed)) % 3) != 0 && c >= 0) c = c & ~((32 >> esz) - 1);
        coord = c;
    end
endfunction

initial begin
    plen = 0; ncomp = 0; lines_moved = 0; halves_moved = 0; halves_st = 0; nx = 0;
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
            rb[d] = 32'h8000 + d * 4096 + ((d == 3) ? (HBIAS ? 32 * (1 + $unsigned($random(seed)) % 7) : 4 * (1 + $unsigned($random(seed)) % 15)) : 0);
            if (rrank[d] == 1) begin
                re1[d] = (2048 + $unsigned($random(seed)) % 1600) >> sesz[d]; re0[d] = 1; rs[d] = 0;
            end else begin
                rs[d] = (d == 5) ? 196 : 256;              // slot 5: a stride that is not a line multiple
                if (HBIAS) rs[d] = (d == 5) ? 196 : (d == 7) ? 224 : (d == 1) ? 96 : 256;   // R42: strides of 32-byte multiples
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
    if (DIRECTED) begin : directed
        // R42: boxes with the number of 32-byte pieces the half-line path must move for each (the
        // engines run in step and are compared after every command). Memory is 64 KiB; the regions overlap.
        //   dslot(slot, esz log2, rank, base, row stride, extent0 (rows), extent1 (columns), box0, box1)
        integer dd;
        dcum = 0;
        for (dd = 0; dd < PMAX; dd = dd + 1) exp_cum[dd] = 0;
        dslot(0, 1, 2, 32'h8000, 512, 64, 256, 16, 16);        // fp16, 16-row tiles of 32-byte rows
        dslot(1, 1, 2, 32'h9000, 96, 64, 48, 16, 16);          // stride 96: a multiple of 32, not of 64
        dslot(2, 1, 2, 32'hA000, 48, 64, 24, 16, 16);          // stride 48: not a multiple of 32
        dslot(3, 1, 2, 32'hB010, 256, 64, 128, 8, 16);         // base on a 16-byte boundary
        dslot(4, 1, 2, 32'hC000, 256, 64, 128, 4, 48);         // rows of 96 bytes
        dslot(5, 1, 2, 32'hD000, 256, 20, 100, 16, 16);        // 20 rows of 100 columns: boxes can straddle the edges
        dslot(6, 2, 2, 32'hE000, 256, 32, 64, 8, 8);           // fp32, rows of 32 bytes
        dslot(7, 0, 2, 32'hF000, 256, 16, 256, 8, 32);         // fp8, rows of 32 bytes
        dslot(8, 1, 1, 32'h8800, 0, 1, 700, 1, 80);            // rank 1, 160 bytes
        dslot(9, 1, 1, 32'h9800, 0, 1, 700, 1, 16);            // rank 1, 32 bytes
        dslot(10, 1, 2, 32'hA800, 512, 32, 256, 5, 16);        // 5 rows of 32 bytes: the last register's second half is not written
        dslot(11, 1, 2, 32'hE800, 512, 8, 256, 8, 32);         // rows of 64 bytes
        for (dd = 0; dd < 12; dd = dd + 1) begin make_desc(dd); add(0, dd, 0, 0, 0, 0, ncomp); dnote(0); end
        for (dd = 0; dd < 16; dd = dd + 1) add(1, dd, dd * 16, 16, 0, 0, ncomp);
        //     kind slot win c0 c1 pieces
        dcmd(2,  0,  0,  0,   0, 16);   dcmd(2,  0,  1, 16,  16, 16);  dcmd(2,  0,  2,  3,  32, 16);
        dcmd(2,  0,  3,  0,   8,  0);   // 16-byte offset: the word path
        dcmd(2,  0,  4,  0, 252,  0);   // 12 columns past the edge: padding, the word path
        dcmd(2,  0,  5, 52,   0,  0);   // rows 52..67 of 64: past the edge
        dcmd(2,  0,  6, 48,   0, 16);   // rows 48..63: inside
        dcmd(2,  0,  7, -1,   0,  0);   // a negative row
        dcmd(2,  1,  8,  0,   0, 16);   dcmd(2,  1,  9,  1,  16, 16);  dcmd(2,  1, 10,  2,  32, 16);
        dcmd(2,  2, 11,  0,   0,  0);   // stride 48
        dcmd(2,  3, 12,  0,   0,  0);   // base + 0: 16-byte aligned
        dcmd(2,  3, 13,  0,   8,  8);   // base + 16: 32-byte aligned
        dcmd(2,  4, 14,  0,   0, 12);   // 4 rows of 96 bytes: three pieces a row
        dcmd(2,  4, 15,  1,  16, 12);   // 32-byte offset, 96-byte rows
        dcmd(2,  5,  0,  0,  96,  0);   // columns 96..111 of 100
        dcmd(2,  5,  1,  4,  32, 16);   // rows 4..19 of 20, columns 32..47
        dcmd(2,  5,  2,  4,  80, 16);   // columns 80..95
        dcmd(2,  6,  3,  0,   0,  8);   dcmd(2,  6,  4,  1,   8,  8);  dcmd(2,  6,  5,  2,   4,  0);
        dcmd(2,  7,  6,  0,  32,  8);   dcmd(2,  7,  7,  0,  64,  8);  dcmd(2,  7,  8,  0,  16,  0);
        dcmd(2,  8,  9,  0,   0,  5);   // rank 1, 160 bytes, 64-byte aligned start: five pieces (160 is not whole lines)
        dcmd(2,  8, 10,  4,   0,  0);   // 8 bytes in
        dcmd(2,  8, 11, 16,   0,  5);   // 32 bytes in
        dcmd(2,  9, 12,  0,   0,  1);   dcmd(2,  9, 13, 16,   0,  1);  dcmd(2,  9, 14,  8,   0,  0);
        dcmd(2, 10, 15,  0,   0,  5);   // an odd number of 32-byte rows
        dcmd(2, 11,  0,  0,   0,  0);   // rows of 64 bytes, 64-aligned: whole lines
        dcmd(2, 11,  1,  0,  32,  0);   // whole lines again (64 bytes in)
        dcmd(2, 11,  2,  0,  16, 16);   // rows of 64 bytes starting 32 bytes into a line: pieces
        dcmd(2,  0,  3,  0,   0,  0);   p_act[plen-1] = 1'b0;     // an inactive command moves nothing
        // stores from the windows the loads above filled
        dcmd(3,  0,  0, 20,   0, 16);   dcmd(3,  0,  1, 22,  32, 16);  dcmd(3,  0,  2,  0,   8,  0);
        dcmd(3,  1,  3,  4,   0, 16);   dcmd(3, 10,  4,  0,   0,  5);  dcmd(3,  6,  5,  8,   0,  8);
        dcmd(3,  9,  6,  8,   0,  0);   dcmd(3,  8,  7, 32,   0,  5);  dcmd(3,  4,  8,  0,  16, 12);
        dcmd(3,  3,  9,  0,   8,  8);
        // a store, then a load of the same rows
        dcmd(3,  0, 10, 40,   0, 16);   dcmd(2,  0, 11, 40,   0, 16);
        // a transposed load of a tile that is half-line shaped: not a half-line command
        dcmd(2,  0, 12,  0,   0,  0);   p_x[plen-1] = 1'b1;
    end else
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
                (rrank[d] == 1) ? coord(re1[d], bl1[d], sesz[d]) : coord(re0[d], bl0[d], 0),
                (rrank[d] == 1) ? 64'd0 : coord(re1[d], bl1[d], sesz[d]), j);
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
    if (HBIAS && LHALF && !LOCK && !XPERF && (halves_moved == 0 || halves_st == 0 || halves_st == halves_moved))
        $fatal(1, "the half-line path moved %0d pieces, %0d of them stores: a load or a store path went unexercised", halves_moved, halves_st);
    $display("TENSOR LINE MITER PASS seed=%0d commands=%0d completions=%0d faults=%0d xpose=%0d lines=%0d halves=%0d half_stores=%0d cycles=%0d",
             SEED, t, ncomp, nfault[0], nx, lines_moved, halves_moved, halves_st, ncyc);
    $finish;
end
endmodule
`default_nettype wire
