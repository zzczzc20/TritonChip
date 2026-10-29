//==============================================================================
// tnca_matrix -- matrix engine and accumulator file
//
//     C[16x16] <- C[16x16] + A[16xK] . B[Kx16]
//
// THE ACCUMULATION CONTRACT (ISA sec 9C.1, arch overview sec 6.3)
//
// The product is formed at full width and never rounded on its own, and there
// is exactly ONE rounding per accumulation step, applied to the sum, in
// increasing k order. So a MAC here is literally one tnca_fpu_multi: the
// operands widen to binary32 exactly, and the binary32 FMA does the rest. The
// result is bit-reproducible across implementations, and an implementation
// that wants a faster reduction order has to expose it as a different opcode;
// it may not make this one quietly cheaper.
//
// FOUR INPUT FORMATS, ONE MULTIPLIER
//
// FP16, BF16 and both OCP FP8 encodings (E5M2, E4M3) all widen to binary32
// exactly, so all four are the same datapath with a different format code on
// the widening step -- ISA sec 9C's mod[3:2]. tnca_fpu_multi is that
// composition, and it is the same module tb_fpu_multi verifies against an
// exact oracle, so the matrix unit has no arithmetic of its own to get wrong.
//
// What does change with the format is the *fragment layout*: an 8-bit element
// packs four to a lane and 64 to a vector register, where a 16-bit element
// packs two and 32. That is one bit of addressing, below.
//
// WHY THE LOOP IS k-OUTER, m-INNER
//
// Sec 6.2 warns against assuming an arithmetic-resource ceiling is achievable
// throughput, and the FP32 accumulate has multi-cycle feedback. Iterating m in
// the inner loop makes 16 consecutive cycles touch 16 different accumulator
// rows, so the FMA pipeline never stalls on its own result: the same row
// returns 16 cycles later against a 2-cycle latency. Sec 6.2's suggested
// remedies were "interleave independent C fragments" or "interleave partial
// sums inside the PE"; this is the first one, and it costs nothing.
//
// It is also cheap on register-file ports. For a fixed k, B[k][.] is one vector
// register, read once and held; A[m][k] moves to the next register only every
// second m. The engine averages well under one vector read per cycle, which is
// the constraint sec 5.3 says to respect before sizing the arithmetic.
//
// THE STATE MACHINE sec 6.2 ASKS FOR
//
//   IDLE -> LOAD_B -> MAC_RUN -> (next k) -> DRAIN -> COMPLETE
//
// ARRAY SIZE. 16 MACs: one output column each, one k per 16 cycles. That is
// 1/16 of the reference configuration's 256 MAC/cycle (sec 3.1). Widening means
// replicating the m dimension, and what limits how far is the vector-file read
// rate, not the multipliers -- which is exactly the point sec 5.3 makes.
//==============================================================================
`include "tnca_defs.vh"

module tnca_matrix #(
    parameter VLEN    = 16,
    parameter ELEN    = 32,
    parameter ACC_N   = 16,
    parameter ACC_W   = 4,
    parameter ROWS    = 16,
    parameter ROW_W   = 4,
    parameter BAR_W   = 5,
    parameter FMA_LAT = 2,
    parameter FMA_IMPL = 0,      // tnca_fp32_fma IMPL (R41 G2 FMA rebuild)
    // R41 G2: 1 = each MAC is tnca_mac_lane (FP32 1x, FP16/BF16/FP8 pairs 2x,
    // E4M3 x E4M3 4x) and the engine executes m.mma.dp.async (ISA sec 9C.4).
    // 0 = tnca_fpu_multi, exactly as before; cmd_dp is then never asserted.
    // MAC_CUTS places the lane's registers (tnca_mac_lane CUTS); the result
    // latency becomes OPND_REG + popcount(MAC_CUTS) and every user of it here
    // (the tag pipeline, the drain) follows from PIPE.
    parameter MAC_DP   = 0,
    parameter [4:0] MAC_CUTS = 5'b11110,
    // Register the MAC operands before the FMA (tnca_fpu_multi IN_REG): one
    // more cycle of latency per command, and the A operand's block-RAM read
    // and element select no longer share a cycle with the multiply.
    parameter OPND_REG = 1,
    // R39-12: read the A fragment into a buffer of its own before the MAC
    // sweep, when it is at most eight registers (FP16/BF16 K16, FP8 K16 and
    // K32), so the sweep issues one row a cycle. Without it every MAC waits
    // for a vector-file read of the register holding its one A element:
    // measured, 551 cycles for an FP16 K16 command, of which the 256 MACs
    // are under half.
    parameter ABUF     = 0
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     flush,

    // ---- command ------------------------------------------------------------
    input  wire                     cmd_valid,       // one-cycle pulse
    input  wire [1:0]               cmd_op,          // 0 zero, 1 mma, 2 row op
    input  wire [ACC_W-1:0]         cmd_slot,
    input  wire [7:0]               cmd_va,
    input  wire [7:0]               cmd_vb,
    input  wire [1:0]               cmd_k,           // 0 K16, 1 K32, 2 K64
    input  wire [1:0]               cmd_dt,          // ISA sec 9C mod[3:2]
    input  wire                     cmd_dp,          // m.mma.dp.async (sec 9C.4)
    input  wire [1:0]               cmd_bdt,         // its B format, mod[6:5]
    input  wire                     cmd_bar_en,
    input  wire [BAR_W-1:0]         cmd_bar,
    // ISA sec 9C: the SECOND event. `free` fires when the fragments have been
    // read for the last time, which is earlier than `done` and is what a
    // double-buffered producer actually needs -- arch overview sec 13.1 names
    // confusing the two as a hazard, and waiting for `done` is the
    // conservative proxy this replaces.
    input  wire                     cmd_sf_en,
    input  wire [BAR_W-1:0]         cmd_sf_bar,
    // row-op detail
    input  wire [1:0]               cmd_rowop,       // 0 vwrite, 1 mulrow, 2 fmarow
    input  wire [ROW_W-1:0]         cmd_row,
    input  wire [VLEN*ELEN-1:0]     cmd_vdata,
    input  wire [31:0]              cmd_scale,
    input  wire [VLEN-1:0]          cmd_mask,

    output wire                     busy,

    // ---- vector register read port (asynchronous) ---------------------------
    output wire [7:0]               vrf_addr,
    input  wire [VLEN*ELEN-1:0]     vrf_data,
    // A synchronous register file answers a cycle later, so the two states
    // that consume vrf_data have to be able to wait. `vrf_req` says a read
    // is wanted this cycle; `vrf_valid` says the data on vrf_data belongs
    // to the address on vrf_addr. With the behavioural file the core ties
    // vrf_valid high and neither state ever waits, which is what keeps the
    // reference implementation bit-identical.
    output wire                     vrf_req,
    input  wire                     vrf_valid,

    // ---- accumulator read, for a.vread --------------------------------------
    input  wire [ACC_W-1:0]         ard_slot,
    input  wire [ROW_W-1:0]         ard_row,
    output wire [VLEN*ELEN-1:0]     ard_data,

    // ---- completion ---------------------------------------------------------
    output reg                      done,            // pulse
    output reg  [ACC_W-1:0]         done_slot,
    output reg                      done_bar_en,
    output reg  [BAR_W-1:0]         done_bar,
    output reg                      src_free,        // pulse
    output reg  [BAR_W-1:0]         src_free_bar
);

    localparam S_IDLE  = 3'd0,
               S_ZERO  = 3'd1,
               S_LOADB = 3'd2,
               S_MAC   = 3'd3,
               S_ROW   = 3'd4,
               S_DRAIN = 3'd5,
               S_DONE  = 3'd6,
               S_LOADA = 3'd7;

    integer i;
    genvar  g;

    reg [2:0]        st;
    reg [ACC_W-1:0]  r_slot;
    reg [7:0]        r_va, r_vb;
    reg [1:0]        r_k;
    reg [1:0]        r_dt;
    reg              r_dp;
    reg [1:0]        r_bdt;
    reg              r_bar_en;
    reg [BAR_W-1:0]  r_bar;
    reg              r_sf_en;
    reg [BAR_W-1:0]  r_sf_bar;
    reg [1:0]        r_rowop;
    reg [ROW_W-1:0]  r_row;
    reg [VLEN*ELEN-1:0] r_vdata;
    reg [31:0]       r_scale;
    reg [VLEN-1:0]   r_mask;

    reg [6:0]        kk;        // current k, 0..63
    reg [ROW_W:0]    mm;        // current m, 0..16
    reg [3:0]        drain;

    wire [6:0] k_limit = (r_k == `TNCA_MMA_K16) ? 7'd16 :
                         (r_k == `TNCA_MMA_K32) ? 7'd32 : 7'd64;

    // ABUF: the A fragment, 16 x k_limit elements, in at most 8 registers.
    reg  [VLEN*ELEN-1:0] abuf [0:7];
    reg  [3:0]           la;          // the A register being read
    reg                  use_abuf;    // this command sweeps out of abuf

    // The fragment dtype, as the converter's format code. The two vocabularies
    // are deliberately different -- a fragment cannot be binary32, so sec 9C
    // needs only two bits -- and this is the one place they meet.
    reg [2:0] r_fmt;
    always @(*) begin
        case (r_dt)
        `TNCA_MMA_F16:  r_fmt = `TNCA_FMT_F16;
        `TNCA_MMA_BF16: r_fmt = `TNCA_FMT_BF16;
        `TNCA_MMA_E5M2: r_fmt = `TNCA_FMT_E5M2;
        default:        r_fmt = `TNCA_FMT_E4M3;
        endcase
    end

    // Both FP8 encodings are one byte per element; FP16 and BF16 are two.
    // elem8 is A's; B's format is its own under m.mma.dp (sec 9C.4).
    wire elem8  = r_dt[1];
    wire dpv    = (MAC_DP != 0) && r_dp;
    wire elemB8 = dpv ? r_bdt[1] : r_dt[1];
    // The group: four E4M3 x E4M3 products, else two (sec 9C.4); 1 = m.mma.
    wire g4     = dpv && (r_dt == `TNCA_MMA_E4M3) && (r_bdt == `TNCA_MMA_E4M3);
    wire [2:0] kstep = !dpv ? 3'd1 : g4 ? 3'd4 : 3'd2;
    // How many registers A takes (ABUF reads them all, when they fit).
    wire [6:0] a_regs   = elem8 ? {2'd0, k_limit[6:2]} : {1'b0, k_limit[6:1]};

    assign busy = (st != S_IDLE);

    //==========================================================================
    // Fragment addressing
    //
    // A fragment is a window's registers read as a flat row-major *byte*
    // array, which is exactly the layout the transfer engine writes, so
    // nothing repacks between a t.ld.async and the m.mma that consumes it.
    // For a 2-byte element, element `flat` lives at vector register `flat/32`,
    // bit `(flat%32)*16`; for a 1-byte element, at `flat/64`, bit `(flat%64)*8`.
    //==========================================================================
    wire [10:0] a_flat = mm[ROW_W-1:0] * k_limit + kk;
    wire [7:0]  a_vreg = r_va + (elem8 ? {3'd0, a_flat[10:6]}
                                       : {2'd0, a_flat[10:5]});
    wire [5:0]  a_elem = elem8 ? a_flat[5:0] : {1'b0, a_flat[4:0]};

    // B[k][.] is 16 consecutive elements starting at k*16: half a vector
    // register at 16 bits per element, a quarter of one at 8.
    wire [7:0]  b_vreg = r_vb + (elemB8 ? {3'd0, kk[6:2]} : {2'd0, kk[6:1]});
    wire [1:0]  b_part = elemB8 ? kk[1:0] : {kk[0], 1'b0};

    assign vrf_addr = (st == S_LOADB) ? b_vreg
                    : (st == S_LOADA) ? (r_va + {4'd0, la})
                    : a_vreg;

    // The two states that read the vector file. Held while the data has
    // not come back: the address is driven from the state, so holding the
    // state holds the request.
    assign vrf_req = (st == S_LOADB) || (st == S_LOADA) || (st == S_MAC && !use_abuf);
    wire vrf_wait = vrf_req && !vrf_valid;

    // B row latch. One 16-bit slot per element whatever the format: an 8-bit
    // element sits right-aligned in its slot, which is where the converter
    // expects a narrow value to be.
    // Under m.mma.dp a lane's word holds its G elements, one from each of G
    // consecutive B rows: 32 bits per lane.
    localparam BW = (MAC_DP != 0) ? 32 : 16;
    reg [VLEN*BW-1:0] b_row;

    // The row's lane words, from the vector read: m.mma's (one element, right
    // aligned) and m.mma.dp's (G elements, one from each of G consecutive B
    // rows, sec 9C.4).
    wire [BW-1:0] b_nd [0:VLEN-1];
    wire [31:0]   b_dp [0:VLEN-1];
    genvar bi;
    generate
        for (bi = 0; bi < VLEN; bi = bi + 1) begin : g_bw
            assign b_nd[bi] = elem8 ? {8'd0, vrf_data[(b_part*VLEN + bi)*8 +: 8]}
                                    : vrf_data[(b_part[1]*VLEN + bi)*16 +: 16];
            assign b_dp[bi] =
                !r_bdt[1] ? {vrf_data[(VLEN + bi)*16 +: 16], vrf_data[bi*16 +: 16]}
              : g4        ? {vrf_data[(3*VLEN + bi)*8 +: 8], vrf_data[(2*VLEN + bi)*8 +: 8],
                             vrf_data[(1*VLEN + bi)*8 +: 8], vrf_data[bi*8 +: 8]}
              :             {8'd0, vrf_data[((b_part + 2'd1)*VLEN + bi)*8 +: 8],
                             8'd0, vrf_data[(b_part*VLEN + bi)*8 +: 8]};
        end
    endgenerate

    // A element for this cycle, right-aligned in 32 bits.
    wire [2:0]           a_bi   = elem8 ? a_flat[8:6] : a_flat[7:5];
    wire [VLEN*ELEN-1:0] a_word = use_abuf ? abuf[a_bi] : vrf_data;
    // m.mma.dp: the G consecutive elements A[m][k..k+G-1] as one lane word --
    // 16-bit elements are one aligned 32-bit word; four E4M3 bytes likewise;
    // two bytes go to the low bytes of two 16-bit slots.
    wire [31:0] a_dp_v = !elem8 ? a_word[a_elem[4:1]*32 +: 32]
                       : g4     ? a_word[a_elem[5:2]*32 +: 32]
                       : {8'd0, a_word[(a_elem + 6'd1)*8 +: 8], 8'd0, a_word[a_elem*8 +: 8]};
    wire [31:0] a_elem_v = dpv  ? a_dp_v
                         : elem8 ? {24'd0, a_word[a_elem*8 +: 8]}
                                 : {16'd0, a_word[a_elem[4:0]*16 +: 16]};

    //==========================================================================
    // The MAC array
    //==========================================================================
    wire [ELEN-1:0] acc_rd [0:VLEN-1];
    wire [VLEN*ELEN-1:0] acc_rd_bus;

    wire [ELEN-1:0] mac_y [0:VLEN-1];
    wire [VLEN*ELEN-1:0] mac_bus;

    // FMA operand selection: a MAC during MAC_RUN, an accumulator row op
    // otherwise. One array serves both, because they contend for the same
    // accumulator port anyway.
    // R41 T2 round 5: S_ROW and its row op as registered flags, set with the
    // command that enters S_ROW (from S_IDLE, for one cycle) -- not decoded
    // from st and r_rowop in front of every MAC's format and operand selects
    // (st -> fmt -> MAC input registers, +0.81 ns tile-0 screen). A bench
    // check compares them with the decode every cycle.
    reg  row_q, row_mul_q, row_fma_q;
    wire row_mul = row_mul_q;
    wire row_fma = row_fma_q;

    // An accumulator row op is binary32 on both inputs, and widen(f32, x) is
    // x, so the same multi-precision MAC serves it with no bypass. The whole
    // S_ROW state takes this path, a.vwrite included: its FMA result is
    // discarded, and presenting a fragment format for an operand that is not
    // a fragment would only be confusing.
    wire        row_any = row_q;
    wire [2:0]  mac_fmt = row_any ? `TNCA_FMT_F32 : r_fmt;

    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_mac
            wire [31:0] fa = row_any ? acc_rd[g] : a_elem_v;
            wire [31:0] b_dpw;
            if (MAC_DP != 0) begin : g_bdp
                assign b_dpw = b_row[g*BW +: 32];
            end else begin : g_nbdp
                assign b_dpw = 32'd0;
            end
            wire [31:0] fb = row_any ? r_scale
                                     : (MAC_DP != 0 && r_dp) ? b_dpw
                                     : {16'd0, b_row[g*BW +: 16]};
            wire [31:0] fc = row_fma ? r_vdata[g*ELEN +: ELEN]
                           : row_mul ? 32'd0
                           : acc_rd[g];

            if (MAC_DP != 0) begin : g_lane
                tnca_mac_lane #(.CUTS(MAC_CUTS), .IN_REG(OPND_REG)) u_mac (
                    .clk(clk), .rst_n(rst_n),
                    .mode((row_any || !r_dp) ? 2'd0 : g4 ? 2'd2 : 2'd1),
                    .fmt(mac_fmt), .fa(r_dt), .fb(r_bdt),
                    .a(fa), .b(fb), .c(fc),
                    .sub_c(1'b0), .mul_only(row_mul), .ftz(1'b0),
                    .y(mac_y[g])
                );
            end else begin : g_fpu
                tnca_fpu_multi #(.LAT(FMA_LAT), .IN_REG(OPND_REG), .IMPL(FMA_IMPL)) u_mac (
                    .clk(clk), .rst_n(rst_n),
                    .fmt(mac_fmt),
                    .a(fa), .b(fb), .c(fc),
                    .sub_c(1'b0), .mul_only(row_mul), .ftz(1'b0),
                    .y(mac_y[g])
                );
            end
            assign mac_bus[g*ELEN +: ELEN] = mac_y[g];
            assign acc_rd[g] = acc_rd_bus[g*ELEN +: ELEN];
        end
    endgenerate

    //==========================================================================
    // Result tag pipeline, alongside the FMA's own registers
    //==========================================================================
    // A result lands PIPE cycles after its operands are presented.
    localparam MAC_LAT = MAC_CUTS[0] + MAC_CUTS[1] + MAC_CUTS[2] + MAC_CUTS[3] + MAC_CUTS[4];
    localparam PIPE = (MAC_DP != 0) ? (OPND_REG + MAC_LAT) : (FMA_LAT + OPND_REG);
    reg              tg_v   [0:PIPE-1];
    reg [ACC_W-1:0]  tg_slot[0:PIPE-1];
    reg [ROW_W-1:0]  tg_row [0:PIPE-1];
    reg [VLEN-1:0]   tg_mask[0:PIPE-1];

    wire wb_v = tg_v[PIPE-1];

    //==========================================================================
    // Accumulator file
    //==========================================================================
    reg              z_en;
    reg [ROW_W-1:0]  z_row;
    reg              direct_w;          // a.vwrite bypasses the FMA
    reg [ACC_W-1:0]  direct_slot;
    reg [ROW_W-1:0]  direct_row;
    reg [VLEN-1:0]   direct_mask;
    reg [VLEN*ELEN-1:0] direct_data;

    wire acc_w_en          = wb_v | direct_w;
    wire [ACC_W-1:0] acc_w_slot = direct_w ? direct_slot : tg_slot[PIPE-1];
    wire [ROW_W-1:0] acc_w_row  = direct_w ? direct_row  : tg_row[PIPE-1];
    wire [VLEN-1:0]  acc_w_lane = direct_w ? direct_mask : tg_mask[PIPE-1];
    wire [VLEN*ELEN-1:0] acc_w_data = direct_w ? direct_data : mac_bus;

    // The read port serves the engine while it is running and a.vread when it
    // is not; a.vread cannot issue while the engine is busy, so there is no
    // contention to arbitrate.
    wire [ACC_W-1:0] acc_r_slot = busy ? r_slot : ard_slot;
    wire [ROW_W-1:0] acc_r_row  = busy ? ((st == S_ROW) ? r_row : mm[ROW_W-1:0])
                                       : ard_row;

    tnca_acc_rf #(
        .ACC_N(ACC_N), .ACC_W(ACC_W), .ROWS(ROWS), .ROW_W(ROW_W),
        .VLEN(VLEN), .ELEN(ELEN)
    ) u_acc (
        .clk(clk), .rst_n(rst_n),
        .r_slot(acc_r_slot), .r_row(acc_r_row), .r_data(acc_rd_bus),
        .w_en(acc_w_en), .w_slot(acc_w_slot), .w_row(acc_w_row),
        .w_lane_en(acc_w_lane), .w_data(acc_w_data),
        .z_en(z_en), .z_slot(r_slot), .z_row(z_row)
    );

    assign ard_data = acc_rd_bus;

`ifndef SYNTHESIS
    // tnca_acc_rf has no write-to-read bypass (see there). That is only right
    // while nothing reads a row in the cycle it is written and uses what it
    // read: the MAC sweep and a row op are the users, so check both.
    wire acc_rd_used = (st == S_MAC && !vrf_wait) || (st == S_ROW && r_rowop != 2'd0);
    wire acc_collide = acc_rd_used && acc_w_en &&
                       ({acc_r_slot, acc_r_row} == {acc_w_slot, acc_w_row});
    // R41 T2 round 5: the registered row flags are the decode they replace.
    always @(posedge clk)
        if (rst_n && ((row_q !== (st == S_ROW)) ||
                      (row_q && ((row_mul_q !== (r_rowop == 2'd1)) || (row_fma_q !== (r_rowop == 2'd2))))))
            $display("FAIL tnca_matrix assert @%0t: registered row flags %b%b%b, st %0d rowop %0d",
                     $time, row_q, row_mul_q, row_fma_q, st, r_rowop);
    always @(posedge clk)
        if (rst_n && acc_collide) begin
            $display("FATAL: tnca_matrix read accumulator %0d row %0d in the cycle it was written",
                     acc_r_slot, acc_r_row);
            $finish;
        end
`endif

    //==========================================================================
    // Sequencer
    //==========================================================================
    always @(posedge clk) begin
        if (!rst_n || flush) begin
            st          <= S_IDLE;
            done        <= 1'b0;
            src_free    <= 1'b0;
            done_bar_en <= 1'b0;
            z_en        <= 1'b0;
            direct_w    <= 1'b0;
            row_q       <= 1'b0;
            row_mul_q   <= 1'b0;
            row_fma_q   <= 1'b0;
            for (i = 0; i < PIPE; i = i + 1) tg_v[i] <= 1'b0;
        end else begin
            done        <= 1'b0;
            src_free    <= 1'b0;
            done_bar_en <= 1'b0;
            z_en        <= 1'b0;
            direct_w    <= 1'b0;
            row_q       <= 1'b0;
            row_mul_q   <= 1'b0;
            row_fma_q   <= 1'b0;
            // R41 T2 round 5: the a.vwrite payload is copied every cycle --
            // it is read only under direct_w, the cycle after S_ROW, and r_*
            // change only in S_IDLE -- so its enable is not r_rowop's decode
            // routed to every bit (r_rowop -> direct_* CE, +1.054 ns tile 1).
            direct_slot <= r_slot;
            direct_row  <= r_row;
            direct_mask <= r_mask;
            direct_data <= r_vdata;

            // advance the result tag pipeline
            for (i = PIPE-1; i > 0; i = i - 1) begin
                tg_v[i]    <= tg_v[i-1];
                tg_slot[i] <= tg_slot[i-1];
                tg_row[i]  <= tg_row[i-1];
                tg_mask[i] <= tg_mask[i-1];
            end
            tg_v[0] <= 1'b0;

            case (st)

            //---------------------------------------------------------------
            S_IDLE: begin
                if (cmd_valid) begin
                    r_slot   <= cmd_slot;
                    r_va     <= cmd_va;
                    r_vb     <= cmd_vb;
                    r_k      <= cmd_k;
                    r_dt     <= cmd_dt;
                    r_dp     <= (MAC_DP != 0) && cmd_dp;
                    r_bdt    <= cmd_bdt;
                    r_bar_en <= cmd_bar_en;
                    r_bar    <= cmd_bar;
                    r_sf_en  <= cmd_sf_en;
                    r_sf_bar <= cmd_sf_bar;
                    r_rowop  <= cmd_rowop;
                    r_row    <= cmd_row;
                    r_vdata  <= cmd_vdata;
                    r_scale  <= cmd_scale;
                    r_mask   <= cmd_mask;
                    kk       <= 7'd0;
                    mm       <= {(ROW_W+1){1'b0}};
                    drain    <= 4'd0;
                    la       <= 4'd0;
                    use_abuf <= 1'b0;
                    case (cmd_op)
                    2'd0: st <= S_ZERO;
                    2'd1: st <= S_LOADB;
                    default: begin
                        st        <= S_ROW;
                        row_q     <= 1'b1;
                        row_mul_q <= (cmd_rowop == 2'd1);
                        row_fma_q <= (cmd_rowop == 2'd2);
                    end
                    endcase
                    // r_k and r_dt are not latched yet: decide from the command.
                    if (cmd_op == 2'd1 && ABUF != 0 &&
                        (cmd_dt[1] ? (cmd_k != `TNCA_MMA_K64) : (cmd_k == `TNCA_MMA_K16))) begin
                        use_abuf <= 1'b1;
                        st       <= S_LOADA;
                    end
                end
            end

            //---------------------------------------------------------------
            // m.zero: one row per cycle through the file's zero port
            S_ZERO: begin
                z_en  <= 1'b1;
                z_row <= mm[ROW_W-1:0];
                if (mm == ROWS-1) begin
                    mm <= {(ROW_W+1){1'b0}};
                    st <= S_DONE;
                end else begin
                    mm <= mm + 1'b1;
                end
            end

            //---------------------------------------------------------------
            // ABUF: the A fragment's registers, one read each, before any MAC.
            S_LOADA: if (!vrf_wait) begin
                abuf[la[2:0]] <= vrf_data;
                if ({3'd0, la} == a_regs - 7'd1) st <= S_LOADB;
                else                             la <= la + 1'b1;
            end

            //---------------------------------------------------------------
            // One vector read holds B[k][.] for the whole m sweep.
            S_LOADB: if (!vrf_wait) begin
                for (i = 0; i < VLEN; i = i + 1)
                    b_row[i*BW +: BW] <= (MAC_DP != 0 && r_dp) ? b_dp[i][BW-1:0] : b_nd[i];
                mm <= {(ROW_W+1){1'b0}};
                st <= S_MAC;
            end

            //---------------------------------------------------------------
            // Nothing is issued into the MAC pipeline while the A element
            // has not arrived: tg_v[0] staying low is what stops the array
            // multiplying whatever happened to be on the bus.
            S_MAC: if (!vrf_wait) begin
                tg_v[0]    <= 1'b1;
                tg_slot[0] <= r_slot;
                tg_row[0]  <= mm[ROW_W-1:0];
                tg_mask[0] <= {VLEN{1'b1}};

                if (mm == ROWS-1) begin
                    mm <= {(ROW_W+1){1'b0}};
                    if (kk == k_limit - {4'd0, kstep}) begin
                        // The last A element and the last B row have now been
                        // read. Nothing in this command will touch the vector
                        // file again, so the fragments are free from here --
                        // PIPE cycles before the result lands.
                        src_free     <= r_sf_en;
                        src_free_bar <= r_sf_bar;
                        drain <= PIPE[3:0] + 4'd1;
                        st    <= S_DRAIN;
                    end else begin
                        kk <= kk + {4'd0, kstep};
                        st <= S_LOADB;
                    end
                end else begin
                    mm <= mm + 1'b1;
                end
            end

            //---------------------------------------------------------------
            // a.vwrite writes directly; a.mul.row and a.fma.row go through the
            // MAC array so they get the single rounding the ISA specifies.
            S_ROW: begin
                if (r_rowop == 2'd0) begin
                    direct_w    <= 1'b1;
                    st <= S_DONE;
                end else begin
                    tg_v[0]    <= 1'b1;
                    tg_slot[0] <= r_slot;
                    tg_row[0]  <= r_row;
                    tg_mask[0] <= r_mask;
                    drain      <= PIPE[3:0] + 4'd1;
                    st         <= S_DRAIN;
                end
            end

            //---------------------------------------------------------------
            S_DRAIN: begin
                if (drain == 4'd0) st <= S_DONE;
                else               drain <= drain - 1'b1;
            end

            //---------------------------------------------------------------
            S_DONE: begin
                done        <= 1'b1;
                done_slot   <= r_slot;
                done_bar_en <= r_bar_en;
                done_bar    <= r_bar;
                st          <= S_IDLE;
            end

            default: st <= S_IDLE;
            endcase
        end
    end

endmodule
