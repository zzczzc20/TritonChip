//==============================================================================
// tnca_matrix -- matrix engine and accumulator file
//
//     C[16x16] <- C[16x16] + A[16xK] . B[Kx16]
//
// THE ACCUMULATION CONTRACT (ISA sec 9C.1, arch overview sec 6.3)
//
// Each product is exact -- an FP16xFP16 or BF16xBF16 product is exactly
// representable in binary32 -- and is added to the binary32 accumulator with
// ONE rounding, in increasing k order. So a MAC here is literally one
// tnca_fp32_fma with widened inputs, and the result is bit-reproducible across
// implementations. An implementation that wants a faster reduction order has to
// expose it as a different opcode; it may not make this one quietly cheaper.
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
    parameter FMA_LAT = 2
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
    input  wire                     cmd_dt,          // 0 FP16, 1 BF16
    input  wire                     cmd_bar_en,
    input  wire [BAR_W-1:0]         cmd_bar,
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

    // ---- accumulator read, for a.vread --------------------------------------
    input  wire [ACC_W-1:0]         ard_slot,
    input  wire [ROW_W-1:0]         ard_row,
    output wire [VLEN*ELEN-1:0]     ard_data,

    // ---- completion ---------------------------------------------------------
    output reg                      done,            // pulse
    output reg  [ACC_W-1:0]         done_slot,
    output reg                      done_bar_en,
    output reg  [BAR_W-1:0]         done_bar
);

    localparam S_IDLE  = 3'd0,
               S_ZERO  = 3'd1,
               S_LOADB = 3'd2,
               S_MAC   = 3'd3,
               S_ROW   = 3'd4,
               S_DRAIN = 3'd5,
               S_DONE  = 3'd6;

    integer i;
    genvar  g;

    reg [2:0]        st;
    reg [ACC_W-1:0]  r_slot;
    reg [7:0]        r_va, r_vb;
    reg [1:0]        r_k;
    reg              r_dt;
    reg              r_bar_en;
    reg [BAR_W-1:0]  r_bar;
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

    assign busy = (st != S_IDLE);

    //==========================================================================
    // Fragment addressing
    //
    // A fragment is a window's registers read as a flat row-major array of
    // 16-bit elements: element `flat` lives at vector register `flat/32`, bit
    // `(flat%32)*16`. That is exactly the byte layout the transfer engine
    // writes, so nothing repacks between a t.ld.async and the m.mma that
    // consumes it.
    //==========================================================================
    wire [10:0] a_flat = mm[ROW_W-1:0] * k_limit + kk;
    wire [7:0]  a_vreg = r_va + a_flat[10:5];
    wire [4:0]  a_elem = a_flat[4:0];

    // B[k][.] is 16 consecutive elements starting at k*16, so it is exactly one
    // half of one vector register.
    wire [7:0]  b_vreg = r_vb + {2'd0, kk[6:1]};
    wire        b_half = kk[0];

    assign vrf_addr = (st == S_LOADB) ? b_vreg : a_vreg;

    // B row latch
    reg [VLEN*16-1:0] b_row;

    // A element for this cycle
    wire [15:0] a_elem_v = vrf_data[a_elem*16 +: 16];

    //==========================================================================
    // Widening and the MAC array
    //==========================================================================
    wire [31:0] a_f32;
    tnca_cvt16_f32 u_cva (.v(a_elem_v), .mode(r_dt), .y(a_f32));

    wire [ELEN-1:0] acc_rd [0:VLEN-1];
    wire [VLEN*ELEN-1:0] acc_rd_bus;

    wire [ELEN-1:0] mac_y [0:VLEN-1];
    wire [VLEN*ELEN-1:0] mac_bus;

    // FMA operand selection: a MAC during MAC_RUN, an accumulator row op
    // otherwise. One array serves both, because they contend for the same
    // accumulator port anyway.
    wire row_mul = (st == S_ROW) && (r_rowop == 2'd1);
    wire row_fma = (st == S_ROW) && (r_rowop == 2'd2);

    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_mac
            wire [31:0] b_f32;
            tnca_cvt16_f32 u_cvb (
                .v(b_row[g*16 +: 16]), .mode(r_dt), .y(b_f32)
            );

            wire [31:0] fa = (row_mul || row_fma) ? acc_rd[g] : a_f32;
            wire [31:0] fb = (row_mul || row_fma) ? r_scale   : b_f32;
            wire [31:0] fc = row_fma ? r_vdata[g*ELEN +: ELEN]
                           : row_mul ? 32'd0
                           : acc_rd[g];

            tnca_fp32_fma u_fma (
                .clk(clk), .rst_n(rst_n),
                .a(fa), .b(fb), .c(fc),
                .sub_c(1'b0), .mul_only(row_mul), .ftz(1'b0),
                .y(mac_y[g])
            );
            assign mac_bus[g*ELEN +: ELEN] = mac_y[g];
            assign acc_rd[g] = acc_rd_bus[g*ELEN +: ELEN];
        end
    endgenerate

    //==========================================================================
    // Result tag pipeline, alongside the FMA's own registers
    //==========================================================================
    reg              tg_v   [0:FMA_LAT-1];
    reg [ACC_W-1:0]  tg_slot[0:FMA_LAT-1];
    reg [ROW_W-1:0]  tg_row [0:FMA_LAT-1];
    reg [VLEN-1:0]   tg_mask[0:FMA_LAT-1];

    wire wb_v = tg_v[FMA_LAT-1];

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
    wire [ACC_W-1:0] acc_w_slot = direct_w ? direct_slot : tg_slot[FMA_LAT-1];
    wire [ROW_W-1:0] acc_w_row  = direct_w ? direct_row  : tg_row[FMA_LAT-1];
    wire [VLEN-1:0]  acc_w_lane = direct_w ? direct_mask : tg_mask[FMA_LAT-1];
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

    //==========================================================================
    // Sequencer
    //==========================================================================
    always @(posedge clk) begin
        if (!rst_n || flush) begin
            st          <= S_IDLE;
            done        <= 1'b0;
            done_bar_en <= 1'b0;
            z_en        <= 1'b0;
            direct_w    <= 1'b0;
            for (i = 0; i < FMA_LAT; i = i + 1) tg_v[i] <= 1'b0;
        end else begin
            done        <= 1'b0;
            done_bar_en <= 1'b0;
            z_en        <= 1'b0;
            direct_w    <= 1'b0;

            // advance the result tag pipeline
            for (i = FMA_LAT-1; i > 0; i = i - 1) begin
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
                    r_bar_en <= cmd_bar_en;
                    r_bar    <= cmd_bar;
                    r_rowop  <= cmd_rowop;
                    r_row    <= cmd_row;
                    r_vdata  <= cmd_vdata;
                    r_scale  <= cmd_scale;
                    r_mask   <= cmd_mask;
                    kk       <= 7'd0;
                    mm       <= {(ROW_W+1){1'b0}};
                    drain    <= 4'd0;
                    case (cmd_op)
                    2'd0: st <= S_ZERO;
                    2'd1: st <= S_LOADB;
                    default: st <= S_ROW;
                    endcase
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
            // One vector read holds B[k][.] for the whole m sweep.
            S_LOADB: begin
                if (b_half)
                    b_row <= vrf_data[VLEN*16 +: VLEN*16];
                else
                    b_row <= vrf_data[0 +: VLEN*16];
                mm <= {(ROW_W+1){1'b0}};
                st <= S_MAC;
            end

            //---------------------------------------------------------------
            S_MAC: begin
                tg_v[0]    <= 1'b1;
                tg_slot[0] <= r_slot;
                tg_row[0]  <= mm[ROW_W-1:0];
                tg_mask[0] <= {VLEN{1'b1}};

                if (mm == ROWS-1) begin
                    mm <= {(ROW_W+1){1'b0}};
                    if (kk == k_limit - 1) begin
                        drain <= FMA_LAT[3:0] + 4'd1;
                        st    <= S_DRAIN;
                    end else begin
                        kk <= kk + 1'b1;
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
                    direct_slot <= r_slot;
                    direct_row  <= r_row;
                    direct_mask <= r_mask;
                    direct_data <= r_vdata;
                    st <= S_DONE;
                end else begin
                    tg_v[0]    <= 1'b1;
                    tg_slot[0] <= r_slot;
                    tg_row[0]  <= r_row;
                    tg_mask[0] <= r_mask;
                    drain      <= FMA_LAT[3:0] + 4'd1;
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
