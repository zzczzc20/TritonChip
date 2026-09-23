//==============================================================================
// tnca_tensor -- TensorMap descriptors, RF windows, and the transfer engine
//
// THE THREE OBJECTS, KEPT APART (arch overview sec 7.1)
//
//   descriptor    what the data IS      base, extents, strides, dtype, padding
//   window        where it may LAND     a range of vector registers
//   hint          how it will be USED   ignorable, and ignored here
//
// They are separate because their lifetimes are: a descriptor outlives many
// transfers, a window is reused every pipeline stage, and a hint may be wrong
// without anything breaking. Sec 1.4 is the load-bearing one -- the second half
// of a TensorMap cannot change an address, a value, an access or an order. This
// engine does not read it at all, which is the strongest form that can take.
//
// A WINDOW IS REGISTER ALIASING, NOT A SCRATCHPAD (sec 5.2)
//
// It names vector registers and nothing else: no byte pointer, no general
// indexed load, no residency for software objects, no storage of its own.
// Without those restrictions it would be a scratchpad under a new name, and
// removing the scratchpad is the whole design. The window is also the bounds
// check: a transfer that would run past `len` registers faults instead of
// corrupting a neighbouring allocation.
//
// SNAPSHOTS (sec 7.4)
//
// td.load copies the descriptor into a slot. A transfer reads that copy, so a
// later td.load cannot change a command already issued and no command ever sees
// a half-updated descriptor. Sec 7.4 puts this above replacement policy in
// importance, and it is why the engine copies rather than re-reading per line.
//
// ORDER OF OPERATIONS (sec 12.11): snapshot, coordinates, addresses, bounds,
// expand into transactions, place into the window, pad, signal ONE completion.
// Step eight happens once per command whatever step five expanded into, which
// is what makes the operation counts in mb.arm a static property.
//
// STAGE-2 GEOMETRY RESTRICTION. Rank 1 and 2 only, and a tile row must be a
// whole number of 8-byte units, aligned. Both are checked and raise
// DESCRIPTOR_FAULT. Sec 7.2 allows rank up to 5 and the general case needs
// byte-granular window writes; faulting is the honest interim, because the
// alternative -- approximating -- would make a wrong answer look like a right
// one. TENSOR_MAX_RANK reports the limit so software can ask.
//==============================================================================
`include "tnca_defs.vh"

module tnca_tensor #(
    parameter VLEN      = 16,
    parameter ELEN      = 32,
    parameter RCTX_W    = 1,
    parameter DESC_N    = 16,
    parameter DESC_W    = 4,
    parameter WIN_N     = 32,
    parameter WIN_W     = 5,
    parameter BAR_W     = 5,
    parameter MEM_DW    = 64,
    parameter MEM_BYTES = 8,
    parameter TAG_W     = 4,        // engine-local tag
    parameter NTAG      = 16,
    // ISA sec 9E mod[5], the transposed write (see TRANSPOSED WRITE below).
    // 0 on every profile before R38; a mod[5] command then faults
    // ILLEGAL_INSTRUCTION instead of transferring untransposed.
    parameter HAS_XPOSE = 0
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     flush,

    // ---- configuration commands (single cycle, from the execute stage) ------
    input  wire                     cfg_valid,
    input  wire [1:0]               cfg_op,      // 0 td.load 1 td.release
                                                 // 2 w.def   3 w.release
    input  wire [DESC_W-1:0]        cfg_desc,
    input  wire [WIN_W-1:0]         cfg_win,
    input  wire [63:0]              cfg_addr,    // TensorMap address
    input  wire [7:0]               cfg_wbase,
    input  wire [7:0]               cfg_wlen,

    // ---- transfer command ---------------------------------------------------
    input  wire                     cmd_valid,
    input  wire                     cmd_store,
    input  wire [DESC_W-1:0]        cmd_desc,
    input  wire [WIN_W-1:0]         cmd_win,
    input  wire [63:0]              cmd_c0,
    input  wire [63:0]              cmd_c1,
    input  wire [2:0]               cmd_esz,     // log2 element bytes
    input  wire                     cmd_xpose,   // mod[5]
    input  wire [1:0]               cmd_layout,  // mod[4:3]; only 0 exists
    input  wire                     cmd_bar_en,
    input  wire [BAR_W-1:0]         cmd_bar,
    input  wire [RCTX_W-1:0]        cmd_rctx,
    // The PC of the issuing instruction. ISA sec 10 has TRAP_PC locate the
    // faulting instruction, and an engine fault surfaces long after that
    // instruction retired, so the PC travels with the command.
    input  wire [63:0]              cmd_pc,
    // Any lane active. ISA sec 2.2 makes tensor instructions take a lane
    // mask, and a fully-masked-off instruction performs no access while still
    // writing its completion event. Without this the engine ignored the mask
    // and performed the transfer regardless, so a predicated-off stage
    // overwrote the window it was meant to leave alone.
    input  wire                     cmd_active,

    output wire                     busy,

    // ---- completion ---------------------------------------------------------
    output reg                      done,
    output reg                      done_bar_en,
    output reg  [BAR_W-1:0]         done_bar,
    output reg                      fault,
    output reg  [7:0]               fault_code,
    // Which Role Context issued the command. ISA sec 10 requires a trap to
    // stop the RAISING context; an engine fault arrives long after the
    // issuing instruction has retired, so the core cannot infer it from
    // whatever happens to be in the execute stage.
    output reg  [RCTX_W-1:0]        fault_rctx,
    output reg  [63:0]              fault_pc,
    // A failing command still signals its barrier -- otherwise the waiter
    // hangs -- but it must signal FAILURE. ISA sec 9A.2 gives the barrier an
    // error bit "set by mb.inval or by a faulting command"; signalling an
    // ordinary completion instead releases the consumer to read a window the
    // transfer never wrote.
    output reg                      done_err,

    // ---- vector register write port -----------------------------------------
    output reg                      vw_en,
    output reg  [RCTX_W-1:0]        vw_rctx,
    output reg  [7:0]               vw_idx,
    output reg  [VLEN-1:0]          vw_lane_en,
    output reg  [VLEN*ELEN-1:0]     vw_data,

    // ---- memory port --------------------------------------------------------
    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [63:0]              mem_req_addr,
    output wire                     mem_req_we,
    output wire [MEM_BYTES-1:0]     mem_req_be,
    output wire [MEM_DW-1:0]        mem_req_wdata,
    output wire [TAG_W-1:0]         mem_req_tag,
    input  wire                     mem_rsp_valid,
    input  wire [TAG_W-1:0]         mem_rsp_tag,
    input  wire [MEM_DW-1:0]        mem_rsp_rdata
);

    integer i;

    //==========================================================================
    // Window table
    //==========================================================================
    reg [WIN_N-1:0] win_v;
    reg [7:0]       win_base [0:WIN_N-1];
    reg [7:0]       win_len  [0:WIN_N-1];

    //==========================================================================
    // Descriptor table -- the snapshot, parsed
    //==========================================================================
    reg [DESC_N-1:0] desc_v;
    reg [63:0] d_base   [0:DESC_N-1];
    reg [63:0] d_e0     [0:DESC_N-1];
    reg [63:0] d_e1     [0:DESC_N-1];
    reg [63:0] d_s0     [0:DESC_N-1];
    reg [63:0] d_s1     [0:DESC_N-1];
    reg [31:0] d_b0     [0:DESC_N-1];
    reg [31:0] d_b1     [0:DESC_N-1];
    reg [7:0]  d_rank   [0:DESC_N-1];
    reg [7:0]  d_dtype  [0:DESC_N-1];
    reg [7:0]  d_pad    [0:DESC_N-1];

    //==========================================================================
    // Engine
    //==========================================================================
    localparam S_IDLE   = 4'd0,
               S_DLOAD  = 4'd1,   // fetching a 256-byte TensorMap
               S_DWAIT  = 4'd2,
               S_DPARSE = 4'd8,
               S_TSETUP = 4'd3,
               S_TSET2  = 4'd9,
               S_TRUN   = 4'd4,
               S_TDRAIN = 4'd5,
               S_DONE   = 4'd6,
               S_FAULT  = 4'd7,
               S_XRUN   = 4'd10,  // transposed: one column block, down the rows
               S_XDRAIN = 4'd11,
               S_XWRITE = 4'd12;

    reg [3:0] st;

    // --- descriptor load state ---
    reg [DESC_W-1:0] dl_slot;
    reg [63:0]       dl_addr;
    reg [4:0]        dl_word;        // 0..31
    reg [63:0]       dl_buf [0:31];

    // --- transfer state ---
    reg               t_store;
    // Which trap a fault raises. The engine used to hard-code DESCRIPTOR_FAULT
    // because geometry was the only thing it could refuse; `t.st.async` needs
    // a different code, and a fault path with one code cannot grow one.
    reg  [7:0]        t_fcode;
    reg [DESC_W-1:0]  t_desc;
    reg [WIN_W-1:0]   t_win;
    reg [63:0]        t_c0, t_c1;
    reg [2:0]         t_esz;
    reg               t_xpose;
    reg               t_bar_en;
    reg [BAR_W-1:0]   t_bar;
    reg [RCTX_W-1:0]  t_rctx;
    reg [63:0]        t_pc;

    reg [31:0]        t_rows, t_cols;
    reg [31:0]        t_row_bytes;    // cols * elem
    reg [31:0]        t_row_units;    // row_bytes / 8
    wire [31:0]       t_valid_bytes;  // in-bounds bytes of the current row
    reg [31:0]        t_row;          // current row
    reg [31:0]        t_unit;         // current 8-byte unit in the row
    reg [7:0]         t_wbase;

    // outstanding reads
    reg [NTAG-1:0]    tg_v;
    reg [7:0]         tg_vreg  [0:NTAG-1];
    reg [3:0]         tg_lane  [0:NTAG-1];   // first of two lanes
    reg [7:0]         tg_keep  [0:NTAG-1];   // per-byte keep mask

    assign busy = (st != S_IDLE);

    reg [TAG_W-1:0] free_tag;
    reg             have_tag;
    always @(*) begin
        free_tag = {TAG_W{1'b0}}; have_tag = 1'b0;
        for (i = NTAG-1; i >= 0; i = i - 1)
            if (!tg_v[i]) begin free_tag = i[TAG_W-1:0]; have_tag = 1'b1; end
    end

    //==========================================================================
    // Address of the current unit
    //
    // The row base is WALKED, not recomputed. Written as
    //     base + (c0 + row)*stride0 + c1*stride1
    // every row costs a 64x64 multiply, and the multiplier then sits in the
    // inner loop of every transfer. The starting address needs the multiplies
    // once, in a setup cycle where there is no throughput to lose; after that a
    // row is one add. This is what a DMA engine is supposed to look like.
    //==========================================================================
    reg  [63:0] r_row_base;
    wire [63:0] unit_addr = r_row_base + {29'd0, t_unit, 3'd0};

    // Out of bounds in the row dimension: the whole row is padding.
    wire row_oob  = ((t_c0 + {32'd0, t_row}) >= d_e0[t_desc]);
    wire [31:0] unit_off = {t_unit[28:0], 3'd0};
    wire unit_oob = row_oob || (unit_off >= t_valid_bytes);

    // Bytes of this unit that came from memory; the rest are padding.
    wire [31:0] rem = t_valid_bytes - unit_off;
    wire [3:0]  nkeep = (rem >= 32'd8) ? 4'd8 : rem[3:0];
    // Its own loop variable: a combinational block that shares a module-level
    // index with another one retriggers it, and simulation time stops.
    reg  [7:0]  keep_mask;
    integer km_i;
    always @(*) begin
        keep_mask = 8'd0;
        for (km_i = 0; km_i < 8; km_i = km_i + 1)
            if (km_i < nkeep) keep_mask[km_i] = 1'b1;
    end

    //==========================================================================
    // TRANSPOSED WRITE (ISA sec 9E mod[5])
    //
    // A source tile of R rows x C columns lands as C rows x R columns, flat
    // row-major: window element (c, r) = source element (r, c). Memory
    // returns 8-byte units of one source ROW, which are slices of 8/esz
    // different destination rows, and the window is written through
    // 32-bit lane enables -- so writing each element where it belongs
    // would clobber its neighbour in a 16-bit lane. Instead the engine
    // corner-turns: it reads one column block down every row into xbuf,
    // then writes the 8/esz completed destination rows, each as whole
    // lanes. xbuf is XR_MAX units, the hardware cost of the feature.
    //==========================================================================
    localparam [31:0] XR_MAX = 32'd16;
    reg  [16*64-1:0] xbuf;             // unit r at [r*64 +: 64]
    wire [7:0]  x_rsp_tagrow = tg_vreg[mem_rsp_tag];
    wire [3:0]  x_rsp_row    = x_rsp_tagrow[3:0];
    wire [7:0]  x_rsp_kbits  = tg_keep[mem_rsp_tag];
    reg  [63:0] x_rsp_keep;
    integer xk;
    always @(*)
        for (xk = 0; xk < 8; xk = xk + 1)
            x_rsp_keep[xk*8 +: 8] = {8{x_rsp_kbits[xk]}};
    reg  [63:0] r_base0;
    reg  [3:0]  x_q;
    // 16-bit elements only (the MMA operand types): a unit is four
    // elements, so one column block is four destination rows, and a
    // destination row of R elements is 2R bytes with R in {4, 8, 16}.
    // Restricting the size is what keeps the placement cheap: the first
    // version accepted 1-, 2- and 4-byte elements at any offset, and its
    // byte crossbar more than doubled the engine's gate count.
    wire [3:0]  x_per_unit   = 4'd4;
    wire [31:0] x_drow_bytes = t_rows << 1;
    wire [31:0] x_ddst  = ((t_unit << 2) + {28'd0, x_q}) * x_drow_bytes;
    wire [7:0]  x_vreg  = t_wbase + x_ddst[13:6];
    wire [3:0]  x_lane0 = x_ddst[5:2];
    wire [31:0] x_b0    = d_b0[t_desc];
    wire        xrow_ok = (t_esz == 3'd1) &&
                          ((x_b0 == 32'd4) || (x_b0 == 32'd8) || (x_b0 == 32'd16));
    reg  [VLEN-1:0]      x_lanes;
    reg  [VLEN*ELEN-1:0] x_data;
    integer xh, xl;
    always @(*) begin
        x_lanes = {VLEN{1'b0}};
        for (xl = 0; xl < VLEN; xl = xl + 1)
            if ((xl >= x_lane0) && (xl < x_lane0 + (x_drow_bytes >> 2)))
                x_lanes[xl] = 1'b1;
        // The destination row, repeated across the register: halfword k
        // holds element x_q of source row (k mod R). The lane enables pick
        // the copy at the row's own offset, so nothing has to be shifted
        // into place. xbuf's only reader, and built only with the feature,
        // so an engine without it has none and synthesis drops xbuf.
        x_data = {(VLEN*ELEN){1'b0}};
        if (HAS_XPOSE != 0)
            for (xh = 0; xh < 2*VLEN; xh = xh + 1)
                case (t_rows[4:0])
                5'd4:  x_data[xh*16 +: 16] = xbuf[((xh & 3)  * 64 + x_q * 16) +: 16];
                5'd8:  x_data[xh*16 +: 16] = xbuf[((xh & 7)  * 64 + x_q * 16) +: 16];
                default: x_data[xh*16 +: 16] = xbuf[((xh & 15) * 64 + x_q * 16) +: 16];
                endcase
    end

    // Destination inside the window
    wire [31:0] flat  = t_row * t_row_bytes + unit_off;
    wire [7:0]  d_vreg = t_wbase + flat[31:6];
    wire [3:0]  d_lane = flat[5:2];

    // A returning unit and a padding unit both want the vector write port. The
    // return has priority -- it is the path that releases a tag and lets the
    // transfer finish -- and the padding unit simply waits a cycle. Letting the
    // padding write win instead silently dropped returned data, which looked
    // like a correct transfer of the wrong numbers.
    wire rsp_write = mem_rsp_valid && (st != S_DLOAD) && (st != S_DWAIT);

    wire issue_ok = ((st == S_TRUN) || (st == S_XRUN)) && !unit_oob && have_tag;
    wire dl_issue = (st == S_DLOAD);

    assign mem_req_valid = dl_issue || issue_ok;
    assign mem_req_addr  = dl_issue ? (dl_addr + {56'd0, dl_word, 3'd0}) : unit_addr;
    assign mem_req_we    = 1'b0;
    assign mem_req_be    = {MEM_BYTES{1'b0}};
    assign mem_req_wdata = {MEM_DW{1'b0}};
    // The descriptor load is serial -- one word outstanding at a time -- so it
    // needs exactly one tag. It happens once per descriptor and never on a
    // critical path, and a parallel fetch would have forced a wider tag through
    // the whole network for no benefit.
    assign mem_req_tag   = dl_issue ? {TAG_W{1'b0}} : free_tag;

    wire mem_fire = mem_req_valid && mem_req_ready;

    //==========================================================================
    // Descriptor parsing, once the 256 bytes are in
    //==========================================================================
    wire [7:0] p_rank  = dl_buf[13][32 +: 8];   // byte 108
    wire [7:0] p_dtype = dl_buf[13][40 +: 8];   // byte 109
    wire [7:0] p_pad   = dl_buf[13][48 +: 8];   // byte 110

    //==========================================================================
    always @(posedge clk) begin
        if (!rst_n || flush) begin
            st        <= S_IDLE;
            win_v     <= {WIN_N{1'b0}};
            desc_v    <= {DESC_N{1'b0}};
            tg_v      <= {NTAG{1'b0}};
            // Reset so that, with HAS_XPOSE = 0, it is a constant 0 that
            // synthesis can prove -- and then remove xbuf and the transposed
            // states rather than build logic no command can reach.
            t_xpose   <= 1'b0;
            vw_en     <= 1'b0;
            done      <= 1'b0;
            fault      <= 1'b0;
            t_fcode    <= `TNCA_TRAP_DESC;
            fault_rctx <= {RCTX_W{1'b0}};
            fault_pc   <= 64'd0;
            done_bar_en <= 1'b0;
            done_err   <= 1'b0;
        end else begin
            vw_en <= 1'b0;
            done  <= 1'b0;
            fault <= 1'b0;
            done_bar_en <= 1'b0;
            // Cleared every cycle, like `done` itself. Left unreset it stayed
            // X until the first fault, and `txf_done_en && !txf_done_err` is
            // X rather than 1 when a transfer completes -- so every
            // completion poisoned the barrier's operation count instead of
            // decrementing it.
            done_err    <= 1'b0;

            //------------------------------------------------------------------
            // configuration
            //------------------------------------------------------------------
            if (cfg_valid) begin
                case (cfg_op)
                2'd1: desc_v[cfg_desc] <= 1'b0;
                2'd2: begin
                    win_v[cfg_win]    <= 1'b1;
                    win_base[cfg_win] <= cfg_wbase;
                    win_len[cfg_win]  <= cfg_wlen;
                end
                2'd3: win_v[cfg_win] <= 1'b0;
                default: ;   // td.load is a sequence, handled below
                endcase
            end

            //------------------------------------------------------------------
            // memory responses
            //------------------------------------------------------------------
            if (mem_rsp_valid) begin
                if (st == S_DLOAD || st == S_DWAIT) begin
                    // captured by the descriptor sequencer below
                end else if (t_xpose) begin
                    // Into the corner-turn buffer, not the window: a unit
                    // holds one source row's slice of 8/esz columns, which
                    // is a slice of 8/esz DIFFERENT destination rows.
                    // Flat, and indexed through a wire: iverilog evaluates a
                    // part-select taken straight off an array word as zero
                    // (see the d_s0 check in S_TSETUP).
                    xbuf[x_rsp_row*64 +: 64] <= mem_rsp_rdata & x_rsp_keep;
                    tg_v[mem_rsp_tag] <= 1'b0;
                end else begin
                    // place the unit into its window register, keeping only the
                    // in-bounds bytes; the rest were already zero
                    vw_en      <= 1'b1;
                    vw_rctx    <= t_rctx;
                    vw_idx     <= tg_vreg[mem_rsp_tag];
                    vw_lane_en <= ({{(VLEN-2){1'b0}}, 2'b11} << tg_lane[mem_rsp_tag]);
                    vw_data    <= {(VLEN*ELEN){1'b0}};
                    for (i = 0; i < 8; i = i + 1)
                        if (tg_keep[mem_rsp_tag][i])
                            vw_data[(tg_lane[mem_rsp_tag]*4 + i)*8 +: 8]
                                <= mem_rsp_rdata[i*8 +: 8];
                    tg_v[mem_rsp_tag] <= 1'b0;
                end
            end

            case (st)

            //---------------------------------------------------------------
            S_IDLE: begin
                if (cfg_valid && (cfg_op == 2'd0)) begin
                    dl_slot <= cfg_desc;
                    dl_addr <= cfg_addr;
                    dl_word <= 5'd0;
                    st      <= S_DLOAD;
                end else if (cmd_valid) begin
                    t_store  <= cmd_store;
                    t_fcode  <= `TNCA_TRAP_DESC;
                    t_desc   <= cmd_desc;
                    t_win    <= cmd_win;
                    t_c0     <= cmd_c0;
                    t_c1     <= cmd_c1;
                    t_esz    <= cmd_esz;
                    // A constant 0 on an engine built without the feature, so
                    // synthesis removes the corner-turn buffer and its states
                    // instead of keeping logic nothing can reach.
                    t_xpose  <= cmd_xpose && (HAS_XPOSE != 0);
                    t_bar_en <= cmd_bar_en;
                    t_bar    <= cmd_bar;
                    t_rctx   <= cmd_rctx;
                    t_pc     <= cmd_pc;
                    // `t.st.async` is encodable (ISA sec 9E func 0x08) and this
                    // engine has no vector READ port, so it cannot perform one.
                    // It used to be excluded from `issue_ok` alone, which meant
                    // a store entered S_TRUN, issued nothing, never saw a
                    // response and sat there until the testbench timed out --
                    // a decodable instruction that deadlocked the core. An
                    // unimplemented transfer direction is a fault: the waiter
                    // is released with the barrier's error bit set and the
                    // issuing context traps, which is what sec 1.6 requires of
                    // a capability the machine does not have.
                    //
                    // Predication does not rescue it. A masked-off command
                    // performs no access and still completes once (sec 9A.3),
                    // but that is about the ACCESS, not about whether the
                    // encoding is executable here; accepting a predicated-off
                    // store would make the same program legal or illegal
                    // depending on a runtime mask.
                    // A window layout this engine does not have, or a
                    // transposed write on an engine built without one, is
                    // refused the same way: the encoding exists, the
                    // machine does not execute it, and running it anyway
                    // would transfer the tile in some other shape and call
                    // it done.
                    if (cmd_store || (cmd_layout != 2'd0) ||
                        (cmd_xpose && (HAS_XPOSE == 0))) begin
                        t_fcode <= `TNCA_TRAP_ILLEGAL;
                        st      <= S_FAULT;
                    end else
                    // A fully-masked-off command reaches S_DONE without a
                    // single memory access or window write, and still signals
                    // its barrier once -- which is what keeps a static
                    // operation count correct under predication (sec 9A.3).
                    st       <= cmd_active ? S_TSETUP : S_DONE;
                end
            end

            //---------------------------------------------------------------
            // A 256-byte TensorMap is 32 words; the snapshot is taken once.
            S_DLOAD: begin
                if (mem_fire) st <= S_DWAIT;
            end

            S_DWAIT: begin
                if (mem_rsp_valid) begin
                    dl_buf[dl_word] <= mem_rsp_rdata;
                    if (dl_word == 5'd31) st <= S_DPARSE;
                    else begin
                        dl_word <= dl_word + 1'b1;
                        st      <= S_DLOAD;
                    end
                end
            end

            S_DPARSE: begin
                d_base[dl_slot]  <= dl_buf[0];
                d_e0[dl_slot]    <= dl_buf[1];
                d_e1[dl_slot]    <= dl_buf[2];
                d_s0[dl_slot]    <= dl_buf[6];
                d_s1[dl_slot]    <= dl_buf[7];
                d_b0[dl_slot]    <= dl_buf[11][31:0];
                d_b1[dl_slot]    <= dl_buf[11][63:32];
                d_rank[dl_slot]  <= p_rank;
                d_dtype[dl_slot] <= p_dtype;
                d_pad[dl_slot]   <= p_pad;
                desc_v[dl_slot]  <= 1'b1;
                st <= S_DONE;
            end

            //---------------------------------------------------------------
            // Bounds and geometry, before a single transaction is issued.
            S_TSETUP: begin
                t_rows      <= d_b0[t_desc];
                t_cols      <= d_b1[t_desc];
                t_row_bytes <= d_b1[t_desc] << t_esz;
                t_row_units <= (d_b1[t_desc] << t_esz) >> 3;
                t_wbase     <= win_base[t_win];
                t_row       <= 32'd0;
                t_unit      <= 32'd0;

                // The row SIZE being a whole number of 8-byte units is not
                // enough: every row must also START on an 8-byte boundary, or
                // the returned units land in the window shifted. That needs
                // the base and the row stride both aligned. Without this the
                // engine returned wrong data for an odd row stride and raised
                // nothing -- a wrong answer wearing the shape of a right one,
                // which is the one failure mode sec 1.6 exists to prevent.
                if (!desc_v[t_desc] || !win_v[t_win] ||
                    (d_rank[t_desc] > 8'd2) || (d_rank[t_desc] == 8'd0) ||
                    (((d_b1[t_desc] << t_esz) & 32'd7) != 32'd0) ||
                    // Masked rather than bit-selected: Verilog-2005 does not
                    // allow a part-select directly off an array word, and
                    // iverilog accepted `d_s0[t_desc][2:0]` while evaluating
                    // it as zero, so the check silently never fired.
                    ((d_base[t_desc] & 64'd7) != 64'd0) ||
                    ((d_s0[t_desc] & 64'd7) != 64'd0) ||
                    ((d_s1[t_desc]) != (64'd1 << t_esz)) ||
                    ((((d_b0[t_desc] * (d_b1[t_desc] << t_esz)) + 63) >> 6)
                      > {24'd0, win_len[t_win]}) ||
                    // TRANSPOSED WRITE geometry: a 2-D tile of 16-bit
                    // elements with 4, 8 or 16 source rows, so every
                    // destination row is whole lanes inside ONE register.
                    (t_xpose && ((d_rank[t_desc] != 8'd2) ||
                                 (d_b0[t_desc] > XR_MAX) ||
                                 !xrow_ok))) begin
                    st <= S_FAULT;
                end else begin
                    st <= S_TSET2;
                end
            end

            //---------------------------------------------------------------
            // The only place the stride multiplies happen: once per command.
            S_TSET2: begin
                r_row_base <= d_base[t_desc]
                            + t_c0 * d_s0[t_desc]
                            + t_c1 * d_s1[t_desc];
                if (HAS_XPOSE != 0) begin     // read only by the transposed walk
                    r_base0 <= d_base[t_desc]
                             + t_c0 * d_s0[t_desc]
                             + t_c1 * d_s1[t_desc];
                    x_q     <= 4'd0;
                end
                st <= t_xpose ? S_XRUN : S_TRUN;
            end

            //---------------------------------------------------------------
            // One 8-byte unit per cycle. Out-of-bounds units are written as
            // zeros with no memory access at all -- padding is not a masked
            // read, it is the absence of one.
            S_TRUN: begin
                // Padding is the ABSENCE of an access, not a masked one: an
                // out-of-bounds unit issues no memory request at all and writes
                // zeros straight into the window.
                if (unit_oob && !rsp_write) begin
                    vw_en      <= 1'b1;
                    vw_rctx    <= t_rctx;
                    vw_idx     <= d_vreg;
                    vw_lane_en <= ({{(VLEN-2){1'b0}}, 2'b11} << d_lane);
                    vw_data    <= {(VLEN*ELEN){1'b0}};
                end

                if ((unit_oob && !rsp_write) || mem_fire) begin
                    if (!unit_oob) begin
                        tg_v[free_tag]    <= 1'b1;
                        tg_vreg[free_tag] <= d_vreg;
                        tg_lane[free_tag] <= d_lane;
                        tg_keep[free_tag] <= keep_mask;
                    end
                    if (t_unit == t_row_units - 1) begin
                        t_unit <= 32'd0;
                        if (t_row == t_rows - 1) st <= S_TDRAIN;
                        else begin
                            t_row      <= t_row + 1'b1;
                            r_row_base <= r_row_base + d_s0[t_desc];
                        end
                    end else begin
                        t_unit <= t_unit + 1'b1;
                    end
                end
            end

            //---------------------------------------------------------------
            // TRANSPOSED WRITE. Column block t_unit (8/esz source columns)
            // is read down all t_rows source rows, one 8-byte unit each,
            // into xbuf[row]. Out-of-bounds units are zeros without an
            // access, exactly as in S_TRUN.
            S_XRUN: begin
                if (unit_oob || mem_fire) begin
                    if (unit_oob)
                        xbuf[t_row[3:0]*64 +: 64] <= 64'd0;
                    else begin
                        tg_v[free_tag]    <= 1'b1;
                        tg_vreg[free_tag] <= {4'd0, t_row[3:0]};
                        tg_keep[free_tag] <= keep_mask;
                    end
                    if (t_row == t_rows - 1) st <= S_XDRAIN;
                    else begin
                        t_row      <= t_row + 1'b1;
                        r_row_base <= r_row_base + d_s0[t_desc];
                    end
                end
            end

            S_XDRAIN: begin
                if ((tg_v == {NTAG{1'b0}}) && !mem_rsp_valid) begin
                    x_q <= 4'd0;
                    st  <= S_XWRITE;
                end
            end

            // Destination row c = t_unit*(8/esz) + x_q holds element x_q of
            // every buffered unit, in source-row order: window element
            // (c, r) = source element (r, c). One whole-lane write per row.
            S_XWRITE: begin
                vw_en      <= 1'b1;
                vw_rctx    <= t_rctx;
                vw_idx     <= x_vreg;
                vw_lane_en <= x_lanes;
                vw_data    <= x_data;
                if (x_q == x_per_unit - 1) begin
                    if (t_unit == t_row_units - 1) st <= S_DONE;
                    else begin
                        t_unit     <= t_unit + 1'b1;
                        t_row      <= 32'd0;
                        r_row_base <= r_base0;
                        st         <= S_XRUN;
                    end
                end else
                    x_q <= x_q + 1'b1;
            end

            //---------------------------------------------------------------
            S_TDRAIN: begin
                if ((tg_v == {NTAG{1'b0}}) && !mem_rsp_valid) st <= S_DONE;
            end

            //---------------------------------------------------------------
            S_DONE: begin
                done        <= 1'b1;
                done_bar_en <= t_bar_en;
                done_bar    <= t_bar;
                t_bar_en    <= 1'b0;
                st          <= S_IDLE;
            end

            //---------------------------------------------------------------
            // A geometry this engine cannot execute is a fault, never an
            // approximation (sec 1.6): a wrong answer that looks right is the
            // one failure mode a capability model exists to prevent.
            S_FAULT: begin
                fault       <= 1'b1;
                fault_code  <= t_fcode;
                fault_rctx  <= t_rctx;
                fault_pc    <= t_pc;
                done        <= 1'b1;
                done_err    <= 1'b1;
                done_bar_en <= t_bar_en;
                done_bar    <= t_bar;
                t_bar_en    <= 1'b0;
                st          <= S_IDLE;
            end

            default: st <= S_IDLE;
            endcase
        end
    end

    //==========================================================================
    // Valid-byte count for the current row, recomputed combinationally
    //==========================================================================
    wire [63:0] cols_left = (d_e1[t_desc] > t_c1) ? (d_e1[t_desc] - t_c1) : 64'd0;
    wire [31:0] cols_ok   = (cols_left >= {32'd0, t_cols}) ? t_cols : cols_left[31:0];
    assign t_valid_bytes  = row_oob ? 32'd0 : (cols_ok << t_esz);

    wire _unused = &{1'b0, t_store, d_dtype[0], d_pad[0], 1'b0};

endmodule
