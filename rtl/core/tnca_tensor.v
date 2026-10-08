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
// GEOMETRY (ISA sec 9D.3, 9E; R38 tensor descriptors, aligned with the TMA
// contract Triton exposes on NVIDIA GPUs)
//
// Rank 1 to 5. Dimension 0 is the outermost, dimension rank-1 the contiguous
// one. A box of block_shape[0..r-1] elements is moved between memory and a
// window in which it is flat row-major: a "row" is one run of the contiguous
// dimension, and the rows are the outer dimensions in odometer order.
// Internally every rank is normalised to five positions with the contiguous
// dimension last; the positions a lower rank does not use have extent 1,
// block 1, stride 0 and coordinate 0, so they are always in bounds and the
// walk below never needs to know the rank.
//
// Coordinates are SIGNED and may point anywhere: before the tensor, past it,
// or straddling an edge, and the contiguous coordinate need not be aligned to
// anything. An element is in bounds when 0 <= coord + index < extent in every
// dimension. Out-of-bounds elements of a load are filled -- zero, or the
// dtype's quiet NaN under padding mode 3 -- and are never read; out-of-bounds
// elements of a store are not written. Padding mode 2 turns any
// out-of-bounds element into DESCRIPTOR_FAULT instead.
//
// What the engine still requires (each checked, each a DESCRIPTOR_FAULT):
// an innermost stride of one element, a box row that is a whole number of
// 8-byte units, every block dimension in 1..256, and a box that fits its
// window. TENSOR_MAX_RANK reports 5. The base and the outer strides may have
// any alignment -- a strict superset of the TMA contract, which wants 16.
//
// A box row that does not start on an 8-byte boundary -- a misaligned base,
// stride or contiguous coordinate, which can differ row to row -- is the one
// case that is not a straight copy: each 8-byte unit of the row is then
// assembled from two memory words. Loads fetch both words of such a unit and
// shift; stores shift the window's units the other way and write each memory
// word through its byte enables, so a store touches exactly the in-bounds
// bytes of its box.
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
    // The transposed window write (ISA sec 9E mod[5], capability
    // `tensor_transpose`). Off, the corner-turn buffer and its states are
    // not built.
    parameter HAS_XPOSE = 0
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     flush,

    // ---- descriptor / window management (single cycle, from X) ----
    input  wire                     cfg_valid,
    input  wire [1:0]               cfg_op,      // 0 td.load 1 td.release
                                                 // 2 w.def   3 w.release
    input  wire [DESC_W-1:0]        cfg_desc,
    input  wire [WIN_W-1:0]         cfg_win,
    input  wire [63:0]              cfg_addr,    // TensorMap address
    input  wire [7:0]               cfg_wbase,
    input  wire [7:0]               cfg_wlen,

    // ---- transfer command ----
    input  wire                     cmd_valid,
    input  wire                     cmd_store,
    input  wire [DESC_W-1:0]        cmd_desc,
    input  wire [WIN_W-1:0]         cmd_win,
    // The coordinates: Sb, Sb+1, ... Sb+4 (ISA sec 9E), one per descriptor
    // dimension in descriptor order. Only the first `rank` are used.
    input  wire [63:0]              cmd_c0,
    input  wire [63:0]              cmd_c1,
    input  wire [63:0]              cmd_c2,
    input  wire [63:0]              cmd_c3,
    input  wire [63:0]              cmd_c4,
    input  wire [5:0]               cmd_sb,      // which register Sb is
    input  wire [2:0]               cmd_esz,     // log2 element bytes
    input  wire                     cmd_xpose,   // mod[5]
    input  wire [1:0]               cmd_layout,  // mod[4:3]; only 0 exists
    input  wire                     cmd_bar_en,
    input  wire [BAR_W-1:0]         cmd_bar,
    input  wire [RCTX_W-1:0]        cmd_rctx,
    // The PC of the instruction that issued this transfer. A geometry fault
    // is detected cycles later, after the core has moved on, and TRAP_PC must
    // name the instruction that caused it -- not whichever one the core
    // happens to be executing when the fault surfaces.
    input  wire [63:0]              cmd_pc,
    // The issuing instruction's predicate (sec 2.2). All-or-nothing: see
    // the ISA's sec 9E note. An inactive command performs no access and
    // still signals its completion once.
    input  wire                     cmd_active,

    output wire                     busy,

    output reg                      done,
    output reg                      done_bar_en,
    output reg  [BAR_W-1:0]         done_bar,
    output reg                      fault,
    output reg  [7:0]               fault_code,
    // A fault is asynchronous: by the time the geometry is checked the core
    // may be issuing from another RCTX. These say which RCTX and which PC.
    output reg  [RCTX_W-1:0]        fault_rctx,
    output reg  [63:0]              fault_pc,
    // Set together with `done` when the transfer faulted. The barrier
    // releases its waiter rather than leaving it to block forever (ISA sec
    // 10), and this is how it knows the operation it is counting failed.
    output reg                      done_err,

    // ---- window write port (into the vector register file) ----
    output reg                      vw_en,
    output reg  [RCTX_W-1:0]        vw_rctx,
    output reg  [7:0]               vw_idx,
    output reg  [VLEN-1:0]          vw_lane_en,
    output reg  [VLEN*ELEN-1:0]     vw_data,

    // ---- window read port (a store's source), held until vr_valid ----
    output wire                     vr_req,
    output wire [RCTX_W-1:0]        vr_rctx,
    output wire [7:0]               vr_idx,
    input  wire                     vr_valid,
    input  wire [VLEN*ELEN-1:0]     vr_data,

    // ---- memory port (one word request per cycle) ----
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
    // Descriptor table -- the snapshot, parsed. Dimension j of the
    // descriptor is d_*j, in descriptor order (0 outermost).
    //==========================================================================
    reg [DESC_N-1:0] desc_v;
    reg [63:0] d_base   [0:DESC_N-1];
    reg [63:0] d_e0 [0:DESC_N-1], d_e1 [0:DESC_N-1], d_e2 [0:DESC_N-1],
               d_e3 [0:DESC_N-1], d_e4 [0:DESC_N-1];
    reg [63:0] d_s0 [0:DESC_N-1], d_s1 [0:DESC_N-1], d_s2 [0:DESC_N-1],
               d_s3 [0:DESC_N-1], d_s4 [0:DESC_N-1];
    // The box's element count, saturated at 2^20, worked out while the
    // TensorMap's tail is still arriving (S_DWAIT, words 14-18: one multiply a
    // word), so the per-transfer window check is a shift and a compare rather
    // than five multiplies in series -- which, in S_TCHK, was the device's
    // worst path once the matrix loop was gone (10.1 ns post-synthesis).
    reg [20:0] d_boxel [0:DESC_N-1];
    reg [20:0] dl_boxel, t_boxel;
    reg [31:0] d_b0 [0:DESC_N-1], d_b1 [0:DESC_N-1], d_b2 [0:DESC_N-1],
               d_b3 [0:DESC_N-1], d_b4 [0:DESC_N-1];
    reg [7:0]  d_rank   [0:DESC_N-1];
    reg [7:0]  d_dtype  [0:DESC_N-1];
    reg [7:0]  d_pad    [0:DESC_N-1];

    //==========================================================================
    // Engine states
    //==========================================================================
    localparam S_IDLE   = 5'd0,
               S_DLOAD  = 5'd1,   // fetching a 256-byte TensorMap
               S_DWAIT  = 5'd2,
               S_DPARSE = 5'd3,
               S_TSETUP = 5'd4,   // copy and normalise the snapshot
               S_TCHK   = 5'd5,   // geometry
               S_TMUL   = 5'd6,   // starting address, one multiply a cycle
               S_TRUN   = 5'd7,   // load: one box unit at a time
               S_TDRAIN = 5'd8,
               S_SRUN   = 5'd9,   // store: one memory word at a time
               S_SREAD  = 5'd10,  // store: reading a window register
               S_SDRAIN = 5'd11,
               S_DONE   = 5'd12,
               S_FAULT  = 5'd13,
               S_XRUN   = 5'd14,  // transposed: one column block, down the rows
               S_XDRAIN = 5'd15,
               S_XWRITE = 5'd16;

    reg [4:0] st;

    // --- descriptor load state ---
    reg [DESC_W-1:0] dl_slot;
    reg [63:0]       dl_addr;
    reg [4:0]        dl_word;        // 0..31
    reg [63:0]       dl_buf [0:31];

    // --- command, as issued ---
    reg               t_store;
    // Which trap a fault raises: DESCRIPTOR_FAULT for geometry, and
    // ILLEGAL_INSTRUCTION for an encoding this engine does not execute.
    reg  [7:0]        t_fcode;
    reg [DESC_W-1:0]  t_desc;
    reg [WIN_W-1:0]   t_win;
    reg [63:0]        cc0, cc1, cc2, cc3, cc4;   // coordinates, descriptor order
    reg [5:0]         t_sb;
    reg [2:0]         t_esz;
    reg               t_xpose;
    reg               t_bar_en;
    reg [BAR_W-1:0]   t_bar;
    reg [RCTX_W-1:0]  t_rctx;
    reg [63:0]        t_pc;

    // --- the transfer, normalised: position 4 is the contiguous dimension,
    //     positions below 5-rank are unused (extent 1, block 1, stride 0,
    //     coordinate 0) ---
    reg [63:0]        t_base;
    reg [63:0]        t_e0, t_e1, t_e2, t_e3, t_e4;
    reg [63:0]        t_s0, t_s1, t_s2, t_s3, t_s4;
    reg [31:0]        t_b0, t_b1, t_b2, t_b3, t_b4;
    reg [63:0]        t_c0, t_c1, t_c2, t_c3, t_c4;
    reg [7:0]         t_rank;
    reg [7:0]         t_dtype;
    reg [7:0]         t_pad;
    reg [7:0]         t_wbase;
    reg [7:0]         t_wlen;

    // --- derived per transfer ---
    reg [63:0]        t_E;           // contiguous extent in bytes
    reg [63:0]        t_CB;          // contiguous coordinate in bytes (signed)
    reg [31:0]        t_row_bytes;   // block[4] * element size
    reg [31:0]        t_row_units;   // t_row_bytes / 8
    reg [63:0]        t_fill;        // what an out-of-bounds load byte becomes

    // --- the starting-address accumulation (S_TMUL) ---
    reg [2:0]         t_mp;          // position being multiplied
    reg [63:0]        t_acc;

    // --- the walk: an odometer over positions 0..3, with the address of
    //     each level's current index kept so a carry is one add ---
    reg [31:0]        t_i0, t_i1, t_i2, t_i3;
    reg [63:0]        t_P0, t_P1, t_P2, t_P3;  // t_P3: this row's element-0 address
    reg [31:0]        t_unit;        // load: box unit; store: memory word
    reg [31:0]        t_flat;        // this row's byte offset in the window
    reg               t_half;        // load: second word of a split unit
    reg [TAG_W-2:0]   t_slot;

    assign busy = (st != S_IDLE);

    //==========================================================================
    // The fill value: zero, or the dtype's quiet NaN repeated across the
    // 8-byte unit. Elements are aligned inside a unit, so a byte of the unit
    // takes the fill byte in the same position. ISA sec 9D.3 dtype codes.
    //==========================================================================
    reg [63:0] fill_nan;
    reg        dtype_float;
    reg [2:0]  dtype_lg;             // log2 bytes of the dtype
    always @(*) begin
        fill_nan = 64'd0; dtype_float = 1'b0; dtype_lg = 3'd0;
        case (t_dtype)
        8'd0: begin fill_nan = {4{16'h7E00}};     dtype_float = 1'b1; dtype_lg = 3'd1; end // FP16
        8'd1: begin fill_nan = {4{16'h7FC0}};     dtype_float = 1'b1; dtype_lg = 3'd1; end // BF16
        8'd2: begin fill_nan = {2{32'h7FC00000}}; dtype_float = 1'b1; dtype_lg = 3'd2; end // FP32
        8'd3: begin dtype_lg = 3'd0; end                                                  // INT8
        8'd4: begin dtype_lg = 3'd1; end                                                  // INT16
        8'd5: begin dtype_lg = 3'd2; end                                                  // INT32
        8'd6: begin dtype_lg = 3'd3; end                                                  // INT64
        8'd7: begin fill_nan = {8{8'h7F}};        dtype_float = 1'b1; dtype_lg = 3'd0; end // FP8 E4M3
        8'd8: begin fill_nan = {8{8'h7F}};        dtype_float = 1'b1; dtype_lg = 3'd0; end // FP8 E5M2
        8'd9: begin fill_nan = 64'h7FF8000000000000; dtype_float = 1'b1; dtype_lg = 3'd3; end // FP64
        default: ;
        endcase
    end

    //==========================================================================
    // Row bounds. A row is in bounds when every outer position's coordinate
    // plus index lies in [0, extent). Signed: a negative coordinate is an
    // ordinary out-of-bounds position, not a huge one.
    //==========================================================================
    function in_dim;
        input [63:0] c;
        input [31:0] i;
        input [63:0] e;
        reg   [63:0] g;
        begin
            g = c + {32'd0, i};
            in_dim = !g[63] && (g < e);
        end
    endfunction

    wire t_rowv = in_dim(t_c0, t_i0, t_e0) && in_dim(t_c1, t_i1, t_e1) &&
                  in_dim(t_c2, t_i2, t_e2) && in_dim(t_c3, t_i3, t_e3);

    // The odometer's next step. Position 3 moves fastest.
    wire inc3 = (t_i3 + 32'd1 < t_b3);
    wire inc2 = !inc3 && (t_i2 + 32'd1 < t_b2);
    wire inc1 = !inc3 && !inc2 && (t_i1 + 32'd1 < t_b1);
    wire inc0 = !inc3 && !inc2 && !inc1 && (t_i0 + 32'd1 < t_b0);
    wire last_row = !inc3 && !inc2 && !inc1 && !inc0;

    //==========================================================================
    // Contiguous-dimension byte ranges. Offsets are relative to the address of
    // the row's element 0 (t_P3), in bytes, signed. The valid bytes are
    // [0, t_E); the box row is [t_CB, t_CB + t_row_bytes).
    //==========================================================================
    // Where this row's box starts within an 8-byte word: (row address +
    // contiguous offset) mod 8. Constant along a row, and possibly different
    // from row to row (an outer stride need not be a multiple of 8).
    wire [2:0]  r_m   = t_P3[2:0] + t_CB[2:0];
    // The box's first byte rounded down to its word, relative to t_P3.
    wire [63:0] t_CBa = t_CB - {61'd0, r_m};

    // A byte j (0..7) of an 8-byte group starting at byte offset `g` is
    // valid when lo <= j < hi, with lo/hi computed once per group.
    function [7:0] byte_range;
        input [63:0] lo;   // signed
        input [63:0] hi;   // signed
        integer      j;
        begin
            for (j = 0; j < 8; j = j + 1)
                byte_range[j] = ($signed({61'd0, j[2:0]}) >= $signed(lo)) &&
                                ($signed({61'd0, j[2:0]}) <  $signed(hi));
        end
    endfunction

    // ---- loads: unit t_unit of the box row ----
    // The unit's 8 bytes are box-row bytes [8k, 8k+8): memory bytes
    // [t_CB + 8k, t_CB + 8k + 8), taken from the words at t_CBa + 8k and,
    // when misaligned, the next one.
    wire [63:0] ld_g    = t_CB  + {29'd0, t_unit, 3'd0};  // first byte of the unit
    wire [63:0] ld_wlo  = t_CBa + {29'd0, t_unit, 3'd0};  // its low word
    wire [63:0] ld_whi  = ld_wlo + 64'd8;
    wire        ld_nlo  = t_rowv && $signed(ld_wlo) > -64'sd8 && $signed(ld_wlo) < $signed(t_E);
    wire        ld_nhi  = t_rowv && (r_m != 3'd0) &&
                          $signed(ld_whi) > -64'sd8 && $signed(ld_whi) < $signed(t_E);
    wire [7:0]  ld_keep = t_rowv ? byte_range(64'd0 - ld_g, t_E - ld_g) : 8'd0;
    wire [31:0] ld_off  = t_flat + {t_unit[28:0], 3'd0};  // byte offset in the window
    wire [7:0]  ld_vreg = t_wbase + ld_off[13:6];
    wire [3:0]  ld_lane = ld_off[5:2];

    // ---- stores: memory word t_unit of the box row ----
    // Word k is memory bytes [t_CBa + 8k, +8). Its bytes are written when
    // they are both inside the tensor and inside the box row.
    wire [31:0] st_nwords = t_row_units + ((r_m != 3'd0) ? 32'd1 : 32'd0);
    wire [63:0] st_g    = t_CBa + {29'd0, t_unit, 3'd0};
    wire [63:0] st_lo0  = 64'd0 - st_g;                   // tensor starts here
    wire [63:0] st_lo1  = t_CB - st_g;                    // box row starts here
    wire [63:0] st_hi0  = t_E - st_g;                     // tensor ends here
    wire [63:0] st_hi1  = t_CB + {32'd0, t_row_bytes} - st_g;  // box row ends
    wire [63:0] st_lo   = ($signed(st_lo0) > $signed(st_lo1)) ? st_lo0 : st_lo1;
    wire [63:0] st_hi   = ($signed(st_hi0) < $signed(st_hi1)) ? st_hi0 : st_hi1;
    wire [7:0]  st_be   = t_rowv ? byte_range(st_lo, st_hi) : 8'd0;
    // The window unit that word k's high bytes come from is unit k; its low
    // bytes come from unit k-1 (st_prev) when misaligned.
    wire        st_has_cur = (t_unit < t_row_units);
    wire [31:0] st_off  = t_flat + {t_unit[28:0], 3'd0};
    wire [7:0]  st_vreg = t_wbase + st_off[13:6];
    wire [2:0]  st_u    = st_off[5:3];

    //==========================================================================
    // Load slots. A box unit that needs memory takes one slot until both of
    // its words (or its one) are back; the tag is {slot, which word}.
    //==========================================================================
    localparam NSLOT  = NTAG / 2;
    localparam SLOT_W = TAG_W - 1;

    reg [NSLOT-1:0] sl_v, sl_nlo, sl_nhi, sl_glo, sl_ghi;
    reg [63:0]      sl_dlo  [0:NSLOT-1];
    reg [63:0]      sl_dhi  [0:NSLOT-1];
    reg [7:0]       sl_vreg [0:NSLOT-1];
    reg [3:0]       sl_lane [0:NSLOT-1];
    reg [7:0]       sl_keep [0:NSLOT-1];
    reg [2:0]       sl_m    [0:NSLOT-1];   // the row's misalignment

    wire [NSLOT-1:0] sl_done = sl_v & (sl_glo | ~sl_nlo) & (sl_ghi | ~sl_nhi);

    reg [SLOT_W-1:0] free_slot, done_slot;
    reg              have_slot, have_done;
    always @(*) begin
        free_slot = {SLOT_W{1'b0}}; have_slot = 1'b0;
        done_slot = {SLOT_W{1'b0}}; have_done = 1'b0;
        for (i = NSLOT-1; i >= 0; i = i - 1) begin
            if (!sl_v[i])   begin free_slot = i[SLOT_W-1:0]; have_slot = 1'b1; end
            if (sl_done[i]) begin done_slot = i[SLOT_W-1:0]; have_done = 1'b1; end
        end
    end

    // The finished unit: shift the word pair into place, then put the fill
    // wherever a byte is outside the tensor.
    wire [63:0]  dn_lo   = sl_dlo[done_slot];
    wire [63:0]  dn_hi   = sl_dhi[done_slot];
    wire [2:0]   dn_m    = sl_m[done_slot];
    wire [127:0] dn_pair = {dn_hi, dn_lo} >> {dn_m, 3'b000};
    wire [7:0]   dn_keep = sl_keep[done_slot];
    reg  [63:0]  dn_data;
    integer      dk;
    always @(*)
        for (dk = 0; dk < 8; dk = dk + 1)
            dn_data[dk*8 +: 8] = dn_keep[dk] ? dn_pair[dk*8 +: 8] : t_fill[dk*8 +: 8];

    //==========================================================================
    // Store state: the window register being read, and the previous unit
    //==========================================================================
    reg               sr_v;          // sr_data holds window register sr_idx
    reg [7:0]         sr_idx;
    reg [VLEN*ELEN-1:0] sr_data;
    reg [63:0]        st_prev;
    reg [NTAG-1:0]    tg_v;          // store: outstanding write responses

    wire [63:0] st_cur  = st_has_cur ? sr_data[st_u*64 +: 64] : 64'd0;
    wire [63:0] st_word = (r_m == 3'd0) ? st_cur
                        : ((st_cur << {r_m, 3'b000}) |
                           (st_prev >> {(4'd8 - {1'b0, r_m}), 3'b000}));
    wire        st_need_reg = st_has_cur && !(sr_v && (sr_idx == st_vreg));

    reg [TAG_W-1:0] free_tag;
    reg             have_tag;
    always @(*) begin
        free_tag = {TAG_W{1'b0}}; have_tag = 1'b0;
        for (i = NTAG-1; i >= 0; i = i - 1)
            if (!tg_v[i]) begin free_tag = i[TAG_W-1:0]; have_tag = 1'b1; end
    end

    assign vr_req  = (st == S_SREAD);
    assign vr_rctx = t_rctx;
    assign vr_idx  = st_vreg;

    //==========================================================================
    // TRANSPOSED WRITE (ISA sec 9E mod[5]) -- rank 2, 16-bit, unchanged.
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
    // It uses normalised positions 3 (rows) and 4 (columns).
    //==========================================================================
    localparam [31:0] XR_MAX = 32'd16;
    reg  [31:0] x_row;                 // current source row
    reg  [63:0] r_row_base;
    wire [63:0] x_unit_addr = r_row_base + {29'd0, t_unit, 3'd0};
    wire        x_row_oob   = !in_dim(t_c3, x_row, t_e3);
    wire [63:0] x_cols_left = (t_e4 > t_c4) ? (t_e4 - t_c4) : 64'd0;
    wire [31:0] x_cols_ok   = (x_cols_left >= {32'd0, t_b4}) ? t_b4 : x_cols_left[31:0];
    wire [31:0] x_valid_bytes = x_row_oob ? 32'd0 : (x_cols_ok << t_esz);
    wire [31:0] x_unit_off  = {t_unit[28:0], 3'd0};
    wire        x_unit_oob  = x_row_oob || (x_unit_off >= x_valid_bytes);
    wire [31:0] x_rem       = x_valid_bytes - x_unit_off;
    wire [3:0]  x_nkeep     = (x_rem >= 32'd8) ? 4'd8 : x_rem[3:0];
    reg  [7:0]  x_keep_mask;
    integer km_i;
    always @(*) begin
        x_keep_mask = 8'd0;
        for (km_i = 0; km_i < 8; km_i = km_i + 1)
            if (km_i < x_nkeep) x_keep_mask[km_i] = 1'b1;
    end

    reg  [16*64-1:0] xbuf;             // unit r at [r*64 +: 64]
    reg  [3:0]  xt_row  [0:NTAG-1];    // which buffered row a tag fills
    reg  [7:0]  xt_keep [0:NTAG-1];
    wire [3:0]  x_rsp_row    = xt_row[mem_rsp_tag];
    wire [7:0]  x_rsp_kbits  = xt_keep[mem_rsp_tag];
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
    wire [3:0]  x_per_unit   = 4'd4;
    wire [31:0] x_drow_bytes = t_b3 << 1;
    wire [31:0] x_ddst  = ((t_unit << 2) + {28'd0, x_q}) * x_drow_bytes;
    wire [7:0]  x_vreg  = t_wbase + x_ddst[13:6];
    wire [3:0]  x_lane0 = x_ddst[5:2];
    wire        xrow_ok = (t_esz == 3'd1) &&
                          ((t_b3 == 32'd4) || (t_b3 == 32'd8) || (t_b3 == 32'd16));
    reg  [VLEN-1:0]      x_lanes;
    reg  [VLEN*ELEN-1:0] x_data;
    integer xh, xl;
    always @(*) begin
        x_lanes = {VLEN{1'b0}};
        for (xl = 0; xl < VLEN; xl = xl + 1)
            if ((xl >= x_lane0) && (xl < x_lane0 + (x_drow_bytes >> 2)))
                x_lanes[xl] = 1'b1;
        x_data = {(VLEN*ELEN){1'b0}};
        if (HAS_XPOSE != 0)
            for (xh = 0; xh < 2*VLEN; xh = xh + 1)
                case (t_b3[4:0])
                5'd4:  x_data[xh*16 +: 16] = xbuf[((xh & 3)  * 64 + x_q * 16) +: 16];
                5'd8:  x_data[xh*16 +: 16] = xbuf[((xh & 7)  * 64 + x_q * 16) +: 16];
                default: x_data[xh*16 +: 16] = xbuf[((xh & 15) * 64 + x_q * 16) +: 16];
                endcase
    end

    //==========================================================================
    // Memory requests
    //==========================================================================
    // Load: unit t_unit, its low word unless t_half (or only the high word
    // is needed).
    wire        ld_issue_hi = t_half || !ld_nlo;
    wire        ld_req = (st == S_TRUN) && (ld_nlo || ld_nhi) &&
                         (t_half || have_slot);
    wire [63:0] ld_addr = t_P3 + (ld_issue_hi ? ld_whi : ld_wlo);
    wire [SLOT_W-1:0] ld_slot = t_half ? t_slot : free_slot;

    wire        st_req = (st == S_SRUN) && t_rowv && !st_need_reg &&
                         (st_be != 8'd0) && have_tag;

    wire        x_req  = (st == S_XRUN) && !x_unit_oob && have_tag;
    wire        dl_issue = (st == S_DLOAD);

    assign mem_req_valid = dl_issue || ld_req || st_req || x_req;
    assign mem_req_addr  = dl_issue ? (dl_addr + {56'd0, dl_word, 3'd0})
                         : ld_req   ? ld_addr
                         : st_req   ? (t_P3 + st_g)
                         :            x_unit_addr;
    assign mem_req_we    = st_req;
    assign mem_req_be    = st_req ? st_be : {MEM_BYTES{1'b0}};
    assign mem_req_wdata = st_req ? st_word : {MEM_DW{1'b0}};
    // The descriptor load is serial -- one word outstanding at a time -- so it
    // needs exactly one tag.
    assign mem_req_tag   = dl_issue ? {TAG_W{1'b0}}
                         : ld_req   ? {ld_slot, ld_issue_hi}
                         :            free_tag;

    wire mem_fire = mem_req_valid && mem_req_ready;

    //==========================================================================
    // Descriptor parsing, once the 256 bytes are in (ISA sec 9D.3)
    //==========================================================================
    wire [7:0] p_rank  = dl_buf[13][32 +: 8];   // byte 108
    wire [7:0] p_dtype = dl_buf[13][40 +: 8];   // byte 109
    wire [7:0] p_pad   = dl_buf[13][48 +: 8];   // byte 110

    // Block dimension j of the TensorMap being loaded, as a factor of the box
    // element count: 1 past the rank, 2^20 (saturating) past the legal 256.
    function [20:0] box_factor;
        input [2:0] j;
        reg [31:0] b;
        begin
            case (j)
            3'd0: b = dl_buf[11][31:0];
            3'd1: b = dl_buf[11][63:32];
            3'd2: b = dl_buf[12][31:0];
            3'd3: b = dl_buf[12][63:32];
            default: b = dl_buf[13][31:0];
            endcase
            if ({5'd0, j} >= p_rank)   box_factor = 21'd1;
            else if (b > 32'd256)      box_factor = 21'h100000;
            else                       box_factor = b[20:0];
        end
    endfunction
    function [20:0] box_mul;
        input [20:0] p;
        input [20:0] f;
        reg   [41:0] x;
        begin
            x = p * f;
            box_mul = (x >= 42'h100000) ? 21'h100000 : x[20:0];
        end
    endfunction

    //==========================================================================
    // Geometry, on the normalised copy
    //==========================================================================
    // Computed from the normalised copy directly: S_TCHK registers the
    // derived quantities in the same cycle it checks them.
    wire [31:0] n_row_bytes = t_b4 << t_esz;
    wire [63:0] n_CB        = t_c4 << t_esz;
    // Bytes in the box: the element count from the descriptor slot (above)
    // times the element size. Saturated at 2^20 elements, far past any window
    // (at most 255 x 64 bytes), so the comparison is exact where it matters;
    // a block dimension over 256 is refused by blk_ok whatever this says.
    wire [52:0] g_box   = {32'd0, t_boxel} << t_esz;
    wire [52:0] g_win   = {39'd0, t_wlen, 6'd0};
    function blk_ok;
        input [31:0] b;
        begin
            blk_ok = (b != 32'd0) && (b <= 32'd256);
        end
    endfunction
    // The whole box inside the tensor, for padding mode 2.
    function box_in;
        input [63:0] c;
        input [31:0] b;
        input [63:0] e;
        reg   [63:0] hi;
        begin
            hi = c + {32'd0, b};
            box_in = !c[63] && (hi <= e);
        end
    endfunction
    wire g_inside = box_in(t_c0, t_b0, t_e0) && box_in(t_c1, t_b1, t_e1) &&
                    box_in(t_c2, t_b2, t_e2) && box_in(t_c3, t_b3, t_e3) &&
                    box_in(t_c4, t_b4, t_e4);
    wire g_bad =
        // The rank the engine executes, and coordinates that exist: Sb+rank-1
        // must not run past S63 (the next index would be S0, which reads 0).
        (t_rank == 8'd0) || (t_rank > 8'd5) ||
        ({2'b00, t_sb} + {2'b00, t_rank[5:0]} > 8'd64) ||
        // Padding: 0 zero, 2 fault, 3 NaN (floating dtypes of this element
        // size only). 1 (clamp) is reserved.
        (t_pad == 8'd1) || (t_pad > 8'd3) ||
        ((t_pad == 8'd3) && (!dtype_float || (dtype_lg != t_esz))) ||
        ((t_pad == 8'd2) && !g_inside) ||
        // Each block dimension 1..256; a row of whole 8-byte units.
        !blk_ok(t_b0) || !blk_ok(t_b1) || !blk_ok(t_b2) ||
        !blk_ok(t_b3) || !blk_ok(t_b4) ||
        ((n_row_bytes & 32'd7) != 32'd0) ||
        // The contiguous dimension contiguous. (Base and outer strides may
        // have any alignment; see the header.)
        (t_s4 != (64'd1 << t_esz)) ||
        // ISA sec 9D.2: the window is the bounds check.
        (g_box > g_win) ||
        // TRANSPOSED WRITE: a 2-D tile of 16-bit elements with 4, 8 or 16
        // source rows, so every destination row is whole lanes inside ONE
        // register, and every source row starting on an 8-byte boundary with
        // a non-negative contiguous coordinate (the corner turn copies whole
        // source units).
        (t_xpose && ((t_rank != 8'd2) || (t_b3 > XR_MAX) || !xrow_ok ||
                     t_c4[63] || (n_CB[2:0] != 3'd0) || t_store ||
                     ((t_base & 64'd7) != 64'd0) || ((t_s3 & 64'd7) != 64'd0)));

    // The multiplier for the starting address: t_acc += c[p] * s[p]
    reg [63:0] mul_c, mul_s;
    always @(*) begin
        case (t_mp)
        3'd0: begin mul_c = t_c0; mul_s = t_s0; end
        3'd1: begin mul_c = t_c1; mul_s = t_s1; end
        3'd2: begin mul_c = t_c2; mul_s = t_s2; end
        default: begin mul_c = t_c3; mul_s = t_s3; end
        endcase
    end
    wire [63:0] mul_p = mul_c * mul_s;

    //==========================================================================
    always @(posedge clk) begin
        if (!rst_n || flush) begin
            st        <= S_IDLE;
            win_v     <= {WIN_N{1'b0}};
            desc_v    <= {DESC_N{1'b0}};
            tg_v      <= {NTAG{1'b0}};
            sl_v      <= {NSLOT{1'b0}};
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
            sr_v       <= 1'b0;
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
                    xbuf[x_rsp_row*64 +: 64] <= mem_rsp_rdata & x_rsp_keep;
                    tg_v[mem_rsp_tag] <= 1'b0;
                end else if (t_store) begin
                    // A write's response says it has been performed.
                    tg_v[mem_rsp_tag] <= 1'b0;
                end else begin
                    if (mem_rsp_tag[0]) begin
                        sl_dhi[mem_rsp_tag[TAG_W-1:1]] <= mem_rsp_rdata;
                        sl_ghi[mem_rsp_tag[TAG_W-1:1]] <= 1'b1;
                    end else begin
                        sl_dlo[mem_rsp_tag[TAG_W-1:1]] <= mem_rsp_rdata;
                        sl_glo[mem_rsp_tag[TAG_W-1:1]] <= 1'b1;
                    end
                end
            end

            //------------------------------------------------------------------
            // A load unit whose words are all back goes into its window
            // register. This has the write port first: it is the path that
            // frees a slot and lets the transfer finish. A unit with no
            // memory behind it (all fill) waits a cycle.
            //------------------------------------------------------------------
            if (have_done && !t_store && !t_xpose) begin
                vw_en      <= 1'b1;
                vw_rctx    <= t_rctx;
                vw_idx     <= sl_vreg[done_slot];
                vw_lane_en <= ({{(VLEN-2){1'b0}}, 2'b11} << sl_lane[done_slot]);
                vw_data    <= {(VLEN*ELEN){1'b0}};
                vw_data[sl_lane[done_slot]*32 +: 64] <= dn_data;
                sl_v[done_slot] <= 1'b0;
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
                    cc0      <= cmd_c0;
                    cc1      <= cmd_c1;
                    cc2      <= cmd_c2;
                    cc3      <= cmd_c3;
                    cc4      <= cmd_c4;
                    t_sb     <= cmd_sb;
                    t_esz    <= cmd_esz;
                    // A constant 0 on an engine built without the feature, so
                    // synthesis removes the corner-turn buffer and its states
                    // instead of keeping logic nothing can reach.
                    t_xpose  <= cmd_xpose && (HAS_XPOSE != 0);
                    t_bar_en <= cmd_bar_en;
                    t_bar    <= cmd_bar;
                    t_rctx   <= cmd_rctx;
                    t_pc     <= cmd_pc;
                    // A window layout this engine does not have, or a
                    // transposed write on an engine built without one, is
                    // refused: the encoding exists, the machine does not
                    // execute it, and running it anyway would transfer the
                    // tile in some other shape and call it done.
                    if ((cmd_layout != 2'd0) || (cmd_xpose && (HAS_XPOSE == 0))) begin
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
                    // Words 11-13 (block dims, rank) are in dl_buf by the time
                    // word 14 arrives: fold one dimension per word after that.
                    if (dl_word >= 5'd14 && dl_word <= 5'd18)
                        dl_boxel <= (dl_word == 5'd14) ? box_factor(3'd0)
                                                       : box_mul(dl_boxel, box_factor(dl_word[2:0] - 3'd6));
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
                d_e2[dl_slot]    <= dl_buf[3];
                d_e3[dl_slot]    <= dl_buf[4];
                d_e4[dl_slot]    <= dl_buf[5];
                d_s0[dl_slot]    <= dl_buf[6];
                d_s1[dl_slot]    <= dl_buf[7];
                d_s2[dl_slot]    <= dl_buf[8];
                d_s3[dl_slot]    <= dl_buf[9];
                d_s4[dl_slot]    <= dl_buf[10];
                d_b0[dl_slot]    <= dl_buf[11][31:0];
                d_b1[dl_slot]    <= dl_buf[11][63:32];
                d_b2[dl_slot]    <= dl_buf[12][31:0];
                d_b3[dl_slot]    <= dl_buf[12][63:32];
                d_b4[dl_slot]    <= dl_buf[13][31:0];
                d_rank[dl_slot]  <= p_rank;
                d_boxel[dl_slot] <= dl_boxel;
                d_dtype[dl_slot] <= p_dtype;
                d_pad[dl_slot]   <= p_pad;
                desc_v[dl_slot]  <= 1'b1;
                st <= S_DONE;
            end

            //---------------------------------------------------------------
            // Copy the snapshot into the transfer and normalise it: descriptor
            // dimension j goes to position 5 - rank + j. Positions below are
            // the identity (extent 1, block 1, stride 0, coordinate 0).
            S_TSETUP: begin
                t_base  <= d_base[t_desc];
                t_rank  <= d_rank[t_desc];
                t_dtype <= d_dtype[t_desc];
                t_pad   <= d_pad[t_desc];
                t_wbase <= win_base[t_win];
                t_wlen  <= win_len[t_win];
                t_boxel <= d_boxel[t_desc];
                t_e0 <= 64'd1; t_e1 <= 64'd1; t_e2 <= 64'd1; t_e3 <= 64'd1; t_e4 <= 64'd1;
                t_s0 <= 64'd0; t_s1 <= 64'd0; t_s2 <= 64'd0; t_s3 <= 64'd0; t_s4 <= 64'd0;
                t_b0 <= 32'd1; t_b1 <= 32'd1; t_b2 <= 32'd1; t_b3 <= 32'd1; t_b4 <= 32'd1;
                t_c0 <= 64'd0; t_c1 <= 64'd0; t_c2 <= 64'd0; t_c3 <= 64'd0; t_c4 <= 64'd0;
                case (d_rank[t_desc])
                8'd1: begin
                    t_e4 <= d_e0[t_desc]; t_s4 <= d_s0[t_desc]; t_b4 <= d_b0[t_desc]; t_c4 <= cc0;
                end
                8'd2: begin
                    t_e3 <= d_e0[t_desc]; t_s3 <= d_s0[t_desc]; t_b3 <= d_b0[t_desc]; t_c3 <= cc0;
                    t_e4 <= d_e1[t_desc]; t_s4 <= d_s1[t_desc]; t_b4 <= d_b1[t_desc]; t_c4 <= cc1;
                end
                8'd3: begin
                    t_e2 <= d_e0[t_desc]; t_s2 <= d_s0[t_desc]; t_b2 <= d_b0[t_desc]; t_c2 <= cc0;
                    t_e3 <= d_e1[t_desc]; t_s3 <= d_s1[t_desc]; t_b3 <= d_b1[t_desc]; t_c3 <= cc1;
                    t_e4 <= d_e2[t_desc]; t_s4 <= d_s2[t_desc]; t_b4 <= d_b2[t_desc]; t_c4 <= cc2;
                end
                8'd4: begin
                    t_e1 <= d_e0[t_desc]; t_s1 <= d_s0[t_desc]; t_b1 <= d_b0[t_desc]; t_c1 <= cc0;
                    t_e2 <= d_e1[t_desc]; t_s2 <= d_s1[t_desc]; t_b2 <= d_b1[t_desc]; t_c2 <= cc1;
                    t_e3 <= d_e2[t_desc]; t_s3 <= d_s2[t_desc]; t_b3 <= d_b2[t_desc]; t_c3 <= cc2;
                    t_e4 <= d_e3[t_desc]; t_s4 <= d_s3[t_desc]; t_b4 <= d_b3[t_desc]; t_c4 <= cc3;
                end
                default: begin    // 5; anything else faults in S_TCHK
                    t_e0 <= d_e0[t_desc]; t_s0 <= d_s0[t_desc]; t_b0 <= d_b0[t_desc]; t_c0 <= cc0;
                    t_e1 <= d_e1[t_desc]; t_s1 <= d_s1[t_desc]; t_b1 <= d_b1[t_desc]; t_c1 <= cc1;
                    t_e2 <= d_e2[t_desc]; t_s2 <= d_s2[t_desc]; t_b2 <= d_b2[t_desc]; t_c2 <= cc2;
                    t_e3 <= d_e3[t_desc]; t_s3 <= d_s3[t_desc]; t_b3 <= d_b3[t_desc]; t_c3 <= cc3;
                    t_e4 <= d_e4[t_desc]; t_s4 <= d_s4[t_desc]; t_b4 <= d_b4[t_desc]; t_c4 <= cc4;
                end
                endcase
                // An unloaded descriptor or an undefined window has nothing
                // to transfer; checked here, before anything is derived.
                if (!desc_v[t_desc] || !win_v[t_win])
                    st <= S_FAULT;
                else
                    st <= S_TCHK;
            end

            //---------------------------------------------------------------
            // Derived quantities, then every geometry rule at once. Nothing
            // has touched memory yet, so a fault here has no side effect.
            S_TCHK: begin
                t_E         <= t_e4 << t_esz;
                t_CB        <= t_c4 << t_esz;
                t_row_bytes <= t_b4 << t_esz;
                t_row_units <= (t_b4 << t_esz) >> 3;
                t_fill      <= (t_pad == 8'd3) ? fill_nan : 64'd0;
                // Positions below 5 - rank are the identity; skip them.
                t_mp        <= 3'd5 - t_rank[2:0];
                t_acc       <= t_base;
                if (g_bad)
                    st <= S_FAULT;
                else
                    st <= S_TMUL;
            end

            //---------------------------------------------------------------
            // base + sum over the outer positions of coordinate * stride,
            // one multiply a cycle. The walk after this is adds only.
            S_TMUL: begin
                if (t_mp < 3'd4) begin
                    t_acc <= t_acc + mul_p;
                    t_mp  <= t_mp + 3'd1;
                end else begin
                    t_P0 <= t_acc; t_P1 <= t_acc; t_P2 <= t_acc; t_P3 <= t_acc;
                    t_i0 <= 32'd0; t_i1 <= 32'd0; t_i2 <= 32'd0; t_i3 <= 32'd0;
                    t_unit <= 32'd0;
                    t_flat <= 32'd0;
                    t_half <= 1'b0;
                    st_prev <= 64'd0;
                    sr_v    <= 1'b0;
                    if (t_xpose) begin
                        r_row_base <= t_acc + t_CB;
                        if (HAS_XPOSE != 0) begin   // read only by the transposed walk
                            r_base0 <= t_acc + t_CB;
                            x_q     <= 4'd0;
                        end
                        x_row <= 32'd0;
                        st <= S_XRUN;
                    end else
                        st <= t_store ? S_SRUN : S_TRUN;
                end
            end

            //---------------------------------------------------------------
            // LOAD. One box unit per step: issue its word (or two), or, when
            // no byte of it is inside the tensor, write the fill straight into
            // the window -- padding is the absence of an access, not a masked
            // one.
            S_TRUN: begin : s_trun
                reg adv;
                adv = 1'b0;
                if (!ld_nlo && !ld_nhi) begin
                    if (!have_done) begin
                        vw_en      <= 1'b1;
                        vw_rctx    <= t_rctx;
                        vw_idx     <= ld_vreg;
                        vw_lane_en <= ({{(VLEN-2){1'b0}}, 2'b11} << ld_lane);
                        vw_data    <= {(VLEN*ELEN){1'b0}};
                        vw_data[ld_lane*32 +: 64] <= t_fill;
                        adv = 1'b1;
                    end
                end else if (mem_fire) begin
                    if (!t_half) begin
                        sl_v[free_slot]    <= 1'b1;
                        sl_nlo[free_slot]  <= ld_nlo;
                        sl_nhi[free_slot]  <= ld_nhi;
                        sl_glo[free_slot]  <= 1'b0;
                        sl_ghi[free_slot]  <= 1'b0;
                        sl_vreg[free_slot] <= ld_vreg;
                        sl_lane[free_slot] <= ld_lane;
                        sl_keep[free_slot] <= ld_keep;
                        sl_m[free_slot]    <= r_m;
                        // A missing word's bytes are all outside the tensor,
                        // so whatever stands in for it is never kept.
                        sl_dlo[free_slot]  <= 64'd0;
                        sl_dhi[free_slot]  <= 64'd0;
                        t_slot <= free_slot;
                    end
                    if (!t_half && ld_nlo && ld_nhi)
                        t_half <= 1'b1;       // the high word next cycle
                    else begin
                        t_half <= 1'b0;
                        adv = 1'b1;
                    end
                end
                if (adv) begin
                    if (t_unit + 32'd1 == t_row_units) begin
                        t_unit <= 32'd0;
                        t_flat <= t_flat + t_row_bytes;
                        if (last_row) st <= S_TDRAIN;
                        else begin
                            if (inc3) begin
                                t_i3 <= t_i3 + 32'd1; t_P3 <= t_P3 + t_s3;
                            end else if (inc2) begin
                                t_i2 <= t_i2 + 32'd1; t_i3 <= 32'd0;
                                t_P2 <= t_P2 + t_s2; t_P3 <= t_P2 + t_s2;
                            end else if (inc1) begin
                                t_i1 <= t_i1 + 32'd1; t_i2 <= 32'd0; t_i3 <= 32'd0;
                                t_P1 <= t_P1 + t_s1; t_P2 <= t_P1 + t_s1; t_P3 <= t_P1 + t_s1;
                            end else begin
                                t_i0 <= t_i0 + 32'd1; t_i1 <= 32'd0; t_i2 <= 32'd0; t_i3 <= 32'd0;
                                t_P0 <= t_P0 + t_s0; t_P1 <= t_P0 + t_s0;
                                t_P2 <= t_P0 + t_s0; t_P3 <= t_P0 + t_s0;
                            end
                        end
                    end else
                        t_unit <= t_unit + 32'd1;
                end
            end

            S_TDRAIN: begin
                if ((sl_v == {NSLOT{1'b0}}) && !mem_rsp_valid) st <= S_DONE;
            end

            //---------------------------------------------------------------
            // STORE. One memory word of the box row per step. A row outside
            // the tensor is skipped whole; a word with no byte to write is
            // skipped; a word whose window unit is not in hand waits for the
            // register to be read.
            S_SRUN: begin : s_srun
                reg adv;
                reg row_end;
                adv = 1'b0;
                row_end = 1'b0;
                if (!t_rowv) begin
                    row_end = 1'b1;
                end else if (st_need_reg) begin
                    st <= S_SREAD;
                end else if (st_be == 8'd0 || mem_fire) begin
                    if (mem_fire) tg_v[free_tag] <= 1'b1;
                    adv = 1'b1;
                end
                if (adv) begin
                    st_prev <= st_cur;
                    if (t_unit + 32'd1 == st_nwords) row_end = 1'b1;
                    else t_unit <= t_unit + 32'd1;
                end
                if (row_end) begin
                    t_unit  <= 32'd0;
                    st_prev <= 64'd0;
                    t_flat  <= t_flat + t_row_bytes;
                    if (last_row) st <= S_SDRAIN;
                    else begin
                        if (inc3) begin
                            t_i3 <= t_i3 + 32'd1; t_P3 <= t_P3 + t_s3;
                        end else if (inc2) begin
                            t_i2 <= t_i2 + 32'd1; t_i3 <= 32'd0;
                            t_P2 <= t_P2 + t_s2; t_P3 <= t_P2 + t_s2;
                        end else if (inc1) begin
                            t_i1 <= t_i1 + 32'd1; t_i2 <= 32'd0; t_i3 <= 32'd0;
                            t_P1 <= t_P1 + t_s1; t_P2 <= t_P1 + t_s1; t_P3 <= t_P1 + t_s1;
                        end else begin
                            t_i0 <= t_i0 + 32'd1; t_i1 <= 32'd0; t_i2 <= 32'd0; t_i3 <= 32'd0;
                            t_P0 <= t_P0 + t_s0; t_P1 <= t_P0 + t_s0;
                            t_P2 <= t_P0 + t_s0; t_P3 <= t_P0 + t_s0;
                        end
                    end
                end
            end

            S_SREAD: begin
                if (vr_valid) begin
                    sr_data <= vr_data;
                    sr_idx  <= st_vreg;
                    sr_v    <= 1'b1;
                    st      <= S_SRUN;
                end
            end

            S_SDRAIN: begin
                if ((tg_v == {NTAG{1'b0}}) && !mem_rsp_valid) st <= S_DONE;
            end

            //---------------------------------------------------------------
            // TRANSPOSED WRITE. Column block t_unit (8/esz source columns)
            // is read down all t_b3 source rows, one 8-byte unit each,
            // into xbuf[row]. Out-of-bounds units are zeros without an
            // access.
            S_XRUN: begin
                if (x_unit_oob || mem_fire) begin
                    if (x_unit_oob)
                        xbuf[x_row[3:0]*64 +: 64] <= 64'd0;
                    else begin
                        tg_v[free_tag]    <= 1'b1;
                        xt_row[free_tag]  <= x_row[3:0];
                        xt_keep[free_tag] <= x_keep_mask;
                    end
                    if (x_row == t_b3 - 1) st <= S_XDRAIN;
                    else begin
                        x_row      <= x_row + 1'b1;
                        r_row_base <= r_row_base + t_s3;
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
                        x_row      <= 32'd0;
                        // Guarded like its write in S_TMUL: without the
                        // feature r_base0 has no driver, and an unguarded
                        // read here is an undriven net in that engine
                        // (yosys `check`, R36-FPGA-02) even though no
                        // command can reach this state.
                        if (HAS_XPOSE != 0)
                            r_row_base <= r_base0;
                        st         <= S_XRUN;
                    end
                end else
                    x_q <= x_q + 1'b1;
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

    wire _unused = &{1'b0, d_dtype[0], 1'b0};

endmodule
