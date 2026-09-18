//==============================================================================
// tnca_vector_alu -- VLEN-lane integer vector unit
//
// Covers ISA majors 0x10 (vector integer), 0x13 (predicate), 0x14 (shuffle)
// and the integer subset of 0x15 (reduction). Single cycle, combinational.
//
// Masking convention (ISA sec 2.2): this unit always computes every lane. The
// decoder's o_mask_dst tells the writeback stage whether inactive lanes should
// be suppressed; the unit itself only needs the mask for
//   - compares, whose inactive lanes must yield 0, and
//   - reductions, whose inactive lanes must contribute the identity.
//==============================================================================
`include "tnca_defs.vh"

module tnca_vector_alu #(
    parameter VLEN = 16,
    parameter ELEN = 32
) (
    input  wire [7:0]              major,
    input  wire [7:0]              func,
    input  wire [9:0]              mod,
    input  wire [VLEN-1:0]         mask,

    input  wire [VLEN*ELEN-1:0]    va,
    input  wire [VLEN*ELEN-1:0]    vb,
    input  wire [63:0]             sa,      // scalar operand (splat, shuffle amount, insert)
    input  wire [63:0]             imm,

    input  wire [VLEN-1:0]         pa,
    input  wire [VLEN-1:0]         pb,

    output reg  [VLEN*ELEN-1:0]    vres,    // vector result
    output reg  [63:0]             sres,    // scalar result (p.any, v.extract, reductions)
    output reg  [VLEN-1:0]         pres     // predicate result (compares, predicate ops)
);

    localparam LW = (VLEN <= 2)  ? 1 :
                    (VLEN <= 4)  ? 2 :
                    (VLEN <= 8)  ? 3 :
                    (VLEN <= 16) ? 4 :
                    (VLEN <= 32) ? 5 : 6;

    integer i;

    // --------------------------------------------------------------------------
    // Lane slices
    // --------------------------------------------------------------------------
    wire [ELEN-1:0] a [0:VLEN-1];
    wire [ELEN-1:0] b [0:VLEN-1];
    genvar g;
    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_slice
            assign a[g] = va[g*ELEN +: ELEN];
            assign b[g] = vb[g*ELEN +: ELEN];
        end
    endgenerate

    // Second operand: register or immediate, chosen by opcode.
    reg use_imm;
    always @(*) begin
        case (func)
        `TNCA_V_ADDI, `TNCA_V_SUBI, `TNCA_V_MULI, `TNCA_V_ANDI, `TNCA_V_ORI,
        `TNCA_V_XORI, `TNCA_V_SHLI, `TNCA_V_LSHRI, `TNCA_V_ASHRI,
        `TNCA_V_CMPI, `TNCA_V_MOVI: use_imm = 1'b1;
        default:                    use_imm = 1'b0;
        endcase
    end

    wire [ELEN-1:0] immv = imm[ELEN-1:0];

    // --------------------------------------------------------------------------
    // Per-lane integer datapath
    // --------------------------------------------------------------------------
    reg  [ELEN-1:0] lane_res [0:VLEN-1];
    reg             lane_cmp [0:VLEN-1];

    // ONE subtractor per lane feeds sub, min, max and every compare.
    //
    // Written as separate relational operators these become six independent
    // comparator trees plus a subtractor, all of which are the same hardware.
    // The borrow out of x - y is the unsigned "less than"; the signed form
    // differs from it only when the operands' signs differ, in which case the
    // answer is x's sign bit.
    reg [ELEN:0]   diff;
    reg            c_ltu, c_lts, c_eq;

    reg [ELEN-1:0] x, y;
    // ONE multiplier per lane.
    //
    // Computing an unsigned and a signed product separately costs two 32x32
    // arrays per lane -- at VLEN=16 that measured 205 k gates, most of the unit,
    // for an operation that is not on any inner loop. The signed high half is
    // derived from the unsigned product instead:
    //
    //   x_s * y_s = x*y - 2^32*(x[31] ? y : 0) - 2^32*(y[31] ? x : 0)  (mod 2^64)
    //
    // so only the high half changes, and it changes by a subtraction of two
    // conditionally-selected operands. This is the arch overview's sec 6.2
    // point applied literally: decide the arithmetic after looking at what it
    // costs, not before.
    reg [2*ELEN-1:0] pu;
    reg [ELEN-1:0]   ps_hi;
    reg [6:0] cnt;
    integer k;

    always @(*) begin
        for (i = 0; i < VLEN; i = i + 1) begin
            x  = a[i];
            y  = use_imm ? immv : b[i];

            diff  = {1'b0, x} - {1'b0, y};
            c_ltu = diff[ELEN];
            c_eq  = (x == y);
            c_lts = (x[ELEN-1] ^ y[ELEN-1]) ? x[ELEN-1] : c_ltu;

            pu    = x * y;
            ps_hi = pu[2*ELEN-1:ELEN]
                  - (x[ELEN-1] ? y : {ELEN{1'b0}})
                  - (y[ELEN-1] ? x : {ELEN{1'b0}});

            case (func)
            `TNCA_V_MOV:                     lane_res[i] = x;
            `TNCA_V_MOVI:                    lane_res[i] = immv;
            `TNCA_V_ADD,  `TNCA_V_ADDI:      lane_res[i] = x + y;
            `TNCA_V_SUB,  `TNCA_V_SUBI:      lane_res[i] = diff[ELEN-1:0];
            `TNCA_V_MUL,  `TNCA_V_MULI:      lane_res[i] = pu[ELEN-1:0];
            `TNCA_V_MULH_S:                  lane_res[i] = ps_hi;
            `TNCA_V_MULH_U:                  lane_res[i] = pu[2*ELEN-1:ELEN];
            `TNCA_V_AND,  `TNCA_V_ANDI:      lane_res[i] = x & y;
            `TNCA_V_OR,   `TNCA_V_ORI:       lane_res[i] = x | y;
            `TNCA_V_XOR,  `TNCA_V_XORI:      lane_res[i] = x ^ y;
            `TNCA_V_ANDN:                    lane_res[i] = x & ~y;
            `TNCA_V_SHL,  `TNCA_V_SHLI:      lane_res[i] = x << y[4:0];
            `TNCA_V_LSHR, `TNCA_V_LSHRI:     lane_res[i] = x >> y[4:0];
            `TNCA_V_ASHR, `TNCA_V_ASHRI:     lane_res[i] = $signed(x) >>> y[4:0];
            `TNCA_V_MIN_S:                   lane_res[i] = c_lts ? x : y;
            `TNCA_V_MAX_S:                   lane_res[i] = c_lts ? y : x;
            `TNCA_V_MIN_U:                   lane_res[i] = c_ltu ? x : y;
            `TNCA_V_MAX_U:                   lane_res[i] = c_ltu ? y : x;
            `TNCA_V_CLZ: begin
                cnt = ELEN;
                for (k = 0; k < ELEN; k = k + 1)
                    if (x[k]) cnt = ELEN - 1 - k;
                if (x == {ELEN{1'b0}}) cnt = ELEN;
                lane_res[i] = {{(ELEN-7){1'b0}}, cnt};
            end
            `TNCA_V_POPC: begin
                cnt = 0;
                for (k = 0; k < ELEN; k = k + 1)
                    cnt = cnt + {6'd0, x[k]};
                lane_res[i] = {{(ELEN-7){1'b0}}, cnt};
            end
            `TNCA_V_SEL:                     lane_res[i] = mask[i] ? x : y;
            `TNCA_V_SPLAT:                   lane_res[i] = sa[ELEN-1:0];
            `TNCA_V_IOTA:                    lane_res[i] = i;
            default:                         lane_res[i] = {ELEN{1'b0}};
            endcase

            // compare, used by v.cmp / v.cmpi
            case (mod[3:0])
            `TNCA_CC_EQ:  lane_cmp[i] =  c_eq;
            `TNCA_CC_NE:  lane_cmp[i] = ~c_eq;
            `TNCA_CC_LT:  lane_cmp[i] =  c_lts;
            `TNCA_CC_LE:  lane_cmp[i] =  c_lts | c_eq;
            `TNCA_CC_GT:  lane_cmp[i] = ~(c_lts | c_eq);
            `TNCA_CC_GE:  lane_cmp[i] = ~c_lts;
            `TNCA_CC_LTU: lane_cmp[i] =  c_ltu;
            `TNCA_CC_LEU: lane_cmp[i] =  c_ltu | c_eq;
            `TNCA_CC_GTU: lane_cmp[i] = ~(c_ltu | c_eq);
            `TNCA_CC_GEU: lane_cmp[i] = ~c_ltu;
            default:      lane_cmp[i] = 1'b0;
            endcase
        end
    end

    // --------------------------------------------------------------------------
    // Shuffle / permute (ISA sec 8.6). Source lanes are read regardless of mask.
    // --------------------------------------------------------------------------
    wire [LW-1:0] shamt = sa[LW-1:0];

    reg [ELEN-1:0] shuf_res [0:VLEN-1];
    reg [LW-1:0]   src_idx  [0:VLEN-1];

    // ONE crossbar, not one per pattern.
    //
    // xor / up / down / rotate / permute differ only in which source lane each
    // destination lane reads. Written as five separate selects they synthesise
    // to five independent VLEN-to-1 crossbars of ELEN bits; computing the
    // source index first and sharing a single crossbar is the same function at
    // a fifth of the mux area. The clamping forms (up, down) express their
    // clamp as "read your own lane", which is just another index.
    always @(*) begin
        for (i = 0; i < VLEN; i = i + 1) begin
            case (func)
            `TNCA_SH_XOR:     src_idx[i] = i[LW-1:0] ^ shamt;
            `TNCA_SH_UP:      src_idx[i] = (i[LW-1:0] >= shamt)
                                           ? (i[LW-1:0] - shamt) : i[LW-1:0];
            `TNCA_SH_DOWN:    src_idx[i] = ((i + shamt) < VLEN)
                                           ? (i[LW-1:0] + shamt) : i[LW-1:0];
            `TNCA_SH_ROTATE:  src_idx[i] = i[LW-1:0] + shamt;   // wraps naturally
            `TNCA_SH_PERMUTE: src_idx[i] = b[i][LW-1:0];
            default:          src_idx[i] = i[LW-1:0];
            endcase
        end

        for (i = 0; i < VLEN; i = i + 1) begin
            // v.insert is not a crossbar over `a`: it rewrites one lane of the
            // destination, so it keeps its own select.
            if (func == `TNCA_SH_INSERT)
                shuf_res[i] = (i[LW-1:0] == mod[LW-1:0]) ? sa[ELEN-1:0] : b[i];
            else
                shuf_res[i] = a[src_idx[i]];
        end
    end

    // --------------------------------------------------------------------------
    // Reduction (ISA sec 8.7)
    //
    // The tree is a balanced binary reduction over lane index, which is what
    // makes vf.reduce.add bit-reproducible. The integer reductions here are
    // associative so the shape does not matter for them, but they share it so
    // the FP unit's tree and this one stay structurally identical.
    // --------------------------------------------------------------------------
    reg [ELEN-1:0] red_in [0:VLEN-1];
    reg [ELEN-1:0] tree   [0:2*VLEN-1];
    reg [ELEN-1:0] ident;
    integer lvl, node, span;

    always @(*) begin
        case (func)
        `TNCA_RD_ADD, `TNCA_RD_OR, `TNCA_RD_XOR: ident = {ELEN{1'b0}};
        `TNCA_RD_AND:                            ident = {ELEN{1'b1}};
        `TNCA_RD_MAX_S:                          ident = {1'b1, {(ELEN-1){1'b0}}};
        `TNCA_RD_MIN_S:                          ident = {1'b0, {(ELEN-1){1'b1}}};
        `TNCA_RD_MAX_U:                          ident = {ELEN{1'b0}};
        `TNCA_RD_MIN_U:                          ident = {ELEN{1'b1}};
        default:                                 ident = {ELEN{1'b0}};
        endcase

        for (i = 0; i < VLEN; i = i + 1)
            red_in[i] = mask[i] ? a[i] : ident;

        // Level 0 of the tree is the (masked) input.
        for (i = 0; i < VLEN; i = i + 1)
            tree[i] = red_in[i];

        // Pairwise combine: span halves each level.
        span = VLEN;
        node = 0;
        for (lvl = 0; lvl < LW; lvl = lvl + 1) begin
            for (i = 0; i < VLEN; i = i + 2) begin
                if (i + 1 < span) begin
                    case (func)
                    `TNCA_RD_ADD:    tree[i>>1] = tree[i] + tree[i+1];
                    `TNCA_RD_AND:    tree[i>>1] = tree[i] & tree[i+1];
                    `TNCA_RD_OR:     tree[i>>1] = tree[i] | tree[i+1];
                    `TNCA_RD_XOR:    tree[i>>1] = tree[i] ^ tree[i+1];
                    `TNCA_RD_MAX_S:  tree[i>>1] = ($signed(tree[i]) > $signed(tree[i+1])) ? tree[i] : tree[i+1];
                    `TNCA_RD_MIN_S:  tree[i>>1] = ($signed(tree[i]) < $signed(tree[i+1])) ? tree[i] : tree[i+1];
                    `TNCA_RD_MAX_U:  tree[i>>1] = (tree[i] > tree[i+1]) ? tree[i] : tree[i+1];
                    `TNCA_RD_MIN_U:  tree[i>>1] = (tree[i] < tree[i+1]) ? tree[i] : tree[i+1];
                    default:         tree[i>>1] = tree[i];
                    endcase
                end
            end
            span = span >> 1;
        end
    end

    wire [ELEN-1:0] red_result = tree[0];

    // --------------------------------------------------------------------------
    // Predicate family (ISA sec 8.5)
    // --------------------------------------------------------------------------
    reg [VLEN-1:0] p_logic;
    reg [63:0]     p_scalar;
    reg [6:0]      p_cnt;

    always @(*) begin
        case (func)
        `TNCA_P_MOV:  p_logic = pa;
        `TNCA_P_AND:  p_logic = pa & pb;
        `TNCA_P_OR:   p_logic = pa | pb;
        `TNCA_P_XOR:  p_logic = pa ^ pb;
        `TNCA_P_NOT:  p_logic = ~pa;
        `TNCA_P_ANDN: p_logic = pa & ~pb;
        // p.frombits takes its source from the scalar port
        `TNCA_P_FROMBITS: p_logic = sa[VLEN-1:0];
        default:      p_logic = {VLEN{1'b0}};
        endcase

        p_cnt = 0;
        for (i = 0; i < VLEN; i = i + 1)
            p_cnt = p_cnt + {6'd0, pa[i]};

        case (func)
        `TNCA_P_ANY:    p_scalar = {63'd0, |pa};
        `TNCA_P_ALL:    p_scalar = {63'd0, &pa};
        `TNCA_P_BALLOT: p_scalar = {{(64-VLEN){1'b0}}, pa};
        `TNCA_P_POPC:   p_scalar = {57'd0, p_cnt};
        default:        p_scalar = 64'd0;
        endcase
    end

    // --------------------------------------------------------------------------
    // Output selection
    // --------------------------------------------------------------------------
    always @(*) begin
        vres = {(VLEN*ELEN){1'b0}};
        sres = 64'd0;
        pres = {VLEN{1'b0}};

        case (major)
        `TNCA_MAJ_VINT: begin
            for (i = 0; i < VLEN; i = i + 1)
                vres[i*ELEN +: ELEN] = lane_res[i];
            for (i = 0; i < VLEN; i = i + 1)
                pres[i] = mask[i] & lane_cmp[i];      // inactive lanes yield 0
        end
        `TNCA_MAJ_PRED: begin
            pres = p_logic;
            sres = p_scalar;
        end
        `TNCA_MAJ_SHUF: begin
            for (i = 0; i < VLEN; i = i + 1)
                vres[i*ELEN +: ELEN] = shuf_res[i];
            // v.extract: lane mod[3:0] of the source, zero-extended
            sres = {{(64-ELEN){1'b0}}, a[mod[LW-1:0]]};
        end
        `TNCA_MAJ_RED: begin
            sres = {{(64-ELEN){1'b0}}, red_result};
            for (i = 0; i < VLEN; i = i + 1)
                vres[i*ELEN +: ELEN] = red_result;    // broadcast form
        end
        default: ;
        endcase
    end

endmodule
