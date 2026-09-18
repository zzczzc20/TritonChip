//==============================================================================
// tnca_decoder -- TNCA ISA v0.1 instruction decoder (purely combinational)
//
// Implements doc/isa/TNCA-ISA-v0.1.md.
//
// Two properties this module is responsible for, beyond producing control
// signals:
//
//   1. ISA sec 2.3 -- every field not assigned a meaning by (major,func) must
//      be zero. Violations raise ILLEGAL_INSTRUCTION instead of being silently
//      ignored, so the unused encoding space stays available for v0.2.
//   2. ISA sec 2.2 -- scalar / control / system / predicate instructions are
//      unpredicated and must encode pred=0, PN=0.
//
// Verilog-2001.
//==============================================================================
`include "tnca_defs.vh"

module tnca_decoder (
    input  wire [63:0]  insn,

    // ---- classification -----------------------------------------------------
    output reg          o_illegal,
    output reg  [2:0]   o_fu,          // TNCA_FU_*
    output reg  [7:0]   o_major,
    output reg  [7:0]   o_func,
    output reg  [9:0]   o_mod,

    // ---- destination --------------------------------------------------------
    output reg  [2:0]   o_dst_class,   // TNCA_DST_*
    output reg  [7:0]   o_dst_idx,

    // ---- scalar source ports ------------------------------------------------
    output reg          o_sa_en,
    output reg  [5:0]   o_sa_idx,
    output reg          o_sb_en,
    output reg  [5:0]   o_sb_idx,
    output reg          o_sc_en,
    output reg  [5:0]   o_sc_idx,

    // ---- vector source ports ------------------------------------------------
    output reg          o_va_en,
    output reg  [7:0]   o_va_idx,
    output reg          o_vb_en,
    output reg  [7:0]   o_vb_idx,
    output reg          o_vc_en,
    output reg  [7:0]   o_vc_idx,

    // ---- predicate source ports (explicit operands, not the lane mask) ------
    output reg          o_pa_en,
    output reg  [3:0]   o_pa_idx,
    output reg          o_pb_en,
    output reg  [3:0]   o_pb_idx,

    // ---- lane mask ----------------------------------------------------------
    output reg          o_pm_en,       // instruction is lane-maskable
    output reg  [3:0]   o_pm_idx,
    output reg          o_pm_neg,
    // 1 => inactive lanes preserve the destination's previous value.
    // 0 => the unit produces every lane itself (v.sel, compares, reductions).
    output reg          o_mask_dst,

    // ---- immediate ----------------------------------------------------------
    output reg  [63:0]  o_imm,

    // ---- control flow -------------------------------------------------------
    output reg          o_is_branch,   // any PC redirect
    output reg          o_br_cond,     // conditional on a scalar source
    output reg          o_br_on_zero,  // 1 => taken when source == 0
    output reg          o_br_indirect, // target from Sa rather than PC+off
    output reg          o_br_link,     // write PC+8 to Sd
    output reg  [63:0]  o_br_off,      // byte displacement, sign-extended
    output reg          o_is_exit,
    output reg          o_is_trap,
    output reg          o_is_yield,

    // ---- memory -------------------------------------------------------------
    output reg          o_mem_en,
    output reg          o_mem_store,
    output reg          o_mem_vector,
    output reg          o_mem_indexed, // gather / scatter
    output reg  [2:0]   o_mem_size,
    output reg          o_mem_signed,
    output reg  [1:0]   o_mem_hint
);

    // --------------------------------------------------------------------------
    // Field extraction
    // --------------------------------------------------------------------------
    wire [7:0]  f_major = insn[`TNCA_F_MAJOR];
    wire [7:0]  f_func  = insn[`TNCA_F_FUNC];
    wire [7:0]  f_d     = insn[`TNCA_F_D];
    wire [7:0]  f_a     = insn[`TNCA_F_A];
    wire [7:0]  f_b     = insn[`TNCA_F_B];
    wire [7:0]  f_c     = insn[`TNCA_F_C];
    wire [3:0]  f_pred  = insn[`TNCA_F_PRED];
    wire        f_x     = insn[`TNCA_F_X];
    wire        f_pn    = insn[`TNCA_F_PN];
    wire [9:0]  f_mod   = insn[`TNCA_F_MOD];

    wire [15:0] imm16   = insn[`TNCA_F_IMM16];
    wire [23:0] imm24   = insn[`TNCA_F_IMM24];

    wire [63:0] imm16_s = {{48{imm16[15]}}, imm16};
    wire [63:0] imm16_z = {48'd0, imm16};
    wire [63:0] imm24_s = {{40{imm24[23]}}, imm24};

    // --------------------------------------------------------------------------
    // Field-usage tracking for the reserved-bits rule (ISA sec 2.3)
    //
    // Each decode arm sets the u_* flag for every field it consumes. Anything
    // left clear must read as zero in the instruction word.
    // --------------------------------------------------------------------------
    reg u_d, u_a, u_b, u_c;
    reg u_imm16;          // implies u_b, u_c
    reg u_imm24;          // implies u_a, u_b, u_c
    reg u_pred;           // pred/PN carry meaning
    reg [9:0] u_mod;      // per-bit mask of meaningful mod bits
    reg bad;              // set by an arm that detected an illegal sub-encoding

    // Register-index legality: scalar indices are 6 bits, predicate 4 bits.
    wire s_d_ok = (f_d[7:6] == 2'b00);
    wire s_a_ok = (f_a[7:6] == 2'b00);
    wire s_b_ok = (f_b[7:6] == 2'b00);
    wire s_c_ok = (f_c[7:6] == 2'b00);
    wire p_d_ok = (f_d[7:4] == 4'b0000);
    wire p_a_ok = (f_a[7:4] == 4'b0000);
    wire p_b_ok = (f_b[7:4] == 4'b0000);

    // Condition-code legality (ISA sec 4).
    wire cc_int_ok = (f_mod[3:0] <= `TNCA_CC_GEU);
    wire cc_fp_ok  = (f_mod[3:0] <= `TNCA_CC_GE) || (f_mod[3:0] == `TNCA_CC_O)
                                                 || (f_mod[3:0] == `TNCA_CC_UO);

    // --------------------------------------------------------------------------
    // Convenience macros for the decode body. Verilog-2001 has no functions
    // returning multiple values, so these are plain assignment groups.
    // --------------------------------------------------------------------------
    `define TNCA_DEC_SRC_SA  begin o_sa_en = 1'b1; o_sa_idx = f_a[5:0]; u_a = 1'b1; bad = bad | ~s_a_ok; end
    `define TNCA_DEC_SRC_SB  begin o_sb_en = 1'b1; o_sb_idx = f_b[5:0]; u_b = 1'b1; bad = bad | ~s_b_ok; end
    `define TNCA_DEC_SRC_SC  begin o_sc_en = 1'b1; o_sc_idx = f_c[5:0]; u_c = 1'b1; bad = bad | ~s_c_ok; end
    `define TNCA_DEC_SRC_VA  begin o_va_en = 1'b1; o_va_idx = f_a;      u_a = 1'b1; end
    `define TNCA_DEC_SRC_VB  begin o_vb_en = 1'b1; o_vb_idx = f_b;      u_b = 1'b1; end
    `define TNCA_DEC_SRC_VC  begin o_vc_en = 1'b1; o_vc_idx = f_c;      u_c = 1'b1; end
    `define TNCA_DEC_SRC_PA  begin o_pa_en = 1'b1; o_pa_idx = f_a[3:0]; u_a = 1'b1; bad = bad | ~p_a_ok; end
    `define TNCA_DEC_SRC_PB  begin o_pb_en = 1'b1; o_pb_idx = f_b[3:0]; u_b = 1'b1; bad = bad | ~p_b_ok; end
    `define TNCA_DEC_DST_S   begin o_dst_class = `TNCA_DST_S; o_dst_idx = {2'b00, f_d[5:0]}; u_d = 1'b1; bad = bad | ~s_d_ok; end
    `define TNCA_DEC_DST_V   begin o_dst_class = `TNCA_DST_V; o_dst_idx = f_d;               u_d = 1'b1; end
    `define TNCA_DEC_DST_P   begin o_dst_class = `TNCA_DST_P; o_dst_idx = {4'b0000, f_d[3:0]}; u_d = 1'b1; bad = bad | ~p_d_ok; end
    // Lane mask: any predicate register is legal, including P0.
    `define TNCA_DEC_MASK    begin o_pm_en = 1'b1; o_pm_idx = f_pred; o_pm_neg = f_pn; u_pred = 1'b1; end

    always @(*) begin
        // ---- defaults --------------------------------------------------------
        o_illegal     = 1'b0;
        o_fu          = `TNCA_FU_NONE;
        o_major       = f_major;
        o_func        = f_func;
        o_mod         = f_mod;
        o_dst_class   = `TNCA_DST_NONE;
        o_dst_idx     = 8'd0;
        o_sa_en       = 1'b0; o_sa_idx = 6'd0;
        o_sb_en       = 1'b0; o_sb_idx = 6'd0;
        o_sc_en       = 1'b0; o_sc_idx = 6'd0;
        o_va_en       = 1'b0; o_va_idx = 8'd0;
        o_vb_en       = 1'b0; o_vb_idx = 8'd0;
        o_vc_en       = 1'b0; o_vc_idx = 8'd0;
        o_pa_en       = 1'b0; o_pa_idx = 4'd0;
        o_pb_en       = 1'b0; o_pb_idx = 4'd0;
        o_pm_en       = 1'b0; o_pm_idx = 4'd0; o_pm_neg = 1'b0;
        o_mask_dst    = 1'b1;
        o_imm         = 64'd0;
        o_is_branch   = 1'b0;
        o_br_cond     = 1'b0;
        o_br_on_zero  = 1'b0;
        o_br_indirect = 1'b0;
        o_br_link     = 1'b0;
        o_br_off      = 64'd0;
        o_is_exit     = 1'b0;
        o_is_trap     = 1'b0;
        o_is_yield    = 1'b0;
        o_mem_en      = 1'b0;
        o_mem_store   = 1'b0;
        o_mem_vector  = 1'b0;
        o_mem_indexed = 1'b0;
        o_mem_size    = 3'd0;
        o_mem_signed  = 1'b0;
        o_mem_hint    = 2'd0;

        u_d = 1'b0; u_a = 1'b0; u_b = 1'b0; u_c = 1'b0;
        u_imm16 = 1'b0; u_imm24 = 1'b0; u_pred = 1'b0;
        u_mod = 10'd0;
        bad = 1'b0;

        case (f_major)

        //======================================================================
        // 0x00 -- System (ISA sec 5)
        //======================================================================
        `TNCA_MAJ_SYS: begin
            o_fu = `TNCA_FU_SALU;
            case (f_func)
            `TNCA_SYS_NOP: begin
                // every other field must be zero
            end
            `TNCA_SYS_RDSPECIAL: begin
                `TNCA_DEC_DST_S
                u_mod = 10'h3ff;                 // mod carries the special-reg id
            end
            `TNCA_SYS_WRSPECIAL: begin
                `TNCA_DEC_SRC_SA
                u_mod = 10'h3ff;
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x01 -- Scalar integer (ISA sec 6)
        //======================================================================
        `TNCA_MAJ_SALU: begin
            o_fu = `TNCA_FU_SALU;
            case (f_func)
            // ---- one source ---------------------------------------------
            `TNCA_S_MOV, `TNCA_S_CLZ, `TNCA_S_POPC,
            `TNCA_S_SEXT_W, `TNCA_S_ZEXT_W: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_SA
            end
            // ---- two sources --------------------------------------------
            `TNCA_S_ADD, `TNCA_S_SUB, `TNCA_S_MUL, `TNCA_S_MULH_S,
            `TNCA_S_MULH_U, `TNCA_S_AND, `TNCA_S_OR, `TNCA_S_XOR,
            `TNCA_S_ANDN, `TNCA_S_SHL, `TNCA_S_LSHR, `TNCA_S_ASHR,
            `TNCA_S_MIN_S, `TNCA_S_MAX_S, `TNCA_S_MIN_U, `TNCA_S_MAX_U: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_SA `TNCA_DEC_SRC_SB
            end
            `TNCA_S_CMP: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_SA `TNCA_DEC_SRC_SB
                u_mod = 10'h00f;
                bad = bad | ~cc_int_ok;
            end
            // ---- three sources ------------------------------------------
            `TNCA_S_SEL: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_SA `TNCA_DEC_SRC_SB `TNCA_DEC_SRC_SC
            end
            // ---- register + imm16 ---------------------------------------
            `TNCA_S_ADDI, `TNCA_S_SUBI, `TNCA_S_MULI: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_SA
                o_imm = imm16_s; u_imm16 = 1'b1;
            end
            `TNCA_S_ANDI, `TNCA_S_ORI, `TNCA_S_XORI: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_SA
                o_imm = imm16_z; u_imm16 = 1'b1;
            end
            `TNCA_S_SHLI, `TNCA_S_LSHRI, `TNCA_S_ASHRI: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_SA
                o_imm = imm16_z; u_imm16 = 1'b1;
            end
            `TNCA_S_CMPI: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_SA
                o_imm = imm16_s; u_imm16 = 1'b1;
                u_mod = 10'h00f;
                bad = bad | ~cc_int_ok;
            end
            // ---- imm16 only ----------------------------------------------
            `TNCA_S_MOVZ, `TNCA_S_MOVN: begin
                `TNCA_DEC_DST_S
                o_imm = imm16_z; u_imm16 = 1'b1;
                u_mod = 10'h003;                 // lsl amount
            end
            `TNCA_S_MOVK: begin
                // movk is read-modify-write on Sd
                `TNCA_DEC_DST_S
                o_sa_en = 1'b1; o_sa_idx = f_d[5:0];   // old Sd value
                o_imm = imm16_z; u_imm16 = 1'b1;
                u_mod = 10'h003;
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x02 -- Control flow (ISA sec 7)
        //======================================================================
        `TNCA_MAJ_CTRL: begin
            o_fu = `TNCA_FU_SALU;
            case (f_func)
            `TNCA_C_BR: begin
                o_is_branch = 1'b1;
                o_br_off = {imm24_s[60:0], 3'b000};
                u_imm24 = 1'b1;
            end
            `TNCA_C_BRNZ, `TNCA_C_BRZ: begin
                o_is_branch  = 1'b1;
                o_br_cond    = 1'b1;
                o_br_on_zero = (f_func == `TNCA_C_BRZ);
                `TNCA_DEC_SRC_SA
                o_br_off = {imm16_s[60:0], 3'b000};
                u_imm16 = 1'b1;
            end
            `TNCA_C_BRR: begin
                o_is_branch   = 1'b1;
                o_br_indirect = 1'b1;
                `TNCA_DEC_SRC_SA
            end
            `TNCA_C_CALL: begin
                o_is_branch = 1'b1;
                o_br_link   = 1'b1;
                `TNCA_DEC_DST_S
                o_br_off = {imm24_s[60:0], 3'b000};
                u_imm24 = 1'b1;
            end
            `TNCA_C_RET: begin
                o_is_branch   = 1'b1;
                o_br_indirect = 1'b1;
                `TNCA_DEC_SRC_SA
            end
            `TNCA_C_YIELD: begin
                o_is_yield = 1'b1;
            end
            `TNCA_C_TRAP: begin
                o_is_trap = 1'b1;
                o_imm = imm24_s; u_imm24 = 1'b1;
            end
            `TNCA_C_EXIT: begin
                o_is_exit = 1'b1;
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x10 -- Vector integer (ISA sec 8.1)
        //======================================================================
        `TNCA_MAJ_VINT: begin
            o_fu = `TNCA_FU_VALU;
            case (f_func)
            `TNCA_V_MOV, `TNCA_V_CLZ, `TNCA_V_POPC: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_MASK
            end
            `TNCA_V_ADD, `TNCA_V_SUB, `TNCA_V_MUL, `TNCA_V_MULH_S,
            `TNCA_V_MULH_U, `TNCA_V_AND, `TNCA_V_OR, `TNCA_V_XOR,
            `TNCA_V_ANDN, `TNCA_V_SHL, `TNCA_V_LSHR, `TNCA_V_ASHR,
            `TNCA_V_MIN_S, `TNCA_V_MAX_S, `TNCA_V_MIN_U, `TNCA_V_MAX_U: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_SRC_VB `TNCA_DEC_MASK
            end
            // v.sel is the select: masking its own write would make it useless
            // (ISA sec 8.1), so it writes every lane.
            `TNCA_V_SEL: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_SRC_VB `TNCA_DEC_MASK
                o_mask_dst = 1'b0;
            end
            `TNCA_V_CMP: begin
                `TNCA_DEC_DST_P `TNCA_DEC_SRC_VA `TNCA_DEC_SRC_VB `TNCA_DEC_MASK
                o_mask_dst = 1'b0;
                u_mod = 10'h00f;
                bad = bad | ~cc_int_ok;
            end
            `TNCA_V_SPLAT: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_SA `TNCA_DEC_MASK
            end
            `TNCA_V_IOTA: begin
                `TNCA_DEC_DST_V `TNCA_DEC_MASK
            end
            `TNCA_V_ADDI, `TNCA_V_SUBI, `TNCA_V_MULI: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_MASK
                o_imm = imm16_s; u_imm16 = 1'b1;
            end
            `TNCA_V_ANDI, `TNCA_V_ORI, `TNCA_V_XORI,
            `TNCA_V_SHLI, `TNCA_V_LSHRI, `TNCA_V_ASHRI: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_MASK
                o_imm = imm16_z; u_imm16 = 1'b1;
            end
            `TNCA_V_CMPI: begin
                `TNCA_DEC_DST_P `TNCA_DEC_SRC_VA `TNCA_DEC_MASK
                o_mask_dst = 1'b0;
                o_imm = imm16_s; u_imm16 = 1'b1;
                u_mod = 10'h00f;
                bad = bad | ~cc_int_ok;
            end
            `TNCA_V_MOVI: begin
                `TNCA_DEC_DST_V `TNCA_DEC_MASK
                o_imm = imm16_s; u_imm16 = 1'b1;
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x11 -- Vector floating point (ISA sec 8.2)
        //======================================================================
        `TNCA_MAJ_VFP: begin
            o_fu = `TNCA_FU_VALU;
            case (f_func)
            `TNCA_VF_ADD, `TNCA_VF_SUB, `TNCA_VF_MUL: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_SRC_VB `TNCA_DEC_MASK
                u_mod = 10'h100;                 // .ftz
            end
            `TNCA_VF_FMA, `TNCA_VF_FMS: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_SRC_VB `TNCA_DEC_SRC_VC `TNCA_DEC_MASK
                u_mod = 10'h100;
            end
            `TNCA_VF_MIN, `TNCA_VF_MAX: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_SRC_VB `TNCA_DEC_MASK
                u_mod = 10'h103;                 // NaN mode + .ftz
                bad = bad | (f_mod[1:0] == 2'd3);
            end
            `TNCA_VF_NEG, `TNCA_VF_ABS: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_MASK
            end
            `TNCA_VF_CMP: begin
                `TNCA_DEC_DST_P `TNCA_DEC_SRC_VA `TNCA_DEC_SRC_VB `TNCA_DEC_MASK
                o_mask_dst = 1'b0;
                u_mod = 10'h00f;
                bad = bad | ~cc_fp_ok;
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x12 -- Convert (ISA sec 8.4)
        //======================================================================
        `TNCA_MAJ_CVT: begin
            o_fu = `TNCA_FU_VALU;
            case (f_func)
            `TNCA_CVT_S32_F32, `TNCA_CVT_U32_F32: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_MASK
            end
            `TNCA_CVT_F32_S32, `TNCA_CVT_F32_U32: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_MASK
                u_mod = 10'h007;                 // rounding mode
                bad = bad | (f_mod[2:0] > `TNCA_RND_RPI);
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x13 -- Predicate (ISA sec 8.5). Unpredicated by construction.
        //======================================================================
        `TNCA_MAJ_PRED: begin
            o_fu = `TNCA_FU_VALU;
            case (f_func)
            `TNCA_P_MOV, `TNCA_P_NOT: begin
                `TNCA_DEC_DST_P `TNCA_DEC_SRC_PA
            end
            `TNCA_P_AND, `TNCA_P_OR, `TNCA_P_XOR, `TNCA_P_ANDN: begin
                `TNCA_DEC_DST_P `TNCA_DEC_SRC_PA `TNCA_DEC_SRC_PB
            end
            `TNCA_P_ANY, `TNCA_P_ALL, `TNCA_P_BALLOT, `TNCA_P_POPC: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_PA
            end
            `TNCA_P_FROMBITS: begin
                `TNCA_DEC_DST_P `TNCA_DEC_SRC_SA
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x14 -- Shuffle / permute (ISA sec 8.6)
        //======================================================================
        `TNCA_MAJ_SHUF: begin
            o_fu = `TNCA_FU_VALU;
            case (f_func)
            `TNCA_SH_XOR, `TNCA_SH_UP, `TNCA_SH_DOWN, `TNCA_SH_ROTATE: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_SRC_SB `TNCA_DEC_MASK
            end
            `TNCA_SH_PERMUTE: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_SRC_VB `TNCA_DEC_MASK
            end
            `TNCA_SH_EXTRACT: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_VA
                u_mod = 10'h00f;
            end
            `TNCA_SH_INSERT: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_SA `TNCA_DEC_MASK
                // read-modify-write on Vd: other lanes are preserved
                o_vb_en = 1'b1; o_vb_idx = f_d;
                u_mod = 10'h00f;
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x15 -- Reduction (ISA sec 8.7). mod[0] selects the destination class.
        //======================================================================
        `TNCA_MAJ_RED: begin
            o_fu = `TNCA_FU_VALU;
            case (f_func)
            `TNCA_RD_ADD, `TNCA_RD_MAX_S, `TNCA_RD_MIN_S, `TNCA_RD_AND,
            `TNCA_RD_OR, `TNCA_RD_XOR, `TNCA_RD_MAX_U, `TNCA_RD_MIN_U,
            `TNCA_RD_FADD, `TNCA_RD_FMAX, `TNCA_RD_FMIN: begin
                `TNCA_DEC_SRC_VA `TNCA_DEC_MASK
                o_mask_dst = 1'b0;      // scalar dest, or broadcast to all lanes
                u_mod = 10'h001;
                if (f_mod[0]) begin
                    `TNCA_DEC_DST_V
                end else begin
                    `TNCA_DEC_DST_S
                end
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x16 -- SFU (ISA sec 8.8)
        //======================================================================
        `TNCA_MAJ_SFU: begin
            o_fu = `TNCA_FU_SFU;
            case (f_func)
            `TNCA_SFU_EXP2, `TNCA_SFU_RCP, `TNCA_SFU_RSQRT, `TNCA_SFU_LOG2: begin
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_VA `TNCA_DEC_MASK
            end
            default: bad = 1'b1;
            endcase
        end

        //======================================================================
        // 0x20 -- Load / store (ISA sec 9.1)
        //======================================================================
        `TNCA_MAJ_LSU: begin
            o_fu         = `TNCA_FU_LSU;
            o_mem_en     = 1'b1;
            o_mem_size   = f_mod[2:0];
            o_mem_signed = f_mod[3];
            o_mem_hint   = f_mod[5:4];
            u_mod        = 10'h03f;
            // size 4..7 and cache hint 3 are reserved
            bad = bad | (f_mod[2:0] > `TNCA_SZ_D) | (f_mod[5:4] == 2'd3);

            case (f_func)
            `TNCA_L_SLD: begin
                `TNCA_DEC_DST_S `TNCA_DEC_SRC_SA `TNCA_DEC_MASK
                o_imm = imm16_s; u_imm16 = 1'b1;
            end
            // Store data lives in field d: fields b and c are the displacement.
            `TNCA_L_SST: begin
                o_mem_store = 1'b1;
                `TNCA_DEC_SRC_SA
                o_sb_en = 1'b1; o_sb_idx = f_d[5:0]; u_d = 1'b1; bad = bad | ~s_d_ok;
                `TNCA_DEC_MASK
                o_imm = imm16_s; u_imm16 = 1'b1;
            end
            `TNCA_L_VLD: begin
                o_mem_vector = 1'b1;
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_SA `TNCA_DEC_MASK
                o_imm = imm16_s; u_imm16 = 1'b1;
                bad = bad | (f_mod[2:0] == `TNCA_SZ_D);   // 64-bit lanes reserved in v0.1
            end
            `TNCA_L_VST: begin
                o_mem_vector = 1'b1; o_mem_store = 1'b1;
                `TNCA_DEC_SRC_SA
                o_vb_en = 1'b1; o_vb_idx = f_d; u_d = 1'b1;
                `TNCA_DEC_MASK
                o_imm = imm16_s; u_imm16 = 1'b1;
                bad = bad | (f_mod[2:0] == `TNCA_SZ_D);
            end
            `TNCA_L_VGATHER: begin
                o_mem_vector = 1'b1; o_mem_indexed = 1'b1;
                `TNCA_DEC_DST_V `TNCA_DEC_SRC_SA `TNCA_DEC_SRC_VB `TNCA_DEC_MASK
                bad = bad | (f_mod[2:0] == `TNCA_SZ_D);
            end
            `TNCA_L_VSCATTER: begin
                o_mem_vector = 1'b1; o_mem_indexed = 1'b1; o_mem_store = 1'b1;
                `TNCA_DEC_SRC_SA `TNCA_DEC_SRC_VB
                o_vc_en = 1'b1; o_vc_idx = f_d; u_d = 1'b1;
                `TNCA_DEC_MASK
                bad = bad | (f_mod[2:0] == `TNCA_SZ_D);
            end
            default: bad = 1'b1;
            endcase
        end

        default: bad = 1'b1;
        endcase

        // ----------------------------------------------------------------------
        // Reserved-field enforcement (ISA sec 2.3)
        // ----------------------------------------------------------------------
        if (u_imm24) begin u_a = 1'b1; u_b = 1'b1; u_c = 1'b1; end
        if (u_imm16) begin u_b = 1'b1; u_c = 1'b1; end

        o_illegal = bad
                  | (~u_d     & (f_d != 8'd0))
                  | (~u_a     & (f_a != 8'd0))
                  | (~u_b     & (f_b != 8'd0))
                  | (~u_c     & (f_c != 8'd0))
                  | (~u_pred  & ((f_pred != 4'd0) | f_pn))
                  | f_x                                    // no ext words in v0.1
                  | (|(f_mod & ~u_mod));

        if (o_illegal) begin
            // A rejected instruction must not present any side effect to the
            // rest of the core.
            o_fu          = `TNCA_FU_NONE;
            o_dst_class   = `TNCA_DST_NONE;
            o_sa_en = 1'b0; o_sb_en = 1'b0; o_sc_en = 1'b0;
            o_va_en = 1'b0; o_vb_en = 1'b0; o_vc_en = 1'b0;
            o_pa_en = 1'b0; o_pb_en = 1'b0; o_pm_en = 1'b0;
            o_mask_dst = 1'b1;
            o_is_branch = 1'b0; o_is_exit = 1'b0;
            o_is_trap   = 1'b0; o_is_yield = 1'b0;
            o_mem_en    = 1'b0;
        end
    end

    `undef TNCA_DEC_SRC_SA
    `undef TNCA_DEC_SRC_SB
    `undef TNCA_DEC_SRC_SC
    `undef TNCA_DEC_SRC_VA
    `undef TNCA_DEC_SRC_VB
    `undef TNCA_DEC_SRC_VC
    `undef TNCA_DEC_SRC_PA
    `undef TNCA_DEC_SRC_PB
    `undef TNCA_DEC_DST_S
    `undef TNCA_DEC_DST_V
    `undef TNCA_DEC_DST_P
    `undef TNCA_DEC_MASK

endmodule
