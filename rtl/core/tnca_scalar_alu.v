//==============================================================================
// tnca_scalar_alu -- 64-bit scalar integer ALU (ISA sec 6)
//
// Register/immediate variants share an arm: the issue stage has already chosen
// between Sb and the immediate and presented it as opb, so s.add and s.addi are
// the same operation here.
//
// The multiplier is NOT here. s.mul and s.mulh.* are executed by tnca_mul64,
// which is pipelined, because combinationally that array measured 135 gate
// levels against a ~50-level budget while everything in this module sits
// comfortably under it. The core routes multiply results from that unit; this
// module returns zero for those opcodes and never sees them selected.
//==============================================================================
`include "tnca_defs.vh"

module tnca_scalar_alu (
    input  wire [7:0]   func,
    input  wire [9:0]   mod,
    input  wire [63:0]  opa,
    input  wire [63:0]  opb,
    input  wire [63:0]  opc,
    output reg  [63:0]  result
);

    // ---- comparison ---------------------------------------------------------
    wire signed [63:0] sa = opa;
    wire signed [63:0] sb = opb;

    wire eq  = (opa == opb);
    wire lts = (sa  <  sb);
    wire ltu = (opa <  opb);

    reg cmp_r;
    always @(*) begin
        case (mod[3:0])
        `TNCA_CC_EQ:  cmp_r =  eq;
        `TNCA_CC_NE:  cmp_r = ~eq;
        `TNCA_CC_LT:  cmp_r =  lts;
        `TNCA_CC_LE:  cmp_r =  lts | eq;
        `TNCA_CC_GT:  cmp_r = ~(lts | eq);
        `TNCA_CC_GE:  cmp_r = ~lts;
        `TNCA_CC_LTU: cmp_r =  ltu;
        `TNCA_CC_LEU: cmp_r =  ltu | eq;
        `TNCA_CC_GTU: cmp_r = ~(ltu | eq);
        `TNCA_CC_GEU: cmp_r = ~ltu;
        default:      cmp_r = 1'b0;
        endcase
    end

    // ---- count leading zeros / population count -----------------------------
    //
    // Both are O(log W) structures (rtl/common). Written as the obvious loops
    // they were a 64-mux chain and a 64-adder chain respectively, and between
    // them they dominated everything in this unit except the multiplier.
    wire [6:0] clz_r;
    wire [6:0] popc_r;

    tnca_lzc  #(.W(64), .LOGW(6)) u_clz  (.v(opa), .clz(clz_r), .msb_idx(), .any());
    tnca_popc #(.W(64), .LOGW(6)) u_popc (.v(opa), .count(popc_r));

    // ---- movz / movk / movn shift amount ------------------------------------
    wire [5:0]  mov_sh   = {mod[1:0], 4'b0000};          // 16 * mod[1:0]
    wire [63:0] mov_val  = opb << mov_sh;
    wire [63:0] mov_mask = 64'hffff << mov_sh;

    always @(*) begin
        case (func)
        `TNCA_S_MOV:                        result = opa;
        `TNCA_S_ADD,  `TNCA_S_ADDI:         result = opa + opb;
        `TNCA_S_SUB,  `TNCA_S_SUBI:         result = opa - opb;
        `TNCA_S_AND,  `TNCA_S_ANDI:         result = opa & opb;
        `TNCA_S_OR,   `TNCA_S_ORI:          result = opa | opb;
        `TNCA_S_XOR,  `TNCA_S_XORI:         result = opa ^ opb;
        `TNCA_S_ANDN:                       result = opa & ~opb;
        `TNCA_S_SHL,  `TNCA_S_SHLI:         result = opa << opb[5:0];
        `TNCA_S_LSHR, `TNCA_S_LSHRI:        result = opa >> opb[5:0];
        `TNCA_S_ASHR, `TNCA_S_ASHRI:        result = $signed(opa) >>> opb[5:0];
        `TNCA_S_MIN_S:                      result = lts ? opa : opb;
        `TNCA_S_MAX_S:                      result = lts ? opb : opa;
        `TNCA_S_MIN_U:                      result = ltu ? opa : opb;
        `TNCA_S_MAX_U:                      result = ltu ? opb : opa;
        `TNCA_S_CLZ:                        result = {57'd0, clz_r};
        `TNCA_S_POPC:                       result = {57'd0, popc_r};
        `TNCA_S_SEXT_W:                     result = {{32{opa[31]}}, opa[31:0]};
        `TNCA_S_ZEXT_W:                     result = {32'd0, opa[31:0]};
        `TNCA_S_CMP,  `TNCA_S_CMPI:         result = {63'd0, cmp_r};
        `TNCA_S_SEL:                        result = (opc != 64'd0) ? opa : opb;
        `TNCA_S_MOVZ:                       result = mov_val;
        `TNCA_S_MOVK:                       result = (opa & ~mov_mask) | mov_val;
        `TNCA_S_MOVN:                       result = ~mov_val;
        default:                            result = 64'd0;
        endcase
    end

endmodule
