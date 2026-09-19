//==============================================================================
// tnca_amo.vh -- the atomic read-modify-write ALU (ISA sec 9.2)
//
// Included into a module body, not a module of its own: it is three pure
// functions on a 64-bit word and both users want them inline.
//
// There are two users and there must only ever be one definition. The home
// slice executes atomics for the four-tile system; the single-core bench's
// memory model executes them for `make core`. If those two disagreed about
// what atom.min.s does, one of the two test suites would be checking the wrong
// machine -- and the one that agreed with the bug would be the one that passed.
//
// The word is 64 bits and an atomic is 4 or 8 bytes, which is what the byte
// enables say: all eight for a doubleword, the upper or lower four for a word.
// `amo_half` extracts the operand at its own offset -- the memory word and the
// request's data are positioned identically -- and `amo_new` puts the result
// back where the byte enables will write it.
//==============================================================================
`include "tnca_defs.vh"

function [63:0] amo_alu;
    input [3:0]  op;
    input        w32;          // 4-byte operation
    input [63:0] old;
    input [63:0] arg;
    input [63:0] cmp;
    reg signed [63:0] so, sa;
    reg signed [31:0] so32, sa32;
    reg               lt_s, lt_u, eq;
    begin
        so = old; sa = arg;
        so32 = old[31:0]; sa32 = arg[31:0];
        lt_s = w32 ? (so32 < sa32) : (so < sa);
        lt_u = w32 ? (old[31:0] < arg[31:0]) : (old < arg);
        eq   = w32 ? (old[31:0] == cmp[31:0]) : (old == cmp);
        case (op)
        `TNCA_AOP_ADD:   amo_alu = old + arg;
        `TNCA_AOP_AND:   amo_alu = old & arg;
        `TNCA_AOP_OR:    amo_alu = old | arg;
        `TNCA_AOP_XOR:   amo_alu = old ^ arg;
        `TNCA_AOP_XCHG:  amo_alu = arg;
        `TNCA_AOP_MIN_S: amo_alu = lt_s ? old : arg;
        `TNCA_AOP_MAX_S: amo_alu = lt_s ? arg : old;
        `TNCA_AOP_MIN_U: amo_alu = lt_u ? old : arg;
        `TNCA_AOP_MAX_U: amo_alu = lt_u ? arg : old;
        `TNCA_AOP_CAS:   amo_alu = eq ? arg : old;
        `TNCA_AOP_LD:    amo_alu = old;
        `TNCA_AOP_ST:    amo_alu = arg;
        default:         amo_alu = old;
        endcase
    end
endfunction

function [63:0] amo_half;
    input [7:0]  be;
    input [63:0] word;
    begin
        if (&be)        amo_half = word;
        else if (be[4]) amo_half = {32'd0, word[63:32]};
        else            amo_half = {32'd0, word[31:0]};
    end
endfunction

function [63:0] amo_new;
    input [3:0]  op;
    input [7:0]  be;
    input [63:0] word;
    input [63:0] wdata;
    input [63:0] cmp;
    reg [63:0] n;
    begin
        n = amo_alu(op, ~(&be), amo_half(be, word), amo_half(be, wdata), cmp);
        amo_new = (&be) ? n : (be[4] ? {n[31:0], 32'd0} : {32'd0, n[31:0]});
    end
endfunction
