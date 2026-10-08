//==============================================================================
// tnca_defs.vh -- TNCA ISA v0.1 encoding constants
//
// Normative source: doc/isa/TNCA-ISA-v0.1.md
// Any disagreement between this file and that document is a bug in this file.
//
// Verilog-2001. Include guard via `ifndef so it may be included by every module.
//==============================================================================
`ifndef TNCA_DEFS_VH
`define TNCA_DEFS_VH

//------------------------------------------------------------------------------
// Instruction word layout (ISA §2)
//------------------------------------------------------------------------------
`define TNCA_ILEN            64

`define TNCA_F_MAJOR         63:56
`define TNCA_F_FUNC          55:48
`define TNCA_F_D             47:40
`define TNCA_F_A             39:32
`define TNCA_F_B             31:24
`define TNCA_F_C             23:16
`define TNCA_F_PRED          15:12
`define TNCA_F_X             11
`define TNCA_F_PN            10
`define TNCA_F_MOD           9:0

// Immediate assembly (ISA §2.1)
`define TNCA_F_IMM16         31:16     // {b,c}
`define TNCA_F_IMM24         39:16     // {a,b,c}

//------------------------------------------------------------------------------
// Architectural sizes (ISA §1)
//------------------------------------------------------------------------------
`define TNCA_XLEN            64        // scalar register width
`define TNCA_SREG_N          64        // scalar registers per RCTX
`define TNCA_SREG_W          6         // index width
`define TNCA_PREG_N          16        // predicate registers per RCTX
`define TNCA_PREG_W          4
`define TNCA_VLEN            16        // vector lanes
`define TNCA_VLEN_W          4
`define TNCA_ELEN            32        // vector element width
`define TNCA_VREG_N          256       // logical vector registers
`define TNCA_VREG_W          8
`define TNCA_VBITS           512       // VLEN * ELEN

//------------------------------------------------------------------------------
// Major opcodes (ISA §3)
//------------------------------------------------------------------------------
`define TNCA_MAJ_SYS         8'h00
`define TNCA_MAJ_SALU        8'h01
`define TNCA_MAJ_CTRL        8'h02
`define TNCA_MAJ_VINT        8'h10
`define TNCA_MAJ_VFP         8'h11
`define TNCA_MAJ_CVT         8'h12
`define TNCA_MAJ_PRED        8'h13
`define TNCA_MAJ_SHUF        8'h14
`define TNCA_MAJ_RED         8'h15
`define TNCA_MAJ_SFU         8'h16
`define TNCA_MAJ_LSU         8'h20
`define TNCA_MAJ_ATOM        8'h21
`define TNCA_MAJ_TDESC       8'h30
`define TNCA_MAJ_TXFER       8'h31
`define TNCA_MAJ_THINT       8'h32
`define TNCA_MAJ_MATRIX      8'h40
`define TNCA_MAJ_ACC         8'h41
`define TNCA_MAJ_MBAR        8'h50
`define TNCA_MAJ_ROLE        8'h51
`define TNCA_MAJ_DEBUG       8'h60

//------------------------------------------------------------------------------
// Condition codes (ISA §4)
//------------------------------------------------------------------------------
`define TNCA_CC_EQ           4'h0
`define TNCA_CC_NE           4'h1
`define TNCA_CC_LT           4'h2
`define TNCA_CC_LE           4'h3
`define TNCA_CC_GT           4'h4
`define TNCA_CC_GE           4'h5
`define TNCA_CC_LTU          4'h6
`define TNCA_CC_LEU          4'h7
`define TNCA_CC_GTU          4'h8
`define TNCA_CC_GEU          4'h9
`define TNCA_CC_O            4'ha
`define TNCA_CC_UO           4'hb

//------------------------------------------------------------------------------
// System family (ISA §5)
//------------------------------------------------------------------------------
`define TNCA_SYS_NOP         8'h00
`define TNCA_SYS_RDSPECIAL   8'h01
`define TNCA_SYS_WRSPECIAL   8'h02

`define TNCA_SPEC_PID_X      10'h000
`define TNCA_SPEC_PID_Y      10'h001
`define TNCA_SPEC_PID_Z      10'h002
`define TNCA_SPEC_NPID_X     10'h004
`define TNCA_SPEC_NPID_Y     10'h005
`define TNCA_SPEC_NPID_Z     10'h006
`define TNCA_SPEC_ROLE_ID    10'h008
`define TNCA_SPEC_NUM_ROLES  10'h009
`define TNCA_SPEC_CORE_ID    10'h00a
`define TNCA_SPEC_NUM_CORES  10'h00b
`define TNCA_SPEC_NUM_LANES  10'h00c
`define TNCA_SPEC_NUM_VREGS  10'h00d
`define TNCA_SPEC_MAX_RANK   10'h00e
`define TNCA_SPEC_NUM_ACC    10'h00f
`define TNCA_SPEC_CLOCK      10'h010
`define TNCA_SPEC_INSTRET    10'h011
// Shared-resource counts. The compiler allocates barriers, windows,
// descriptors and physical vector registers for the whole program
// (compiler_overview.md sec 10.1), so it must be able to interrogate
// the machine it targets rather than trust a target description file.
`define TNCA_SPEC_NUM_BAR    10'h012
`define TNCA_SPEC_NUM_WIN    10'h013
`define TNCA_SPEC_NUM_DESC   10'h014
`define TNCA_SPEC_NUM_PHYSV  10'h015
`define TNCA_SPEC_ARG_BASE   10'h020
`define TNCA_SPEC_ISA_VER    10'h030
`define TNCA_SPEC_CAP0       10'h031
`define TNCA_SPEC_TRAP_PC    10'h040
`define TNCA_SPEC_TRAP_CODE  10'h041

`define TNCA_ISA_VERSION     64'h0000_0000_0000_0001

//------------------------------------------------------------------------------
// Scalar integer family (ISA §6)
//------------------------------------------------------------------------------
`define TNCA_S_MOV           8'h00
`define TNCA_S_ADD           8'h01
`define TNCA_S_SUB           8'h02
`define TNCA_S_MUL           8'h03
`define TNCA_S_MULH_S        8'h04
`define TNCA_S_MULH_U        8'h05
`define TNCA_S_AND           8'h08
`define TNCA_S_OR            8'h09
`define TNCA_S_XOR           8'h0a
`define TNCA_S_ANDN          8'h0b
`define TNCA_S_SHL           8'h0c
`define TNCA_S_LSHR          8'h0d
`define TNCA_S_ASHR          8'h0e
`define TNCA_S_MIN_S         8'h10
`define TNCA_S_MAX_S         8'h11
`define TNCA_S_MIN_U         8'h12
`define TNCA_S_MAX_U         8'h13
`define TNCA_S_CLZ           8'h14
`define TNCA_S_POPC          8'h15
`define TNCA_S_SEXT_W        8'h16
`define TNCA_S_ZEXT_W        8'h17
`define TNCA_S_CMP           8'h18
`define TNCA_S_SEL           8'h19
`define TNCA_S_ADDI          8'h20
`define TNCA_S_SUBI          8'h21
`define TNCA_S_MULI          8'h22
`define TNCA_S_ANDI          8'h28
`define TNCA_S_ORI           8'h29
`define TNCA_S_XORI          8'h2a
`define TNCA_S_SHLI          8'h2c
`define TNCA_S_LSHRI         8'h2d
`define TNCA_S_ASHRI         8'h2e
`define TNCA_S_CMPI          8'h38
`define TNCA_S_MOVZ          8'h40
`define TNCA_S_MOVK          8'h41
`define TNCA_S_MOVN          8'h42

//------------------------------------------------------------------------------
// Control flow family (ISA §7)
//------------------------------------------------------------------------------
`define TNCA_C_BR            8'h00
`define TNCA_C_BRNZ          8'h01
`define TNCA_C_BRZ           8'h02
`define TNCA_C_BRR           8'h04
`define TNCA_C_CALL          8'h08
`define TNCA_C_RET           8'h09
`define TNCA_C_YIELD         8'h10
`define TNCA_C_TRAP          8'h18
`define TNCA_C_EXIT          8'h1f

//------------------------------------------------------------------------------
// Vector integer family (ISA §8.1) -- shares func codes with scalar where sane
//------------------------------------------------------------------------------
`define TNCA_V_MOV           8'h00
`define TNCA_V_ADD           8'h01
`define TNCA_V_SUB           8'h02
`define TNCA_V_MUL           8'h03
`define TNCA_V_MULH_S        8'h04
`define TNCA_V_MULH_U        8'h05
`define TNCA_V_AND           8'h08
`define TNCA_V_OR            8'h09
`define TNCA_V_XOR           8'h0a
`define TNCA_V_ANDN          8'h0b
`define TNCA_V_SHL           8'h0c
`define TNCA_V_LSHR          8'h0d
`define TNCA_V_ASHR          8'h0e
`define TNCA_V_MIN_S         8'h10
`define TNCA_V_MAX_S         8'h11
`define TNCA_V_MIN_U         8'h12
`define TNCA_V_MAX_U         8'h13
`define TNCA_V_CLZ           8'h14
`define TNCA_V_POPC          8'h15
`define TNCA_V_CMP           8'h18
`define TNCA_V_SEL           8'h19
`define TNCA_V_SPLAT         8'h1a
`define TNCA_V_IOTA          8'h1b
`define TNCA_V_ADDI          8'h20
`define TNCA_V_SUBI          8'h21
`define TNCA_V_MULI          8'h22
`define TNCA_V_ANDI          8'h28
`define TNCA_V_ORI           8'h29
`define TNCA_V_XORI          8'h2a
`define TNCA_V_SHLI          8'h2c
`define TNCA_V_LSHRI         8'h2d
`define TNCA_V_ASHRI         8'h2e
`define TNCA_V_CMPI          8'h38
`define TNCA_V_MOVI          8'h40

// The functions whose second operand is the immediate. A macro rather than a
// case in tnca_vector_alu so the core can decode it a stage early (R) and the
// unit can take it registered; the list exists once.
`define TNCA_V_USES_IMM(fn) \
    (((fn) == `TNCA_V_ADDI)  || ((fn) == `TNCA_V_SUBI)  || \
     ((fn) == `TNCA_V_MULI)  || ((fn) == `TNCA_V_ANDI)  || \
     ((fn) == `TNCA_V_ORI)   || ((fn) == `TNCA_V_XORI)  || \
     ((fn) == `TNCA_V_SHLI)  || ((fn) == `TNCA_V_LSHRI) || \
     ((fn) == `TNCA_V_ASHRI) || ((fn) == `TNCA_V_CMPI)  || \
     ((fn) == `TNCA_V_MOVI))

//------------------------------------------------------------------------------
// Vector FP family (ISA §8.2)
//------------------------------------------------------------------------------
`define TNCA_VF_ADD          8'h01
`define TNCA_VF_SUB          8'h02
`define TNCA_VF_MUL          8'h03
`define TNCA_VF_FMA          8'h06
`define TNCA_VF_FMS          8'h07
`define TNCA_VF_MIN          8'h10
`define TNCA_VF_MAX          8'h11
`define TNCA_VF_NEG          8'h14
`define TNCA_VF_ABS          8'h15
`define TNCA_VF_CMP          8'h18

// min/max NaN mode, mod[1:0] (ISA §8.3)
`define TNCA_FMM_IEEE        2'd0
`define TNCA_FMM_PROP        2'd1
`define TNCA_FMM_FAST        2'd2

//------------------------------------------------------------------------------
// Convert family (ISA §8.4)
//------------------------------------------------------------------------------
`define TNCA_CVT_S32_F32     8'h00
`define TNCA_CVT_U32_F32     8'h01
`define TNCA_CVT_F32_S32     8'h02
`define TNCA_CVT_F32_U32     8'h03
`define TNCA_CVT_FMT         8'h04

// Float format codes, ISA sec 8.4. These are the numbers xls/tnca_formats.x
// and tools/model/floatfmt.py use, so the RTL, the generator and the oracle
// cannot disagree about what a 3 means.
`define TNCA_FMT_F32         3'd0
`define TNCA_FMT_BF16        3'd1
`define TNCA_FMT_F16         3'd2
`define TNCA_FMT_E5M2        3'd3
`define TNCA_FMT_E4M3        3'd4
`define TNCA_FMT_MAX         3'd4

`define TNCA_RND_RTZ         3'd0
`define TNCA_RND_RNE         3'd1
`define TNCA_RND_RMI         3'd2
`define TNCA_RND_RPI         3'd3

//------------------------------------------------------------------------------
// Predicate family (ISA §8.5)
//------------------------------------------------------------------------------
`define TNCA_P_MOV           8'h00
`define TNCA_P_AND           8'h01
`define TNCA_P_OR            8'h02
`define TNCA_P_XOR           8'h03
`define TNCA_P_NOT           8'h04
`define TNCA_P_ANDN          8'h05
`define TNCA_P_ANY           8'h10
`define TNCA_P_ALL           8'h11
`define TNCA_P_BALLOT        8'h12
`define TNCA_P_FROMBITS      8'h13
`define TNCA_P_POPC          8'h14

//------------------------------------------------------------------------------
// Shuffle family (ISA §8.6)
//------------------------------------------------------------------------------
`define TNCA_SH_XOR          8'h00
`define TNCA_SH_UP           8'h01
`define TNCA_SH_DOWN         8'h02
`define TNCA_SH_PERMUTE      8'h03
`define TNCA_SH_ROTATE       8'h04
`define TNCA_SH_EXTRACT      8'h08
`define TNCA_SH_INSERT       8'h09

//------------------------------------------------------------------------------
// Reduction family (ISA §8.7)
//------------------------------------------------------------------------------
`define TNCA_RD_ADD          8'h00
`define TNCA_RD_MAX_S        8'h01
`define TNCA_RD_MIN_S        8'h02
`define TNCA_RD_AND          8'h03
`define TNCA_RD_OR           8'h04
`define TNCA_RD_XOR          8'h05
`define TNCA_RD_MAX_U        8'h06
`define TNCA_RD_MIN_U        8'h07
`define TNCA_RD_FADD         8'h08
`define TNCA_RD_FMAX         8'h09
`define TNCA_RD_FMIN         8'h0a

//------------------------------------------------------------------------------
// SFU family (ISA §8.8)
//------------------------------------------------------------------------------
`define TNCA_SFU_EXP2        8'h00
`define TNCA_SFU_RCP         8'h01
`define TNCA_SFU_RSQRT       8'h02
`define TNCA_SFU_LOG2        8'h03

//------------------------------------------------------------------------------
// Load/store family (ISA §9.1)
//------------------------------------------------------------------------------
`define TNCA_L_SLD           8'h00
`define TNCA_L_SST           8'h08
`define TNCA_L_VLD           8'h10
`define TNCA_L_VST           8'h18
`define TNCA_L_VGATHER       8'h20
`define TNCA_L_VSCATTER      8'h28

// Atomic / fence, ISA sec 9.2 (major 0x21)
`define TNCA_AT_ADD          8'h00
`define TNCA_AT_AND          8'h01
`define TNCA_AT_OR           8'h02
`define TNCA_AT_XOR          8'h03
`define TNCA_AT_XCHG         8'h04
`define TNCA_AT_MIN_S        8'h05
`define TNCA_AT_MAX_S        8'h06
`define TNCA_AT_MIN_U        8'h07
`define TNCA_AT_MAX_U        8'h08
`define TNCA_AT_CAS          8'h10
`define TNCA_AT_LD           8'h18
`define TNCA_AT_ST           8'h19
`define TNCA_AT_FENCE        8'h20

// The op the memory carries. Deliberately a separate, dense encoding from the
// `func` above: it travels on every request flit, so it is four bits and not
// eight, and the mapping lives in one place (tnca_decoder).
`define TNCA_AOP_W           4
`define TNCA_AOP_ADD         4'd0
`define TNCA_AOP_AND         4'd1
`define TNCA_AOP_OR          4'd2
`define TNCA_AOP_XOR         4'd3
`define TNCA_AOP_XCHG        4'd4
`define TNCA_AOP_MIN_S       4'd5
`define TNCA_AOP_MAX_S       4'd6
`define TNCA_AOP_MIN_U       4'd7
`define TNCA_AOP_MAX_U       4'd8
`define TNCA_AOP_CAS         4'd9
`define TNCA_AOP_LD          4'd10
`define TNCA_AOP_ST          4'd11

// mod fields, ISA sec 9.2
`define TNCA_AMO_ORD_RLX     2'd0
`define TNCA_AMO_ORD_ACQ     2'd1
`define TNCA_AMO_ORD_REL     2'd2
`define TNCA_AMO_ORD_AR      2'd3
`define TNCA_AMO_SCOPE_CORE  2'd0
`define TNCA_AMO_SCOPE_DEV   2'd1
`define TNCA_AMO_SCOPE_SYS   2'd2

`define TNCA_SZ_B            3'd0
`define TNCA_SZ_H            3'd1
`define TNCA_SZ_W            3'd2
`define TNCA_SZ_D            3'd3

`define TNCA_CH_DEFAULT      2'd0
`define TNCA_CH_STREAM       2'd1
`define TNCA_CH_HIGHPRI      2'd2

//------------------------------------------------------------------------------
// Atomic family (ISA §9.2)
//------------------------------------------------------------------------------
`define TNCA_AT_ADD          8'h00
`define TNCA_AT_AND          8'h01
`define TNCA_AT_OR           8'h02
`define TNCA_AT_XOR          8'h03
`define TNCA_AT_XCHG         8'h04
`define TNCA_AT_MIN_S        8'h05
`define TNCA_AT_MAX_S        8'h06
`define TNCA_AT_MIN_U        8'h07
`define TNCA_AT_MAX_U        8'h08
`define TNCA_AT_CAS          8'h10
`define TNCA_AT_LD           8'h18
`define TNCA_AT_ST           8'h19
`define TNCA_AT_FENCE        8'h20

`define TNCA_MO_RELAXED      2'd0
`define TNCA_MO_ACQUIRE      2'd1
`define TNCA_MO_RELEASE      2'd2
`define TNCA_MO_ACQ_REL      2'd3

`define TNCA_SCOPE_CORE      2'd0
`define TNCA_SCOPE_DEVICE    2'd1
`define TNCA_SCOPE_SYSTEM    2'd2

//------------------------------------------------------------------------------
// Mbarrier family (ISA §9A)
//------------------------------------------------------------------------------
`define TNCA_MB_INIT         8'h00
`define TNCA_MB_ARM          8'h01
`define TNCA_MB_ARRIVE       8'h02
`define TNCA_MB_TEST         8'h03
`define TNCA_MB_WAIT         8'h04
`define TNCA_MB_INVAL        8'h05

`define TNCA_BAR_N           32
`define TNCA_BAR_W           5
`define TNCA_TOKEN_BAR       15:0
`define TNCA_TOKEN_EPOCH     63:16

//------------------------------------------------------------------------------
// Accumulator family (ISA §9B)
//------------------------------------------------------------------------------
`define TNCA_A_VREAD         8'h00
`define TNCA_A_VWRITE        8'h01
`define TNCA_A_MULROW        8'h02
`define TNCA_A_FMAROW        8'h03
`define TNCA_A_ZERO          8'h08

`define TNCA_ACC_N           16
`define TNCA_ACC_W           4
`define TNCA_ACC_ROWS        16

//------------------------------------------------------------------------------
// Matrix family (ISA §9C)
//------------------------------------------------------------------------------
`define TNCA_M_ZERO          8'h00
`define TNCA_M_MMA           8'h01

`define TNCA_MMA_K16         2'd0
`define TNCA_MMA_K32         2'd1
`define TNCA_MMA_K64         2'd2

`define TNCA_MMA_F16         2'd0
`define TNCA_MMA_BF16        2'd1
`define TNCA_MMA_E5M2        2'd2
`define TNCA_MMA_E4M3        2'd3

//------------------------------------------------------------------------------
// Tensor descriptor / RF window family (ISA §9D)
//------------------------------------------------------------------------------
`define TNCA_TD_LOAD         8'h00
`define TNCA_TD_RELEASE      8'h02
`define TNCA_W_DEF           8'h10
`define TNCA_W_RELEASE       8'h11

`define TNCA_DESC_N          16
`define TNCA_DESC_W          4
`define TNCA_WIN_N           32
`define TNCA_WIN_W           5

// TensorMap dtype encodings
`define TNCA_DT_F16          8'd0
`define TNCA_DT_BF16         8'd1
`define TNCA_DT_F32          8'd2
`define TNCA_DT_I8           8'd3

// padding modes
`define TNCA_PAD_ZERO        8'd0
`define TNCA_PAD_CLAMP       8'd1
`define TNCA_PAD_FAULT       8'd2

//------------------------------------------------------------------------------
// Tensor transfer family (ISA §9E)
//------------------------------------------------------------------------------
`define TNCA_T_LD            8'h00
`define TNCA_T_ST            8'h08
`define TNCA_T_PREFETCH      8'h10

//------------------------------------------------------------------------------
// Traps (ISA §10)
//------------------------------------------------------------------------------
`define TNCA_TRAP_NONE       8'h00
`define TNCA_TRAP_ILLEGAL    8'h01
`define TNCA_TRAP_MISPC      8'h02
`define TNCA_TRAP_ACCESS     8'h03
`define TNCA_TRAP_DESC       8'h04
`define TNCA_TRAP_BARRIER    8'h05
`define TNCA_TRAP_RESOURCE   8'h06
`define TNCA_TRAP_USER       8'h40

//------------------------------------------------------------------------------
// Capability bitmap (ISA §11)
//------------------------------------------------------------------------------
`define TNCA_CAP_VECTOR_FP32 0
`define TNCA_CAP_SFU         1
`define TNCA_CAP_GATHER      2
`define TNCA_CAP_ATOM_DEV    3
`define TNCA_CAP_ATOM_SYS    4
`define TNCA_CAP_MATRIX_F16  5
`define TNCA_CAP_MATRIX_BF16 6
`define TNCA_CAP_TENSOR_DMA  7
`define TNCA_CAP_MBARRIER    8
`define TNCA_CAP_PACKED_F16  9
`define TNCA_CAP_CACHE_HINTS 10
`define TNCA_CAP_MATRIX_FP8  11

//------------------------------------------------------------------------------
// Internal microarchitecture encodings (NOT architectural)
//------------------------------------------------------------------------------
// Destination register class produced by the decoder.
`define TNCA_DST_NONE        3'd0
`define TNCA_DST_S           3'd1
`define TNCA_DST_V           3'd2
`define TNCA_DST_P           3'd3
`define TNCA_DST_PC          3'd4
`define TNCA_DST_A           3'd5   // accumulator slot
`define TNCA_DST_B           3'd6   // mbarrier
`define TNCA_DST_RES         3'd7   // descriptor or window handle

// Functional unit class -- one instruction may issue per class per cycle.
`define TNCA_FU_NONE         3'd0
`define TNCA_FU_SALU         3'd1   // scalar ALU + control + system
`define TNCA_FU_VALU         3'd2   // vector int/fp/cvt/pred/shuffle/reduce
`define TNCA_FU_SFU          3'd3
`define TNCA_FU_LSU          3'd4
`define TNCA_FU_MATRIX       3'd5
`define TNCA_FU_TXFER        3'd6
`define TNCA_FU_BAR          3'd7

`endif // TNCA_DEFS_VH
