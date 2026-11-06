// TNCA ISA v0.1 -- GENERATED FILE, DO NOT EDIT.
// Source: isa/tnca-isa-v0.1.json (itself generated from
// tools/asm/tnca_isa.py). Regenerate with `make isa`.
// Normative text: doc/isa/TNCA-ISA-v0.1.md.

// This file carries encodings only. Selection patterns, register
// classes and scheduling belong in hand-written .td beside it;
// sec 11.2 keeps the LLVM target thin and off the critical path.

class TNCAInst<bits<8> major, bits<8> func, string mnem>
    : Instruction {
  field bits<64> Inst;
  let Namespace = "TNCA";
  let Size = 8;
  let Inst{63-56} = major;
  let Inst{55-48} = func;
  let AsmString = mnem;
  bits<8> Major = major;
  bits<8> Func  = func;
  bit Maskable = 0;
}

def TNCA_A_FMA_ROW : TNCAInst<65, 3, "a.fma.row"> {
  let Maskable = 1;
}
def TNCA_A_MUL_ROW : TNCAInst<65, 2, "a.mul.row"> {
  let Maskable = 1;
}
def TNCA_A_VREAD : TNCAInst<65, 0, "a.vread"> {
  let Maskable = 1;
}
def TNCA_A_VWRITE : TNCAInst<65, 1, "a.vwrite"> {
  let Maskable = 1;
}
def TNCA_A_ZERO : TNCAInst<65, 8, "a.zero"> {
  let Maskable = 1;
}
def TNCA_ATOM_ADD : TNCAInst<33, 0, "atom.add"> {
  let Maskable = 1;
}
def TNCA_ATOM_AND : TNCAInst<33, 1, "atom.and"> {
  let Maskable = 1;
}
def TNCA_ATOM_CAS : TNCAInst<33, 16, "atom.cas"> {
  let Maskable = 1;
}
def TNCA_ATOM_LD : TNCAInst<33, 24, "atom.ld"> {
  let Maskable = 1;
}
def TNCA_ATOM_MAX_S : TNCAInst<33, 6, "atom.max.s"> {
  let Maskable = 1;
}
def TNCA_ATOM_MAX_U : TNCAInst<33, 8, "atom.max.u"> {
  let Maskable = 1;
}
def TNCA_ATOM_MIN_S : TNCAInst<33, 5, "atom.min.s"> {
  let Maskable = 1;
}
def TNCA_ATOM_MIN_U : TNCAInst<33, 7, "atom.min.u"> {
  let Maskable = 1;
}
def TNCA_ATOM_OR : TNCAInst<33, 2, "atom.or"> {
  let Maskable = 1;
}
def TNCA_ATOM_ST : TNCAInst<33, 25, "atom.st"> {
  let Maskable = 1;
}
def TNCA_ATOM_XCHG : TNCAInst<33, 4, "atom.xchg"> {
  let Maskable = 1;
}
def TNCA_ATOM_XOR : TNCAInst<33, 3, "atom.xor"> {
  let Maskable = 1;
}
def TNCA_BR : TNCAInst<2, 0, "br"> {
}
def TNCA_BR_NZ : TNCAInst<2, 1, "br.nz"> {
}
def TNCA_BR_R : TNCAInst<2, 4, "br.r"> {
}
def TNCA_BR_Z : TNCAInst<2, 2, "br.z"> {
}
def TNCA_CALL : TNCAInst<2, 8, "call"> {
}
def TNCA_EXIT : TNCAInst<2, 31, "exit"> {
}
def TNCA_FENCE : TNCAInst<33, 32, "fence"> {
  let Maskable = 1;
}
def TNCA_M_MMA_ASYNC : TNCAInst<64, 1, "m.mma.async"> {
}
def TNCA_M_MMA_DP_ASYNC : TNCAInst<64, 2, "m.mma.dp.async"> {
}
def TNCA_M_ZERO : TNCAInst<64, 0, "m.zero"> {
}
def TNCA_MB_ARM : TNCAInst<80, 1, "mb.arm"> {
}
def TNCA_MB_ARRIVE : TNCAInst<80, 2, "mb.arrive"> {
}
def TNCA_MB_INIT : TNCAInst<80, 0, "mb.init"> {
}
def TNCA_MB_INVAL : TNCAInst<80, 5, "mb.inval"> {
}
def TNCA_MB_TEST : TNCAInst<80, 3, "mb.test"> {
}
def TNCA_MB_WAIT : TNCAInst<80, 4, "mb.wait"> {
}
def TNCA_NOP : TNCAInst<0, 0, "nop"> {
}
def TNCA_P_ALL : TNCAInst<19, 17, "p.all"> {
}
def TNCA_P_AND : TNCAInst<19, 1, "p.and"> {
}
def TNCA_P_ANDN : TNCAInst<19, 5, "p.andn"> {
}
def TNCA_P_ANY : TNCAInst<19, 16, "p.any"> {
}
def TNCA_P_BALLOT : TNCAInst<19, 18, "p.ballot"> {
}
def TNCA_P_FROMBITS : TNCAInst<19, 19, "p.frombits"> {
}
def TNCA_P_MOV : TNCAInst<19, 0, "p.mov"> {
}
def TNCA_P_NOT : TNCAInst<19, 4, "p.not"> {
}
def TNCA_P_OR : TNCAInst<19, 2, "p.or"> {
}
def TNCA_P_POPC : TNCAInst<19, 20, "p.popc"> {
}
def TNCA_P_XOR : TNCAInst<19, 3, "p.xor"> {
}
def TNCA_RET : TNCAInst<2, 9, "ret"> {
}
def TNCA_S_ADD : TNCAInst<1, 1, "s.add"> {
}
def TNCA_S_ADDI : TNCAInst<1, 32, "s.addi"> {
}
def TNCA_S_AND : TNCAInst<1, 8, "s.and"> {
}
def TNCA_S_ANDI : TNCAInst<1, 40, "s.andi"> {
}
def TNCA_S_ANDN : TNCAInst<1, 11, "s.andn"> {
}
def TNCA_S_ASHR : TNCAInst<1, 14, "s.ashr"> {
}
def TNCA_S_ASHRI : TNCAInst<1, 46, "s.ashri"> {
}
def TNCA_S_CLZ : TNCAInst<1, 20, "s.clz"> {
}
def TNCA_S_CMP : TNCAInst<1, 24, "s.cmp"> {
}
def TNCA_S_CMPI : TNCAInst<1, 56, "s.cmpi"> {
}
def TNCA_S_LD : TNCAInst<32, 0, "s.ld"> {
  let Maskable = 1;
}
def TNCA_S_LSHR : TNCAInst<1, 13, "s.lshr"> {
}
def TNCA_S_LSHRI : TNCAInst<1, 45, "s.lshri"> {
}
def TNCA_S_MAX_S : TNCAInst<1, 17, "s.max.s"> {
}
def TNCA_S_MAX_U : TNCAInst<1, 19, "s.max.u"> {
}
def TNCA_S_MIN_S : TNCAInst<1, 16, "s.min.s"> {
}
def TNCA_S_MIN_U : TNCAInst<1, 18, "s.min.u"> {
}
def TNCA_S_MOV : TNCAInst<1, 0, "s.mov"> {
}
def TNCA_S_MOVK : TNCAInst<1, 65, "s.movk"> {
}
def TNCA_S_MOVN : TNCAInst<1, 66, "s.movn"> {
}
def TNCA_S_MOVZ : TNCAInst<1, 64, "s.movz"> {
}
def TNCA_S_MUL : TNCAInst<1, 3, "s.mul"> {
}
def TNCA_S_MULH_S : TNCAInst<1, 4, "s.mulh.s"> {
}
def TNCA_S_MULH_U : TNCAInst<1, 5, "s.mulh.u"> {
}
def TNCA_S_MULI : TNCAInst<1, 34, "s.muli"> {
}
def TNCA_S_OR : TNCAInst<1, 9, "s.or"> {
}
def TNCA_S_ORI : TNCAInst<1, 41, "s.ori"> {
}
def TNCA_S_POPC : TNCAInst<1, 21, "s.popc"> {
}
def TNCA_S_RDSPECIAL : TNCAInst<0, 1, "s.rdspecial"> {
}
def TNCA_S_SEL : TNCAInst<1, 25, "s.sel"> {
}
def TNCA_S_SEXT_W : TNCAInst<1, 22, "s.sext.w"> {
}
def TNCA_S_SHL : TNCAInst<1, 12, "s.shl"> {
}
def TNCA_S_SHLI : TNCAInst<1, 44, "s.shli"> {
}
def TNCA_S_ST : TNCAInst<32, 8, "s.st"> {
  let Maskable = 1;
}
def TNCA_S_SUB : TNCAInst<1, 2, "s.sub"> {
}
def TNCA_S_SUBI : TNCAInst<1, 33, "s.subi"> {
}
def TNCA_S_WRSPECIAL : TNCAInst<0, 2, "s.wrspecial"> {
}
def TNCA_S_XOR : TNCAInst<1, 10, "s.xor"> {
}
def TNCA_S_XORI : TNCAInst<1, 42, "s.xori"> {
}
def TNCA_S_ZEXT_W : TNCAInst<1, 23, "s.zext.w"> {
}
def TNCA_T_LD_ASYNC : TNCAInst<49, 0, "t.ld.async"> {
  let Maskable = 1;
}
def TNCA_T_PREFETCH : TNCAInst<49, 16, "t.prefetch"> {
  let Maskable = 1;
}
def TNCA_T_ST_ASYNC : TNCAInst<49, 8, "t.st.async"> {
  let Maskable = 1;
}
def TNCA_TD_LOAD : TNCAInst<48, 0, "td.load"> {
}
def TNCA_TD_RELEASE : TNCAInst<48, 2, "td.release"> {
}
def TNCA_TRAP : TNCAInst<2, 24, "trap"> {
}
def TNCA_V_ADD : TNCAInst<16, 1, "v.add"> {
  let Maskable = 1;
}
def TNCA_V_ADDI : TNCAInst<16, 32, "v.addi"> {
  let Maskable = 1;
}
def TNCA_V_AND : TNCAInst<16, 8, "v.and"> {
  let Maskable = 1;
}
def TNCA_V_ANDI : TNCAInst<16, 40, "v.andi"> {
  let Maskable = 1;
}
def TNCA_V_ANDN : TNCAInst<16, 11, "v.andn"> {
  let Maskable = 1;
}
def TNCA_V_ASHR : TNCAInst<16, 14, "v.ashr"> {
  let Maskable = 1;
}
def TNCA_V_ASHRI : TNCAInst<16, 46, "v.ashri"> {
  let Maskable = 1;
}
def TNCA_V_CLZ : TNCAInst<16, 20, "v.clz"> {
  let Maskable = 1;
}
def TNCA_V_CMP : TNCAInst<16, 24, "v.cmp"> {
  let Maskable = 1;
}
def TNCA_V_CMPI : TNCAInst<16, 56, "v.cmpi"> {
  let Maskable = 1;
}
def TNCA_V_EXP2_APPROX : TNCAInst<22, 0, "v.exp2.approx"> {
  let Maskable = 1;
}
def TNCA_V_EXTRACT : TNCAInst<20, 8, "v.extract"> {
  let Maskable = 1;
}
def TNCA_V_INSERT : TNCAInst<20, 9, "v.insert"> {
  let Maskable = 1;
}
def TNCA_V_IOTA : TNCAInst<16, 27, "v.iota"> {
  let Maskable = 1;
}
def TNCA_V_LD : TNCAInst<32, 16, "v.ld"> {
  let Maskable = 1;
}
def TNCA_V_LD_GATHER : TNCAInst<32, 32, "v.ld.gather"> {
  let Maskable = 1;
}
def TNCA_V_LOG2_APPROX : TNCAInst<22, 3, "v.log2.approx"> {
  let Maskable = 1;
}
def TNCA_V_LSHR : TNCAInst<16, 13, "v.lshr"> {
  let Maskable = 1;
}
def TNCA_V_LSHRI : TNCAInst<16, 45, "v.lshri"> {
  let Maskable = 1;
}
def TNCA_V_MAX_S : TNCAInst<16, 17, "v.max.s"> {
  let Maskable = 1;
}
def TNCA_V_MAX_U : TNCAInst<16, 19, "v.max.u"> {
  let Maskable = 1;
}
def TNCA_V_MIN_S : TNCAInst<16, 16, "v.min.s"> {
  let Maskable = 1;
}
def TNCA_V_MIN_U : TNCAInst<16, 18, "v.min.u"> {
  let Maskable = 1;
}
def TNCA_V_MOV : TNCAInst<16, 0, "v.mov"> {
  let Maskable = 1;
}
def TNCA_V_MOVI : TNCAInst<16, 64, "v.movi"> {
  let Maskable = 1;
}
def TNCA_V_MUL : TNCAInst<16, 3, "v.mul"> {
  let Maskable = 1;
}
def TNCA_V_MULH_S : TNCAInst<16, 4, "v.mulh.s"> {
  let Maskable = 1;
}
def TNCA_V_MULH_U : TNCAInst<16, 5, "v.mulh.u"> {
  let Maskable = 1;
}
def TNCA_V_MULI : TNCAInst<16, 34, "v.muli"> {
  let Maskable = 1;
}
def TNCA_V_OR : TNCAInst<16, 9, "v.or"> {
  let Maskable = 1;
}
def TNCA_V_ORI : TNCAInst<16, 41, "v.ori"> {
  let Maskable = 1;
}
def TNCA_V_PERMUTE : TNCAInst<20, 3, "v.permute"> {
  let Maskable = 1;
}
def TNCA_V_POPC : TNCAInst<16, 21, "v.popc"> {
  let Maskable = 1;
}
def TNCA_V_RCP_APPROX : TNCAInst<22, 1, "v.rcp.approx"> {
  let Maskable = 1;
}
def TNCA_V_REDUCE_ADD : TNCAInst<21, 0, "v.reduce.add"> {
  let Maskable = 1;
}
def TNCA_V_REDUCE_AND : TNCAInst<21, 3, "v.reduce.and"> {
  let Maskable = 1;
}
def TNCA_V_REDUCE_MAX_S : TNCAInst<21, 1, "v.reduce.max.s"> {
  let Maskable = 1;
}
def TNCA_V_REDUCE_MAX_U : TNCAInst<21, 6, "v.reduce.max.u"> {
  let Maskable = 1;
}
def TNCA_V_REDUCE_MIN_S : TNCAInst<21, 2, "v.reduce.min.s"> {
  let Maskable = 1;
}
def TNCA_V_REDUCE_MIN_U : TNCAInst<21, 7, "v.reduce.min.u"> {
  let Maskable = 1;
}
def TNCA_V_REDUCE_OR : TNCAInst<21, 4, "v.reduce.or"> {
  let Maskable = 1;
}
def TNCA_V_REDUCE_XOR : TNCAInst<21, 5, "v.reduce.xor"> {
  let Maskable = 1;
}
def TNCA_V_ROTATE : TNCAInst<20, 4, "v.rotate"> {
  let Maskable = 1;
}
def TNCA_V_RSQRT_APPROX : TNCAInst<22, 2, "v.rsqrt.approx"> {
  let Maskable = 1;
}
def TNCA_V_SEL : TNCAInst<16, 25, "v.sel"> {
  let Maskable = 1;
}
def TNCA_V_SHL : TNCAInst<16, 12, "v.shl"> {
  let Maskable = 1;
}
def TNCA_V_SHLI : TNCAInst<16, 44, "v.shli"> {
  let Maskable = 1;
}
def TNCA_V_SHUFFLE_DOWN : TNCAInst<20, 2, "v.shuffle.down"> {
  let Maskable = 1;
}
def TNCA_V_SHUFFLE_UP : TNCAInst<20, 1, "v.shuffle.up"> {
  let Maskable = 1;
}
def TNCA_V_SHUFFLE_XOR : TNCAInst<20, 0, "v.shuffle.xor"> {
  let Maskable = 1;
}
def TNCA_V_SPLAT : TNCAInst<16, 26, "v.splat"> {
  let Maskable = 1;
}
def TNCA_V_ST : TNCAInst<32, 24, "v.st"> {
  let Maskable = 1;
}
def TNCA_V_ST_SCATTER : TNCAInst<32, 40, "v.st.scatter"> {
  let Maskable = 1;
}
def TNCA_V_SUB : TNCAInst<16, 2, "v.sub"> {
  let Maskable = 1;
}
def TNCA_V_SUBI : TNCAInst<16, 33, "v.subi"> {
  let Maskable = 1;
}
def TNCA_V_XOR : TNCAInst<16, 10, "v.xor"> {
  let Maskable = 1;
}
def TNCA_V_XORI : TNCAInst<16, 42, "v.xori"> {
  let Maskable = 1;
}
def TNCA_VF_ABS : TNCAInst<17, 21, "vf.abs"> {
  let Maskable = 1;
}
def TNCA_VF_ADD : TNCAInst<17, 1, "vf.add"> {
  let Maskable = 1;
}
def TNCA_VF_CMP : TNCAInst<17, 24, "vf.cmp"> {
  let Maskable = 1;
}
def TNCA_VF_CVT : TNCAInst<18, 4, "vf.cvt"> {
  let Maskable = 1;
}
def TNCA_VF_CVT_F32_S32 : TNCAInst<18, 2, "vf.cvt.f32.s32"> {
  let Maskable = 1;
}
def TNCA_VF_CVT_F32_U32 : TNCAInst<18, 3, "vf.cvt.f32.u32"> {
  let Maskable = 1;
}
def TNCA_VF_CVT_S32_F32 : TNCAInst<18, 0, "vf.cvt.s32.f32"> {
  let Maskable = 1;
}
def TNCA_VF_CVT_U32_F32 : TNCAInst<18, 1, "vf.cvt.u32.f32"> {
  let Maskable = 1;
}
def TNCA_VF_FMA : TNCAInst<17, 6, "vf.fma"> {
  let Maskable = 1;
}
def TNCA_VF_FMS : TNCAInst<17, 7, "vf.fms"> {
  let Maskable = 1;
}
def TNCA_VF_MAX : TNCAInst<17, 17, "vf.max"> {
  let Maskable = 1;
}
def TNCA_VF_MIN : TNCAInst<17, 16, "vf.min"> {
  let Maskable = 1;
}
def TNCA_VF_MUL : TNCAInst<17, 3, "vf.mul"> {
  let Maskable = 1;
}
def TNCA_VF_NEG : TNCAInst<17, 20, "vf.neg"> {
  let Maskable = 1;
}
def TNCA_VF_REDUCE_ADD : TNCAInst<21, 8, "vf.reduce.add"> {
  let Maskable = 1;
}
def TNCA_VF_REDUCE_MAX : TNCAInst<21, 9, "vf.reduce.max"> {
  let Maskable = 1;
}
def TNCA_VF_REDUCE_MIN : TNCAInst<21, 10, "vf.reduce.min"> {
  let Maskable = 1;
}
def TNCA_VF_SUB : TNCAInst<17, 2, "vf.sub"> {
  let Maskable = 1;
}
def TNCA_W_DEF : TNCAInst<48, 16, "w.def"> {
}
def TNCA_W_RELEASE : TNCAInst<48, 17, "w.release"> {
}
def TNCA_YIELD : TNCAInst<2, 16, "yield"> {
}
