#!/usr/bin/env python3
"""
check_consistency.py -- the two existing ISA sources must agree.

doc/compiler_overview.md sec 11.3: there must be ONE machine-readable source of
opcode numbers, not a third one added for the compiler. Before making the schema
the source, this checks whether the two that already exist -- the Verilog
defines the RTL decodes with, and the Python tables the assembler encodes with
-- say the same thing. A silent disagreement between them would mean the
assembler and the decoder have been testing each other against different ISAs.
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'asm'))
from tnca_isa import ISA, MAJ, CC, SIZE, SPECIAL, MMA_K, MMA_DT   # noqa: E402

DEFS = os.path.join(ROOT, 'rtl', 'common', 'tnca_defs.vh')

# `define NAME  <width>'<base><value>
DEF_RE = re.compile(r"^`define\s+(\w+)\s+(\d+)'([hdb])([0-9a-fA-F_]+)")


def load_defs():
    out = {}
    for line in open(DEFS):
        m = DEF_RE.match(line.strip())
        if not m:
            continue
        name, _w, base, val = m.groups()
        val = val.replace('_', '')
        out[name] = int(val, {'h': 16, 'd': 10, 'b': 2}[base])
    return out


# Verilog define name -> (python table, key). Only entries that exist in both.
def build_expectations():
    exp = {}
    for k, v in MAJ.items():
        exp[f'TNCA_MAJ_{k}'] = v
    for k, v in CC.items():
        exp[f'TNCA_CC_{k.upper()}'] = v
    for k, v in SIZE.items():
        exp[f'TNCA_SZ_{k.upper()}'] = v

    # func codes, by the mnemonic-to-define mapping the two sources share
    func_map = {
        's.mov': 'TNCA_S_MOV', 's.add': 'TNCA_S_ADD', 's.sub': 'TNCA_S_SUB',
        's.mul': 'TNCA_S_MUL', 's.mulh.s': 'TNCA_S_MULH_S',
        's.mulh.u': 'TNCA_S_MULH_U', 's.and': 'TNCA_S_AND', 's.or': 'TNCA_S_OR',
        's.xor': 'TNCA_S_XOR', 's.andn': 'TNCA_S_ANDN', 's.shl': 'TNCA_S_SHL',
        's.lshr': 'TNCA_S_LSHR', 's.ashr': 'TNCA_S_ASHR',
        's.min.s': 'TNCA_S_MIN_S', 's.max.s': 'TNCA_S_MAX_S',
        's.min.u': 'TNCA_S_MIN_U', 's.max.u': 'TNCA_S_MAX_U',
        's.clz': 'TNCA_S_CLZ', 's.popc': 'TNCA_S_POPC',
        's.sext.w': 'TNCA_S_SEXT_W', 's.zext.w': 'TNCA_S_ZEXT_W',
        's.cmp': 'TNCA_S_CMP', 's.sel': 'TNCA_S_SEL',
        's.addi': 'TNCA_S_ADDI', 's.subi': 'TNCA_S_SUBI', 's.muli': 'TNCA_S_MULI',
        's.andi': 'TNCA_S_ANDI', 's.ori': 'TNCA_S_ORI', 's.xori': 'TNCA_S_XORI',
        's.shli': 'TNCA_S_SHLI', 's.lshri': 'TNCA_S_LSHRI',
        's.ashri': 'TNCA_S_ASHRI', 's.cmpi': 'TNCA_S_CMPI',
        's.movz': 'TNCA_S_MOVZ', 's.movk': 'TNCA_S_MOVK', 's.movn': 'TNCA_S_MOVN',
        'br': 'TNCA_C_BR', 'br.nz': 'TNCA_C_BRNZ', 'br.z': 'TNCA_C_BRZ',
        'br.r': 'TNCA_C_BRR', 'call': 'TNCA_C_CALL', 'ret': 'TNCA_C_RET',
        'yield': 'TNCA_C_YIELD', 'trap': 'TNCA_C_TRAP', 'exit': 'TNCA_C_EXIT',
        'v.mov': 'TNCA_V_MOV', 'v.add': 'TNCA_V_ADD', 'v.sub': 'TNCA_V_SUB',
        'v.mul': 'TNCA_V_MUL', 'v.cmp': 'TNCA_V_CMP', 'v.sel': 'TNCA_V_SEL',
        'v.splat': 'TNCA_V_SPLAT', 'v.iota': 'TNCA_V_IOTA',
        'v.movi': 'TNCA_V_MOVI', 'v.clz': 'TNCA_V_CLZ', 'v.popc': 'TNCA_V_POPC',
        'vf.add': 'TNCA_VF_ADD', 'vf.sub': 'TNCA_VF_SUB', 'vf.mul': 'TNCA_VF_MUL',
        'vf.fma': 'TNCA_VF_FMA', 'vf.fms': 'TNCA_VF_FMS',
        'vf.min': 'TNCA_VF_MIN', 'vf.max': 'TNCA_VF_MAX',
        'vf.neg': 'TNCA_VF_NEG', 'vf.abs': 'TNCA_VF_ABS', 'vf.cmp': 'TNCA_VF_CMP',
        'p.mov': 'TNCA_P_MOV', 'p.and': 'TNCA_P_AND', 'p.or': 'TNCA_P_OR',
        'p.xor': 'TNCA_P_XOR', 'p.not': 'TNCA_P_NOT', 'p.andn': 'TNCA_P_ANDN',
        'p.any': 'TNCA_P_ANY', 'p.all': 'TNCA_P_ALL', 'p.ballot': 'TNCA_P_BALLOT',
        'p.frombits': 'TNCA_P_FROMBITS', 'p.popc': 'TNCA_P_POPC',
        'v.shuffle.xor': 'TNCA_SH_XOR', 'v.shuffle.up': 'TNCA_SH_UP',
        'v.shuffle.down': 'TNCA_SH_DOWN', 'v.permute': 'TNCA_SH_PERMUTE',
        'v.rotate': 'TNCA_SH_ROTATE', 'v.extract': 'TNCA_SH_EXTRACT',
        'v.insert': 'TNCA_SH_INSERT',
        'v.reduce.add': 'TNCA_RD_ADD', 'v.reduce.max.s': 'TNCA_RD_MAX_S',
        'v.reduce.min.s': 'TNCA_RD_MIN_S', 'v.reduce.and': 'TNCA_RD_AND',
        'v.reduce.or': 'TNCA_RD_OR', 'v.reduce.xor': 'TNCA_RD_XOR',
        'v.reduce.max.u': 'TNCA_RD_MAX_U', 'v.reduce.min.u': 'TNCA_RD_MIN_U',
        'vf.reduce.add': 'TNCA_RD_FADD', 'vf.reduce.max': 'TNCA_RD_FMAX',
        'vf.reduce.min': 'TNCA_RD_FMIN',
        'v.exp2.approx': 'TNCA_SFU_EXP2', 'v.rcp.approx': 'TNCA_SFU_RCP',
        'v.rsqrt.approx': 'TNCA_SFU_RSQRT', 'v.log2.approx': 'TNCA_SFU_LOG2',
        's.ld': 'TNCA_L_SLD', 's.st': 'TNCA_L_SST', 'v.ld': 'TNCA_L_VLD',
        'v.st': 'TNCA_L_VST', 'v.ld.gather': 'TNCA_L_VGATHER',
        'v.st.scatter': 'TNCA_L_VSCATTER',
        'mb.init': 'TNCA_MB_INIT', 'mb.arm': 'TNCA_MB_ARM',
        'mb.arrive': 'TNCA_MB_ARRIVE', 'mb.test': 'TNCA_MB_TEST',
        'mb.wait': 'TNCA_MB_WAIT', 'mb.inval': 'TNCA_MB_INVAL',
        'a.vread': 'TNCA_A_VREAD', 'a.vwrite': 'TNCA_A_VWRITE',
        'a.mul.row': 'TNCA_A_MULROW', 'a.fma.row': 'TNCA_A_FMAROW',
        'a.zero': 'TNCA_A_ZERO',
        'm.zero': 'TNCA_M_ZERO', 'm.mma.async': 'TNCA_M_MMA',
        'td.load': 'TNCA_TD_LOAD', 'td.release': 'TNCA_TD_RELEASE',
        'w.def': 'TNCA_W_DEF', 'w.release': 'TNCA_W_RELEASE',
        't.ld.async': 'TNCA_T_LD', 't.st.async': 'TNCA_T_ST',
        't.prefetch': 'TNCA_T_PREFETCH',
        'vf.cvt.s32.f32': 'TNCA_CVT_S32_F32',
        'vf.cvt.u32.f32': 'TNCA_CVT_U32_F32',
        'vf.cvt.f32.s32': 'TNCA_CVT_F32_S32',
        'vf.cvt.f32.u32': 'TNCA_CVT_F32_U32',
        'nop': 'TNCA_SYS_NOP', 's.rdspecial': 'TNCA_SYS_RDSPECIAL',
        's.wrspecial': 'TNCA_SYS_WRSPECIAL',
    }
    for mnem, define in func_map.items():
        if mnem in ISA:
            exp[define] = ISA[mnem][1]

    for k, v in SPECIAL.items():
        name = 'TNCA_SPEC_' + {
            'pid.x': 'PID_X', 'pid.y': 'PID_Y', 'pid.z': 'PID_Z',
            'npid.x': 'NPID_X', 'npid.y': 'NPID_Y', 'npid.z': 'NPID_Z',
            'role_id': 'ROLE_ID', 'num_roles': 'NUM_ROLES',
            'core_id': 'CORE_ID', 'num_cores': 'NUM_CORES',
            'num_lanes': 'NUM_LANES', 'num_vregs': 'NUM_VREGS',
            'tensor_max_rank': 'MAX_RANK', 'num_acc': 'NUM_ACC',
            'clock': 'CLOCK', 'instret': 'INSTRET', 'arg_base': 'ARG_BASE',
            'num_barriers': 'NUM_BAR', 'num_windows': 'NUM_WIN',
            'num_descriptors': 'NUM_DESC', 'num_phys_vregs': 'NUM_PHYSV',
            'isa_version': 'ISA_VER', 'cap0': 'CAP0',
            'trap_pc': 'TRAP_PC', 'trap_code': 'TRAP_CODE',
        }.get(k, k.upper())
        exp[name] = v

    for k, v in MMA_K.items():
        exp['TNCA_MMA_' + k.upper()] = v
    for k, v in MMA_DT.items():
        exp['TNCA_MMA_' + k.upper()] = v
    return exp


def main():
    defs = load_defs()
    exp = build_expectations()

    missing, mismatch = [], []
    for name, want in sorted(exp.items()):
        if name not in defs:
            missing.append(name)
        elif defs[name] != want:
            mismatch.append((name, defs[name], want))

    print(f"checked {len(exp)} shared constants against {DEFS}")
    for name in missing:
        print(f"  MISSING in Verilog: {name} (Python says {exp[name]:#x})")
    for name, got, want in mismatch:
        print(f"  MISMATCH {name}: Verilog {got:#x}, Python {want:#x}")

    if missing or mismatch:
        print(f"FAIL: {len(missing)} missing, {len(mismatch)} disagree")
        return 1
    print("PASS: the RTL defines and the assembler tables agree")
    return 0


if __name__ == '__main__':
    sys.exit(main())
