#!/usr/bin/env python3
"""
Generate encoding-legality vectors for tb_decoder.

Every valid instruction the assembler can emit is fed through once, then
mutated in each of the ways ISA sec 2.2 / 2.3 forbid. A decoder that ignores
reserved bits passes the first set and fails the second, which is the point.
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'asm'))
from tnca_isa import ISA, MAJ, decode_fields, encode   # noqa: E402
from tnca_asm import Assembler                          # noqa: E402

# Families the Stage-1 decoder implements. The rest are allocated but reserved
# and must be rejected (ISA sec 3).
IMPLEMENTED = {MAJ[k] for k in
               ('SYS', 'SALU', 'CTRL', 'VINT', 'VFP', 'CVT', 'PRED',
                'SHUF', 'RED', 'SFU', 'LSU')}

# One representative, assemblable instruction per mnemonic.
SAMPLES = """
nop
s.rdspecial s1, clock
s.wrspecial #0x100, s2
s.mov s1, s2
s.add s1, s2, s3
s.sub s1, s2, s3
s.mul s1, s2, s3
s.mulh.s s1, s2, s3
s.mulh.u s1, s2, s3
s.and s1, s2, s3
s.or s1, s2, s3
s.xor s1, s2, s3
s.andn s1, s2, s3
s.shl s1, s2, s3
s.lshr s1, s2, s3
s.ashr s1, s2, s3
s.min.s s1, s2, s3
s.max.u s1, s2, s3
s.clz s1, s2
s.popc s1, s2
s.sext.w s1, s2
s.zext.w s1, s2
s.cmp.lt s1, s2, s3
s.sel s1, s2, s3, s4
s.addi s1, s2, #100
s.subi s1, s2, #-100
s.muli s1, s2, #7
s.andi s1, s2, #0xff
s.ori s1, s2, #0xff
s.xori s1, s2, #0xff
s.shli s1, s2, #3
s.lshri s1, s2, #3
s.ashri s1, s2, #3
s.cmpi.ne s1, s2, #5
s.movz s1, #0x1234
s.movk s1, #0x1234, lsl #32
s.movn s1, #0x1234
br 0
br.nz s1, 0
br.z s1, 0
br.r s1
call s30, 0
ret s30
yield
trap #5
exit
v.mov v1, v2
v.add v1, v2, v3
v.sub v1, v2, v3
v.mul v1, v2, v3
v.mulh.s v1, v2, v3
v.and v1, v2, v3
v.shl v1, v2, v3
v.min.s v1, v2, v3
v.clz v1, v2
v.popc v1, v2
v.cmp.lt p1, v2, v3
v.sel v1, v2, v3
v.splat v1, s2
v.iota v1
v.addi v1, v2, #3
v.andi v1, v2, #3
v.shli v1, v2, #3
v.cmpi.ge p1, v2, #3
v.movi v1, #-7
vf.add v1, v2, v3
vf.sub v1, v2, v3
vf.mul v1, v2, v3
vf.fma v1, v2, v3, v4
vf.fms v1, v2, v3, v4
vf.min v1, v2, v3
vf.max.prop v1, v2, v3
vf.neg v1, v2
vf.abs v1, v2
vf.cmp.uo p1, v2, v3
vf.cvt.s32.f32 v1, v2
vf.cvt.u32.f32 v1, v2
vf.cvt.f32.s32.rne v1, v2
vf.cvt.f32.u32 v1, v2
p.mov p1, p2
p.and p1, p2, p3
p.not p1, p2
p.any s1, p2
p.all s1, p2
p.ballot s1, p2
p.frombits p1, s2
p.popc s1, p2
v.shuffle.xor v1, v2, s3
v.shuffle.up v1, v2, s3
v.shuffle.down v1, v2, s3
v.permute v1, v2, v3
v.rotate v1, v2, s3
v.extract s1, v2, #5
v.insert v1, s2, #5
v.reduce.add s1, v2
v.reduce.add.b v1, v2
v.reduce.max.s s1, v2
vf.reduce.add s1, v2
vf.reduce.max.b v1, v2
v.exp2.approx v1, v2
v.rcp.approx v1, v2
v.rsqrt.approx v1, v2
v.log2.approx v1, v2
s.ld.w s1, [s2, #16]
s.ld.sb s1, [s2, #-1]
s.ld.d.stream s1, [s2]
s.st.d [s2, #8], s1
s.st.b [s2], s1
v.ld.w v1, [s2, #0]
v.ld.sh v1, [s2, #0]
v.st.w [s2, #0], v1
v.ld.gather.w v1, [s2, v3]
v.st.scatter.b [s2, v3], v1
@p3 v.add v1, v2, v3
@!p3 v.ld.w v1, [s2, #0]
"""


def assemble_one(text):
    a = Assembler()
    recs = a.assemble(text)
    return [v for (_, n, v, _) in recs if n == 8]


# Which instruction fields each operand kind consumes. A field that carries part
# of an immediate is "in use" even when that part happens to be zero, so it must
# not be mutated to prove the reserved-bit rule.
KIND_FIELDS = {
    'sd': 'd', 'vd': 'd', 'pd': 'd', 'rdst': 'd',
    'sa': 'a', 'va': 'a', 'pa': 'a',
    'sb': 'b', 'vb': 'b', 'pb': 'b',
    'sc': 'c', 'vc': 'c',
    'i16s': 'bc', 'i16z': 'bc', 'lbl16': 'bc', 'mem_si': 'abc_off',
    'i24': 'abc', 'lbl24': 'abc',
    'mem_sv': 'ab',
    'spec': '', 'lane': '',
}


def used_fields(mnemonic):
    """Set of instruction fields (d/a/b/c) the encoding actually uses."""
    _, _, kinds, _ = ISA[mnemonic]
    used = set()
    for k in kinds:
        f = KIND_FIELDS[k]
        if f == 'abc':
            used |= {'a', 'b', 'c'}
        elif f == 'bc':
            used |= {'b', 'c'}
        elif f == 'ab':
            used |= {'a', 'b'}
        elif f == 'abc_off':
            used |= {'a', 'b', 'c'}     # base in a, displacement in b and c
        elif f:
            used.add(f)
    # s.movk reads its own destination, so d is a source too -- already counted.
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True)
    args = ap.parse_args()

    lines = []

    def emit(word, illegal, label):
        lines.append(f"{word:016x} {1 if illegal else 0} {label}")

    valid = []
    for src in SAMPLES.strip().splitlines():
        src = src.strip()
        if not src:
            continue
        try:
            ws = assemble_one(".org 0\n" + src + "\n")
        except Exception as e:                       # pragma: no cover
            print(f"cannot assemble sample {src!r}: {e}", file=sys.stderr)
            return 1
        body = src[src.index(' ') + 1:] if src.startswith('@') else src
        mnem, _ = Assembler.resolve_mnemonic(body.split()[0])
        for wd in ws:
            valid.append((wd, src.replace(' ', '_').replace(',', ''), mnem))

    for wd, label, _ in valid:
        emit(wd, False, f"ok:{label}")

    # ---- mutations that must be rejected --------------------------------
    for wd, label, mnem in valid:
        f = decode_fields(wd)
        used = used_fields(mnem)

        # the X bit is reserved in v0.1
        emit(wd | (1 << 11), True, f"xbit:{label}")

        # a reserved field forced non-zero
        for name, shift in (('d', 40), ('a', 32), ('b', 24), ('c', 16)):
            if name not in used:
                emit(wd | (0x5a << shift), True, f"resv_{name}:{label}")

        # a predicate on an unpredicated family
        if f['major'] in (MAJ['SYS'], MAJ['SALU'], MAJ['CTRL'], MAJ['PRED']):
            emit(wd | (3 << 12), True, f"pred_on_scalar:{label}")
            emit(wd | (1 << 10), True, f"pn_on_scalar:{label}")

        # mod bit 9 is reserved everywhere except the special-register family,
        # which uses the whole 10-bit field for the register id
        if f['major'] != MAJ['SYS']:
            emit(wd | (1 << 9), True, f"mod9:{label}")

    # ---- unallocated families and functions -----------------------------
    for major in (0x03, 0x17, 0x22, 0x30, 0x31, 0x32, 0x40, 0x41,
                  0x50, 0x51, 0x60, 0x7f, 0xff):
        emit(encode(major, 0x00, d=1, a=2), True, f"major_{major:02x}")
    for major in sorted(IMPLEMENTED):
        emit(encode(major, 0xfe, d=1, a=2), True, f"func_fe_major_{major:02x}")

    # ---- out-of-range register indices ----------------------------------
    emit(encode(MAJ['SALU'], 0x01, d=0x40, a=1, b=2), True, "sreg64_dst")
    emit(encode(MAJ['SALU'], 0x01, d=1, a=0x40, b=2), True, "sreg64_srcA")
    emit(encode(MAJ['PRED'], 0x01, d=0x10, a=1, b=2), True, "preg16_dst")
    emit(encode(MAJ['PRED'], 0x01, d=1, a=0x10, b=2), True, "preg16_srcA")

    # ---- illegal modifiers ----------------------------------------------
    emit(encode(MAJ['SALU'], 0x18, d=1, a=2, b=3, mod=0x0a), True, "cc_o_on_int")
    emit(encode(MAJ['VFP'], 0x18, d=1, a=2, b=3, mod=0x06), True, "cc_ltu_on_fp")
    emit(encode(MAJ['VFP'], 0x10, d=1, a=2, b=3, mod=0x03), True, "fminmax_mode3")
    emit(encode(MAJ['LSU'], 0x00, d=1, a=2, mod=0x04), True, "size4")
    emit(encode(MAJ['LSU'], 0x00, d=1, a=2, mod=0x32), True, "cachehint3")
    emit(encode(MAJ['LSU'], 0x10, d=1, a=2, mod=0x03), True, "vector_size64")
    emit(encode(MAJ['CVT'], 0x02, d=1, a=2, mod=0x05), True, "rndmode5")

    # ---- writes to S0 / P0 assemble and decode fine (ISA sec 12.6) ------
    emit(encode(MAJ['SALU'], 0x01, d=0, a=1, b=2), False, "write_s0_ok")
    emit(encode(MAJ['PRED'], 0x01, d=0, a=1, b=2), False, "write_p0_ok")

    with open(args.output, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(lines)} vectors -> {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
