#!/usr/bin/env python3
"""
tnca_dis.py -- disassembler for TNCA ISA v0.1.

    tnca_dis.py image.hex [--base 0] [--count N]
    tnca_dis.py --word 0140010012340000

Reads the same encoding tables as the assembler, so a round-trip
(assemble -> disassemble -> assemble) is a real check on both. `--roundtrip`
does exactly that over a source file and reports any instruction whose
re-assembly does not reproduce the original word.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnca_isa import (ISA, MAJ, MAJ_NAME, CC_NAME, SPECIAL_NAME, SIZE,
                      CACHE_HINT, FMINMAX_MODE, RND, decode_fields,
                      MASKABLE_MAJORS, MMA_K, MMA_DT, FMT_NAME)

BY_OPCODE = {}
for _m, (_maj, _fn, _kinds, _sfx) in ISA.items():
    BY_OPCODE[(MAJ[_maj], _fn)] = (_m, _kinds, _sfx)

SIZE_NAME = {v: k for k, v in SIZE.items()}
MMA_K_NAME  = {v: k for k, v in MMA_K.items()}
MMA_DT_NAME = {v: k for k, v in MMA_DT.items()}
HINT_NAME = {v: k for k, v in CACHE_HINT.items()}
FMM_NAME = {v: k for k, v in FMINMAX_MODE.items()}
RND_NAME = {v: k for k, v in RND.items()}


def disassemble(word, addr=0):
    f = decode_fields(word)
    key = (f['major'], f['func'])
    if key not in BY_OPCODE:
        return f".dword 0x{word:016x}    // undecodable: major={f['major']:#04x} func={f['func']:#04x}"

    mnem, kinds, sfx = BY_OPCODE[key]
    mod = f['mod']
    name = mnem
    trailing = []

    if sfx == 'cc_int' or sfx == 'cc_fp':
        name += '.' + CC_NAME.get(mod & 0xf, f"cc{mod & 0xf}")
    elif sfx == 'ls':
        signed = (mod >> 3) & 1
        name += '.' + ('s' if signed else '') + SIZE_NAME.get(mod & 7, '?')
        h = (mod >> 4) & 3
        if h:
            name += '.' + HINT_NAME.get(h, f"hint{h}")
    elif sfx == 'fmm':
        name += '.' + FMM_NAME.get(mod & 3, '?')
    elif sfx == 'rnd':
        name += '.' + RND_NAME.get(mod & 7, '?')
    elif sfx == 'ftz':
        if (mod >> 8) & 1:
            name += '.ftz'
    elif sfx == 'redb':
        if mod & 1:
            name += '.b'
    elif sfx == 'fmt2':
        name += '.' + FMT_NAME.get(mod & 7, f'fmt?{mod & 7}')
        name += '.' + FMT_NAME.get((mod >> 3) & 7, f'fmt?{(mod >> 3) & 7}')
    elif sfx == 'mma':
        k, dt = mod & 3, (mod >> 2) & 3
        if k:
            name += '.' + MMA_K_NAME.get(k, f'k?{k}')
        if dt:
            name += '.' + MMA_DT_NAME.get(dt, f'dt?{dt}')
    elif sfx == 'txf':
        name += '.' + SIZE_NAME.get(mod & 7, '?')
        if (mod >> 5) & 1:
            name += '.t'
    elif sfx == 'lsl':
        if mod & 3:
            trailing.append(f"lsl #{16 * (mod & 3)}")

    ops = []
    for k in kinds:
        if k == 'sd':   ops.append(f"s{f['d'] & 0x3f}")
        elif k == 'sa': ops.append(f"s{f['a'] & 0x3f}")
        elif k == 'sb': ops.append(f"s{f['b'] & 0x3f}")
        elif k == 'sc': ops.append(f"s{f['c'] & 0x3f}")
        elif k == 'vd': ops.append(f"v{f['d']}")
        elif k == 'va': ops.append(f"v{f['a']}")
        elif k == 'vb': ops.append(f"v{f['b']}")
        elif k == 'vc': ops.append(f"v{f['c']}")
        elif k == 'pd': ops.append(f"p{f['d'] & 0xf}")
        elif k == 'pa': ops.append(f"p{f['a'] & 0xf}")
        elif k == 'pb': ops.append(f"p{f['b'] & 0xf}")
        elif k == 'ad': ops.append(f"a{f['d'] & 0xf}")
        elif k == 'aa': ops.append(f"a{f['a'] & 0xf}")
        elif k == 'bd': ops.append(f"b{f['d'] & 0x1f}")
        elif k == 'ba': ops.append(f"b{f['a'] & 0x1f}")
        elif k == 'dd': ops.append(f"d{f['d'] & 0xf}")
        elif k == 'da': ops.append(f"d{f['a'] & 0xf}")
        elif k == 'wd': ops.append(f"w{f['d'] & 0x1f}")
        elif k == 'wa': ops.append(f"w{f['a'] & 0x1f}")
        elif k == 'cnt2':
            ops.append(f"#{f['b']}")
            ops.append(f"#{f['c']}")
        elif k == 'winspec':
            ops.append(f"#{f['b']}")
            ops.append(f"#{f['c']}")
        elif k == 'mem_s':
            ops.append(f"[s{f['a'] & 0x3f}]")
        elif k == 'rdst':
            ops.append(f"v{f['d']}" if (mod & 1) else f"s{f['d'] & 0x3f}")
        elif k == 'i16s':
            v = f['imm16']
            ops.append(f"#{v - 0x10000 if v & 0x8000 else v}")
        elif k == 'i16z':
            ops.append(f"#{f['imm16']}")
        elif k == 'i24':
            v = f['imm24']
            ops.append(f"#{v - 0x1000000 if v & 0x800000 else v}")
        elif k in ('lbl16', 'lbl24'):
            raw = f['imm16'] if k == 'lbl16' else f['imm24']
            bits = 16 if k == 'lbl16' else 24
            disp = raw - (1 << bits) if raw & (1 << (bits - 1)) else raw
            ops.append(f"0x{addr + 8 * disp:x}")
        elif k == 'spec':
            ops.append(SPECIAL_NAME.get(mod, f"#{mod}"))
        elif k == 'lane':
            ops.append(f"#{mod & 0xf}")
        elif k == 'mem_si':
            v = f['imm16']
            off = v - 0x10000 if v & 0x8000 else v
            ops.append(f"[s{f['a'] & 0x3f}]" if off == 0
                       else f"[s{f['a'] & 0x3f}, #{off}]")
        elif k == 'mem_sv':
            ops.append(f"[s{f['a'] & 0x3f}, v{f['b']}]")

    ops += trailing
    pfx = ''
    if f['major'] in MASKABLE_MAJORS and (f['pred'] or f['pn']):
        pfx = f"@{'!' if f['pn'] else ''}p{f['pred']} "
    return (pfx + name + (' ' + ', '.join(ops) if ops else '')).rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image', nargs='?')
    ap.add_argument('--word', help="disassemble a single 64-bit hex word")
    ap.add_argument('--base', type=lambda x: int(x, 0), default=0)
    ap.add_argument('--count', type=int, default=0)
    ap.add_argument('--roundtrip', help="assemble this .s, disassemble it, "
                                        "re-assemble and compare")
    args = ap.parse_args()

    if args.word:
        w = int(args.word, 16)
        print(f"{w:016x}  {disassemble(w)}")
        return 0

    if args.roundtrip:
        from tnca_asm import Assembler
        src = open(args.roundtrip).read()
        recs = Assembler().assemble(src)
        bad = 0
        n = 0
        for addr, nbytes, word, srctext in recs:
            # data directives also emit 8-byte records; they are not code
            if nbytes != 8 or srctext.strip().split()[0].lower() in (
                    '.dword', '.word', '.float'):
                continue
            n += 1
            text = disassemble(word, addr)
            if text.startswith('.dword'):
                print(f"{addr:08x}: not decodable: {word:016x}")
                bad += 1
                continue
            try:
                back = Assembler()
                back.symbols = {}
                back.addr = addr
                back.line_no, back.line_text = 0, text
                again = back.encode_insn(text)
            except Exception as e:
                print(f"{addr:08x}: re-assembly failed: {text!r}: {e}")
                bad += 1
                continue
            if again != word:
                print(f"{addr:08x}: mismatch {word:016x} -> {text!r} -> {again:016x}")
                bad += 1
        print(f"round-trip: {n} instructions, {bad} mismatches")
        return 1 if bad else 0

    if not args.image:
        ap.error("give an image, --word or --roundtrip")

    with open(args.image) as fh:
        words = [int(l.strip(), 16) for l in fh if l.strip()]
    if args.count:
        words = words[:args.count]
    for i, w in enumerate(words):
        a = args.base + 8 * i
        print(f"{a:08x}  {w:016x}  {disassemble(w, a)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
