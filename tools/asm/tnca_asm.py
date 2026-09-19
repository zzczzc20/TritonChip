#!/usr/bin/env python3
"""
tnca_asm.py -- assembler for TNCA ISA v0.1.

Usage:
    tnca_asm.py input.s -o image.hex [--lst image.lst] [--size WORDS]

Produces a flat little-endian memory image as one 64-bit hex word per line,
suitable for $readmemh. Text and data share one address space; the Stage-1
testbench memory serves both the instruction port and the data port from it.

Syntax
------
    label:                      define a label at the current address
    @p3  v.add v1, v2, v3       lane-mask prefix; @!p3 inverts it
    s.addi s1, s2, #4           immediates are '#'-prefixed
    s.ld.w s1, [s2, #16]        memory operands are bracketed
    v.ld.gather v1, [s2, v3]    indexed form

Pseudo-instructions
-------------------
    .li sN, #imm64                  4 words: movz + 3x movk
    .assert_eq sN, #imm64, #id      7 words: fault with TRAP_USER #id on mismatch
    .assert_lane vN, #lane, #imm32, #id
                                    8 words: extract then .assert_eq

    The .assert_* forms clobber s61, s62 and s63. Each expands to a fixed
    number of words so that label addresses are known in pass 1.

Directives
----------
    .text / .data               (informational; both emit into the same image)
    .org  <addr>                set the emit address (must be 8-byte aligned
                                for instructions)
    .align <n>                  align to 2^n bytes
    .equ  NAME, <expr>          define an assembly-time constant
    .word  a, b, ...            emit 32-bit values
    .dword a, b, ...            emit 64-bit values
    .float f, ...               emit binary32 values
    .space <n>                  reserve n zero bytes
    .global NAME                (informational)
"""

import argparse
import re
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnca_isa import (ISA, MAJ, CC, CC_INT, CC_FP, SPECIAL, SIZE, CACHE_HINT,
                      FMINMAX_MODE, RND, MASKABLE_MAJORS, encode,
                      SREG_N, PREG_N, VREG_N, VLEN,
                      AREG_N, BREG_N, DREG_N, WREG_N, MMA_K, MMA_DT, FMT)


class AsmError(Exception):
    def __init__(self, msg, line_no=None, text=None):
        self.msg, self.line_no, self.text = msg, line_no, text
        super().__init__(msg)

    def __str__(self):
        if self.line_no is None:
            return self.msg
        return f"line {self.line_no}: {self.msg}\n    {self.text}"


IDENT = r'[A-Za-z_.$][A-Za-z0-9_.$]*'


# ---------------------------------------------------------------------------
# Operand scanning
# ---------------------------------------------------------------------------

def split_operands(s):
    """Split on commas that are not inside brackets."""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth < 0:
                raise AsmError("unbalanced ']'")
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if depth != 0:
        raise AsmError("unbalanced '['")
    if cur.strip():
        out.append(cur.strip())
    return out


class Assembler:
    def __init__(self):
        self.symbols = {}
        self.records = []      # (addr, nbytes, value_or_thunk, source)
        self.addr = 0
        self.line_no = 0
        self.line_text = ''

    # -- helpers ----------------------------------------------------------
    def err(self, msg):
        raise AsmError(msg, self.line_no, self.line_text)

    def parse_int(self, tok, pass2=True):
        """Evaluate an integer expression: literals, symbols, + and -."""
        tok = tok.strip()
        if not tok:
            self.err("empty expression")
        # Split into terms on top-level +/- (no parentheses supported).
        terms = re.findall(r'[+-]?[^+-]+', tok)
        total = 0
        for t in terms:
            t = t.strip()
            sign = 1
            if t.startswith('+'):
                t = t[1:].strip()
            elif t.startswith('-'):
                sign, t = -1, t[1:].strip()
            total += sign * self.parse_atom(t, pass2)
        return total

    def parse_atom(self, t, pass2):
        tl = t.lower()
        try:
            if tl.startswith('0x'):
                return int(tl, 16)
            if tl.startswith('0b'):
                return int(tl, 2)
            if re.fullmatch(r'\d+', tl):
                return int(tl, 10)
        except ValueError:
            self.err(f"bad numeric literal '{t}'")
        if re.fullmatch(r"'.'", t):
            return ord(t[1])
        if re.fullmatch(IDENT, t):
            if t in self.symbols:
                return self.symbols[t]
            if pass2:
                self.err(f"undefined symbol '{t}'")
            return 0
        self.err(f"cannot parse '{t}'")

    def reg(self, tok, kind):
        tok = tok.strip().lower()
        pfx, lim = {'s': ('s', SREG_N), 'v': ('v', VREG_N), 'p': ('p', PREG_N),
                    'a': ('a', AREG_N), 'b': ('b', BREG_N),
                    'd': ('d', DREG_N), 'w': ('w', WREG_N)}[kind]
        m = re.fullmatch(pfx + r'(\d+)', tok)
        if not m:
            self.err(f"expected a {pfx}-register, got '{tok}'")
        n = int(m.group(1))
        if n >= lim:
            self.err(f"{pfx}{n} out of range (0..{lim - 1})")
        return n

    # -- mnemonic + suffix resolution --------------------------------------
    @staticmethod
    def resolve_mnemonic(tok):
        tok = tok.lower()
        best = None
        for k in ISA:
            if tok == k:
                return k, []
            if tok.startswith(k + '.') and (best is None or len(k) > len(best)):
                best = k
        if best is None:
            return None, None
        return best, tok[len(best) + 1:].split('.')

    # -- main passes -------------------------------------------------------
    def assemble(self, text):
        lines = text.splitlines()
        # Pass 1: fix addresses and collect labels.
        self.addr = 0
        pending = []
        for i, raw in enumerate(lines, 1):
            self.line_no, self.line_text = i, raw
            line = raw.split('//')[0].split(';')[0].split('#!')[0]
            line = line.strip()
            if not line:
                continue
            while True:
                m = re.match(r'^(' + IDENT + r'):\s*(.*)$', line)
                if not m:
                    break
                name = m.group(1)
                if name in self.symbols:
                    self.err(f"duplicate label '{name}'")
                self.symbols[name] = self.addr
                line = m.group(2).strip()
            if not line:
                continue
            size = self.size_of(line)
            pending.append((i, raw, line, self.addr))
            self.addr += size

        # Pass 2: emit.
        for i, raw, line, addr in pending:
            self.line_no, self.line_text = i, raw
            self.addr = addr
            self.emit(line)
        return self.records

    def size_of(self, line):
        """Pass-1 sizing. Directives that move the address handle it here."""
        head = line.split()[0].lower()
        if head.startswith('.'):
            rest = line[len(head):].strip()
            if head in ('.text', '.data', '.global'):
                return 0
            if head == '.equ':
                parts = split_operands(rest)
                if len(parts) != 2:
                    self.err(".equ needs NAME, value")
                self.symbols[parts[0]] = self.parse_int(parts[1], pass2=False)
                return 0
            if head == '.org':
                new = self.parse_int(rest, pass2=False)
                if new < self.addr:
                    self.err(f".org moves backwards ({new} < {self.addr})")
                return new - self.addr
            if head == '.align':
                n = self.parse_int(rest, pass2=False)
                a = 1 << n
                return (-self.addr) % a
            if head == '.space':
                return self.parse_int(rest, pass2=False)
            if head == '.li':
                return 8 * 4
            if head == '.assert_eq':
                return 8 * 7
            if head == '.assert_lane':
                return 8 * 8
            if head == '.word':
                return 4 * len(split_operands(rest))
            if head in ('.dword',):
                return 8 * len(split_operands(rest))
            if head == '.float':
                return 4 * len(split_operands(rest))
            self.err(f"unknown directive '{head}'")
        # Instruction. Every v0.1 instruction is exactly 8 bytes (no ext words).
        if self.addr % 8 != 0:
            self.err(f"instruction at unaligned address {self.addr}")
        return 8

    def emit(self, line):
        head = line.split()[0].lower()
        rest = line[len(head):].strip()
        if head.startswith('.'):
            if head in ('.li', '.assert_eq', '.assert_lane'):
                for sub in self.expand_pseudo(head, split_operands(rest)):
                    self.records.append((self.addr, 8, self.encode_insn(sub), sub))
                    self.addr += 8
                return
            if head in ('.text', '.data', '.global', '.equ', '.org', '.align', '.space'):
                return
            if head == '.word':
                for t in split_operands(rest):
                    self.records.append((self.addr, 4, self.parse_int(t) & 0xffffffff, line))
                    self.addr += 4
                return
            if head == '.dword':
                for t in split_operands(rest):
                    self.records.append((self.addr, 8, self.parse_int(t) & ((1 << 64) - 1), line))
                    self.addr += 8
                return
            if head == '.float':
                for t in split_operands(rest):
                    v = struct.unpack('<I', struct.pack('<f', float(t)))[0]
                    self.records.append((self.addr, 4, v, line))
                    self.addr += 4
                return
            self.err(f"unknown directive '{head}'")

        word = self.encode_insn(line)
        self.records.append((self.addr, 8, word, line))
        self.addr += 8

    # -- instruction encoding ---------------------------------------------
    def encode_insn(self, line):
        pred, pn = 0, 0
        m = re.match(r'^@(!?)(p\d+)\s+(.*)$', line, re.IGNORECASE)
        if m:
            pn = 1 if m.group(1) == '!' else 0
            pred = self.reg(m.group(2), 'p')
            line = m.group(3).strip()

        toks = line.split(None, 1)
        mnem_tok = toks[0]
        operand_text = toks[1] if len(toks) > 1 else ''

        mnem, suffixes = self.resolve_mnemonic(mnem_tok)
        if mnem is None:
            self.err(f"unknown mnemonic '{mnem_tok}'")

        major_name, func, kinds, sclass = ISA[mnem]
        major = MAJ[major_name]

        if (pred or pn) and major not in MASKABLE_MAJORS:
            self.err(f"'{mnem}' is unpredicated (ISA sec 2.2); "
                     f"remove the @p prefix")

        ops = split_operands(operand_text)
        mod, extra = self.apply_suffixes(mnem, sclass, suffixes, ops)
        ops = extra

        f = {'d': 0, 'a': 0, 'b': 0, 'c': 0}
        oi = 0

        def next_op(what):
            nonlocal oi
            if oi >= len(ops):
                self.err(f"'{mnem}' expects an operand ({what}) at position {oi + 1}")
            v = ops[oi]
            oi += 1
            return v

        for kind in kinds:
            if kind in ('sd', 'sa', 'sb', 'sc'):
                f[kind[1]] = self.reg(next_op(kind), 's')
            elif kind in ('vd', 'va', 'vb', 'vc'):
                f[kind[1]] = self.reg(next_op(kind), 'v')
            elif kind in ('pd', 'pa', 'pb'):
                f[kind[1]] = self.reg(next_op(kind), 'p')
            elif kind in ('ad', 'aa', 'ab'):
                f[kind[1]] = self.reg(next_op(kind), 'a')
            elif kind in ('bd', 'ba'):
                f[kind[1]] = self.reg(next_op(kind), 'b')
            elif kind in ('dd', 'da'):
                f[kind[1]] = self.reg(next_op(kind), 'd')
            elif kind in ('wd', 'wa'):
                f[kind[1]] = self.reg(next_op(kind), 'w')
            elif kind == 'cnt2':
                # two 8-bit counts: "#arrivals, #operations"
                n_arr = self.imm(next_op('arrivals'))
                n_ops = self.imm(next_op('operations'))
                for nm, vv in (('arrivals', n_arr), ('operations', n_ops)):
                    if not (0 <= vv <= 255):
                        self.err(f"{nm} {vv} out of range 0..255")
                f['b'], f['c'] = n_arr, n_ops
            elif kind == 'winspec':
                # "#base, #len" packed into imm16 as {base[7:0], len[7:0]}
                base = self.imm(next_op('window base vreg'))
                ln   = self.imm(next_op('window length'))
                if not (0 <= base <= 255):
                    self.err(f"window base {base} out of range 0..255")
                if not (1 <= ln <= 255):
                    self.err(f"window length {ln} out of range 1..255")
                f['b'], f['c'] = base, ln
            elif kind == 'mem_s':
                # "[sN]" -- a plain base register, no displacement
                tok = next_op('memory operand').strip()
                m = re.fullmatch(r'\[\s*(s\d+)\s*\]', tok, re.IGNORECASE)
                if not m:
                    self.err(f"expected '[sN]', got '{tok}'")
                f['a'] = self.reg(m.group(1), 's')
            elif kind == 'rdst':
                tok = next_op('destination')
                if mod & 1:
                    f['d'] = self.reg(tok, 'v')
                else:
                    f['d'] = self.reg(tok, 's')
            elif kind in ('i16s', 'i16z'):
                v = self.imm(next_op('immediate'))
                lo, hi = (-(1 << 15), (1 << 15) - 1) if kind == 'i16s' else (0, (1 << 16) - 1)
                if not (lo <= v <= hi):
                    self.err(f"immediate {v} out of range [{lo},{hi}] for '{mnem}'")
                v &= 0xffff
                f['b'], f['c'] = (v >> 8) & 0xff, v & 0xff
            elif kind == 'i24':
                v = self.imm(next_op('immediate'))
                if not (-(1 << 23) <= v <= (1 << 24) - 1):
                    self.err(f"immediate {v} does not fit in 24 bits")
                v &= 0xffffff
                f['a'], f['b'], f['c'] = (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff
            elif kind in ('lbl16', 'lbl24'):
                target = self.parse_int(next_op('label'))
                if target % 8 != 0:
                    self.err(f"branch target {target:#x} is not 8-byte aligned")
                disp = (target - self.addr) // 8
                bits = 16 if kind == 'lbl16' else 24
                if not (-(1 << (bits - 1)) <= disp < (1 << (bits - 1))):
                    self.err(f"branch displacement {disp} words does not fit in "
                             f"{bits} bits")
                disp &= (1 << bits) - 1
                if bits == 16:
                    f['b'], f['c'] = (disp >> 8) & 0xff, disp & 0xff
                else:
                    f['a'], f['b'], f['c'] = ((disp >> 16) & 0xff, (disp >> 8) & 0xff,
                                              disp & 0xff)
            elif kind == 'spec':
                tok = next_op('special register').strip().lower()
                if tok in SPECIAL:
                    mod |= SPECIAL[tok]
                else:
                    # a raw id, with or without the '#' immediate marker
                    v = self.parse_int(tok[1:] if tok.startswith('#') else tok)
                    if not (0 <= v < 1024):
                        self.err(f"special-register id {v} out of range")
                    mod |= v
            elif kind == 'lane':
                v = self.imm(next_op('lane index'))
                if not (0 <= v < VLEN):
                    self.err(f"lane index {v} out of range (0..{VLEN - 1})")
                mod |= v
            elif kind == 'mem_si':
                base, off = self.mem_si(next_op('memory operand'))
                f['a'] = base
                if not (-(1 << 15) <= off <= (1 << 15) - 1):
                    self.err(f"memory displacement {off} does not fit in 16 bits")
                off &= 0xffff
                f['b'], f['c'] = (off >> 8) & 0xff, off & 0xff
            elif kind == 'mem_sv':
                base, idx = self.mem_sv(next_op('memory operand'))
                f['a'], f['b'] = base, idx
            else:
                self.err(f"internal: unhandled operand kind '{kind}'")

        if oi != len(ops):
            self.err(f"'{mnem}' takes {oi} operand(s), got {len(ops)}")

        # ISA sec 9E / sec 12.7: a transfer takes its coordinates from
        # consecutive scalar registers starting at Sb, so Sb+1 must exist.
        # At s63 the index wraps to s0, which reads as zero and discards
        # writes, so the second coordinate could never be set and the transfer
        # would silently run at 0 on that axis. The decoder refuses this; say
        # so here, where the source line is still in hand.
        if mnem in ('t.ld.async', 't.st.async') and f['b'] == 63:
            self.err("'%s' cannot take its coordinates from s63: the second "
                     "coordinate would be s0, which reads as zero (ISA sec 9E)"
                     % mnem)

        return encode(major, func, d=f['d'], a=f['a'], b=f['b'], c=f['c'],
                      pred=pred, pn=pn, mod=mod)

    SCRATCH_VAL = 's62'
    SCRATCH_CMP = 's63'
    SCRATCH_TMP = 's61'

    def expand_pseudo(self, head, ops):
        """Expand a pseudo-instruction to a fixed-length instruction sequence."""
        if head == '.li':
            if len(ops) != 2:
                self.err(".li takes sN, #imm64")
            return self.li_seq(ops[0], self.imm(ops[1]))

        if head == '.assert_eq':
            if len(ops) != 3:
                self.err(".assert_eq takes sN, #imm64, #id")
            reg, val, tid = ops[0], self.imm(ops[1]), self.imm(ops[2])
            # 4 (li) + cmp + br.z + trap = 7 words; the branch skips the trap.
            skip = self.addr + 8 * 7
            return (self.li_seq(self.SCRATCH_VAL, val) + [
                f"s.cmp.ne {self.SCRATCH_CMP}, {reg}, {self.SCRATCH_VAL}",
                f"br.z {self.SCRATCH_CMP}, {skip}",
                f"trap #{tid}",
            ])

        if head == '.assert_lane':
            if len(ops) != 4:
                self.err(".assert_lane takes vN, #lane, #imm32, #id")
            vreg, lane, val, tid = ops[0], self.imm(ops[1]), self.imm(ops[2]), self.imm(ops[3])
            skip = self.addr + 8 * 8
            return ([f"v.extract {self.SCRATCH_TMP}, {vreg}, #{lane}"] +
                    self.li_seq(self.SCRATCH_VAL, val) + [
                        f"s.cmp.ne {self.SCRATCH_CMP}, {self.SCRATCH_TMP}, {self.SCRATCH_VAL}",
                        f"br.z {self.SCRATCH_CMP}, {skip}",
                        f"trap #{tid}",
                    ])

        self.err(f"internal: unknown pseudo '{head}'")

    @staticmethod
    def li_seq(reg, val):
        """Always four words, so pseudo-op sizing is exact."""
        v = val & ((1 << 64) - 1)
        return [f"s.movz {reg}, #{v & 0xffff}"] + [
            f"s.movk {reg}, #{(v >> (16 * k)) & 0xffff}, lsl #{16 * k}"
            for k in (1, 2, 3)
        ]

    def imm(self, tok):
        tok = tok.strip()
        if not tok.startswith('#'):
            self.err(f"expected a '#'-prefixed immediate, got '{tok}'")
        return self.parse_int(tok[1:])

    def mem_si(self, tok):
        tok = tok.strip()
        m = re.fullmatch(r'\[\s*(s\d+)\s*(?:,\s*(.+?)\s*)?\]', tok, re.IGNORECASE)
        if not m:
            self.err(f"expected '[sN]' or '[sN, #imm]', got '{tok}'")
        base = self.reg(m.group(1), 's')
        off = self.imm(m.group(2)) if m.group(2) else 0
        return base, off

    def mem_sv(self, tok):
        tok = tok.strip()
        m = re.fullmatch(r'\[\s*(s\d+)\s*,\s*(v\d+)\s*\]', tok, re.IGNORECASE)
        if not m:
            self.err(f"expected '[sN, vM]', got '{tok}'")
        return self.reg(m.group(1), 's'), self.reg(m.group(2), 'v')

    def apply_suffixes(self, mnem, sclass, suffixes, ops):
        """Fold mnemonic suffixes into mod. Returns (mod, operands)."""
        mod = 0
        sfx = list(suffixes)

        if sclass is None:
            if sfx:
                self.err(f"'{mnem}' takes no suffix, got '.{'.'.join(sfx)}'")
            return 0, ops

        if sclass in ('cc_int', 'cc_fp'):
            if len(sfx) != 1:
                self.err(f"'{mnem}' needs exactly one condition-code suffix")
            cc = sfx[0]
            allowed = CC_INT if sclass == 'cc_int' else CC_FP
            if cc not in allowed:
                self.err(f"condition code '.{cc}' is not valid for '{mnem}' "
                         f"(allowed: {', '.join(sorted(allowed))})")
            return CC[cc], ops

        if sclass == 'ls':
            if not sfx:
                self.err(f"'{mnem}' needs an access-size suffix (.b/.h/.w/.d)")
            size_tok = sfx.pop(0)
            signed = 0
            if size_tok.startswith('s') and size_tok[1:] in SIZE:
                signed, size_tok = 1, size_tok[1:]
            if size_tok not in SIZE:
                self.err(f"unknown access size '.{size_tok}'")
            mod |= SIZE[size_tok] | (signed << 3)
            for s in sfx:
                if s not in CACHE_HINT:
                    self.err(f"unknown cache hint '.{s}'")
                mod |= CACHE_HINT[s] << 4
            return mod, ops

        if sclass == 'fmm':
            mode = 'ieee'
            for s in sfx:
                if s not in FMINMAX_MODE:
                    self.err(f"unknown FP min/max mode '.{s}' "
                             f"(expected ieee/prop/fast)")
                mode = s
            return FMINMAX_MODE[mode], ops

        if sclass == 'rnd':
            mode = 'rtz'
            for s in sfx:
                if s not in RND:
                    self.err(f"unknown rounding mode '.{s}'")
                mode = s
            return RND[mode], ops

        if sclass == 'ftz':
            for s in sfx:
                if s != 'ftz':
                    self.err(f"'{mnem}' only accepts the '.ftz' suffix")
                mod |= 1 << 8
            return mod, ops

        if sclass == 'redb':
            for s in sfx:
                if s != 'b':
                    self.err(f"'{mnem}' only accepts the '.b' broadcast suffix")
                mod |= 1
            return mod, ops

        if sclass == 'fmt2':
            # exactly ".<src>.<dst>", both from the five-format vocabulary
            if len(sfx) != 2 or sfx[0] not in FMT or sfx[1] not in FMT:
                self.err(f"'{mnem}' needs a source and a destination format "
                         f"suffix, e.g. 'vf.cvt.f32.e4m3' "
                         f"(formats: {', '.join(FMT)})")
            return FMT[sfx[0]] | (FMT[sfx[1]] << 3), ops

        if sclass == 'mma':
            # ".k32", ".bf16" in any order; defaults k16/f16
            kk, dt = 0, 0
            for sf in sfx:
                if sf in MMA_K:
                    kk = MMA_K[sf]
                elif sf in MMA_DT:
                    dt = MMA_DT[sf]
                else:
                    self.err(f"unknown m.mma suffix '.{sf}' "
                             f"(expected k16/k32/k64, f16/bf16)")
            return kk | (dt << 2), ops

        if sclass == 'txf':
            # ".b/.h/.w" element size, optional ".t" transpose
            esz, tr = 1, 0
            for sf in sfx:
                if sf in SIZE:
                    esz = SIZE[sf]
                elif sf == 't':
                    tr = 1
                else:
                    self.err(f"unknown transfer suffix '.{sf}'")
            return esz | (tr << 5), ops

        if sclass == 'lsl':
            # trailing "lsl #n" operand, if present
            if ops and ops[-1].strip().lower().startswith('lsl'):
                t = ops[-1].strip()[3:].strip()
                n = self.imm(t)
                if n not in (0, 16, 32, 48):
                    self.err(f"'lsl #{n}' must be 0, 16, 32 or 48")
                mod |= n // 16
                ops = ops[:-1]
            if sfx:
                self.err(f"'{mnem}' takes no suffix")
            return mod, ops

        self.err(f"internal: unknown suffix class '{sclass}'")


def build_image(records, size_words=None):
    """Lay records out into a flat byte image, then pack into 64-bit words."""
    if not records:
        return []
    end = max(a + n for a, n, _, _ in records)
    mem = bytearray((end + 7) // 8 * 8)
    for addr, nbytes, value, _ in records:
        mem[addr:addr + nbytes] = value.to_bytes(nbytes, 'little')
    words = [int.from_bytes(mem[i:i + 8], 'little') for i in range(0, len(mem), 8)]
    if size_words is not None:
        if len(words) > size_words:
            raise AsmError(f"image needs {len(words)} words but --size is {size_words}")
        words += [0] * (size_words - len(words))
    return words


def main():
    ap = argparse.ArgumentParser(description="TNCA ISA v0.1 assembler")
    ap.add_argument('source')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--lst', help="write a human-readable listing")
    ap.add_argument('--size', type=int, help="pad the image to this many 64-bit words")
    args = ap.parse_args()

    with open(args.source) as fh:
        text = fh.read()

    asm = Assembler()
    try:
        records = asm.assemble(text)
        words = build_image(records, args.size)
    except AsmError as e:
        print(f"{args.source}: {e}", file=sys.stderr)
        return 1

    with open(args.output, 'w') as fh:
        for w in words:
            fh.write(f"{w:016x}\n")

    if args.lst:
        with open(args.lst, 'w') as fh:
            fh.write(f"// listing for {args.source}\n")
            for addr, nbytes, value, src in records:
                fh.write(f"{addr:08x}  {value:0{nbytes * 2}x}  {src}\n")
            fh.write("\n// symbols\n")
            for k, v in sorted(asm.symbols.items(), key=lambda kv: kv[1]):
                fh.write(f"{v:08x}  {k}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
