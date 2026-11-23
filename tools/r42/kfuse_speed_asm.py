#!/usr/bin/env python3
"""Instruction census of compiled TNCA kernels (domain: static, from the compiler's .s dump).

    python3 tools/r42/kfuse_speed_asm.py KERNEL.s [...]

For each kernel: instruction count, the matrix commands, and for every backward-branch loop that contains a matrix command and
no inner matrix loop (the innermost matrix loops) the instruction count per iteration by class: scalar/branch bookkeeping
(`s.*`, `.li`, `br.*`: what a hardware loop, hoisting and strength reduction can remove), window/barrier/transfer commands,
matrix commands, vector/other. These classes are the ones the kfuse-speed model's `issue_free_mode` applies to.
"""
import re
import sys
from collections import Counter
from pathlib import Path

LABEL = re.compile(r'^([A-Za-z_][\w.]*):')


def parse(path):
    insts, labels = [], {}
    for line in Path(path).read_text().splitlines():
        s = line.split('//')[0].strip()
        if not s:
            continue
        m = LABEL.match(s)
        if m:
            labels[m.group(1)] = len(insts)
            s = s[m.end():].strip()
            if not s:
                continue
        mn = s.split()[0]
        insts.append((mn, s))
    return insts, labels


def klass(mn):
    if mn.startswith('m.mma'):
        return 'mma'
    if mn.startswith('m.') or mn.startswith('a.'):
        return 'matrix-other'
    if mn.startswith(('s.', '.li')) or mn.startswith(('br', 'jmp', 'call', 'ret', 'exit', 'nop')):
        return 'scalar-branch'
    if mn.startswith(('mb.', 'w.def', 'fence', 'td.', 't.')):
        return 'window-barrier-transfer'
    return 'vector-other'


def loops(insts, labels):
    out = []
    for i, (mn, s) in enumerate(insts):
        if mn.startswith('br'):
            tgt = s.replace(',', ' ').split()[-1]
            if tgt in labels and labels[tgt] <= i:
                out.append((labels[tgt], i))
    return out


def main():
    for path in sys.argv[1:]:
        insts, labels = parse(path)
        print(f'== {Path(path).name}: {len(insts)} instructions')
        mm = [i for i, (mn, _) in enumerate(insts) if mn.startswith('m.mma')]
        print('   matrix commands (static):', Counter(insts[i][0] for i in mm))
        sp = [(a, b) for a, b in loops(insts, labels) if any(a <= i <= b for i in mm)]
        inner = [s for s in sp if not any(o != s and o[0] >= s[0] and o[1] <= s[1] and (o[0] > s[0] or o[1] < s[1]) for o in sp)]
        for a, b in inner:
            c = Counter(klass(insts[i][0]) for i in range(a, b + 1))
            n = sum(c.values())
            print(f'   innermost matrix loop [{a}..{b}]: {n} instructions/iteration, ' +
                  ', '.join(f'{k} {v}' for k, v in sorted(c.items())))
            print('      mnemonics:', dict(Counter(insts[i][0] for i in range(a, b + 1)).most_common(12)))


if __name__ == '__main__':
    main()
