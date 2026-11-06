#!/usr/bin/env python3
"""Instructions per iteration of the innermost loop of compiled TNCA programs.

  loop_issue_count.py DIR_OR_FILE [...] [--mnemonics]

For each `*.s` under the arguments, counts the instruction lines (comments,
directives and labels excluded) between the last `for_N:` label that precedes
the first `m.mma.*` and its `endfor_(N+1):` label, and sorts them by what they
are: matrix and accumulator, vector load/store, vector floating point, vector
integer/move/shuffle, scalar/predicate/control. The same classes as
evidence/r41/hbm_bw/BALANCE.md section 5, counted from the text a kernel was
compiled to, so a lowering change can be judged by one number per iteration.
"""
import re
import sys
from collections import Counter
from pathlib import Path

LABEL = re.compile(r'^(\w+):')


def klass(m):
    if m.startswith(('m.', 'a.', 'mb.')):
        return 'matrix/accumulator/barrier'
    if m.startswith(('v.ld', 'v.st', 't.', 'td.', 'w.')):
        return 'vector or transfer load/store'
    if m.startswith(('vf.', 'v.exp2', 'v.rcp', 'v.rsqrt', 'v.log2')):
        return 'vector fp'
    if m.startswith('v.'):
        return 'vector int/move/shuffle'
    return 'scalar/predicate/control'


def instructions(path):
    """[(label_before, mnemonic)] for every instruction, plus label positions."""
    out, labels = [], {}
    for raw in Path(path).read_text(errors='replace').splitlines():
        line = raw.split('//', 1)[0].strip()
        if not line:
            continue
        m = LABEL.match(line)
        if m:
            labels[m.group(1)] = len(out)
            continue
        if line.startswith(('.', '#')):
            continue
        toks = line.split()
        if toks[0].startswith('@'):
            toks = toks[1:]
        out.append(toks[0])
    return out, labels


def main(argv):
    show = '--mnemonics' in argv
    paths = []
    for a in argv:
        if a.startswith('--'):
            continue
        p = Path(a)
        paths += sorted(p.rglob('*.s')) if p.is_dir() else [p]
    for p in paths:
        ins, labels = instructions(p)
        first_mma = next((i for i, m in enumerate(ins) if m.startswith('m.mma')), None)
        if first_mma is None:
            continue
        fors = sorted((pos, n) for n, pos in labels.items() if re.fullmatch(r'for_\d+', n) and pos <= first_mma)
        if not fors:
            continue
        pos, name = fors[-1]
        end = f"endfor_{int(name.split('_')[1]) + 1}"
        if end not in labels:
            continue
        body = ins[pos:labels[end]]
        by = Counter(klass(m) for m in body)
        macs = sum(1 for m in body if m.startswith('m.mma'))
        print(f"{p.parent.name}: loop {name}..{end}: {len(body)} instructions, {macs} matrix commands")
        for k, v in by.most_common():
            print(f"    {k:34s} {v:6d} {100 * v / len(body):5.1f} %")
        if show:
            for m, v in Counter(body).most_common(25):
                print(f"      {m:20s} {v}")


if __name__ == '__main__':
    main(sys.argv[1:])
